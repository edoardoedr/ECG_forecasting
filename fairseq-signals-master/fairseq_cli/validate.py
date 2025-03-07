#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from itertools import chain
import pprint

from omegaconf import DictConfig

import torch

from fairseq_signals import distributed_utils
from fairseq_signals.utils import checkpoint_utils, options, utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.logging import metrics, progress_bar
from fairseq_signals.utils.store import STORE_KEYS, initialize_stores_to_criterion

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")

def main(cfg: DictConfig, override_args=None):
    utils.import_user_module(cfg.common)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)
    
    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
    else:
        overrides = {}

    if "task" in overrides:
        overrides["task"]["data"] = cfg.task.data
    else:
        overrides["task"] = {"data": cfg.task.data}

    checkpoint_overrides = eval(getattr(cfg.common_eval, "model_overrides", "{}"))
    # force not to load pretrained checkpoint when building a new model instance
    checkpoint_overrides.update({"model_path": None, "no_pretrained_weights": True})
    overrides.update(checkpoint_overrides)

    # Process 'extract' argument
    model_overrides = {}
    if cfg.common_eval.extract:
        if cfg.common_eval.results_path is None:
            raise ValueError(
                "common_eval.results_path must be set with common_eval.extract."
            )
        # Convert to a list
        cfg.common_eval.extract = [
            item.strip() for item in cfg.common_eval.extract
        ]

        # If extracting saliency, update the model definition 
        if "saliency" in cfg.common_eval.extract:
            model_overrides["saliency"] = True

    # Load model
    logger.info(f"loading model from {cfg.common_eval.path}")
    model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        cfg.common_eval.path,
        checkpoint_overrides=overrides,
        model_overrides=model_overrides,
        suffix=cfg.checkpoint.checkpoint_suffix
    )

    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad)
        )
    )

    # Move model to GPU
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(pprint.pformat(dict(saved_cfg)))

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    def _fp_convert_sample(sample):
        def apply_half(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.half)
            return t
            # return t.to(dtype = torch.half)
        
        def apply_float(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.float)
            return t
        
        if use_fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        else:
            sample = utils.apply_to_sample(apply_float, sample)
        
        return sample

    for subset in cfg.dataset.valid_subset.split(","):
        subset = subset.strip()
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=cfg.task, shuffle=False)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)
        logger.info("begin validation on {} subset".format(subset))

        # Initialize data iterator
        batch_iterator = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_signals=cfg.dataset.batch_size,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size
        )
        itr = batch_iterator.next_epoch_itr(shuffle=False)
        _dummy_batch = batch_iterator.first_batch
        is_dummy_batch = False

        # Initialize stores for tensor extraction
        if cfg.common_eval.extract is not None:
            dummies = {}
            with torch.no_grad():
                # Extract dummy batch
                dummy_batch = utils.move_to_cuda(_dummy_batch) if use_cuda else _dummy_batch
                dummy_batch = _fp_convert_sample(dummy_batch)

                model.eval()

                if any(
                    obj in cfg.common_eval.extract for obj in \
                    ['output', 'encoder_out', 'saliency']
                ):
                    net_output = model(**dummy_batch["net_input"])

                if 'output' in cfg.common_eval.extract:
                    dummies['output'] = model.get_logits(net_output)
                    dummies['target'] = model.get_targets(dummy_batch, net_output)

                if 'encoder_out' in cfg.common_eval.extract:
                    if 'encoder_out' not in net_output:
                        raise ValueError(
                            'Encoder extraction is not supported for this model.'
                        )
                    dummies['encoder_out'] = net_output['encoder_out']
                    if 'padding_mask' in net_output:
                        dummies['padding_mask'] = net_output['padding_mask']

                if 'saliency' in cfg.common_eval.extract:
                    if 'saliency' not in net_output:
                        raise ValueError(
                            'Saliency extraction is not supported for this model.'
                        )
                    dummies['saliency'] = net_output['saliency']

            underlying_criterions = getattr(
                criterion,
                "underlying_criterions",
                [criterion],
            )

            # Define the stores inside the criterion(s)
            if isinstance(list(dummies.values())[0], list):
                # If the tensors are multiple for the purpose of composite criterion
                for store_key, tensor_value in dummies.items():
                    store_info = {}
                    for i, tensor in enumerate(tensor_value):
                        shape = (len(dataset),) + tuple(tensor.shape[1:])
                        store_info[store_key] = \
                            (f'{STORE_KEYS[store_key]}_{subset}_{i}.npy', shape)

                    initialize_stores_to_criterion(
                        dtype="float16" if cfg.common.fp16 else "float32",
                        criterion=underlying_criterions[i],
                        store_info=store_info,
                        save_directory=cfg.common_eval.results_path,
                    )
            else:
                # If the tensors are single, i.e., not a composite criterion
                store_info = {}
                for store_key, tensor in dummies.items():
                    if tensor is None:
                        continue

                    shape = (len(dataset),) + tuple(tensor.shape[1:])
                    store_info[store_key] = (
                        f'{STORE_KEYS[store_key]}_{subset}.npy',
                        shape,
                    )

                initialize_stores_to_criterion(
                    dtype="float16" if cfg.common.fp16 else "float32",
                    criterion=underlying_criterions[0],
                    store_info=store_info,
                    save_directory=cfg.common_eval.results_path,
                )

        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_file = cfg.common.log_file,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=None,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_entity=(
                cfg.common.wandb_entity
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
            azureml_logging=False
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            with torch.no_grad():
                if sample is None or len(sample) == 0:
                    is_dummy_batch = True
                    sample = _dummy_batch

                sample = utils.move_to_cuda(sample) if use_cuda else sample
                sample = _fp_convert_sample(sample)

                _loss, _sample_size, log_output = task.valid_step(
                    sample,
                    model,
                    criterion,
                    subset,
                )
                if not is_dummy_batch:
                    log_outputs.append(log_output)
                is_dummy_batch = False

        # Close any initialized stores
        criterion.close_stores()

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group()
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate(new_root=True) as agg:
            task.reduce_metrics(log_outputs, criterion)
            del log_outputs
            log_output = agg.get_smoothed_values()

        if hasattr(task, "post_validate"):
            task.post_validate(model, log_output, agg, num_updates=0)
        
        progress.print(log_output, tag=subset, step=None)

def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults = True
    )

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )

if __name__ == "__main__":
    cli_main()