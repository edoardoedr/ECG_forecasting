import contextlib
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn

from fairseq_signals.models.resnet import VanillaResnetConfig, VanillaResnetModel
from fairseq_signals.models.lstm import LanguageLSTMConfig, LanguageLSTMModel
from fairseq_signals.models.pretraining_model import PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task

logger = logging.getLogger(__name__)

@dataclass
class ResnetLSTMConfig(VanillaResnetConfig, LanguageLSTMConfig):
    """
    Resnet (for ECG) with LSTM (for question) model configs.
    This is a modified version of (https://arxiv.org/pdf/1505.00468.pdf) which utilizes Resnet
    instead of VGG-16
    """
    apply_norm: bool = field(
        default=False,
        metadata={
            "help": "whether to apply l2 normalization to the output of the last hidden layer of resnet"
        }
    )
    final_dim: int = field(
        default=1024,
        metadata={
            "help": "project final embeddings of resnet and lstm to this many dimensions"
        }
    )

class ResnetLSTMModel(PretrainingModel):
    """
    Resnet (for ECG) with LSTM (for question) model implementation.
    This is a modified version of (https://arxiv.org/pdf/1505.00468.pdf) which utilizes Resnet
    instead of VGG-16
    """
    def __init__(self, cfg: ResnetLSTMConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.apply_norm = cfg.apply_norm

        self.resnet = VanillaResnetModel(cfg)
        self.resnet_final_proj = nn.Linear(512 * self.resnet.block.expansion, cfg.final_dim)

        self.lstm = LanguageLSTMModel(cfg)
        self.lstm_final_proj = nn.Linear(2 * cfg.lstm_num_layer * cfg.lstm_embed_dim, cfg.final_dim)

        self.pad_token = cfg.pad_token
        self.sep_token = cfg.sep_token
        self.special_tokens = [cfg.pad_token, cfg.sep_token]
        self.vocab_size = cfg.vocab_size

        self.num_updates = 0
    
    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrate a (possibly old) state dict for new versions."""
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def forward(
        self,
        ecg,
        text,
        text_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        ecg_features = self.resnet(ecg)["x"]
        ecg_features = self.resnet_final_proj(ecg_features)
        if ecg_2 is not None:
            ecg_features_2 = self.resnet(ecg_2)["x"]
            ecg_features_2 = self.resnet_final_proj(ecg_features_2)
            ecg_features_2[
                torch.where((ecg_2_padding_mask == 1).all(dim=2).all(dim=1))
            ] = 0
            ecg_features = ecg_features + ecg_features_2

        if text_padding_mask is not None and not text_padding_mask.any():
            text_padding_mask = None

        text_features = self.lstm(text, padding_mask=text_padding_mask)["x"]
        text_features = self.lstm_final_proj(text_features)

        # point-wise multiplication
        x = ecg_features * text_features

        return {"x": x}
    
    def extract_features(self, ecg, text, text_padding_mask, ecg_2, ecg_2_padding_mask):
        return self.forward(ecg, text, text_padding_mask, ecg_2, ecg_2_padding_mask)
    
    def get_logits(self, net_output, **kwargs):
        raise NotImplementedError()
    
    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ResnetLSTMConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.ResnetLSTMModel` from a pre-trained model checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ResnetLSTMConfig): cfg to override some arguments of pre-trained model
        """

        arg_overrides = {
            "dropout": cfg.dropout,
            "load_bert_embedding": False
        }
        
        return super().from_pretrained(model_path, cfg, arg_overrides, **kwargs)

@dataclass
class ResnetLSTMFinetuningConfig(FinetuningConfig, ResnetLSTMConfig):
    # overriding arguments
    pass

class ResnetLSTMFinetuningModel(FinetuningModel):
    def __init__(self, cfg: ResnetLSTMFinetuningConfig, encoder: ResnetLSTMModel):
        super().__init__(cfg, encoder)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg: ResnetLSTMFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = ResnetLSTMModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = ResnetLSTMModel(cfg)
        
        return cls(cfg, encoder)

    def forward(
        self,
        ecg,
        text,
        text_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        args = {
            "ecg": ecg,
            "text": text,
            "text_padding_mask": text_padding_mask,
            "ecg_2": ecg_2,
            "ecg_2_padding_mask": ecg_2_padding_mask
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)

        return res