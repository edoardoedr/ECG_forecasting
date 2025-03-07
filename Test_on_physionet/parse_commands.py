import os
import scipy.io
import subprocess

root = os.getcwd()



# PHYSIONET_ROOT=".../physionet.org/files/challenge-2021/1.0.3/training"
# EVALUATION_ROOT=".../evaluation-2021"
root_1 ='physionet.org/files/challenge-2021/1.0.3/training'
root_2 = 'Test_on_physionet/processed_root/evaluation-2021'

PHYSIONET_ROOT= os.path.join(root,root_1)
EVALUATION_ROOT= os.path.join(root,root_2)
# print("Il path è:" , PHYSIONET_ROOT)

# Add subprocess to execute the specified command
command = [
    "python", "Test_on_physionet/physionet2021_records.py",
    "--processed_root", os.path.join(root, "Test_on_physionet/processed_root/physionet2021"),
    "--raw_root", PHYSIONET_ROOT
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

command = [
    "python", "Test_on_physionet/physionet2021_signals.py", "--help"
]
print("Eseguendo il comando:", " ".join(command))
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stdout.decode())
print(stderr.decode())

command = [
        "python", "Test_on_physionet/physionet2021_signals.py",
        "--processed_root", "Test_on_physionet/processed_root/physionet2021",
        "--raw_root", PHYSIONET_ROOT,
        "--manifest_file", "Test_on_physionet/processed_root/manifest.csv"
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout.decode())
print(stderr.decode())

# -------------------------------------------------------------


command = [
        "python", "Test_on_physionet/splits.py",
        "--strategy", "random",
        "--processed_root", "Test_on_physionet/processed_root/physionet2021",
        "--filter_cols", "nan_any,constant_leads_any",
        "--dataset_subset", "cpsc_2018, cpsc_2018_extra, georgia, ptb-xl, chapman_shaoxing, ningbo" # Excludes 'ptb' and 'st_petersburg_incart'

]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout.decode())
print(stderr.decode())

os.makedirs(EVALUATION_ROOT, exist_ok=True)
os.makedirs("Test_on_physionet/processed_root/physionet2021/labels", exist_ok=True)

command = [
        # "mkdir", EVALUATION_ROOT,
        # "mkdir", "Test_on_physionet/processed_root/physionet2021/labels",
        "python", "Test_on_physionet/physionet2021_labels.py",
        "--processed_root", "Test_on_physionet/processed_root/physionet2021",
        "--weights_path", "Test_on_physionet/processed_root/evaluation-2021/weights.csv",
        "--weight_abbrev_path", "Test_on_physionet/processed_root/evaluation-2021/weights_abbreviations.csv" 
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout.decode())
print(stderr.decode())


command = [
        "python", "Test_on_physionet/prepare_clf_labels.py",
        "--output_dir", "Test_on_physionet/processed_root/physionet2021/labels",
        "--labels", "Test_on_physionet/processed_root/physionet2021/labels/labels.csv",
        "--meta_splits", "Test_on_physionet/processed_root/physionet2021/meta_split.csv"
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout.decode())
print(stderr.decode())


print("Finished processing Physionet 2021 dataset")




'''
TO DO:
In inference.py: 
        -Controllare la funzione task.load_dataset (non mi ricordo se il nome preciso è questo)
        e controllare le varie funzioni che vengono chiamate all'interno di questa.

        -Capire pure a cosa serve model.half().

 
'''