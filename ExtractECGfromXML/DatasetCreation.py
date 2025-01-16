import pickle
import os
from .ExportXMLECG import ExportXLMECG, Config
import matplotlib.pyplot as plt

class DatasetCheckerCreator:
    def __init__(self, 
                 dataset_path, 
                 n_seconds,
                 debug = False):
        
        self.dataset_path = os.path.abspath(dataset_path)
        self.n_seconds = n_seconds
        self.debug = debug
        assert self._check_structure()
        self.stemi_path, self.nstemi_path = self.join_data_path()
        self.stemi_data_path, self.nstemi_data_path = self._create_new_data_directories()
        
        
    def _create_new_data_directories(self):
        stemi_data_path = self.stemi_path + '_data'
        nstemi_data_path = self.nstemi_path + '_data'
        
        os.makedirs(stemi_data_path, exist_ok=True)
        os.makedirs(nstemi_data_path, exist_ok=True)
        
        return stemi_data_path, nstemi_data_path
    
    def save_info(self):
        stemi_count = len(os.listdir(self.stemi_data_path))
        nstemi_count = len(os.listdir(self.nstemi_data_path))
        info = (
            f"Number of STEMI ECGs: {stemi_count}\n"
            f"Number of NSTEMI ECGs: {nstemi_count}\n"
            )
        info_path = os.path.join(self.dataset_path, 'info.txt')
        with open(info_path, 'w') as file:
            file.write(info)
        print(f"Info saved to {info_path}")
        
    def _check_structure(self):
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"The path {self.dataset_path} is not a valid directory.")
        
        subdirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if len(subdirs) != 2:
            raise ValueError("The dataset directory must contain exactly two subdirectories.")
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.dataset_path, subdir)
            pdf_files = [f for f in os.listdir(subdir_path) if f.endswith('.xml')]
            
            if not pdf_files:
                raise ValueError(f"The subdirectory {subdir} does not contain any xml files.")
        
        return True
    
    def join_data_path(self):
        
        subdirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        stemi_path = None
        nstemi_path = None

        for subdir in subdirs:
            if 'nstemi' in subdir.lower():
                nstemi_path = os.path.join(self.dataset_path, subdir)
            elif 'stemi' in subdir.lower():
                stemi_path = os.path.join(self.dataset_path, subdir)

        if not stemi_path or not nstemi_path:
            raise ValueError("Both 'stemi' and 'nstemi' subdirectories must be present in the dataset.")

        return stemi_path, nstemi_path
    
    def save_signals(self, data: dict, file_path: str) -> None:

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def process_derivation(self):
        
        stemi_image_paths = [os.path.join(self.stemi_path, img_pth) for img_pth in os.listdir(self.stemi_path) if img_pth.endswith('.xml')]
        nstemi_image_paths = [os.path.join(self.nstemi_path, img_pth) for img_pth in os.listdir(self.nstemi_path) if img_pth.endswith('.xml')]
        stemi_data_path = [os.path.join(self.stemi_data_path, os.path.splitext(os.path.basename(img_pth))[0] + '.pkl') for img_pth in stemi_image_paths]
        nstemi_data_path = [os.path.join(self.nstemi_data_path, os.path.splitext(os.path.basename(img_pth))[0] + '.pkl') for img_pth in nstemi_image_paths]
        
        for stemi_path, stemi_data_path in zip(stemi_image_paths, stemi_data_path):
            stemi_ecg = ExportXLMECG(stemi_path)
            stemi_signals = stemi_ecg.get_ecgs_seconds(self.n_seconds)
            if self.debug:
               stemi_ecg.plot_ecgs(self.n_seconds, stemi_data_path.replace('.pkl', '.png')) 
            self.save_signals(stemi_signals, stemi_data_path)
            
        for nstemi_path, nstemi_data_path in zip(nstemi_image_paths, nstemi_data_path):
            nstemi_ecg = ExportXLMECG(nstemi_path)
            nstemi_signals = nstemi_ecg.get_ecgs_seconds(self.n_seconds)
            if self.debug:
                nstemi_ecg.plot_ecgs(self.n_seconds, nstemi_data_path.replace('.pkl', '.png'))
            self.save_signals(nstemi_signals, nstemi_data_path)
        
        self.save_info()
                      
    
if __name__ == '__main__':
    config = Config('config.yaml')
    
    dataset_checker_creator = DatasetCheckerCreator(config.data_path, 
                                                    config.n_seconds,
                                                    config.debug)
    
    dataset_checker_creator.process_derivation()
    print("Signals saved successfully!")
            
        
        
        

