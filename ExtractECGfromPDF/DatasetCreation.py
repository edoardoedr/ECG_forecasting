import pickle
from functools import partial
import fitz
import io
import os
from PIL import Image
from .DatasetCreationTools import PreprocessImage, SignalExtractor, Background, Config
import matplotlib.pyplot as plt

class DatasetCheckerCreator:
    def __init__(self, 
                 dataset_path, 
                 divisore_threshold = 4, 
                 soglia = 128, 
                 raggio = 1,
                 soglia_distanza = 10,
                 interpolate = False,
                 num_points = None,
                 coordinates_pdf = (10, 690, 2540, 1900),
                 debug = False):
        
        self.dataset_path = os.path.abspath(dataset_path)
        self.divisore_threshold = divisore_threshold
        self.soglia = soglia
        self.raggio = raggio
        self.soglia_distanza = soglia_distanza
        self.interpolate = interpolate
        self.num_points = num_points
        self.coordinates_pdf = coordinates_pdf
        self.get_image_u_coordinates = partial(self._get_image_from_pdf, coordinates=self.coordinates_pdf)
        self.debug = debug
        assert self._check_structure()
        self.background_image = self._check_or_create_background()
        self.stemi_path, self.nstemi_path = self.join_data_path()
        self.stemi_data_path, self.nstemi_data_path = self._create_new_data_directories()
        

    def _get_image_from_pdf(self, document_path, coordinates):
        
        pdf_document = fitz.open(document_path)
        page = pdf_document[0]
        image_info = page.get_images(full=True)[0]
        xref = image_info[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image_cropped = pil_image.crop(coordinates)
        pdf_document.close()
        # Converti l'immagine in scala di grigi
        pil_image_grayscale = pil_image_cropped.convert('L')

        return pil_image_grayscale
        
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
            f"Parameters used:\n"
            f"  divisore_threshold: {self.divisore_threshold}\n"
            f"  soglia: {self.soglia}\n"
            f"  raggio: {self.raggio}\n"
            f"  soglia_distanza: {self.soglia_distanza}\n"
            f"  interpolate: {self.interpolate}\n"
            f"  num_points: {self.num_points}\n"
            f"  coordinates_pdf: {self.coordinates_pdf}\n"
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
            pdf_files = [f for f in os.listdir(subdir_path) if f.endswith('.pdf')]
            
            if not pdf_files:
                raise ValueError(f"The subdirectory {subdir} does not contain any PDF files.")
        
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

    def _check_or_create_background(self):
        background_image_path = os.path.join(self.dataset_path, 'background.tif')
        
        if not os.path.isfile(background_image_path):
            print("Background image not found. Creating background image...")
            background_creator = Background(self.dataset_path, self.get_image_u_coordinates ,self.divisore_threshold)
            background_image = background_creator.get_background()
            background_image_pil = Image.fromarray(background_image)
            background_image_pil.save(background_image_path)
            print(f"Background image saved at {background_image_path}")
            
        else:
            print("Background image already exists.")
            background_image = Image.open(background_image_path)
        
        return background_image
    
    def save_signals(self, data: dict, file_path: str) -> None:

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            
    def plot_and_compare(self, ecg_signal, ecg_image, save_path):
        
        num_keys = 12 # 12 derivazioni
        fig, axes = plt.subplots(num_keys, 1, figsize=(10, 5 * num_keys))
        
        for ax, key in zip(axes, ecg_signal.keys()):
            x_vals, y_vals = ecg_signal[key]
            ax.imshow(ecg_image[key], cmap='gray', aspect='auto')
            ax.set_title(f'Segnali Estratti Sovrapposti all\'Immagine ECG Originale - {key}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.plot(x_vals, y_vals, '-', color='red', label=f'{key}', linewidth=1, alpha=1, markersize=4)
            ax.legend()
            ax.grid(False)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def process_derivation(self):
        
        stemi_image_paths = [os.path.join(self.stemi_path, img_pth) for img_pth in os.listdir(self.stemi_path) if img_pth.endswith('.pdf')]
        nstemi_image_paths = [os.path.join(self.nstemi_path, img_pth) for img_pth in os.listdir(self.nstemi_path) if img_pth.endswith('.pdf')]
        stemi_data_path = [os.path.join(self.stemi_data_path, os.path.splitext(os.path.basename(img_pth))[0] + '.pkl') for img_pth in stemi_image_paths]
        nstemi_data_path = [os.path.join(self.nstemi_data_path, os.path.splitext(os.path.basename(img_pth))[0] + '.pkl') for img_pth in nstemi_image_paths]
        
        for stemi_path, stemi_data_path in zip(stemi_image_paths, stemi_data_path):
            stemi_image = PreprocessImage(stemi_path, self.get_image_u_coordinates ,self.background_image, self.soglia, self.raggio, self.debug)
            if self.debug:
                stemi_regions, stemi_regions_bg = stemi_image.get_derivation_images()
            else:
                stemi_regions = stemi_image.get_derivation_images()
            signal_extractor_stemi = SignalExtractor(stemi_regions, self.soglia_distanza, self.interpolate, self.num_points)
            stemi_signals = signal_extractor_stemi.extract_signals()
            if self.debug:
                self.plot_and_compare(stemi_signals, stemi_regions_bg, stemi_data_path.replace('.pkl', '.png'))
            self.save_signals(stemi_signals, stemi_data_path)
            
        for nstemi_path, nstemi_data_path in zip(nstemi_image_paths, nstemi_data_path):
            nstemi_image = PreprocessImage(nstemi_path, self.get_image_u_coordinates, self.background_image, self.soglia, self.raggio, self.debug)
            if self.debug:
                nstemi_regions, nstemi_regions_bg = stemi_image.get_derivation_images()
            else:
                nstemi_regions = nstemi_image.get_derivation_images()
            signal_extractor_nstemi = SignalExtractor(nstemi_regions, self.soglia_distanza, self.interpolate, self.num_points)
            nstemi_signals = signal_extractor_nstemi.extract_signals()
            if self.debug:
                self.plot_and_compare(nstemi_signals, nstemi_regions_bg, nstemi_data_path.replace('.pkl', '.png'))  
            self.save_signals(nstemi_signals, nstemi_data_path)
        
        self.save_info()
                      
    
if __name__ == '__main__':
    config = Config('config.yaml')
    
    dataset_checker_creator = DatasetCheckerCreator(config.data_path, 
                                                    config.divisore_threshold, 
                                                    config.soglia, 
                                                    config.raggio, 
                                                    config.soglia_distanza, 
                                                    config.interpolate, 
                                                    config.num_points, 
                                                    config.coordinates_pdf,
                                                    config.debug)
    
    dataset_checker_creator.process_derivation()
    print("Signals saved successfully!")
            
        
        
        

