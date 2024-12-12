import os
import fitz  # PyMuPDF
from PIL import Image
import io
import random
import numpy as np


def get_image_from_pdf(document_path):
    pdf_document = fitz.open(document_path)
    page = pdf_document[0]
    image_info = page.get_images(full=True)[0]
    xref = image_info[0]
    base_image = pdf_document.extract_image(xref)
    image_bytes = base_image["image"]
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image_cropped = pil_image.crop((10, 690, 2540, 1900))
    pdf_document.close()
    # Converti l'immagine in scala di grigi
    pil_image_grayscale = pil_image_cropped.convert('L')

    return pil_image_grayscale


class Background:
    def __init__(self, path):
        self.base_path = os.path.abspath(path)
        self.tolleranza = 50000
        self.get_image_from_pdf = get_image_from_pdf
        self.random_pdf_paths = self.get_pdf_paths()
        
    def get_pdf_paths(self):
        subdirs = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        pdf_paths = []
            
        for subdir in subdirs:
            subdir_path = os.path.join(self.base_path, subdir)
            pdf_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.pdf')]
            pdf_paths.extend(pdf_files)
            
        sample_size = max(1, len(pdf_paths) // 10)
        sampled_paths = random.sample(pdf_paths, sample_size)
            
        return sampled_paths
    
    def _get_similarity(self, image1, image2):
        
        array_img1 = np.array(image1)
        array_img2 = np.array(image2)

        differenza = np.linalg.norm(array_img1 - array_img2, axis=0) > self.tolleranza
        nuova_immagine = np.where(differenza, 255, array_img1)

        return nuova_immagine

    def iter_similarity(self):
        
        list_images = [self.get_image_from_pdf(pdf_path) for pdf_path in self.random_pdf_paths]
        
        similarities = []
        for i in range(len(list_images)):
            for j in range(i + 1, len(list_images)):
                similarity = self._get_similarity(list_images[i], list_images[j])
                similarities.append(similarity)
    
        return similarities
    
    def calcola_pixel_comuni(self, array_immagini):
        
        stack = np.stack(array_immagini, axis=-1)
        pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)

        stack_flat = stack.reshape(-1, stack.shape[-1])
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=stack_flat)
        pixel_comuni = mode.reshape(array_immagini[0].shape)
        
        return pixel_comuni
    
    def get_background(self):
        
        array_immagini = self.iter_similarity()
        sfondo = self.calcola_pixel_comuni(array_immagini)
        
        return sfondo.astype(np.uint8)
    
class PreprocessImage:
    def __init__(self, pdf_path, background_image):
        self.image_ecg = get_image_from_pdf(pdf_path)
        self.background = background_image
        self.soglia = 127
        self.tolleranza = 0

    def remove_background(self):

        image_np = np.array(self.image_ecg)
        background_np = np.array(self.background)

        image_bin = (image_np > self.soglia).astype(np.uint8)
        background_bin = (background_np > self.soglia).astype(np.uint8)
    
        diff_image = np.abs(image_bin - background_bin) > self.tolleranza
        np_without_background = np.where(diff_image, image_np, 255)
        
        return Image.fromarray(np_without_background)
    
    def get_image_derivation(self) -> dict:
            
            image_without_background = self.remove_background()
            altezza = image_without_background.size[1]//4
            
            crop_coordinates = {
                "R1) I derivazione": (0, 0, 610, altezza),
                "R1) aVR": (615, 0, 1225, altezza),
                "R1) V1": (1230, 0, 1840, altezza),
                "R1) V4": (1845, 0, 2455, altezza),
                "R2) II derivazione": (0, altezza, 610, altezza*2),
                "R2) aVL": (615, altezza, 1225, altezza*2),
                "R2) V2": (1230, altezza, 1840, altezza*2),
                "R2) V5": (1845, altezza, 2455, altezza*2),
                "R3) III derivazione": (0, altezza*2, 610, altezza*3),
                "R3) aVF": (615, altezza*2, 1225, altezza*3+50),
                "R3) V3": (1230, altezza*2, 1840, altezza*3+50),
                "R3) V6": (1845, altezza*2, 2455, altezza*3+50),
                "R4) IV derivazione": (0, altezza*3+50, 2454, altezza*4)
                }
                    
            crop_region = {}
            for region, coordinates in crop_coordinates.items():
                cropped_region = image_without_background.crop(coordinates)
                crop_region[region] = cropped_region
             
            return crop_region
    



class DatasetChecker:
    def __init__(self, dataset_path):
        self.dataset_path = os.path.abspath(dataset_path)
        assert self._check_structure()
        self.background_image = self._check_or_create_background()
        self.stemi_path, self.nstemi_path = self.join_data_path()
        

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
            background_creator = Background(self.dataset_path)
            background_image = background_creator.get_background()
            background_image_pil = Image.fromarray(background_image)
            background_image_pil.save(background_image_path)
            print(f"Background image saved at {background_image_path}")
            
        else:
            print("Background image already exists.")
            background_image = Image.open(background_image_path)
        
        return background_image
    
    def process_derivation(self):
        
        stemi_image_paths = [os.path.join(self.stemi_path, img_pth) for img_pth in os.listdir(self.stemi_path)]
        nstemi_image_paths = [os.path.join(self.nstemi_path, img_pth) for img_pth in os.listdir(self.nstemi_path)]
        
        for stemi_path in stemi_image_paths:
            stemi_image = PreprocessImage(stemi_path, self.background_image)
            stemi_regions = stemi_image.get_image_derivation()
            
            
        
        
        

