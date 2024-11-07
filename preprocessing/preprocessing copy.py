from data_extraction import *
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from scipy.ndimage import convolve
# from scipy.interpolate import interp1d
# from sklearn.cluster import KMeans
from scipy.ndimage import median_filter


class Background:
    def __init__(self, path):
        self.path = path
        
    def get_image_from_pdf(self,document_path):
        self.document_path = document_path
        pdf_document = fitz.open(self.document_path)
        page = pdf_document[0]
        image_info = page.get_images(full=True)[0]
        xref = image_info[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image_cropped = pil_image.crop((10, 690, 2540, 1900 ))
        pdf_document.close()
        
        return pil_image_cropped
    
    def save_similarities (self, img1,img2):
        array_img1 = np.array(img1)
        array_img2 = np.array(img2)
        tolleranza = 50000
        tolleranza = 0
        differenza = np.linalg.norm(array_img1 - array_img2, axis=-1) > tolleranza
        nuova_immagine = np.where(differenza[..., None], 255, array_img1)
        return nuova_immagine    
    
    def calcola_pixel_comuni(self,array_immagini):
        stack = np.stack(array_immagini, axis=-1)
        pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)
        for canale in range(3):
            canale_array = stack[:, :, canale, :]
            canale_array_flat = canale_array.reshape(-1, canale_array.shape[-1])
            mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=canale_array_flat)
            pixel_comuni[:, :, canale] = mode.reshape(array_immagini[0].shape[:2])
        return pixel_comuni   
    
    
    def iter_similarity(self):
        repo_imgs = []
        num_campioni = len([nome for nome in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, nome))])+1
        for contatore in range (1,num_campioni,1):
            for combinazioni in range (contatore +1 ,num_campioni):
                if(contatore < 10 and combinazioni < 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/00{contatore}.pdf')
                    immagine2 = self.get_image_from_pdf(f'{self.path}/00{combinazioni}.pdf')
                if(contatore < 10 and combinazioni >= 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/00{contatore}.pdf')
                    immagine2 = self.get_image_from_pdf(f'{self.path}/0{combinazioni}.pdf')
                if(contatore >= 10 and combinazioni >= 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/0{contatore}.pdf')
                    immagine2 = self.get_image_from_pdf(f'{self.path}/0{combinazioni}.pdf')
                
                img_grayscale = immagine1.convert('L')
                immagine1 = Image.merge('RGB', (img_grayscale, img_grayscale, img_grayscale))
                img_grayscale = immagine2.convert('L')
                immagine2 = Image.merge('RGB', (img_grayscale, img_grayscale, img_grayscale))
                repo_imgs.append(self.save_similarities(immagine1,immagine2))
        array_immagini = [np.array(immagine) for immagine in repo_imgs]       
        return array_immagini
    
    
    def workflow(self):
        array_immagini = self.iter_similarity()
        sfondo = self.calcola_pixel_comuni(array_immagini)
        return sfondo


class Rimozione_sfondo_e_tagli:
    def __init__(self, path):
        self.path = path
        
    def get_image_from_pdf(self,document_path):
        self.document_path = document_path
        pdf_document = fitz.open(self.document_path)
        page = pdf_document[0]
        image_info = page.get_images(full=True)[0]
        xref = image_info[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image_cropped = pil_image.crop((10, 690, 2540, 1900 ))
        pdf_document.close()       
        
        return pil_image_cropped 
        
    def importa_file(self,contatore):
        
        if(contatore < 10):
            immagine1 = self.get_image_from_pdf(f'{self.path}/00{contatore}.pdf')
        else:
            immagine1 = self.get_image_from_pdf(f'{self.path}/0{contatore}.pdf')
        img_grayscale = immagine1.convert('L')
        immagine1 = Image.merge('RGB', (img_grayscale, img_grayscale, img_grayscale))
        return immagine1
    
    
    def differenze(self,immagine,sfondo):
        
        prima_immagine = immagine
        seconda_immagine = sfondo
        

        array_prima = np.array(prima_immagine)
        array_seconda = np.array(seconda_immagine)

        
        soglia = 127

        array_img1_bin = (array_prima > soglia).astype(np.uint8)
        array_img2_bin = (array_seconda > soglia).astype(np.uint8)
    
        tolleranza =0
        differenza = np.linalg.norm(array_img1_bin - array_img2_bin, axis=-1) > tolleranza
        nuova_immagine = np.where(differenza[..., None], array_img1_bin, 255 )
        
        return Image.fromarray(nuova_immagine)
    
    

    def get_image_derivation(self, no_grid_image: Image) -> dict:
            altezza = no_grid_image.size[1]//4
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
                cropped_region = no_grid_image.crop(coordinates)
                crop_region[region] = cropped_region
             
            return crop_region
        
        
    def stampa(self,images): 
        # Numero di immagini
        num_images = len(images)
        # Numero di colonne desiderato
        num_cols = 3
        # Numero di righe necessario
        num_rows = (num_images + num_cols - 1) // num_cols

        # Creiamo la figura e gli assi
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axes = axes.flatten()

        # Visualizziamo ogni immagine in una cella della griglia
        for ax, (name, image) in zip(axes, images.items()):
            ax.imshow(image, cmap='gray')
            ax.set_title(name)
            ax.axis('off')

        # Nascondiamo eventuali assi vuoti
        for ax in axes[num_images:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    def workflow(self,contatore,sfondo):
        immagine1 = self.importa_file(contatore)
        img_no_bkgr= self.differenze(immagine1,sfondo)
        crop_dict = self.get_image_derivation(img_no_bkgr)
        # self.stampa(crop_dict)
        return crop_dict
        

            
            
            

if __name__ == '__main__':
    
    if not os.path.isfile(os.path.join("preprocessing", "sfondo.png")):
        repo_imgs = Background('data')
        sfondo= repo_imgs.workflow()
        Image.fromarray(sfondo).save('preprocessing/sfondo.png')
    else:
        sfondo = Image.open("preprocessing/sfondo.png")


    immagine = Rimozione_sfondo_e_tagli('data')
    # for pdf in range(1,len([nome for nome in os.listdir('data') if os.path.isfile(os.path.join('data', nome))])+1):
    #     risultato_finale = immagine.workflow(pdf,sfondo)
    risultato_finale = immagine.workflow(1,sfondo)
    

    for key,image in risultato_finale.items():
        
        image = np.array(image)
        import_functions_export_data(key,image)