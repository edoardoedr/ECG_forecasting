from data_extraction import *
# from pt_background import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from scipy.ndimage import convolve
from scipy.ndimage import median_filter


class Background:
    def __init__(self, path):
        self.path = path
        
    def get_image_from_pdf(self, document_path):
        self.document_path = document_path
        pdf_document = fitz.open(self.document_path)
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
  
    
    def calcola_pixel_comuni(self, array_immagini):
        stack = np.stack(array_immagini, axis=-1)
        pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)

        stack_flat = stack.reshape(-1, stack.shape[-1])
        threshold = len(array_immagini) // 4
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax() if np.bincount(x).max() > threshold else 255, axis=1, arr=stack_flat)
        pixel_comuni = mode.reshape(array_immagini[0].shape)
        
        return pixel_comuni

    
    
    def iter_similarity(self):
        repo_imgs = []
        num_campioni = len([nome for nome in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, nome))])+1
        for contatore in range (1,num_campioni,1):
            for combinazioni in range (contatore +1 ,num_campioni):
                if(contatore < 10 and combinazioni < 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/00{contatore}.pdf')
                if(contatore < 10 and combinazioni >= 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/00{contatore}.pdf')
                if(contatore >= 10 and combinazioni >= 10):
                    immagine1 = self.get_image_from_pdf(f'{self.path}/0{contatore}.pdf')

                repo_imgs.append(immagine1)
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
        immagine1 = immagine1.convert('L')
        return immagine1
    
    def trasformata(self,array,soglia):
        f = np.fft.fft2(array)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # Definisci la maschera per rimuovere una parte delle componenti di frequenza
        rows, cols = array.shape
        crow, ccol = rows // 2 , cols // 2  # Centro dell'immagine

        # Crea una maschera con un quadrato centrale di basse frequenze
        mask = np.ones((rows, cols), np.uint8)
        r = soglia  # Raggio del quadrato centrale
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0

        # Applica la maschera alle componenti di frequenza
        fshift_masked = fshift * mask

        # Calcola l'IFFT per ottenere l'immagine modificata
        f_ishift = np.fft.ifftshift(fshift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        soglia = 128
        img_back = (img_back > soglia).astype(np.uint8)
        # img_back_inv = cv2.bitwise_not(img_back)
        img_back_inv = 1 - img_back
        return img_back_inv
    


    
    
    
    def differenze(self,immagine,sfondo):

        
        array_prima = np.array(immagine)
        array_seconda = np.array(sfondo)

        img_back_1 = self.trasformata(array_prima,1)
       
        img_back_2 = self.trasformata(array_seconda,1)
        
        
        

        nuova_immagine= img_back_1.copy() 
        nuova_immagine[img_back_2 == 0] = 1


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


    def workflow(self,contatore,sfondo):
        immagine1 = self.importa_file(contatore)
        img_no_bkgr= self.differenze(immagine1,sfondo)
        crop_dict = self.get_image_derivation(img_no_bkgr)
        crop_dict_bkgr = self.get_image_derivation(immagine1)
        
        return crop_dict,crop_dict_bkgr
        


if __name__ == '__main__':
    
    if not os.path.isfile(os.path.join("preprocessing", "sfondo.png")):
        repo_imgs = Background('data')
        sfondo= repo_imgs.workflow()
        sfondo = sfondo.astype(np.uint8)
        Image.fromarray(sfondo).save('preprocessing/sfondo.png')
    else:
        sfondo = Image.open("preprocessing/sfondo.png")


    immagine = Rimozione_sfondo_e_tagli('data')
    
    coordinate_x_y_di_ogni_campione =  {}
    
    # for pdf in range(1,len([nome for nome in os.listdir('data') if os.path.isfile(os.path.join('data', nome))])+1):
    for pdf in range(1,2):
        risultato_finale, immagine_con_background = immagine.workflow(pdf,sfondo)
    # risultato_finale,immagine_con_background = immagine.workflow(1,sfondo)
        
        for (key,image),(key_bkr,image_bkr) in zip(risultato_finale.items(), immagine_con_background.items()):
            
            nuovo_dizionario = {}
            
            image = np.array(image)
            x_val,y_val = import_functions_export_data(key,image,image_bkr)
            # nuovo_sfondo = Rimozione_ordini_superiori()
            # nuovo_sfondo.workflow()
            
            
            
        #     nuovo_dizionario = {chiave: (x_val.tolist(),y_val.tolist()) for chiave in risultato_finale.keys()}
        # coordinate_x_y_di_ogni_campione[f'{pdf}'] =nuovo_dizionario
    
    