import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ECGExtractor:
    def __init__(self, document_path):
        self.document_path = document_path
        
    def get_image_from_pdf(self):
        """
        Estrae l'unica immagine da un PDF a singola pagina e singola immagine.
        """
        # Apri il documento PDF
        pdf_document = fitz.open(self.document_path)

        # Carica la prima (e unica) pagina
        page = pdf_document[0]
        
        # Estrai l'unica immagine
        image_info = page.get_images(full=True)[0]
        xref = image_info[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Converti i byte dell'immagine in formato PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image_cropped = pil_image.crop((10, 690, 2540, 1900 ))
        
        # Chiudi il documento PDF
        pdf_document.close()
        
        return pil_image_cropped
    
    def remove_grid(self, ROI_image: Image) -> Image:
        
        img  = np.array(ROI_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        lower_red = np.array([90, 0, 0])
        upper_red = np.array([255, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([90, 110, 110])
        # Creare maschere per i colori
        mask_red = cv2.inRange(img_rgb, lower_red, upper_red)
        mask_black = cv2.inRange(img_rgb, lower_black, upper_black)

        # Applicare maschere e cambiare i colori
        img_rgb[mask_red != 0] = [255, 255, 255]
        img_rgb[mask_black != 0] = [0, 0, 0]
        img_rgb[(mask_red == 0) & (mask_black == 0)] = [255, 255, 255]

        # Converti e salva l'immagine modificata
        no_grid_image = Image.fromarray(img_rgb)
        
        return no_grid_image
    
    def get_image_derivation(self, no_grid_image: Image) -> dict:
        
        crop_coordinates = {
            "I derivazione": (0, 45, 610, 250),
            "aVR": (615, 45, 1225, 250),
            "V1": (1230, 45, 1840, 250),
            "V4": (1850, 45, 2455, 250),
            "II derivazione": (0, 345, 610, 550),
            "aVL": (615, 345, 1225, 550),
            "V2": (1230, 345, 1840, 550),
            "V5": (1850, 345, 2455, 550),
            "III derivazione": (0, 645, 610, 850),
            "aVF": (615, 645, 1225, 850),
            "V3": (1230, 600, 1840, 950),
            "V6": (1850, 580, 2455, 890),
            "IV derivazione": (0, 980, 2454, 1180)
            }
                
        crop_region = {}
        
        for region, coordinates in crop_coordinates.items():
            
            cropped_region = no_grid_image.crop(coordinates)
            
            crop_region[region] = cropped_region
            
        return crop_region
    
    def preprocess(self):
        ROI_image = self.get_image_from_pdf()
        ROI_image_cleaned = self.remove_grid(ROI_image)
        crop_dict = self.get_image_derivation(ROI_image_cleaned)
        
        return crop_dict
        
    
if __name__ == '__main__':
    
    pdf2image = ECGExtractor('data/ECG_STEMI antero-laterale.pdf')
    dict_image = pdf2image.preprocess()
    
    for nome, derivazione in dict_image.items():
        
        print(nome)
        plt.imshow(derivazione)
        plt.title(nome)
        plt.show()
    
    