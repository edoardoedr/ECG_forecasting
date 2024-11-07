import cv2
import numpy as np
import matplotlib.pyplot as plt
from ECG_extractor import ECGExtractor
from PIL import Image
import os
from scipy.ndimage import convolve


def save_similarities (img1,img2):


    array_img1 = np.array(img1)
    array_img2 = np.array(img2)
    tolleranza = 50000

    differenza = np.linalg.norm(array_img1 - array_img2, axis=-1) > tolleranza

    nuova_immagine = np.where(differenza[..., None], 255, array_img1)
    return nuova_immagine
    
    
def calcola_pixel_comuni(array_immagini):
    stack = np.stack(array_immagini, axis=-1)
    pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)

    for canale in range(3):
        canale_array = stack[:, :, canale, :]
        canale_array_flat = canale_array.reshape(-1, canale_array.shape[-1])
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=canale_array_flat)
        pixel_comuni[:, :, canale] = mode.reshape(array_immagini[0].shape[:2])
    return pixel_comuni



    
def iter_similarity():
    repo_imgs = []
    
    cartella = 'data'
    num_campioni = len([nome for nome in os.listdir(cartella) if os.path.isfile(os.path.join(cartella, nome))])+1
    
    for contatore in range (1,num_campioni,1):
        
        for combinazioni in range (contatore +1 ,num_campioni):
            if(contatore < 10 and combinazioni < 10):
                immagine1 = ECGExtractor(f'data/00{contatore}.pdf')
                immagine2 = ECGExtractor(f'data/00{combinazioni}.pdf')
            if(contatore < 10 and combinazioni >= 10):
                immagine1 = ECGExtractor(f'data/00{contatore}.pdf')
                immagine2 = ECGExtractor(f'data/0{combinazioni}.pdf')
            if(contatore >= 10 and combinazioni >= 10):
                immagine1 = ECGExtractor(f'data/0{contatore}.pdf')
                immagine2 = ECGExtractor(f'data/0{combinazioni}.pdf')
                
            immagine1 = immagine1.get_image_from_pdf()
            immagine2 = immagine2.get_image_from_pdf()
            
            img_grayscale = immagine1.convert('L')
            immagine1 = Image.merge('RGB', (img_grayscale, img_grayscale, img_grayscale))
            img_grayscale = immagine2.convert('L')
            immagine2 = Image.merge('RGB', (img_grayscale, img_grayscale, img_grayscale))
            
            
            repo_imgs.append(save_similarities(immagine1,immagine2))
    
    array_immagini = [np.array(immagine) for immagine in repo_imgs]
    sfondo = calcola_pixel_comuni(array_immagini)
    Image.fromarray(sfondo).save('sfondo.png')
    return sfondo



def differenze(immagine,sfondo):
    
    prima_immagine = immagine
    seconda_immagine = sfondo
    

    array_prima = np.array(prima_immagine)
    array_seconda = np.array(seconda_immagine)

    
    soglia = 120

    array_img1_bin = (array_prima > soglia).astype(np.uint8)
    array_img2_bin = (array_seconda > soglia).astype(np.uint8)
 
    tolleranza =100
    differenza = np.linalg.norm(array_img1_bin - array_img2_bin, axis=-1) > tolleranza
    nuova_immagine = np.where(differenza[..., None], array_img1_bin, 255 )
    

    Image.fromarray(nuova_immagine).save('rimozione_sfondo.png')
    Image.fromarray(array_prima).save('originale.png')
    
    






