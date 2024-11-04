import cv2
import numpy as np
import matplotlib.pyplot as plt
from ECG_extractor import ECGExtractor
from PIL import Image
import os
from scipy.ndimage import convolve


def save_similarities (img1,img2):

    # img1  = np.array(img1)
    # img2  = np.array(img2)

    # immagine_confronto = np.zeros_like(img1)
    # immagine_confronto[np.all(img1 == img2, axis=-1)] = img1[np.all(img1 == img2, axis=-1)]
    # immagine_confronto[np.all(img1 != img2, axis=-1)] = [0, 255, 0]
 
    # Converti le immagini in array numpy
    array_img1 = np.array(img1)
    array_img2 = np.array(img2)

    # Definisci una tolleranza per considerare i pixel "simili"
    tolleranza = 12550

    # Crea una maschera di differenza che considera i pixel simili
    differenza = np.linalg.norm(array_img1 - array_img2, axis=-1) > tolleranza

    # Applica la maschera alla prima immagine, sostituendo i pixel simili con quelli della prima immagine
    nuova_immagine = np.where(differenza[..., None], 255, array_img1)

    # Salva la nuova immagine

    return nuova_immagine
    
    # return immagine_confronto

    
# Definisci la funzione per calcolare il pixel più comune
def calcola_pixel_comuni(array_immagini):
    # Stack delle immagini lungo un nuovo asse
    stack = np.stack(array_immagini, axis=-1)
    # Inizializza la matrice per i pixel più comuni
    pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)

    # Itera su ogni canale RGB
    for canale in range(3):
        # Ottieni il canale corrente delle immagini
        canale_array = stack[:, :, canale, :]
        # Appiattisci per contare i valori unici
        canale_array_flat = canale_array.reshape(-1, canale_array.shape[-1])
        # Conta i valori unici lungo l'ultimo asse
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=canale_array_flat)
        # Ricostruisci la matrice del canale
        pixel_comuni[:, :, canale] = mode.reshape(array_immagini[0].shape[:2])
    return pixel_comuni



    
def iter_similarity():
    repo_imgs = []
    
    # Specifica il percorso della cartella
    cartella = 'data'
    # Conta il numero di file nella cartella
    num_campioni = len([nome for nome in os.listdir(cartella) if os.path.isfile(os.path.join(cartella, nome))])
    
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
            repo_imgs.append(save_similarities(immagine1,immagine2))
    
    array_immagini = [np.array(immagine) for immagine in repo_imgs]
    sfondo = calcola_pixel_comuni(array_immagini)
    Image.fromarray(sfondo).save('sfondo.png')
    return sfondo



def differenze(immagine,sfondo):
    
    import numpy as np

    # Carica le immagini
    prima_immagine = immagine.get_image_from_pdf()
    seconda_immagine = sfondo
    

    # Converti le immagini in array numpy
    array_prima = np.array(prima_immagine)
    array_seconda = np.array(seconda_immagine)

    
    # Definisci una soglia per la quantizzazione
    soglia = 70

    # Applica la soglia per quantizzare i pixel in 0 e 1
    array_img1_bin = (array_prima > soglia).astype(np.uint8)
    array_img2_bin = (array_seconda > soglia).astype(np.uint8)
 
    # Tolleranza metodo
    tolleranza = 0
    differenza = np.linalg.norm(array_img1_bin - array_img2_bin, axis=-1) > tolleranza
    nuova_immagine = np.where(differenza[..., None], array_img1_bin, 255 )
    

    # Salva la nuova immagine
    Image.fromarray(nuova_immagine).save('rimozione_sfondo.png')
    Image.fromarray(array_prima).save('originale.png')
    
    






