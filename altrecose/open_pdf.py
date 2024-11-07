import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def read_image_from_pdf(document_path):
    # Apri il documento PDF
    pdf_document = fitz.open(document_path)

    # Scorri le pagine del PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)

        # Estrai ogni immagine
        images_exstraceted = []
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            images_exstraceted.append(image_bytes)
            
            # Salva l'immagine
            #with open(f"immagine_{page_num+1}_{img_index+1}.png", "wb") as img_file:
            #    img_file.write(image_bytes)
    #leggiamo solo la prima immagine
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    return pil_image

def crop_image(image, left, top, right, bottom):
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def get_ECG_images(image):
    
    start_pixel = 50
    len_quad = 295
    primo_ecg = crop_image(image, 0, start_pixel, 2530, start_pixel+len_quad)
    secondo_ecg = crop_image(image, 0, start_pixel+len_quad*1, 2530, start_pixel+len_quad*2)
    terzo_ecg = crop_image(image, 0, start_pixel+len_quad*2, 2530, start_pixel+len_quad*3)
    quarto_ecg = crop_image(image, 0, start_pixel+len_quad*3, 2530, start_pixel+len_quad*4)
    
    primo_ecg_recropped = crop_image(primo_ecg, 0, 60, primo_ecg.width, primo_ecg.height)
    #primo_ecg.save("primo_ECG.png")
    #secondo_ecg.save("secondo_ECG.png")
    #terzo_ecg.save("terzo_ECG.png")
    #quarto_ecg.save("quarto_ECG.png")
    
    
    return primo_ecg_recropped, secondo_ecg, terzo_ecg, quarto_ecg


def process_ecg_images(image):
    
    image_np = np.copy(image)

    mask = np.all(image_np <= 150, axis=-1)
    
    # Creare una nuova immagine con la stessa forma riempita di bianco
    new_image = np.ones_like(image_np) * 255
    
    # Applicare la maschera per mantenere i pixel neri o quasi neri
    new_image[mask] = 0
    Image.fromarray(new_image).save("processed.png")
    
    columns_with_black_lines = []
    for col in range(new_image.shape[1]):
        # Trovare le sequenze di pixel neri nella colonna
        black_pixel_sequences = np.where(new_image[:, col, :] == 0)[0]
        if len(black_pixel_sequences) >= 60:
            columns_with_black_lines.append(col)
            
    processed = new_image.copy()
    for col in columns_with_black_lines:
        # Sostituire i pixel neri con il colore di sfondo (bianco in questo caso)
        processed[:, col] = [255, 255, 255]

    Image.fromarray(processed).save("processed.png")
    return processed
    

def get_graph(image):
    
    y_dim, x_dim, channel = image.shape
    print(image.shape)
    
    x_values = [x for x in range(x_dim)]
    y_values = []
    
    for x in x_values:
        indexes = np.where(image[:,x,0] == 0)
        if indexes[0].size > 0:
            #print(indexes[0].max())
            y_values.append(y_dim - indexes[0].max())
        else:
            y_values.append(np.nan)
            
    plt.plot(x_values, y_values, marker='o', linestyle='-', markersize = 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
        


if __name__ == "__main__":
    
    image = read_image_from_pdf("data/001.pdf")
    cropped_image = crop_image(image, 6, 688, 2536, 1919)
    cropped_image.save("prova_crop.png")
    four_ecg_images = get_ECG_images(cropped_image)
    processed_image = process_ecg_images(four_ecg_images[0])
    
    get_graph(processed_image)
    
    
    
    

