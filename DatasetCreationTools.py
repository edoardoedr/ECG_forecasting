import os
from PIL import Image as PILImage
import random
import numpy as np
from dataclasses import field, dataclass
from typing import Iterable
from math import ceil
from itertools import groupby
from operator import itemgetter
import yaml

class Background:
    def __init__(self, path, get_image_from_pdf, divisore_threshold = 4):
        self.base_path = os.path.abspath(path)
        self.get_image_from_pdf = get_image_from_pdf
        self.random_pdf_paths = self.get_pdf_paths()
        self.divisore_threshold = divisore_threshold #da controllare il valore del threshold
        
    def get_pdf_paths(self):
        subdirs = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        pdf_paths = []
            
        for subdir in subdirs:
            subdir_path = os.path.join(self.base_path, subdir)
            pdf_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.pdf')]
            pdf_paths.extend(pdf_files)
            
            
        sample_size = max(1, len(pdf_paths) // 1)
        sampled_paths = random.sample(pdf_paths, sample_size)
        
        return sampled_paths
    
    def calcola_pixel_comuni(self, array_immagini):
        
        stack = np.stack(array_immagini, axis=-1)
        pixel_comuni = np.zeros(array_immagini[0].shape, dtype=np.uint8)

        stack_flat = stack.reshape(-1, stack.shape[-1])
        #da controllare il valore del threshold
        threshold = len(array_immagini) // self.divisore_threshold
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax() if np.bincount(x).max() > threshold else 255, axis=1, arr=stack_flat)
        pixel_comuni = mode.reshape(array_immagini[0].shape)
        
        return pixel_comuni
    
    def get_background(self):
        
        array_immagini = [np.array(self.get_image_from_pdf(pdf_path)) for pdf_path in self.random_pdf_paths]
        sfondo = self.calcola_pixel_comuni(array_immagini)
        
        return sfondo.astype(np.uint8)
    
class PreprocessImage:
    def __init__(self, pdf_path, get_image_from_pdf, background_image, soglia = 128, raggio = 1):
        self.image_ecg = get_image_from_pdf(pdf_path)
        self.background = background_image
        self.soglia = soglia
        self.raggio = raggio

    def get_image_fft(self, image_np):
        
        f = np.fft.fft2(image_np)
        fshift = np.fft.fftshift(f)

        # Definisci la maschera per rimuovere una parte delle componenti di frequenza
        rows, cols = image_np.shape
        crow, ccol = rows // 2 , cols // 2  # Centro dell'immagine

        # Crea una maschera con un quadrato centrale di basse frequenze
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-self.raggio:crow+self.raggio, ccol-self.raggio:ccol+self.raggio] = 0

        # Applica la maschera alle componenti di frequenza
        fshift_masked = fshift * mask

        # Calcola l'IFFT per ottenere l'immagine modificata
        f_ishift = np.fft.ifftshift(fshift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = (img_back > self.soglia).astype(np.uint8)
        # img_back_inv = cv2.bitwise_not(img_back)
        img_back_inv = 1 - img_back
        
        return img_back_inv
    
    def remove_bg_fft(self):

        image_np = np.array(self.image_ecg)
        background_np = np.array(self.background)
        
        filtered_image = self.get_image_fft(image_np)
        filtered_sfondo = self.get_image_fft(background_np)
        
        filtered_image_copy = filtered_image.copy() 
        filtered_image_copy[filtered_sfondo == 0] = 1
        print("----------------------",type(filtered_image_copy))
        print("----------------------",filtered_image_copy.dtype)
        filtered_image_copy = PILImage.fromarray(filtered_image_copy)
        print("----------------------",type(filtered_image_copy))

        return filtered_image_copy
    
    def crop_image_regions(self, image) -> dict:
            
        altezza = image.size[1]//4
            
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
            cropped_region = image.crop(coordinates)
            crop_region[region] = cropped_region
             
        return crop_region
    
    def get_derivation_images(self):

        image_without_background = self.remove_bg_fft()
        image_without_background_cropped = self.crop_image_regions(image_without_background)
        
        return image_without_background_cropped
    
@dataclass
class Point:
    """
    Abstract representation of an integer Point in 2D. It is defined by a (x,y) tuple.
    """
    x: int = field()
    y: int = field()

class Image:
    def __init__(self, data: np.ndarray):
        self.__data = data
        
    @property
    def width(self):
        return self.__data.shape[1]

    @property
    def height(self):
        return self.__data.shape[0]

    def __getitem__(self, index):
        return self.__data[index]

class SignalExtractor:
    def __init__(self, ecg_image_dict, soglia_distanza = 10, interpolate = False, num_points = None) -> None:
        self.__n = 1
        self.__ecg_image_dict = ecg_image_dict
        self.soglia_distanza = soglia_distanza  # Puoi regolare questa soglia in base alle tue esigenze
        self.interpolate = interpolate
        if self.interpolate:
            assert num_points is not None, "You must provide the number of points for interpolation."
            self.__num_points = num_points

    def extract_single_signal(self, ecg: Image) -> Iterable[Iterable[Point]]:
        """
        Extract the signals of the ECG image.

        Args:
            ecg (Image): ECG image from which to extract the signals.

        
        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        N = ecg.width
        LEN, SCORE = (2, 3)  # Cache values
        
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        cache = {}

        for col in range(1, N):
            prev_clusters = self.__get_clusters(ecg, col - 1)
            if not len(prev_clusters):
                continue
            clusters = self.__get_clusters(ecg, col)
            for c in clusters:
                # For each row get best cluster center based on minimizing the score
                cache[col, c] = [None] * self.__n
                
                costs = {}
                for pc in prev_clusters:
                    node = (col - 1, pc)
                    ctr = ceil(mean(pc))
                    if node not in cache.keys():
                        val = [ctr, None, 1, 0]
                        cache[node] = [val] * self.__n
                    ps = cache[node][0][SCORE]  # Previous score
                    g = self.__gap(pc, c)  # Disconnection level
                    costs[pc] = ps + N / 10 * g

                best = min(costs, key=costs.get)
                y = ceil(mean(best))
                p = (col - 1, best)
                l = cache[p][0][LEN] + 1
                s = costs[best]
                cache[col, c][0] = (y, p, l, s)

        # Backtracking
        raw_signals = self.__backtracking(cache)
        
        return raw_signals

    def __get_clusters(self, ecg, col: Iterable[int]) -> Iterable[Iterable[int]]:
        BLACK = 0
        clusters = []
        
        black_p = np.where(ecg[:, col] == BLACK)[0]
        
        for _, g in groupby(enumerate(black_p), lambda idx_val: idx_val[0] - idx_val[1]):
            clu = tuple(map(itemgetter(1), g))
            clusters.append(clu)
        return clusters

    def __gap(self, pc: Iterable[int], c: Iterable[int]) -> int:
        pc_min, pc_max = (pc[0], pc[-1])
        c_min, c_max = (c[0], c[-1])
        d = 0
        if pc_min <= c_min and pc_max <= c_max:
            d = len(range(pc_max + 1, c_min))
        elif pc_min >= c_min and pc_max >= c_max:
            d = len(range(c_max + 1, pc_min))
        return d

    def __backtracking(self, cache: dict) -> Iterable[Iterable[Point]]:
        """
        Performs a backtracking process over the cache of links between clusters
        to extract the signals.

        Args:
            cache (dict): Cache with the links between clusters.

        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        # Crea una lista vuota per i segnali grezzi
        raw_signals = []
        
        # Soglia di distanza per considerare un punto estremamente distante dal suo intorno
        
        # Crea una lista vuota per i punti di ciascun segnale
        raw_s = []
        for key, value in cache.items():
            # Ottieni la coordinata X dalla chiave
            x_coord = key[0]
            # Ottieni la coordinata Y dal valore
            y_coord = value[0][0]
            # Aggiungi il punto alla lista
            raw_s.append(Point(x_coord, y_coord))
        
        # Filtro dei punti estremamente distanti dal loro intorno
        filtered_raw_s = []
        drop_next = False

        for i, p in enumerate(raw_s):
            if drop_next:
                drop_next = False
                continue
            # filtered_raw_s.append(p)
            if i > 0 and i < len(raw_s) - 1:
                dist_precedente = abs(p.y - raw_s[i - 1].y)
                dist_successiva = abs(p.y - raw_s[i + 1].y)
                if dist_precedente < self.soglia_distanza:
                    filtered_raw_s.append(p)
                # else:
                #     drop_next = True
            else:
                filtered_raw_s.append(p)

        raw_signals.append(filtered_raw_s)

        return raw_signals

    def interpolate_signal(self, x_signal, y_signal): 
        
        x = np.array(x_signal)
        y = np.array(y_signal)
        
        # Crea un nuovo array con il numero di punti definito
        x_new = np.linspace(x.min(), x.max(), self.num_points)
        
        # Esegui l'interpolazione
        y_new = np.interp(x_new, x, y)
        
        return x_new, y_new

    def extract_signals(self):
        signals = {}
        for region, image in self.__ecg_image_dict.items():
            signal_iterables = self.extract_single_signal(image)
            for signal in signal_iterables:
                x_signal = [point.x for point in signal]
                y_signal = [point.y for point in signal]
                if self.interpolate:
                    x_new, y_new = self.interpolate_signal(x_signal, y_signal)
                    signals[region] = (x_new, y_new)
                else:
                    signals[region] = (np.array(x_signal), np.array(y_signal))
        
        return signals

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, section, key):
        """Restituisce il valore di una chiave nella sezione specificata."""
        return self.config.get(section, {}).get(key)

    @property
    def data_path(self):
        return self.get('parameters', 'data_path')
    
    @property
    def divisore_threshold(self):
        return self.get('parameters', 'divisore_threshold')

    @property
    def soglia(self):
        return self.get('parameters', 'soglia')

    @property
    def raggio(self):
        return self.get('parameters', 'raggio')

    @property
    def soglia_distanza(self):
        return self.get('parameters', 'soglia_distanza')

    @property
    def interpolate(self):
        return self.get('parameters', 'interpolate')

    @property
    def num_points(self):
        return self.get('parameters', 'num_points')

    @property
    def coordinates_pdf(self):
        return tuple(self.get('parameters', 'coordinates_pdf'))
    