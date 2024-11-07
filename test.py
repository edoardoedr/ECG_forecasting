from __future__ import annotations
from dataclasses import field, dataclass
from math import ceil
from itertools import groupby
from operator import itemgetter
from typing import Iterable
import numpy as np
from scipy.signal import find_peaks

# Definizione delle classi necessarie

@dataclass
class Point:
    """
    Abstract representation of an integer Point in 2D. It is defined by a (x,y) tuple.
    """
    x: int = field()
    y: int = field()

class DigitizationError(Exception):
    """
    Error occurred during digitization process.
    """
    pass

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
    def __init__(self, n: int) -> None:
        self.__n = n

    def extract_signals(self, ecg: Image) -> Iterable[Iterable[Point]]:
        """
        Extract the signals of the ECG image.

        Args:
            ecg (Image): ECG image from which to extract the signals.

        Raises:
            DigitizationError: The indicated number of ROI could not be detected.
        
        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        N = ecg.width
        LEN, SCORE = (2, 3)  # Cache values
        rois = self.__get_roi(ecg)
        print(rois)
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
                    d = abs(ctr - rois[0])  # Vertical distance to roi
                    g = self.__gap(pc, c)  # Disconnection level
                    costs[pc] = ps + d + N / 10 * g

                best = min(costs, key=costs.get)
                y = ceil(mean(best))
                p = (col - 1, best)
                l = cache[p][0][LEN] + 1
                s = costs[best]
                cache[col, c][0] = (y, p, l, s)

        # Backtracking
        raw_signals = self.__backtracking(cache, rois)
        return raw_signals

    def __get_roi(self, ecg: Image) -> Iterable[int]:
        """
        Get the coordinates of the ROI of the ECG image.

        Args:
            ecg (Image): ECG image from which to extract the ROI.

        Raises:
            DigitizationError: The indicated number of ROI could not be detected.
        
        Returns:
            Iterable[int]: List of row coordinates of the ROI.
        """
        WINDOW = 10
        SHIFT = (WINDOW - 1) // 2
        stds = np.zeros(ecg.height)
        for i in range(ecg.height - WINDOW + 1):
            x0, x1 = (0, ecg.width)
            y0, y1 = (i, i + WINDOW - 1)
            std = ecg[y0:y1, x0:x1].reshape(-1).std()
            stds[i + SHIFT] = std
        # Find peaks
        min_distance = int(ecg.height * 0.1)
        peaks, _ = find_peaks(stds, distance=min_distance)
        print(peaks)
        rois = sorted(peaks, key=lambda x: stds[x], reverse=True)
        rois = [0]
        print(rois)
        if len(rois) < self.__n:
            raise DigitizationError("The indicated number of rois could not be detected.")
        rois = rois[0 : self.__n]
        rois = sorted(rois)
        
        print(rois)
        return rois





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




    def __backtracking(
        self, cache: dict, rois: Iterable[int]
    ) -> Iterable[Iterable[Point]]:
        """
        Performs a backtracking process over the cache of links between clusters
        to extract the signals.

        Args:
            cache (dict): Cache with the links between clusters.
            rois (Iterable[int]): List with the row coordinates of the rois.

        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        # Crea una lista vuota per i segnali grezzi
        raw_signals = []
        
        # Soglia di distanza per considerare un punto estremamente distante dal suo intorno
        soglia_distanza = 10  # Puoi regolare questa soglia in base alle tue esigenze
        
        # Itera su ciascuna ROI
        for roi_i in range(self.__n):
            # Crea una lista vuota per i punti di ciascun segnale
            raw_s = []
            for key, value in cache.items():
                # Ottieni la coordinata X dalla chiave
                x_coord = key[0]
                # Ottieni la coordinata Y dal valore
                y_coord = value[roi_i][0]
                # Aggiungi il punto alla lista
                raw_s.append(Point(x_coord, y_coord))
            
            # Filtro dei punti estremamente distanti dal loro intorno
            filtered_raw_s = []
            for i, p in enumerate(raw_s):
                if i > 0 and i < len(raw_s) - 1:
                    dist_precedente = abs(p.y - raw_s[i - 1].y)
                    dist_successiva = abs(p.y - raw_s[i + 1].y)
                    if dist_precedente < soglia_distanza and dist_successiva < soglia_distanza:
                        filtered_raw_s.append(p)
                else:
                    filtered_raw_s.append(p)
            
            # Aggiungi la lista di punti filtrati al segnale grezzo
            raw_signals.append(filtered_raw_s)
        
        return raw_signals





import matplotlib.pyplot as plt
import cv2 as cv



# Caricamento dell'immagine ECG dal file processed.png
ecg_image_data = cv.imread('processed.png', cv.IMREAD_GRAYSCALE)
ecg_image = Image(ecg_image_data)

# Inizializzazione del SignalExtractor
extractor = SignalExtractor(n=1)

# Estrazione dei segnali utilizzando l'immagine modificata
signals = extractor.extract_signals(ecg_image)

# Creazione dei plot per sovrapporre i segnali estratti sull'immagine originale
plt.figure(figsize=(12, 8))

# Mostra l'immagine originale
plt.imshow(ecg_image_data, cmap='gray', aspect='auto')
plt.title('Segnali Estratti Sovrapposti all\'Immagine ECG Originale')
plt.xlabel('X')
plt.ylabel('Y')

# Sovrappone i segnali estratti sull'immagine originale
for i, signal in enumerate(signals):
    x_vals = [point.x for point in signal]
    y_vals = [point.y for point in signal]
    plt.plot(x_vals, y_vals, label=f'Segnale {i+1}', linewidth=2.2)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
