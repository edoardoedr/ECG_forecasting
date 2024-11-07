# Import delle librerie necessarie

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

# Definizione della classe SignalExtractor
class SignalExtractor:
    def __init__(self, n: int) -> None:
        self.__n = n

    def extract_signals(self, ecg) -> Iterable[Iterable[Point]]:
        """
        Extract the signals of the entire ECG image.

        Args:
            ecg (Image): ECG image from which to extract the signals.

        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        N = ecg.width
        LEN, SCORE = (2, 3)  # Cache values
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        cache = {}

        # Use every row in the image as ROI
        rois = range(ecg.height)

        for col in range(1, N):
            
            prev_clusters = self.__get_clusters(ecg, col - 1)
            if not len(prev_clusters):
                
                continue
            clusters = self.__get_clusters(ecg, col)
            if not clusters:
                continue  # Skip columns with no black pixels
            # print(clusters)
            for c in clusters:
                
                cache[col, c] = [None] * len(rois)
                for roi_i in range(len(rois)):
                    
                    costs = {}
                    for pc in prev_clusters:
                        
                        node = (col - 1, pc)
                        ctr = ceil(mean(pc))
                        if node not in cache.keys():
                            val = [ctr, None, 1, 0]
                            cache[node] = [val] * len(rois)
                            
                        ps = cache[node][roi_i][SCORE]  # Previous score
                        d = abs(ctr - rois[roi_i])  # Vertical distance to roi
                        g = self.__gap(pc, c)  # Disconnection level
                        costs[pc] = ps + d + N / 10 * g
                    best = min(costs, key=costs.get)
                    y = ceil(mean(best))
                    p = (col - 1, best)
                    l = cache[p][roi_i][LEN] + 1
                    s = costs[best]
                    cache[col, c][roi_i] = (y, p, l, s)
                    

        raw_signals = self.__backtracking(cache, rois)
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
        X_COORD, CLUSTER = (0, 1)  # Cache keys
        Y_COORD, PREV, LEN = (0, 1, 2)  # Cache values
        
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        raw_signals = [None] * self.__n
        for roi_i in range(self.__n):
            # Get candidate points (max signal length)
            roi = rois[roi_i]
            max_len = max([v[roi_i][LEN] for v in cache.values()])
            cand_nodes = [
                node
                for node, stats in cache.items()
                if stats[roi_i][LEN] == max_len
            ]
            # Best last point is the one more closer to ROI
            best = min(
                cand_nodes,
                key=lambda node: abs(ceil(mean(node[CLUSTER])) - roi),
            )
            raw_s = []
            clusters = []
            while best is not None:
                y = cache[best][roi_i][Y_COORD]
                raw_s.append(Point(best[X_COORD], y))
                clusters.append(best[CLUSTER])
                best = cache[best][roi_i][PREV]
            raw_s = list(reversed(raw_s))
            clusters = list(reversed(clusters))
            # Peak delineation
            roi_dist = [abs(p.y - roi) for p in raw_s]
            peaks, _ = find_peaks(roi_dist)
            for p in peaks:
                cluster = clusters[p - 1]
                farthest = max(cluster, key=lambda x: abs(x - roi))
                raw_s[p] = Point(raw_s[p].x, farthest)
            raw_signals[roi_i] = raw_s
        return raw_signals





import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv  # Importa cv2 per caricare l'immagine

# Caricamento dell'immagine ECG dal file processed.png
ecg_image_data = cv.imread('processed.png', cv.IMREAD_GRAYSCALE)
ecg_image = Image(ecg_image_data)

# Inizializzazione del SignalExtractor
extractor = SignalExtractor(n=1)

# Estrazione dei segnali
signals = extractor.extract_signals(ecg_image)

# Creazione dei plot per sovrapporre i segnali estratti sull'immagine originale
plt.figure(figsize=(12, 8))

# Mostra l'immagine originale
plt.imshow(ecg_image_data, cmap='gray', aspect='auto')
plt.title('Segnali Estratti Sovrapposti all\'Immagine ECG Originale')
plt.xlabel('X')
plt.ylabel('Y')

# Sovrappone i segnali estratti sull'immagine
for i, signal in enumerate(signals):
    x_vals = [point.x for point in signal]
    y_vals = [point.y for point in signal]
    plt.plot(x_vals, y_vals, label=f'Segnale {i+1}', linewidth=2)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

