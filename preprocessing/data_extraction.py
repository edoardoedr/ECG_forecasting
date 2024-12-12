from __future__ import annotations
from dataclasses import field, dataclass
from math import ceil
from itertools import groupby
from operator import itemgetter
from typing import Iterable
import numpy as np
from scipy.signal import find_peaks
import cv2 as cv
import matplotlib.pyplot as plt


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
    def __init__(self) -> None:
        self.__n = 1

    def extract_signals(self, ecg: Image) -> Iterable[Iterable[Point]]:
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
        soglia_distanza = 10  # Puoi regolare questa soglia in base alle tue esigenze
        
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
                if dist_precedente < soglia_distanza :
                    filtered_raw_s.append(p)
                # else:
                #     drop_next = True
            else:
                filtered_raw_s.append(p)

        raw_signals.append(filtered_raw_s)

        return raw_signals

    def plot_grafici(self, key, x_vals, y_vals,ecg_image_data):
        
        plt.imshow(ecg_image_data, cmap='gray', aspect='auto')
        plt.title('Segnali Estratti Sovrapposti all\'Immagine ECG Originale')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.plot(x_vals, y_vals,'-',color = 'red',label = f'{key}', linewidth=1,alpha=1,markersize=4)
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        
    def interpolazione(self,lista1,lista2):
        
        # Converti le liste in array numpy
        x = np.array(lista1)
        y = np.array(lista2)
        
        # Crea un nuovo array con la spaziatura definita
        x_new = np.arange(0, x.max(), 0.1)
        
        # Esegui l'interpolazione
        y_new = np.interp(x_new, x, y)
        
        return x_new, y_new
        
        
        
        
        


def import_functions_export_data(key,ecg_image_data,image_bkr):

    ecg_image = Image(ecg_image_data)
    extractor = SignalExtractor(n=1)
    signals = extractor.extract_signals(ecg_image)
    # plt.imshow(ecg_image_data)
    # plt.show()
    # Sovrappone i segnali estratti sull'immagine originale
    for i, signal in enumerate(signals):
        x_vals = [point.x for point in signal]
        y_vals = [point.y for point in signal]
        
        x_vals,y_vals = extractor.interpolazione(x_vals,y_vals)
        
        # extractor.plot_grafici(key,x_vals, y_vals,ecg_image_data)
        extractor.plot_grafici(key,x_vals, y_vals,image_bkr)
        return x_vals,y_vals