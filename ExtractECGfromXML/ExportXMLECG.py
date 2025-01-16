import os
import sys
import matplotlib.pyplot as plt
from neurokit2.ecg.ecg_clean import ecg_clean
from neurokit2.signal.signal_sanitize import signal_sanitize
from neurokit2.signal.signal_filter import signal_filter
import yaml
sys.path.append(os.path.abspath('ExtractECGfromXML/pysierraecg/src'))
from sierraecg import read_file


class ExportXLMECG:
    def __init__(self, file_path):
        
        self.xml_ecg = read_file(file_path)
        self.clinical_data = self.xml_ecg.clinical_data
        self.raw_ecg = self.extract_raw_data() 
        self.clean_ecg = self.clean_ecg_data()
        
    def extract_raw_data(self):
        
        ecg_raw_dict = {}
        
        for lead in self.xml_ecg.leads:
            ecg_raw_dict[lead.label] = lead.samples
            
        return ecg_raw_dict
    
    def clean_ecg_data(self):
        
        ecg_clean_dict = {}
        sampling_rate = int(self.xml_ecg.signal_characteristic['sampling_rate'])
        
        for lead_label, ecg_raw in self.raw_ecg.items():
            ecg_clean_dict[lead_label] = self._process_ecg(ecg_raw, sampling_rate)
        
        return ecg_clean_dict
    
    def get_ecgs_seconds(self, seconds):
        
        ecg_seconds_dict = {}
        
        sampling_rate = int(self.xml_ecg.signal_characteristic['sampling_rate'])
        slicing = int(seconds * sampling_rate)
        
        for lead_label, ecg_clean in self.clean_ecg.items():
            ecg_seconds_dict[lead_label] = ecg_clean[0:slicing]
        
        return ecg_seconds_dict
    
    def _process_ecg(self, ecg_signal, sampling_rate):
    
        ecg_signal = signal_sanitize(ecg_signal[0:5300])
        ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit') #taglia le frequenze sotto i 0.5 e la frequenza della corrente a 50Hz
        ecg_cleaned = signal_filter(ecg_cleaned, sampling_rate=sampling_rate, lowcut=None, highcut=150, method='butterworth', order=2) #taglia le frequenze sopra i 150Hz
        
        return ecg_cleaned
    
    def plot_ecgs(self, seconds, dir = None):
        
        ecg_seconds = self.get_ecgs_seconds(seconds)
        num_leads = len(ecg_seconds)
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (lead_label, ecg) in enumerate(ecg_seconds.items()):
            axes[i].plot(ecg)
            axes[i].set_title(lead_label)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if dir is not None:
            plt.savefig(dir)
            plt.close()
        else:
            plt.show()

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, section: str, key: str):
        """Restituisce il valore di una chiave nella sezione specificata."""
        return self.config.get(section, {}).get(key)

    @property
    def data_path(self):
        return self.get('parameters', 'data_path')
    
    @property
    def data_path(self):
        return self.get('parameters', 'n_seconds')
    
    @property
    def debug(self):
        return self.get('parameters', 'debug')
    

if __name__ == '__main__':
    
    file_path = 'data.xml'
    ecg_export = ExportXLMECG(file_path)
    ecg_export.plot_ecgs(seconds=2.5)
