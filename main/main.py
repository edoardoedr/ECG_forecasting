import os
import pickle
import random

def shuffle_data(data):
    """
    Mescola i dati in modo casuale.
    :param data: Una lista di dati (ad esempio, una lista di file o elementi da mescolare)
    :return: La lista di dati mescolata
    """
    random.shuffle(data)  # Mescola i dati in modo casuale
    return data

def load_and_show_data(data_dir):
    # Cartelle da esaminare e le relative etichette
    classes = {
        'ECG STEMI_data': 'STEMI',  # Etichetta "STEMI" per ECG STEMI_data
        'ECG NSTEMI_data': 'NSTEMI'  # Etichetta "NSTEMI" per ECG NSTEMI_data
    }
    
    # Lista per raccogliere tutti i file .pkl e le etichette
    all_files = []

    # Scansione delle cartelle
    for folder, label in classes.items():
        label_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(label_path):
            print(f"Caricando dati dalla cartella: {folder} (Etichetta: {label})")
            
            # Scansione dei file .pkl nella cartella
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                
                if file_path.endswith('.pkl'):  # Aggiungi solo i file .pkl
                    all_files.append((file_path, label))  # Aggiungi il file e l'etichetta alla lista

    # Mescola i file raccolti
    all_files = shuffle_data(all_files)

    # Carica e mostra i dati (ma solo le chiavi o altre informazioni)
    print("\nLista dei file .pkl, le etichette associate e le chiavi dei dizionari:")

    for file_path, label in all_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # Carica il dizionario dal file .pkl
            
        # Controllo se è un dizionario
        if isinstance(data, dict):
            print(f"\nFile: {file_path} | Etichetta: {label}")
            print(f"Chiavi del dizionario: {list(data.keys())}")
        else:
            print(f"\nIl file {file_path} non contiene un dizionario.")

def main():
    # Percorso della cartella principale che contiene ECG STEMI_data e ECG NSTEMI_data
    data_dir = r'G:\Il mio Drive\Codici Python\ECG_forecasting\Dataset'  # Percorso aggiornato
    
    # Carica e mostra i dati (solo la lista delle chiavi del dizionario e le etichette)
    load_and_show_data(data_dir)

if __name__ == '__main__':
    main()
