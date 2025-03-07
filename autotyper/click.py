import pandas as pd
import time
import pyautogui
import csv
import threading
from pynput import keyboard

def load_names_df(path_df):
    """Carica i nomi da un file CSV in un DataFrame di Pandas."""
    try:
        df = pd.read_csv(path_df, header=0)
        print("Anteprima dati:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Errore nel caricamento del DataFrame: {e}")
        return None

def load_coordinates(file_path):
    """Carica le coordinate dal file CSV."""
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            click_positions = [(int(row[1]), int(row[2])) for row in reader]
        print(f"Coordinate caricate: {click_positions}")
        return click_positions
    except Exception as e:
        print(f"Errore nel caricamento delle coordinate: {e}")
        return []

def auto_clicker(pos):
    """Esegue un click automatico sulla posizione specificata."""
    if pos:
        pyautogui.click(x=pos[0], y=pos[1])
        print(f"Click a {pos}")
        time.sleep(2)  # Riduzione del tempo per migliorare la velocità
    else:
        print("Posizione non valida per il click.")
        
def move_mouse(pos):
    """Sposta il mouse alla posizione specificata senza cliccare."""
    if pos:
        pyautogui.moveTo(x=pos[0], y=pos[1])
        print(f"Mouse spostato a {pos}")
        time.sleep(2)
    else:
        print("Posizione non valida per lo spostamento del mouse.")        


def auto_typer(testo_da_scrivere):
    """Scrive automaticamente il testo fornito."""
    if isinstance(testo_da_scrivere, str):
        pyautogui.typewrite(testo_da_scrivere)
        print(f"Scritto: {testo_da_scrivere}")
        time.sleep(2)
    else:
        print("Testo non valido per la digitazione.")

def main_func(path_coordinate, path_patients):
    click_positions = load_coordinates(path_coordinate)
    df = load_names_df(path_patients)
    
    if not click_positions or df is None:
        print("Errore nel caricamento dei dati. Uscita dal programma.")
        return
    
    for _, row in df.iterrows():
        # Ricerca_paziente
        auto_clicker(click_positions[0])
        # Cognome
        auto_clicker(click_positions[1])
        auto_typer(row.get('COGNOME', ''))
        # Nome
        auto_clicker(click_positions[2])
        auto_typer(row.get('NOME', ''))
        # Cerca
        auto_clicker(click_positions[3])
        
        
        # Esporta 1ECG
        auto_clicker(click_positions[4])
        auto_clicker(click_positions[14])
        move_mouse(click_positions[15])
        auto_clicker(click_positions[16])
        time.sleep(10)
        
        
        # Esporta 2ECG
        auto_clicker(click_positions[4])
        auto_clicker(click_positions[5])
        auto_clicker(click_positions[14])
        move_mouse(click_positions[15])
        auto_clicker(click_positions[16])
        time.sleep(10)
        
        # Esporta 3ECG
        auto_clicker(click_positions[5])
        auto_clicker(click_positions[6])
        auto_clicker(click_positions[14])
        move_mouse(click_positions[15])
        auto_clicker(click_positions[16])
        time.sleep(10)        

        # Esporta 4ECG
        auto_clicker(click_positions[6])
        auto_clicker(click_positions[7])
        auto_clicker(click_positions[14])
        move_mouse(click_positions[15])
        auto_clicker(click_positions[16])
        time.sleep(10)
        
        # Pulisci campi
        auto_clicker(click_positions[1])
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.5)
        pyautogui.press('delete')
        auto_clicker(click_positions[2])
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.5)
        pyautogui.press('delete')

        break  # Rimuovere per eseguire su tutti i pazienti

def main():
    
    time.sleep(10)
    
    """Funzione principale per eseguire l'autoclicker e autotyper."""
    path_coordinate = "coordinates.csv"  # Modifica con il percorso corretto
    path_patients = "names.csv"  # Modifica con il percorso corretto
    main_func(path_coordinate, path_patients)

if __name__ == "__main__":
    main()
