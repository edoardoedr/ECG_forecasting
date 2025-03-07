import time
import pyautogui
import csv
import threading
from pynput import keyboard

# Variabili globali
click_positions = []
text_list = []
running = False
click_thread = None
type_thread = None

def load_coordinates(file_path):
    """Carica le coordinate dal file CSV."""
    global click_positions
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        click_positions = [(int(row[0]), int(row[1])) for row in reader]
    print(f"Coordinate caricate: {click_positions}")

def load_text(file_path):
    """Carica le parole da scrivere dal file CSV."""
    global text_list
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        text_list = [row[0] for row in reader]
    print(f"Testi caricati: {text_list}")

def auto_clicker(interval=1):
    """Esegue un click automatico sulle posizioni salvate."""
    global running
    while running:
        for pos in click_positions:
            pyautogui.click(x=pos[0], y=pos[1])
            print(f"Click a {pos}")
            time.sleep(interval)

def auto_typer(interval=1):
    """Scrive automaticamente i testi caricati."""
    global running
    while running:
        for text in text_list:
            pyautogui.typewrite(text)
            pyautogui.press('enter')
            print(f"Scritto: {text}")
            time.sleep(interval)

def toggle_running():
    """Attiva o disattiva il programma con un tasto."""
    global running, click_thread, type_thread
    running = not running
    print(f"Running: {running}")
    
    if running:
        click_thread = threading.Thread(target=auto_clicker, daemon=True)
        type_thread = threading.Thread(target=auto_typer, daemon=True)
        click_thread.start()
        type_thread.start()

# Assegna il tasto per avviare/fermare
hotkey_listener = keyboard.GlobalHotKeys({'<f10>': toggle_running})
print("Premi F10 per avviare/fermare il programma.")
load_coordinates("coordinates.csv")
load_text("text.csv")
hotkey_listener.start()

# Mantieni il programma in esecuzione
type_thread = threading.Thread(target=hotkey_listener.join)
type_thread.start()
type_thread.join()
