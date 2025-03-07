import time
import pyautogui
import csv
from pynput import mouse, keyboard

# Variabili globali
click_positions = []
text_list = []
running = False

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
            pyautogui.click(pos)
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
    global running
    running = not running
    print(f"Running: {running}")

# Assegna il tasto per avviare/fermare
with keyboard.GlobalHotKeys({'<ctrl>+<alt>+s': toggle_running}) as listener:
    print("Premi CTRL+ALT+S per avviare/fermare il programma.")
    load_coordinates("coordinates.csv")
    load_text("text.csv")
    listener.join()
