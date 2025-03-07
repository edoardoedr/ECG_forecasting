import pyautogui
import csv
import time
from pynput import mouse, keyboard

coordinates = []

def on_click(x, y, button, pressed):
    """Salva le coordinate del click nel file CSV."""
    if pressed:
        print(f"Coordinate catturate: ({x}, {y})")
        coordinates.append((x, y))


def save_coordinates_to_csv(file_path):
    """Salva le coordinate catturate in un file CSV."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for coord in coordinates:
            writer.writerow(coord)
    print(f"Coordinate salvate in {file_path}")


def on_press(key):
    """Termina la registrazione e salva il file CSV quando si preme ESC."""
    if key == keyboard.Key.esc:
        print("Terminazione della registrazione coordinate.")
        save_coordinates_to_csv("coordinates_1.csv")
        return False  # Ferma l'ascoltatore


def main():
    """Avvia l'ascolto degli eventi mouse per catturare le coordinate."""
    print("Clicca nei punti che vuoi salvare. Premi ESC per terminare e salvare.")
    with mouse.Listener(on_click=on_click) as listener:
        with keyboard.Listener(on_press=on_press) as key_listener:
            listener.join()
            key_listener.join()

if __name__ == "__main__":
    main()
