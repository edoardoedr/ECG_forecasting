from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class WebScraper:
    def __init__(self, url, percorso_file):
        self.url = url
        self.percorso_file = percorso_file
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    def leggi_nomi_da_file(self):
        with open(self.percorso_file, 'r') as file:
            nomi = file.readlines()
        return [nome.strip() for nome in nomi]

    def avvia(self):
        self.driver.get(self.url)
        nomi = self.leggi_nomi_da_file()

        for nome in nomi:
            self.cerca_e_scarica(nome)

        self.driver.quit()

    def cerca_e_scarica(self, nome):
        # Trova la barra di ricerca e inserisci il nome
        barra_ricerca = self.driver.find_element(By.ID, 'ID_BARRA_RICERCA')
        barra_ricerca.clear()
        barra_ricerca.send_keys(nome)
        barra_ricerca.send_keys(Keys.RETURN)
        
        time.sleep(2)  # Attendi che i risultati vengano caricati
        
        # Seleziona il risultato desiderato
        risultato = self.driver.find_element(By.XPATH, 'XPATH_RISULTATO')
        risultato.click()
        
        time.sleep(2)  # Attendi che la pagina del risultato venga caricata
        
        # Clicca il bottone di download
        bottone_download = self.driver.find_element(By.XPATH, 'XPATH_BOTTONE_DOWNLOAD')
        bottone_download.click()
        
        time.sleep(2)  # Attendi che il download inizi

# Utilizzo della classe
if __name__ == "__main__":
    url = 'URL_DELLA_PAGINA_WEB'
    percorso_file = 'percorso/del/file.txt'
    scraper = WebScraper(url, percorso_file)
    scraper.avvia()