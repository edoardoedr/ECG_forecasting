# ECG Forecasting

Questo progetto riguarda la previsione dei segnali ECG utilizzando vari script Python. Di seguito viene fornita una spiegazione dei file presenti nella cartella `ECG_FORECASTING`.

## Struttura della Cartella

La cartella `ECG_FORECASTING` contiene i seguenti file e cartelle:

- `Test_on_physionet/`
  - `parse_commands.py`
  - `physionet2021_records.py`
  - `physionet2021_signals.py`
  - `physionet2021_labels.py`
  - `prepare_clf_labels.py`
  - `splits.py`


## Descrizione dei File

### `Test_on_physionet/parse_commands.py`

Questo script esegue una serie di comandi per processare i dati ECG. Utilizza `subprocess` per eseguire altri script Python che gestiscono la preparazione e la trasformazione dei dati. I comandi includono la generazione di record, segnali, etichette e la preparazione dei dati per la classificazione.

### `Test_on_physionet/physionet2021_records.py`

Questo script gestisce la creazione e la gestione dei record dei dati ECG. Prepara i dati grezzi per l'elaborazione successiva.

### `Test_on_physionet/physionet2021_signals.py`

Questo script elabora i segnali ECG. Fornisce funzionalità per il caricamento, la trasformazione e la visualizzazione dei segnali.

### `Test_on_physionet/physionet2021_labels.py`

Questo script genera le etichette per i dati ECG. Utilizza i pesi e le abbreviazioni dei pesi per creare un file di etichette che può essere utilizzato per l'addestramento e la valutazione del modello.

### `Test_on_physionet/prepare_clf_labels.py`

Questo script prepara le etichette per la classificazione. Carica le etichette generate e le organizza in un formato adatto per l'addestramento del modello di classificazione.

### `Test_on_physionet/splits.py`

Questo script gestisce la suddivisione dei dati in set di addestramento, validazione e test. Utilizza diverse strategie di suddivisione e filtri per creare i set di dati.

#
## Come Eseguire il Codice

1. Assicurati di avere tutti i requisiti installati, inclusi `scipy`, `matplotlib`, e `torch`.

2. Esegui lo script `parse_commands.py` per processare i dati:
    ```sh
    python Test_on_physionet/parse_commands.py
    ```


## Note

- Assicurati che i percorsi dei file e delle cartelle siano corretti.
- Verifica che tutti i file necessari siano presenti nelle directory specificate.



` TEST DI INFERENZA`

### [inference_tutorial.py](http://_vscodecontentref_/0)

Questo script fornisce un tutorial su come eseguire l'inferenza sui dati ECG utilizzando un modello pre-addestrato. Ecco una spiegazione dettagliata di cosa fa:

1. **Importazione dei Moduli**: Importa i moduli necessari per l'inferenza, inclusi `torch` per il deep learning, `numpy` per le operazioni numeriche e `matplotlib` per la visualizzazione.

2. **Caricamento del Modello**: Carica un modello pre-addestrato da un file di checkpoint. Questo modello è stato addestrato su dati ECG e può essere utilizzato per fare previsioni sui nuovi dati.

3. **Preparazione dei Dati**: Carica i dati ECG da un file `.mat` e li prepara per l'inferenza. Questo include la normalizzazione dei dati e la conversione in un formato compatibile con il modello.

4. **Esecuzione dell'Inferenza**: Utilizza il modello per fare previsioni sui dati ECG preparati. Le previsioni possono includere la classificazione dei segnali ECG in diverse categorie.

5. **Visualizzazione dei Risultati**: Visualizza i risultati dell'inferenza utilizzando `matplotlib`. Questo può includere la visualizzazione dei segnali ECG originali e delle previsioni del modello.
