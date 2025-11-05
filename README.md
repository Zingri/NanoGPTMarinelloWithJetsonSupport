# NanoGPT Marinello — Interfaccia Conversazionale Locale

**NanoGPT Marinello** è una versione minimalista di un modello GPT progettata per l’esecuzione **locale (CPU/GPU)** e su **dispositivi NVIDIA Jetson**.  
Fornisce un assistente conversazionale completo con interfaccia web, API REST e funzionalità di training personalizzato.

---

## Caratteristiche principali

- Interfaccia web (Gradio) per chat testuale e vocale
- API REST (Flask) per integrazione remota e uso su Jetson
- Pipeline di training semplice con salvataggio dei checkpoint
- Trascrizione vocale tramite Whisper (opzionale)
- Text-To-Speech (TTS) tramite gTTS (opzionale)
- Salvataggio cronologia conversazioni in SQLite
- Adattamento automatico dei parametri in base alle risorse di sistema

---

## Requisiti

### Dipendenze principali
- Python ≥ 3.8
- PyTorch (CPU o CUDA compatibile)
- tokenizers, transformers (fallback tokenizer GPT-2)
- gradio, flask, pandas, psutil, ftfy
- sqlite3 (builtin)

### Opzionali
- whisper → trascrizione vocale
- gTTS → sintesi vocale
- pygame → riproduzione audio locale

---

## Installazione

1. Clona il repository
```bash
git clone https://github.com/Zingri/NanoGPTMarinelloWithJetsonSupport.git
cd NanoGPTMarinelloWithJetsonSupport
```

2. Crea e attiva l’ambiente virtuale
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate
```

3. Installa le dipendenze
```bash
pip install -r requirements.txt
```

---

## File principali

- `NanoGPT_Marinello.py` — Core del progetto: modello GPT, training, UI Gradio, DB conversazioni  
- `LlmServer.py` — Server Flask con API REST  
- `checkpoint.pth` — Checkpoint del modello salvato (se presente)  
- `tokenizer.json` — Tokenizer BPE personalizzato (se presente)  
- `chat_memory.db` — Database SQLite delle conversazioni (se presente)  
- `train_tokenizer.py` — Script per generare un nuovo tokenizer

Se `tokenizer.json` non è presente, esegui:
```bash
python train_tokenizer.py
```

---

## Interfaccia Gradio

Avvia l'interfaccia Gradio:
```bash
python NanoGPT_Marinello.py
```
Gradio sarà disponibile su: http://127.0.0.1:7860

Funzionalità principali:
- Chat testuale o vocale (microfono / textbox)
- Training interattivo del modello
- Visualizzazione cronologia conversazioni
- Informazioni di sistema (CPU / RAM / GPU / Whisper status)

---

## Server REST (per Jetson / integrazione remota)

Avvia il server LLM:
```bash
python LlmServer.py
```
Ricordarsi di modificare il file e inserire in NOTEBOOK_API_URL l'ip del pc principale.

Endpoint principali:

- POST /api/train  
  Body JSON:
  ```json
  {
    "training_text": "...",
    "steps": 1000
  }
  ```
  Risposte:
  - 200 OK: `{"complete": "training completed"}`
  - 400/500: `{"error": "..."}`

  Nota: testi inferiori a 100 caratteri vengono rifiutati.

- POST /api/chat  
  Body JSON:
  ```json
  {
    "question": "...",
    "temperature": 0.7,
    "history": []
  }
  ```
  Risposta:
  ```json
  {
    "response": "testo generato"
  }
  ```
  La gestione della cronologia è basata sul database interno.

- GET /api/load  
  Risposta:
  ```json
  {
    "overview": [
      {
        "conversation_id": 1,
        "last_user": "...",
        "last_assistant": "..."
      }
    ]
  }
  ```

---

## Uso su Jetson (Modalità Ibrida)

1. Avvia il server LLM sul PC principale:
```bash
python LlmServer.py
```

2. Sul Jetson, esegui l’interfaccia grafica (GUI):
```bash
python GUI.py
```

3. Modifica la variabile `NOTEBOOK_API_URL` nel codice Jetson inserendo l’IP del PC principale.  
Questo consente la connessione tra GUI e server remoto.

Suggerimenti per Jetson:
- Utilizza versioni leggere di Whisper (tiny o small)
- Disattiva TTS se le risorse sono limitate

---

## Debug & Note operative

Problema: Model not loaded  
- Soluzione: Esegui il training o verifica la presenza di `checkpoint.pth`.

Problema: CUDA Out of Memory  
- Soluzione: Riduci `batch_size` o il numero di layer, oppure esegui su CPU.

Problema: Whisper non disponibile  
- Soluzione: La trascrizione verrà ignorata; la chat testuale funzionerà comunque.

---

## Suggerimenti

- Su GPU con VRAM limitata, riduci `n_layer` o lascia che `get_adaptive_config()` adatti automaticamente i parametri.
- Se esponi il server in rete, implementa autenticazione e rate-limiting per maggiore sicurezza.

---

## Riferimenti

Progetto sviluppato nell’ambito della tesi:

Matteo Zingrillara (2025)  
Assistenti LLM per l’Interazione con i Sistemi Autonomi  
Università degli Studi di Padova

Repository di origine:  
https://github.com/FrannPizz/NanoGPT-Marinello

Tesi di riferimento:

Francesco Pizzato (2025)  
Conversing with Robots: Building LLM Assistants to Understand and Utilize Autonomous Systems  
Università degli Studi di Padova
