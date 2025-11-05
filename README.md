 NanoGPT Marinello — Interfaccia Conversazionale Locale

NanoGPT Marinello è una versione minimalista di un modello GPT progettata per l’esecuzione locale (CPU/GPU) e su dispositivi NVIDIA Jetson.
Fornisce un assistente conversazionale completo con interfaccia web, API REST e funzionalità di training personalizzato.

 Caratteristiche principali

Interfaccia web (Gradio) per chat testuale e vocale.

API REST (Flask) per integrazione remota e uso su Jetson.

Pipeline di training semplice con salvataggio dei checkpoint.

Trascrizione vocale tramite Whisper (opzionale).

Text-To-Speech (TTS) tramite gTTS (opzionale).

Salvataggio cronologia conversazioni in SQLite.

Adattamento automatico dei parametri in base alle risorse di sistema.

⚙️ Requisiti

Python ≥ 3.8

PyTorch (CPU o CUDA compatibile)

tokenizers, transformers (fallback tokenizer GPT-2)

gradio, flask, pandas, psutil, ftfy

sqlite3 (builtin)

Opzionali:

whisper → trascrizione vocale

gTTS → sintesi vocale

pygame → riproduzione audio locale

 Installazione
# Clona il repository
git clone https://github.com/Zingri/NanoGPTMarinelloWithJetsonSupport.git
cd NanoGPTMarinelloWithJetsonSupport

# Crea e attiva l’ambiente virtuale
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Installa le dipendenze
pip install -r requirements.txt

 File principali
File	Descrizione
NanoGPT_Marinello.py	Core del progetto: modello GPT, training, UI Gradio, DB conversazioni
LlmServer.py	Server Flask con API REST
checkpoint.pth	Checkpoint del modello salvato
tokenizer.json	Tokenizer BPE personalizzato
chat_memory.db	Database SQLite delle conversazioni
train_tokenizer.py	Script per generare un nuovo tokenizer

Se tokenizer.json non è presente, esegui:

python train_tokenizer.py

 Interfaccia Gradio

Avvia con:

python NanoGPT_Marinello.py


Gradio sarà disponibile su http://127.0.0.1:7860

Funzionalità principali:

Chat testuale o vocale (microfono/textbox)

Training interattivo del modello

Visualizzazione cronologia conversazioni

Informazioni di sistema (CPU/RAM/GPU/Whisper status)

 Server REST (per Jetson / integrazione remota)

Avvia con:

python LlmServer.py


Server attivo su 0.0.0.0:5000

Endpoint principali
POST /api/train

Body JSON:

{"training_text": "...", "steps": 1000}


Risposte:

 200 OK: {"complete": "training completed"}

 400/500: {"error": "..."}

Nota: testi <100 caratteri vengono rifiutati.

POST /api/chat

Body JSON:

{"question":"...", "temperature":0.7, "history":[]}


Risposta:

{"response": "testo generato"}


La gestione della cronologia è basata sul DB interno.

GET /api/load

Risposta:

{
  "overview": [
    {"conversation_id": 1, "last_user": "...", "last_assistant": "..."}
  ]
}

🤖 Uso su Jetson (Modalità Ibrida)

Avvia il server LLM sul PC principale:

python LlmServer.py


Sul Jetson, esegui l’interfaccia grafica:

python GUI.py


Modifica NOTEBOOK_API_URL nel codice Jetson inserendo l’IP del PC principale.
Questo consente la connessione tra GUI e server remoto.

 Debug & Note operative
Problema	Soluzione
 Model not loaded	Esegui il training o verifica la presenza di checkpoint.pth.
 CUDA Out of Memory	Riduci batch_size o il numero di layer, o esegui su CPU.
 Whisper non disponibile	La trascrizione verrà ignorata, chat testuale funzionante.
 Suggerimenti

Su GPU con VRAM limitata, riduci n_layer o lascia che get_adaptive_config() scelga i parametri migliori.

Su Jetson, preferisci modelli tiny/small di Whisper e disattiva TTS se necessario.

Se esponi il server in rete, implementa autenticazione e rate-limiting.

📚 Riferimenti

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
