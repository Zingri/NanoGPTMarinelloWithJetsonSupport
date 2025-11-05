# 🤖 NanoGPT Marinello — Interfaccia Conversazionale Locale

**NanoGPT Marinello** è una versione minimalista di un modello GPT progettata per l’esecuzione **locale (CPU/GPU)** e su **dispositivi NVIDIA Jetson**.  
Fornisce un assistente conversazionale completo con interfaccia web, API REST e funzionalità di training personalizzato.

---

## ✨ Caratteristiche principali

- **Interfaccia web (Gradio)** per chat testuale e vocale  
- **API REST (Flask)** per integrazione remota e uso su Jetson  
- **Pipeline di training** semplice con salvataggio dei checkpoint  
- **Trascrizione vocale** tramite *Whisper* (opzionale)  
- **Text-To-Speech (TTS)** tramite *gTTS* (opzionale)  
- **Salvataggio cronologia conversazioni** in SQLite  
- **Adattamento automatico** dei parametri in base alle risorse di sistema  

---

## ⚙️ Requisiti

### Dipendenze principali
- Python ≥ 3.8  
- PyTorch (CPU o CUDA compatibile)  
- `tokenizers`, `transformers` (fallback tokenizer GPT-2)  
- `gradio`, `flask`, `pandas`, `psutil`, `ftfy`  
- `sqlite3` (builtin)

### Opzionali
- `whisper` → trascrizione vocale  
- `gTTS` → sintesi vocale  
- `pygame` → riproduzione audio locale  

---

## 🧩 Installazione

### 1️⃣ Clona il repository
```bash
git clone https://github.com/Zingri/NanoGPTMarinelloWithJetsonSupport.git
cd NanoGPTMarinelloWithJetsonSupport
2️⃣ Crea e attiva l’ambiente virtuale
bash
Copia codice
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
3️⃣ Installa le dipendenze
bash
Copia codice
pip install -r requirements.txt
📁 File principali
File	Descrizione
NanoGPT_Marinello.py	Core del progetto: modello GPT, training, UI Gradio, DB conversazioni
LlmServer.py	Server Flask con API REST
checkpoint.pth	Checkpoint del modello salvato
tokenizer.json	Tokenizer BPE personalizzato
chat_memory.db	Database SQLite delle conversazioni
train_tokenizer.py	Script per generare un nuovo tokenizer

Se tokenizer.json non è presente, esegui:

bash
Copia codice
python train_tokenizer.py
💬 Interfaccia Gradio
Avvia con:

bash
Copia codice
python NanoGPT_Marinello.py
Gradio sarà disponibile su: http://127.0.0.1:7860

Funzionalità principali
Chat testuale o vocale (microfono/textbox)

Training interattivo del modello

Visualizzazione cronologia conversazioni

Informazioni di sistema (CPU/RAM/GPU/Whisper status)

🌐 Server REST (per Jetson / integrazione remota)
Avvia con:

bash
Copia codice
python LlmServer.py
Server attivo su 0.0.0.0:5000

Endpoint principali
🔹 POST /api/train
Body JSON:

json
Copia codice
{"training_text": "...", "steps": 1000}
Risposte:

✅ 200 OK: {"complete": "training completed"}

❌ 400/500: {"error": "..."}

Nota: testi inferiori a 100 caratteri vengono rifiutati.

🔹 POST /api/chat
Body JSON:

json
Copia codice
{"question": "...", "temperature": 0.7, "history": []}
Risposta:

json
Copia codice
{"response": "testo generato"}
La gestione della cronologia è basata sul database interno.

🔹 GET /api/load
Risposta:

json
Copia codice
{
  "overview": [
    {
      "conversation_id": 1,
      "last_user": "...",
      "last_assistant": "..."
    }
  ]
}
🧠 Uso su Jetson (Modalità Ibrida)
Avvia il server LLM sul PC principale:

bash
Copia codice
python LlmServer.py
Sul Jetson, esegui l’interfaccia grafica:

bash
Copia codice
python GUI.py
Modifica la variabile NOTEBOOK_API_URL nel codice Jetson inserendo l’IP del PC principale.
Questo consente la connessione tra GUI e server remoto.

🧪 Debug & Note operative
Problema	Soluzione
⚠️ Model not loaded	Esegui il training o verifica la presenza di checkpoint.pth.
💥 CUDA Out of Memory	Riduci batch_size o il numero di layer, oppure esegui su CPU.
🔇 Whisper non disponibile	La trascrizione verrà ignorata, la chat testuale funzionerà comunque.

💡 Suggerimenti
Su GPU con VRAM limitata, riduci n_layer o lascia che get_adaptive_config() adatti automaticamente i parametri.

Su Jetson, utilizza versioni leggere di Whisper (tiny o small) e disattiva TTS se necessario.

Se esponi il server in rete, implementa autenticazione e rate-limiting per maggiore sicurezza.

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
