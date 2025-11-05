NanoGPT Marinello — Interfaccia Conversazionale Locale


NanoGPT Marinello è una versione minimalista sperimentale di un piccolo modello GPT pensata per esecuzione locale (CPU/GPU) e dispositivi Jetson. Fornisce:
- Interfaccia web con Gradio per chat testo/voce.
- API REST (Flask) per integrazione e uso remoto.
- Pipeline di training semplice con salvataggio checkpoint.
- Trascrizione (Whisper opzionale) e Text-To-Speech (gTTS).
- Salvataggio cronologia conversazioni in SQLite.

Prerequisiti principali
- Python 3.8+
- PyTorch (CPU o CUDA compatibile con la tua macchina)
- tokenizers, transformers (fallback tokenizer GPT-2)
- gradio
- whisper (opzionale, per trascrizione)
- gTTS (opzionale, per TTS)
- pygame (opzionale, riproduzione audio locale)
- pandas, psutil, ftfy
- sqlite3 (builtin)

Note per Jetson
- Installa la versione di PyTorch compatibile con Jetson (seguire le guide NVIDIA).
- Usa modelli Whisper più leggeri (tiny/small) o disabilita voce se risorse limitate.
- get_adaptive_config() nel codice adatta parametri in base a RAM/VRAM.

Installazione rapida (esempio)
1. Clona il repository:
   git clone https://github.com/Zingri/NanoGPTMarinelloWithJetsonSupport.git
   cd NanoGPTMarinelloWithJetsonSupport

2. Crea un virtualenv e attivalo:
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows

3. Installa tutte le dipendenze necessarie con: pip install -r requirements.txt

File chiave
- NanoGPT_Marinello.py — core: modello GPT, dataset BPE, training, Gradio UI, DB conversazioni.
- LlmServer.py — server Flask con endpoint REST (/api/train, /api/chat, /api/load).
- checkpoint.pth — checkpoint salvato dal training (generato da training).
- tokenizer.json — tokenizer BPE personalizzato (se presente viene caricato).
- chat_memory.db — SQLite con cronologia conversazioni.

Se non presente il tokenizer.json fare: python train_tokenizer.py

Come avviare l'interfaccia Gradio
- Esegui:
  python NanoGPT_Marinello.py
- Default: Gradio avviato su 127.0.0.1:7860
- Funzionalità UI:
  - Voice & Text Chat (microfono / textbox)
  - Train the Model (inserisci testo + imposti steps)
  - History (visualizza conversazioni salvate)
  - System Info (CPU/RAM/GPU/Whisper status)

Come avviare il server REST (per Jetson / integrazione)
- Esegui:
  python LlmServer.py
- Default: ascolta su 0.0.0.0:5000

API REST principali
- POST /api/train
  - Body JSON: {"training_text": "...", "steps": 1000}
  - Risposte:
    - 200: {"complete": "training comleted"} (testo originario nel codice)
    - 400/500: {"error": "..."} in caso di problemi
  - Nota: il codice rifiuta testi troppo corti (<100 caratteri).

- POST /api/chat
  - Body JSON: {"question":"...", "temperature":0.7, "history":[]}
  - Risposta: {"response": "testo generato"}
  - Il parametro history è accettato, ma la funzione attuale sovrascrive la gestione della cronologia basandosi su DB interno.

- GET /api/load
  - Risposta: {"overview": [ {conversation_id, last_user, last_assistant}, ... ]}

Sulla jetson in caso si voglia procedere con implementazione ibrida:
Dopo aver avviato il server LLM sul pc principale, avviare anche GUI.py sulla jetson.
Si aprirà un'interfaccia grafica con chat e trainer.
Ricordarsi di modificare NOTEBOOK_API_URL con l'IP del pc così che possano collegarsi.

Dettagli implementativi importanti
- Tokenizer: se `tokenizer.json` è presente viene caricato; altrimenti si usa il GPT-2 tokenizer Fast come fallback.
- Checkpoint: salvato in `checkpoint.pth` con stato modello, ottimizzatore e config. All'avvio il file viene caricato se presente.
- Dataset: classe BPEDataset che genera batch per training autoregressivo.
- Validazione: la generazione viene validata tramite calcolo della perplexity per evitare risposte incoerenti.
- Cronologia: salvata in SQLite (`chat_memory.db`) con funzioni per overview, caricamento e pulizia.

Suggerimenti pratici
- Se hai GPU con memoria limitata, riduci batch_size / n_layer tramite get_adaptive_config() o lascia che il codice auto-adatti i parametri.
- Per Jetson: preferisci versioni leggere dei modelli, disabilita TTS/Whisper se necessario.
- Proteggi gli endpoint REST se esponi il server in rete (autenticazione, rate limit).

Debug comune
- "Model not loaded": esegui il training o verifica che `checkpoint.pth` esista e sia caricabile.
- Errori di memoria GPU: esegui su CPU o riduci dimensioni modello / batch.
- Whisper non disponibile: la trascrizione verrà ignorata e la chat testuale funzionerà comunque.

-----

Note finali: 

Il seguente progetto è stato sviluppato assieme alla tesi:
Matteo Zingrillara(2025)
Assistenti LLM per l’Interazione con i Sistemi Autonomi ​
Università degli studi di Padova

Il programma è stato sviluppato dal codice che si trova nella repository:

https://github.com/FrannPizz/NanoGPT-Marinello

Tesi di riferimento:

-Francesco Pizzato (2025).
CONVERSING WITH ROBOTS: BUILDING LLM ASSISTANTS TO UNDERSTAND AND UTILIZE AUTONOMOUS SYSTEMS.
Bachelor’s thesis, Università degli Studi di Padova.
