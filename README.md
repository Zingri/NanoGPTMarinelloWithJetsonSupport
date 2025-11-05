NanoGPT Marinello — Interfaccia Conversazionale Locale

Questo progetto implementa un’interfaccia Gradio per interagire con un piccolo modello linguistico (NanoGPT) in modo locale, oppure direttamente su un dispositivo jetson in modalità ibrida, il pc principale svolge le operazioni di training e risposta mentre il disositivo la visualizzazione.

Funzionalità principali

-Interfaccia grafica (Gradio) per chat testuale e vocale.
-Gestione conversazioni tramite database SQLite (chat_history.db).
-Sintesi vocale (TTS) con gTTS e pygame per risposte parlate.
-GUI implementabile su dispositivo esterno che permette la comunicazione tra i dispositivi

Requisiti 

Assicurarsi di avere installato:
-Almeno python 3.9.18
-tutte le dipendenze necessarie 

Le dipendenze sono installabili attraverso il comando: pip install -r requirements
