import os
import io
import math
import tempfile

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import re
import html
import logging
import importlib
try:
    ftfy = importlib.import_module('ftfy')
    _have_ftfy = True
except Exception:
    ftfy = None
    _have_ftfy = False

import gradio as gr
import psutil
import whisper
from gtts import gTTS
import pygame
import pandas as pd 

import math

import sqlite3

from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast

db = None
cursor = None
conv_id = int(time.time())

# -----------------------
# Implementing perplexity
# -----------------------

def calculate_perplexity(model, input_ids):
    """Calcola la perplexity su input_ids con controllo block_size."""
    with torch.no_grad():
        block_size = getattr(model, "config", {}).get("block_size", 256)
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, -block_size:]

        outputs = model(input_ids)
        logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "logits", outputs)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        try:
            ppl = torch.exp(loss).item()
        except OverflowError:
            ppl = float("inf")
        return ppl

    
def validate_response(model, tokenizer, context, generated, prompt_ids, max_ppl=50.0, min_length=1):
    """
    Valida la risposta generata:
    - Calcola Perplexity
    - Pulisce il testo
    - Verifica coerenza e artefatti
    """
    try:
        full_sequence = torch.cat([context, generated[:, len(prompt_ids):]], dim=1)
        block_size = getattr(model, "config", {}).get("block_size", 256)
        if full_sequence.size(1) > block_size:
            full_sequence = full_sequence[:, -block_size:]

        ppl = calculate_perplexity(model, full_sequence)

        gen_tokens = generated[0][len(prompt_ids):].tolist()
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        cleaned = clean_generated_text(raw, tokenizer=tokenizer)

        print(f"{ppl, len(cleaned.split()) < min_length}")

        if ppl > max_ppl:
            return False, cleaned, ppl
        if not cleaned or len(cleaned.split()) < min_length:
            return False, cleaned, ppl
        if any(bad in cleaned for bad in ["âĢ", "Âł", "[UNK]", "Ġ"]):
            return False, cleaned, ppl

        return True, cleaned, ppl

    except Exception as e:
        print(f"Errore nella validazione: {e}")
        return False, "", None



# -----------------------
# Load/create database
# -----------------------
db = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
)
""")
db.commit()

def save_message(conversation_id, role, content):
    with db:
        global conv_id
        conversation_id = conv_id
        print(f"Saving message: conv_id={conversation_id}, role={role}, content={content[:30]}...")
        cursor = db.cursor()
        cursor.execute("INSERT INTO conversation (conversation_id, role, content) VALUES (?, ?, ?)", (conversation_id, role, content))
        db.commit()

def get_last_messages(conversation_id, n):
    with db:
        cursor = db.cursor()
        cursor.execute("SELECT role, content FROM conversation WHERE conversation_id=? ORDER BY id DESC LIMIT ?", (conversation_id, n))
        rows = cursor.fetchall()
        return rows[::-1]

def get_conversations_overview():
    # crea un cursore locale
    with db: 
        cur = db.cursor()
        cur.execute("SELECT DISTINCT conversation_id FROM conversation ORDER BY conversation_id DESC")
        conv_ids = [row[0] for row in cur.fetchall()]

        overview = []
        for cid in conv_ids:
            # ultimo messaggio user
            cur.execute("SELECT content FROM conversation WHERE conversation_id=? AND role='user' ORDER BY id DESC LIMIT 1", (cid,))
            last_user = cur.fetchone()
            last_user = last_user[0] if last_user else ""
            # ultimo messaggio assistant
            cur.execute("SELECT content FROM conversation WHERE conversation_id=? AND role='assistant' ORDER BY id DESC LIMIT 1", (cid,))
            last_assistant = cur.fetchone()
            last_assistant = last_assistant[0] if last_assistant else ""
            overview.append({
                "conversation_id": cid,
                "last_user": last_user,
                "last_assistant": last_assistant
            })

    return overview


def clear_chat():
    with db:
        cursor = db.cursor()
        cursor.execute("DELETE FROM conversation WHERE conversation_id=?", (conv_id,))
        db.commit()

def clear_database():
    global db, cursor
    with db:
        cursor = db.cursor()
        cursor.execute("DELETE FROM conversation")  # cancella tutti i record
        db.commit()
    return pd.DataFrame(columns=["ID", "Last Question", "Last Answer"])

def load_conversation(conv_id):
    with db:
        cursor = db.cursor()
        cursor.execute("SELECT role, content FROM conversation WHERE conversation_id=? ORDER BY id ASC", (conv_id,))
        rows = cursor.fetchall()
    chat_history = [{"role": row[0], "content": row[1]} for row in rows]
    return chat_history
    


# -----------------------
# Adaptive config
# -----------------------
def get_adaptive_config():
    ram_gb = psutil.virtual_memory().total / (1024**3)

    # Impostazioni iniziali
    b, nl, nh, bs = 8, 2, 2, 128
    ne = nh * 32  # n_embd sempre multiplo di n_head

    if ram_gb > 8:
        b, nl, nh, bs = 16, 4, 4, 256
        ne = nh * 32
    if ram_gb > 16:
        b, nl, nh, bs = 32, 6, 6, 256
        ne = nh * 32

    try:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb > 6:
                b = max(b, 32)
                nl = max(nl, 6)
                nh = max(nh, 6)
                bs = max(bs, 256)
                ne = nh * 32
            if vram_gb > 12:
                b = max(b, 64)
                nl = max(nl, 8)
                nh = max(nh, 8)
                bs = max(bs, 512)
                ne = nh * 64  # più capacità, più embedding
    except Exception:
        pass

    return b, nl, nh, ne, bs


# -----------------------
# Tokenizer
# -----------------------

tokenizer_file = "tokenizer.json"
bpe_tokenizer = None

# Tentativo di caricamento del tokenizer locale
if os.path.exists(tokenizer_file):
    try:
        bpe_tokenizer = Tokenizer.from_file(tokenizer_file)
        vocab_size = len(bpe_tokenizer.get_vocab())
        if vocab_size > 1000:
            print(f"✅ Tokenizer caricato da '{tokenizer_file}' (vocab_size={vocab_size})")
        else:
            print(f"⚠️ Tokenizer caricato da '{tokenizer_file}' ma sembra non essere addestrato su dati reali (vocab_size={vocab_size}).")
    except Exception as e:
        print(f"❌ Impossibile caricare '{tokenizer_file}': {e}")
        bpe_tokenizer = None

# Fallback su GPT-2 tokenizer pre-addestrato
if bpe_tokenizer is None:
    print("⚠️ Utilizzo di un tokenizer di fallback pre-addestrato (GPT-2).")
    # GPT-2 tokenizer Fast restituisce un oggetto compatibile con tokenizers
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # Convertiamo in oggetto tokenizers (solo se necessario)
    bpe_tokenizer = gpt2_tok.backend_tokenizer
    vocab_size = len(bpe_tokenizer.get_vocab())
    print(f"✅ Tokenizer di fallback GPT-2 caricato (vocab_size={vocab_size})")


# -----------------------
# Globals / Hyperparameters
# -----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
batch_size, n_layer, n_head, n_embd, block_size = get_adaptive_config()

print(f"""
Configurazione adattiva scelta:
- Batch size: {batch_size}
- N° Layers: {n_layer}
- N° Heads: {n_head}
- Embedding size: {n_embd}
- Block size: {block_size}
""")

model = None
optimizer = None
dataset = None

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def safe_decode(tokenizer, ids, skip_special_tokens=True):
    try:
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tokenizer.decode(ids)

# -----------------------
# Optional components
# -----------------------
try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper loaded successfully")
except Exception as e:
    whisper_model = None
    print(f"⚠️ Whisper not available: {e}")

try:
    pygame.mixer.init()
    print("✅ Pygame mixer initialized")
except Exception as e:
    print(f"⚠️ Pygame mixer not initialized: {e}")

# -----------------------
# Utility
# -----------------------
def bytes_to_tempfile(data, suffix=".mp3"):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    return f.name

def new_conversation(chat_state):
    global conv_id
    conv_id = int(time.time())
    chat_state = []     # resetta lo stato
    return [], None, "", chat_state

# Funzione di utilità per troncare testo lungo
def truncate_text(text, max_len=50):
    if text is None:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "..."


def show_history():
    overview = get_conversations_overview()
    if not overview:
        return pd.DataFrame(columns=["ID", "Last Question", "Last Answer"])
    
    for conv in overview:
        conv["last_user"] = truncate_text(conv["last_user"], 50)
        conv["last_assistant"] = truncate_text(conv["last_assistant"], 50)
    
    # Creiamo un dataframe per renderlo tabellare in Gradio
    df = pd.DataFrame(overview)
    df = df.rename(columns={
        "conversation_id": "ID",
        "last_user": "Last Question",
        "last_assistant": "Last Answer"
    })
    return df


# Robust text cleaning for generated model output
def clean_generated_text(text, tokenizer=None, max_len=2000):
    if not text:
        return text
    # If tokenizer provided and supports decoding from ids, leave that to caller.
    # Normalize HTML entities and control characters
    text = html.unescape(text)
    # remove common control chars
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)

    if ' . Ġ' in text:
            idx = text.find(' . Ġ') + len(' . Ġ')  # trova la posizione e aggiungi la lunghezza del pattern
            text = text[:idx]

    if ').' in text:
        idx = text.find(').') + len(').')      # mantiene il pattern
        text = text[:idx]

    # Common garbage token sequences mapping -> replacement
    replacements = {
        "[UNK]": "",
        "Ġ": " ",
        "âĢ¯": "",
        "âĢĳ": "-",
        "Âł": "",
        "âĢĻ": "'",
        "\u200b": "",
        " ' ": "'",
        " - ":"-",
        " âĢij ":"-",
        " âī¥ ": "≥",
        " ÃĹ " : "x",
        "( " : "(",
        " )" : ")"
    }
    # Apply ftfy if available to fix encoding artifacts before manual replacements
    if _have_ftfy:
        try:
            text = ftfy.fix_text(text)
        except Exception:
            pass

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove role tags accidentally included in the generation (case-insensitive)
    # Cut at the earliest likely role-token marker to avoid the model continuing the conversation
    split_markers = ["user:", "ia:", "ai:", "assistant:", "user:"]
    text_lower = text.lower()
    earliest_idx = None
    for m in split_markers:
        idx = text_lower.find(m)
        if idx != -1:
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx
    if earliest_idx is not None:
        text = text[:earliest_idx]

    # Fix spacing around punctuation: no space before ,.;:?! and exactly one space after
    text = re.sub(r"\s+([,\.;:\?!])", r"\1", text)
    text = re.sub(r"([,\.;:\?!])(\S)", r"\1 \2", text)

    # Collapse repeated spaces and newlines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace multiple punctuation (e.g., "!!!" or "???") with a max of 3
    text = re.sub(r"([!?]){3,}", r"\1\1\1", text)

    # Trim leading/trailing whitespace and stray punctuation
    text = text.strip()
    # Limit length to avoid bizarre runaway outputs
    if max_len and len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
    return text

def open_conversation(chat_state):
    chat_history = load_conversation(conv_id)
    chat_state = chat_history  # aggiorna lo stato
    return chat_history, chat_state

def load_selected_conversation(selection):
    if not selection:
        return [], []
    global conv_id
    conv_id = int(selection.split(" - ")[0])
    print(f"Loading conversation ID {conv_id}...")
    db_local = sqlite3.connect("chat_memory.db", check_same_thread=False)
    cur = db_local.cursor()
    cur.execute("SELECT role, content FROM conversation WHERE conversation_id=? ORDER BY id ASC", (conv_id,))
    rows = cur.fetchall()
    chat_history = [{"role": row[0], "content": row[1]} for row in rows]
    db_local.close()
    return chat_history, chat_history


def get_conversations_for_dropdown():
    overview = get_conversations_overview()
    choices = [
        f"{conv.get('conversation_id', 'N/A')} - {truncate_text(conv.get('last_user', ''), 50)}"
        for conv in overview
    ]
    return choices

def update_dropdown():
    choices = get_conversations_for_dropdown()
    return gr.update(choices=choices)

def build_prompt(history, user_text, max_turns=3):
    """
    Costruisce un prompt in stile USER / AI per NanoGPT.
    - Usa solo gli ultimi max_turns messaggi utili.
    - Non include risposte fallite.
    """
    prompt_text = ""
    
    # Filtra la cronologia rimuovendo messaggi 'falliti' (es. placeholder "Sorry, I couldn't generate")
    filtered_history = [msg for msg in history if msg.get("content") and "Sorry" not in msg["content"]]
    
    # Prendi solo gli ultimi max_turns *interazioni complete* (USER+AI)
    recent = filtered_history[-2*max_turns:] if len(filtered_history) > 2*max_turns else filtered_history

    for msg in recent:
        role = "USER" if msg["role"] == "user" else "AI"
        prompt_text += f"{role}: {msg['content'].strip()}\n"
    
    # Aggiungi l'ultimo input dell'utente
    prompt_text += f"USER: {user_text.strip()}\nAI:"
    return prompt_text




# -----------------------
# Dataset
# -----------------------
class BPEDataset:
    def __init__(self, text=None, tokenizer=bpe_tokenizer, block_size=128):
        if tokenizer is None:
            raise ValueError("Tokenizer is None: train or provide a valid tokenizer.json before creating dataset.")
        self.tokenizer = tokenizer
        self.block_size = block_size
        if text:
            encoding = tokenizer.encode(text)
            self.data = torch.tensor(encoding.ids, dtype=torch.long)
        else:
            self.data = None
        vocab = tokenizer.get_vocab()
        self.stoi = vocab
        self.itos = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

    def get_batch(self, batch_size, device='cpu'):
        if self.data is None:
            raise ValueError("Dataset empty: train or load first.")
        n = len(self.data)
        b_size = self.block_size
        if n <= b_size:
            repeated = self.data.repeat((math.ceil((b_size + 1) / n),))[:b_size + 1]
            x = repeated[:b_size].unsqueeze(0).repeat(batch_size, 1)
            y = repeated[1:b_size + 1].unsqueeze(0).repeat(batch_size, 1)
            return x.to(device), y.to(device)
        max_start = n - b_size - 1
        ix = torch.randint(0, max_start + 1, (batch_size,))
        # convert tensor indices to plain Python ints to avoid indexing/slicing issues
        starts = ix.tolist()
        x = torch.stack([self.data[s:s + b_size] for s in starts])
        y = torch.stack([self.data[s + 1:s + 1 + b_size] for s in starts])
        return x.to(device), y.to(device)

# -----------------------
# GPT Model
# -----------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.c_attn = nn.Linear(config['n_embd'], 3*config['n_embd'], bias=False)
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                             .view(1,1,config['block_size'],config['block_size']))

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(C, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4*config['n_embd'], bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config['n_embd'], config['n_embd'], bias=False)
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd'])
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B,T = idx.size()
        # Guard against exceeding configured block size
        if T > self.config['block_size']:
            raise ValueError(f"Sequence length T={T} > block_size={self.config['block_size']}. Truncate input before calling model.")
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k, top_p= 0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:] if idx.size(1) > self.config['block_size'] else idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / (temperature if temperature>0 else 1.0)

            # Clamp top_k to vocab size
            V = logits.size(-1)
            if top_k is None:
                top_k = 0
            k = min(max(int(top_k), 0), V)
            if k > 0:
                v, _ = torch.topk(logits, k)
                min_v = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_v, torch.tensor(-float('Inf')).to(logits.device), logits)

            # Apply top-p (nucleus) filtering if requested
            if top_p and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Determine tokens to remove
                sorted_indices_to_remove = cumulative_probs > float(top_p)
                # Always keep at least the top token
                sorted_indices_to_remove[..., 0] = False
                # Create a mask of tokens to remove in original indexing
                for b_i in range(sorted_indices.size(0)):
                    remove_idx = sorted_indices[b_i][sorted_indices_to_remove[b_i]]
                    if remove_idx.numel() > 0:
                        logits[b_i, remove_idx] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# -----------------------
# Checkpoint
# -----------------------
def save_checkpoint(model, optimizer, dataset, training_text="", filename="checkpoint.pth"):
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **({'tokenizer_file': tokenizer_file} if os.path.exists(tokenizer_file) else {}),
        'config': model.config,
        'training_text': training_text
    }
    try:
        if dataset is not None and getattr(dataset, 'tokenizer', None) is not None:
            payload['tokenizer_vocab_size'] = len(dataset.tokenizer.get_vocab())
    except Exception:
        payload['tokenizer_vocab_size'] = None
    torch.save(payload, filename)
    print(f"✅ Checkpoint saved: {filename}")

def load_checkpoint(filename):
    global model, optimizer, dataset, bpe_tokenizer
    if not os.path.exists(filename):
        print("❌ Checkpoint not found")
        return "Checkpoint not found"
    ckpt = torch.load(filename, map_location=device)
    config = ckpt.get('config')
    if config is None:
        return "❌ Checkpoint missing config."
    if 'tokenizer_file' in ckpt and os.path.exists(ckpt['tokenizer_file']):
        try:
            bpe_tokenizer = Tokenizer.from_file(ckpt['tokenizer_file'])
        except Exception as e:
            print(f"⚠️ Impossibile caricare tokenizer dal checkpoint: {e}")
            bpe_tokenizer = None
    ds_block = config.get('block_size', block_size)
    if 'training_text' in ckpt:
        try:
            dataset = BPEDataset(text=ckpt['training_text'], tokenizer=bpe_tokenizer, block_size=ds_block)
        except Exception as e:
            print(f"⚠️ Impossibile creare dataset dal checkpoint: {e}")
            dataset = None
    else:
        try:
            dataset = BPEDataset(text=None, tokenizer=bpe_tokenizer, block_size=ds_block)
        except Exception as e:
            print(f"⚠️ Impossibile creare dataset vuoto: {e}")
            dataset = None
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print("✅ Checkpoint loaded")
    return "Checkpoint loaded successfully"

checkpoint_file = "checkpoint.pth"
if os.path.exists(checkpoint_file):
    load_checkpoint(checkpoint_file)
else:
    print("⚠️ Nessun checkpoint trovato. Addestra il modello prima di chattare.")

if model is not None:
    print("✅ Checkpoint vocab size:", model.config.get('vocab_size', 'Unknown'))
    if bpe_tokenizer is not None:
        print("✅ Tokenizer vocab size:", len(bpe_tokenizer.get_vocab()))
    else:
        print("⚠️ Tokenizer non disponibile")
else:
    print("❌ Model not loaded. Skip vocab size check.")

import time

# -----------------------
# Training con Validation
# -----------------------
def train_model(input_text, steps):
    global dataset, model, optimizer, batch_size, n_layer, n_head, n_embd, block_size

    # Parametri di stabilità
    use_amp = True
    max_grad_norm = 1.0
    accum_steps = 1

    # Early Stopping sulla validation loss
    early_stop_patience = 100
    min_delta = 1e-3
    best_val_loss = float('inf')
    steps_since_improvement = 0

    if not input_text or len(input_text.strip()) < 100:
        return "❌ Text too short!"
    if bpe_tokenizer is None:
        return "❌ Tokenizer not available. Run train_tokenizer.py first."

    # Dividi dataset in train/validation
    full_dataset = BPEDataset(text=input_text, tokenizer=bpe_tokenizer, block_size=block_size)
    split_idx = int(0.9 * len(full_dataset.data))
    train_data = full_dataset.data[:split_idx]
    val_data = full_dataset.data[split_idx:]
    train_set = BPEDataset(text=None, tokenizer=bpe_tokenizer, block_size=block_size)
    val_set = BPEDataset(text=None, tokenizer=bpe_tokenizer, block_size=block_size)
    train_set.data = train_data
    val_set.data = val_data

    config = {
        'vocab_size': full_dataset.vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    model.train()

    scaler = torch.cuda.amp.GradScaler() if use_amp and device == 'cuda' else None

    running_loss = 0.0
    running_count = 0

    total_steps = int(steps)
    start_time = time.time()
    block_time = time.time()

    optimizer.zero_grad()

    def evaluate_validation_loss():
        model.eval()
        total_val_loss = 0.0
        n_batches = 20  # batch di validazione per la media
        with torch.no_grad():
            for _ in range(n_batches):
                xb, yb = val_set.get_batch(batch_size, device)
                if xb is None or yb is None:
                    continue
                logits, loss = model(xb, yb)
                total_val_loss += loss.item()
        model.train()
        return total_val_loss / n_batches

    for step in range(total_steps):
        xb, yb = train_set.get_batch(batch_size, device)
        if xb is None or yb is None:
            return f"❌ get_batch failed at step {step}"

        # forward/backward con AMP opzionale
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = model(xb, yb)
            loss = loss / accum_steps
            scaler.scale(loss).backward()
        else:
            logits, loss = model(xb, yb)
            loss = loss / accum_steps
            loss.backward()

        running_loss += loss.item() * accum_steps
        running_count += 1

        # ottimizzazione
        if (step + 1) % accum_steps == 0:
            if max_grad_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # logging ogni 50 step
        if (step + 1) % 50 == 0:
            avg_train_loss = running_loss / running_count
            val_loss = evaluate_validation_loss()
            elapsed = time.time() - block_time
            print(f"Step {step+1}/{total_steps}, train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, time={elapsed:.2f}s")

            # early stopping sulla val_loss
            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                steps_since_improvement = 0
                try:
                    save_checkpoint(model, optimizer, train_set, training_text=input_text)
                    print(f"💾 Checkpoint salvato (miglior modello) a step {step+1}")
                except Exception as e:
                    print(f"⚠️ Errore salvataggio checkpoint: {e}")
            else:
                steps_since_improvement += 50

            if steps_since_improvement >= early_stop_patience:
                print(f"⏹️ Early stopping attivato a step {step+1} (val_loss={val_loss:.4f})")
                break

            running_loss = 0.0
            running_count = 0
            block_time = time.time()

    total_time = time.time() - start_time
    try:
        save_checkpoint(model, optimizer, train_set, training_text=input_text)
    except Exception as e:
        print(f"⚠️ Errore salvataggio checkpoint finale: {e}")

    return f"✅ Training done. Best val_loss={best_val_loss:.4f}, total_time={total_time:.2f}s"





# -----------------------
# Audio / Chat
# -----------------------
def transcribe_audio(audio_file):
    if audio_file is None: return "No audio", ""
    if whisper_model is None: return "Whisper not available", ""
    result = whisper_model.transcribe(audio_file, language="en")
    return result["text"], result["text"]

def text_to_speech(text, lang="en"):
    if not text: return None
    buf=io.BytesIO()
    tts=gTTS(text=text,lang=lang,slow=False)
    tts.write_to_fp(buf)
    buf.seek(0)
    data=buf.read()
    buf.close()
    return data

def unified_chat(audio_input, manual_text, temperature, history):
    """
    Funzione principale di chat:
    - Gestisce input audio o testo
    - Costruisce prompt
    - Genera risposta con gestione dei token e troncamento
    - Aggiorna cronologia e salva su DB
    """
    global model, dataset, conv_id

    print(f"SONO NELLA FUNZIONE")

    if model is None:
        return history + [{"role": "system", "content": "❌ Model not loaded."}], None, "", history

    print(f"DETERMINO INPUT")

    # 1. Determina input utente
    user_text = ""
    if audio_input:
        try:
            _, user_text = transcribe_audio(audio_input)
        except Exception:
            return history + [{"role": "system", "content": "❌ Transcription failed."}], None, "", history
    elif manual_text and manual_text.strip():
        user_text = manual_text.strip()

    if not user_text:
        return history + [{"role": "system", "content": "❌ No text detected."}], None, "", history

    print(f"AGGIORNO CROLOLOGIA")

    print(f"HISTORY: {history}")

    # 2. Aggiorna cronologia
    if history is not None:
        new_history = list(history)
    else:
        new_history= []

    print(f"NEW HISTORY")
    new_history.append({"role": "user", "content": user_text})

    print(f"COSTRUISCO PROMPT")

    # 3. Costruisci prompt ottimizzato
    prompt_text = build_prompt(new_history, user_text)

    print(f"TOKENIZZAZIONE")

    # 4. Tokenizzazione e troncamento al block_size
    tokenizer_to_use = dataset.tokenizer if dataset and getattr(dataset,'tokenizer',None) else bpe_tokenizer
    if tokenizer_to_use is None:
        return history + [{"role": "system", "content": "❌ No tokenizer available."}], None, "", history

    prompt_enc = tokenizer_to_use.encode(prompt_text)
    prompt_ids = prompt_enc.ids[-block_size:] if len(prompt_enc.ids) > block_size else prompt_enc.ids
    context = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    print(f"INIZIO A GENERARE")

    # 5. Generazione testo
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=100,       # più lunghezza per risposte complesse
                temperature=float(temperature),
                top_k=100,                # più libertà nel vocabolario
                top_p=0.95
            )
    except Exception as e:
        print(f"Errore nella generazione: {e}")
        return new_history + [{"role": "system", "content": "❌ Model generation failed."}], None, "", history

    # 6. Decodifica
    gen_tokens = generated[0][len(prompt_ids):].tolist()
    try:
        raw = tokenizer_to_use.decode(gen_tokens, skip_special_tokens=True)
        is_valid, ai_text, perplexity_score = validate_response(model, tokenizer_to_use, context, generated, prompt_ids)
        if not is_valid or not ai_text.strip():
            ai_text = "Sorry, I couldn't generate a coherent response. Could you rephrase your question?"
    except Exception:
        ai_text = "Sorry, I couldn't generate a coherent response."

    ai_text = clean_generated_text(ai_text, tokenizer=tokenizer_to_use)
    print(f"[AI]: {ai_text} | Perplexity: {perplexity_score:.2f}")

    # 7. Salvataggio su DB
    save_message(conv_id, "user", user_text)
    save_message(conv_id, "assistant", ai_text)

    # 8. Aggiorna cronologia
    new_history.append({"role": "assistant", "content": ai_text})
    return new_history, None, "", new_history


def clear_chat_history(chat_state):
    clear_chat()        # cancella dal DB
    chat_state = []     # resetta lo stato
    return [], None, "", chat_state

def get_system_info():
    """
    Collect minimal system info for display.
    """
    info = f"""System Information:
- CPU: {psutil.cpu_count()} cores
- RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB
- CUDA: {torch.cuda.is_available()}
"""
    if torch.cuda.is_available():
        info += f"""- GPU: {torch.cuda.get_device_name(0)}
- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB
- PyTorch CUDA: {torch.version.cuda}
"""
    else:
        info += "- GPU: Not available (training on CPU)"

    info += f"\n- Whisper: {'✅ Available' if whisper_model else '❌ Not available'}"
    info += f"\n- gTTS: ✅ Available"
    info += f"\n- Pygame: ✅ Available"

    return info

with gr.Blocks(title="NanoGPT Marinello") as demo:
    chat_state = gr.State([])
    tabs = gr.Tabs()
    with tabs:
        with gr.Tab("1️⃣ Voice & Text Chat") as chat:

            
            with gr.Row():
                # Left column: controls
                with gr.Column(scale=1, min_width=300, elem_classes=["left-panel"]):
                    gr.Markdown("#### 🎙️ Voice Input")
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        label="Voice input",
                        type="filepath"
                    )

                    gr.Markdown("#### ⌨️ Text Input")
                    text_input = gr.Textbox(
                        label="Type your question",
                        lines=3,
                        max_lines=5,
                        placeholder="Type your question...", 
                        autofocus=True
                    )

                    gr.Markdown("#### ⚙️ Parameters")
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="🌡️ Temperature",
                            info="Creativity (0.1=conservative, 2.0=creative)"
                        )

                    with gr.Row():
                        send_button = gr.Button("💫 Send", variant="primary")
                        clear_button = gr.Button("🗑️ Clear Chat", variant="secondary")
                        listen_button = gr.Button("🎧 Listen to response", variant="secondary")
                        new_conv_button = gr.Button("New conversation")

                # Right column: chat area
                with gr.Column(scale=2, min_width=500, elem_classes=["right-panel"]):
                    gr.Markdown("#### select past conversations")
                    conversation_dropdown = gr.Dropdown(
                        choices=get_conversations_for_dropdown(),
                        label="Select a past conversation",
                        interactive=True
                    )

                    gr.Markdown("#### 💭 Conversation")
                    chatbot = gr.Chatbot(
                        label=None,
                        height=450,
                        show_label=False,
                        bubble_full_width=False,
                        layout="panel",
                        type="messages"
                    )

                    gr.Markdown("#### 🔊 Audio Response")
                    audio_output = gr.Audio(
                        label="Listen to response",
                        type="filepath",
                        interactive=False,
                        autoplay=True
                    )

            # Event handlers
            send_button.click(
                unified_chat,
                [audio_input, text_input, temperature, chat_state],
                [chatbot, audio_output, text_input, chat_state],
                show_progress=True
            )

            clear_button.click(
                clear_chat_history,
                inputs=[chat_state],
                outputs=[chatbot, audio_output, text_input, chat_state]
            )

            new_conv_button.click(
                new_conversation,
                inputs=[chat_state],
                outputs=[chatbot, audio_output, text_input, chat_state]
            )

        with gr.Tab("2️⃣ Train the Model"):
            gr.Markdown("### 📚 Train your NanoGPT model")

            with gr.Row():
                with gr.Column():
                    training_text = gr.Textbox(
                        label="📝 Training Text",
                        lines=15,
                        max_lines = 25,
                        placeholder="Paste training text..."
                        
                    )
                
                    with gr.Row():
                        training_steps = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="🔄 Training Steps",
                            info="More steps = better quality but longer time"
                        )

                    train_button = gr.Button("🚀 Start Training", variant="primary")
            
                with gr.Column():
                    training_output = gr.Textbox(
                        label="📊 Training Status",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )

                    system_info_button = gr.Button("💻 System Info", variant="secondary")

            train_button.click(
                train_model,
                [training_text, training_steps],
                [training_output],
            )
            
            system_info_button.click(
                fn=get_system_info,
                inputs=[],
                outputs=[training_output]
            )

        with gr.Tab("ℹ️ Information"):
            gr.Markdown("""
            ## 🛥️ NanoGPT Marinello - Quick Guide

            ### Getting Started
            1. **Train**: go to the "Train the Model" tab, paste at least 100 characters, then run 1000+ steps.
            2. **Chat**: use voice or text, set temperature, click **Send**, optionally listen to TTS.

            ### Parameters
            - **Temperature**: 0.1–2.0 (lower = safer, higher = more creative)
            - **Steps**: 100–5000 (more = better but slower)

            ### Notes
            - Context length: the context depends on the corpus.
            - Whisper + gTTS are optional but recommended for voice I/O.
            - A checkpoint is saved after every 500 steps and after completing the training.
            """)

        with gr.Tab("History") as history_tab:
            history_table = gr.Dataframe(headers=["ID", "Last Question", "Last Answer"], interactive=False, type  = "pandas")
            delete_history_button = gr.Button("Delete History")
            delete_history_button.click(clear_database,outputs=[history_table] )

    #come apossibile aggiunta si potrebbe fare in modo che si possa apire una vecchia chat direttamente dal tab history
            
    
    conversation_dropdown.change(
    fn=update_dropdown,
    inputs=[],
    outputs=[conversation_dropdown]
    )

    chat.select(
        fn=update_dropdown,
        inputs=[],
        outputs=[conversation_dropdown]
    )

    conversation_dropdown.select(
    fn=load_selected_conversation,
    inputs=[conversation_dropdown],
    outputs=[chatbot, chat_state]
    )

    history_tab.select(
    fn=show_history,
    inputs=[],
    outputs=[history_table]
)

if __name__ == "__main__":
    print("🛥️ Starting NanoGPT Marinello...")
    ckpt_msg = load_checkpoint("checkpoint.pth")
    print(ckpt_msg)
    print("🌐 Opening web interface...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
