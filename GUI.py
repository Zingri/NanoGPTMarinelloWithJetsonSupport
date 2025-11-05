# gui.py
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
from NanoGPT_Marinello import unified_chat, clear_chat_history, get_system_info, train_model
import requests
import json

NOTEBOOK_API_URL = "http://192.168.1.32:5000" #VA INSERITO L'IP DEL NOTEBOOK CHE SI VUOLE USARE COME SERVER

all_conv = []



class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NanoGPT Chat")
        #self.root.geometry("1250x600")
        self.root.resizable(True,True)

        self.main_frame= tk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky = "nsew")
        root.grid_rowconfigure(0, weight = 1)
        root.grid_columnconfigure(0, weight = 1)
        self.main_frame.grid_rowconfigure(0, weight = 1)
        self.main_frame.grid_columnconfigure(0, weight = 1)

        self.frame_chat = tk.Frame(self.main_frame)
        self.frame_training = tk.Frame(self.main_frame)
        self.frame_chat.grid(row=0, column=0, sticky="nsew")

        self._setup_chat_ui()
 
        self._setup_training_ui()

        # Cronologia chat
        self.chat_history = []

    def _setup_chat_ui(self):
        for i in range(13):
              self.frame_chat.grid_rowconfigure(i, weight=1)
        for j in range(3): 
              self.frame_chat.grid_columnconfigure(j, weight=1)
        self.txt_chat = scrolledtext.ScrolledText(self.frame_chat, wrap=tk.WORD, state='disabled')
        self.txt_chat.grid(row=0, column = 0, columnspan=3, rowspan= 10, padx=5, pady= 5, sticky = "nsew")

        # Input utente
        self.entry_user = tk.Entry(self.frame_chat)
        self.entry_user.grid(row=11, column = 0, columnspan=3, padx=5,  sticky = "new")
        self.entry_user.bind("<Return>", self.send_message)

        # Pulsanti
        frame_buttons = tk.Frame(self.frame_chat)
        frame_buttons.grid(row=12, column = 0, columnspan=3, padx=5, sticky = "new") 

        frame_buttons.grid_columnconfigure((0,1,2,3), weight = 1)
       
        tk.Button(frame_buttons, text="Invia", command=self.send_message).grid(row=0, column = 0, sticky = "we")
        tk.Button(frame_buttons, text="Cancella", command=self.clear_chat).grid(row=0, column = 1, sticky ="we")
        tk.Button(frame_buttons, text="Vai al Training", command =self.show_training).grid(row=0, column = 3, sticky = "we")

        #slider per la temperatura	
        tk.Label(frame_buttons, text="Creatività (temperature):").grid(row=2, column = 0, padx=10, pady = 10, sticky = "w")
        self.temp_slider= tk.Scale(frame_buttons, from_=0.1, to=2, resolution=0.1, orient=tk.HORIZONTAL)
        self.temp_slider.set(0.8)
        self.temp_slider.grid(row=2, column = 1, columnspan= 3, sticky ="ew")

        self.combo = ttk.Combobox(self.frame_chat, values = all_conv, state="readonly")
        self.combo.grid(row=13, column = 0, columnspan=3, padx=5, pady=5, sticky ="ew")
        #self.combo.bind("<<ComboboxSelected>>", on_select)
        self.get_all_conversations()

    def _setup_training_ui(self):
              
        """crea schermata di training"""
        for i in range(20):
                 self.frame_training.grid_rowconfigure(i, weight = 1)
        for j in range(1):
                 self.frame_training.grid_columnconfigure(j, weight = 1)

        tk.Label(self.frame_training, text="Inserisci testo per il training:").grid(row=0, column = 0, padx= 10, pady = 10, sticky ="nw")
        self.training_text = scrolledtext.ScrolledText(self.frame_training, wrap=tk.WORD)
        self.training_text.grid(row=1, column = 0,padx=10, pady=5,rowspan= 15, sticky= "nsew")
        frame_steps = tk.Frame(self.frame_training)
        frame_steps.grid(row=16, column = 0,padx=10, pady=5, sticky ="ew")
 
        #slider per gli step 
        tk.Label(frame_steps, text="Numero di step:").grid(row=0, column = 0, padx=10, sticky="w")
        self.step_slider= tk.Scale(frame_steps, from_=100, to=5000, resolution=100, orient=tk.HORIZONTAL)
        self.step_slider.set(0.8)
        self.step_slider.grid(row=0, column = 1, columnspan = 2,  sticky ="ew")

        tk.Button(frame_steps, text = "Avvia Training", command = self.start_training).grid(row=1, column = 1, sticky ="ew")
        tk.Button(frame_steps, text = "Torna alla chat", command = self.show_chat).grid(row=1, column = 2, sticky = "ew")


    def show_training(self):
        self.frame_chat.grid_forget()
        self.frame_training.grid(row=0, column = 0, sticky = "nsew")

    def show_chat(self):
        self.frame_training.grid_forget()
        self.frame_chat.grid(row=0, column = 0, sticky = "nsew")

    def send_message(self, event=None):
        user_text = self.entry_user.get().strip()
        if not user_text:
            return
        self.entry_user.delete(0, tk.END)


        # Mostra messaggio utente
        self.append_chat("You", user_text)

        # Chiama il modello
        try:
            #self.chat_history, _, _, _ = unified_chat(
            #    audio_input=None,
            #    manual_text=user_text,
            #    temperature=self.temp_slider.get(),
            #    history=self.chat_history
            #)
            #ai_text = self.chat_history[-1]["content"]
            
            data = {"question": user_text, "temperature": self.temp_slider.get()}
            print(f"data:{data}")
            response = requests.post(f"{NOTEBOOK_API_URL}/api/chat", json=data, timeout = 20)
            print(f"response:{response.json()}")
            ai_text = response.json().get("response", "")
            self.append_chat("AI", ai_text)
            print(f"ai_text: {ai_text}")
            if response.status_code == 200:
                   return ai_text
            else:
                   return f"Errore dal server: {response.status_code}"
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella generazione: {e}")

    def append_chat(self, sender, message):
        self.txt_chat.config(state='normal')
        self.txt_chat.insert(tk.END, f"{sender}: {message}\n\n")
        self.txt_chat.see(tk.END)
        self.txt_chat.config(state='disabled')

    def clear_chat(self):
        self.chat_history = []
        clear_chat_history(self.chat_history)
        self.txt_chat.config(state='normal')
        self.txt_chat.delete(1.0, tk.END)
        self.txt_chat.config(state='disabled')
 
    def start_training(self):
        text = self.training_text.get("1.0", tk.END).strip()
        try:
                 step = int(self.step_slider.get())
        except ValueError: 
                 messagebox.showwarning("Errore", "Inserisci un numero valido per gli step tra 0 e 5000")
                 return
        if not text: 
                 messagebox.showwarning("Errore", "Inserisci almeno 100 caratteri")
                 return
        #result = train_model(text, step)
        data = {"training_text": text, "steps": step}
        result = requests.post(f"{NOTEBOOK_API_URL}/api/train", json=data)
        if result.status_code == 200:
                 messagebox.showinfo("Training","Training completed")
                 return
        else:
                 messagebox.showwarning("training","Training failed")
   
    def get_all_conversations(self):
        try:
                 result = requests.get(f"{NOTEBOOK_API_URL}/api/load", timeout= 5)
                 conversations = result.json()
                 if result.status_code== 200:
                         conversation = result.json()
                         print(f"conversation:{conversations}")
                         all_conv = [f"{c['conversation_id']} - {c['last_user']}" for c in conversation["overview"]]
                         print(f"CONVERSAZIONI:{all_conv}")
                         self.combo['values'] = all_conv
                         if all_conv:
                                combo.current(0)
                 else: 
                         print(f"Errore server: {response.status_code}")
        except Exception as e: 
                 print  (f"Errore richiesta: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
