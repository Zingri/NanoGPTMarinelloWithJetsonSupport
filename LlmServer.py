from flask import Flask, request, jsonify
from NanoGPT_Marinello import unified_chat, train_model, get_conversations_overview

app = Flask(__name__)

modelName = "NanoGPT_Marinello"

@app.route("/api/train", methods=["POST"])
def train():
    print(f"INIZIO")
    data = request.get_json()
    if not data or "training_text" not in data:
        return jsonify({"error": "Missing 'training_tesx' field"}), 400
    

    training_text= data.get("training_text", "")
    steps = data.get("steps", 1000)

    try:
        train_model(training_text,steps)
        return jsonify({"complete": "training comleted"}) , 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 



@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint principale: riceve una domanda e restituisce la risposta generata dal modello."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400


    question = data.get("question", "")
    temperature = data.get("temperature", 0.7)

    print(f"question:{question}")

    history = data.get("history", [])

    # Chiama il modello
    try:
        print(f"ENTRO")
        history, _, _, _ = unified_chat(
            audio_input=None,
            manual_text=question,
            temperature=temperature,
            history=None
        )
        print(f"HO LA MIA RIPSOSTA")

        ai_reply = history[-1]["content"] if history else "Nessuna risposta generata."
        print(f"ai_reply:{ai_reply}")
        return jsonify({"response": ai_reply})
    
    except Exception as e:
        print(f"cannolo")
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/load", methods=["GET"])
def load():
    try:
        print(f"INIZIO")
        overview = get_conversations_overview()
        print(f"OVERVIEW:{overview}")
        return jsonify({"overview": overview}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ascolta su tutte le interfacce (utile per collegare Jetson + Notebook)
    app.run(host="0.0.0.0", port=5000)
