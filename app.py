from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.main import ask_claude_with_context

app = Flask(
    __name__,
    static_folder="../portfolio",
    static_url_path=""
)

CORS(app)

# Serve index.html
@app.route("/health")
def health():
    return "ok"

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"].lower()
    reply = ask_claude_with_context(user_message)   
    return jsonify({"reply": reply})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
