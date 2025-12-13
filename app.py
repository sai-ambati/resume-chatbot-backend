from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(
    __name__,
    static_folder="../portfolio",
    static_url_path=""
)

CORS(app)

# Serve index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"].lower()

    if "skill" in user_message:
        reply = "I have skills in Python, Machine Learning, AWS, SQL, and Data Structures."
    elif "project" in user_message:
        reply = "My projects include a Resume Q&A Chatbot and a personal portfolio website."
    elif "aws" in user_message:
        reply = "I have hands-on experience with AWS EC2, S3, IAM, and Bedrock."
    else:
        reply = "I can answer questions about my skills, projects, and experience."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
