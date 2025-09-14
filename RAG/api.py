from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import handle_query

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    answer = handle_query(user_query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)