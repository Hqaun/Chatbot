from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Tải biến môi trường từ file .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Khởi tạo mô hình LangChain
llm = OpenAI(openai_api_key=api_key, temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = conversation.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
