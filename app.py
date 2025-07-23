from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

try:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory="rag_db", embedding_function=embedding)
    retriever = db.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
except Exception as e:
    print(f"Error during initialization: {e}")
    retriever = None
    qa = None

@app.route('/ask', methods=['POST'])
def ask_question():
    if not qa:
        return jsonify({"error": "QA system failed to initialize"}), 500

    data = request.json
    question = data.get('message', '').strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = qa.run(question)
        return jsonify({"Response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005)
