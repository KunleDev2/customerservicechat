from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Optional, for local development

app = Flask(__name__)

# Use environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma(persist_directory="rag_db", embedding_function=embedding)
retriever = db.as_retriever()
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        answer = qa.run(question)
        return jsonify({"Response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)