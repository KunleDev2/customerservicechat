from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

# Load the vector DB with OpenAI embedding
embedding = OpenAIEmbeddings()
db = Chroma(persist_directory="rag_db", embedding_function=embedding)
retriever = db.as_retriever()

# Use GPT-3.5 or GPT-4
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

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
        answer = qa.run(question)  # Use `.run()` to get string, not a dict
        print(f"Received answer: {answer}")
        return jsonify({"Response": answer})  # Capital 'R' to match C# model
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)
