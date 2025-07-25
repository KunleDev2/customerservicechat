from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = "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"  # replace with secure loading for production

if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in the environment.")

os.environ["OPENAI_API_KEY"] = openai_api_key

app = Flask(__name__)

embedding = OpenAIEmbeddings()
CHROMA_DB_PATH = "rag_db"

if not os.path.exists(CHROMA_DB_PATH):
    raise FileNotFoundError(f"Chroma DB folder '{CHROMA_DB_PATH}' not found.")

db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('message', '').strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = qa.invoke({"query": question})
        print(f"Answer: {answer}")
        return jsonify({"Response": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)
