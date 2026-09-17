from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in the environment.")

os.environ["OPENAI_API_KEY"] = openai_api_key

embedding = OpenAIEmbeddings()
CHROMA_DB_PATH = "rag_db"

if not os.path.exists(CHROMA_DB_PATH):
    raise FileNotFoundError(f"Chroma DB folder '{CHROMA_DB_PATH}' not found.")

db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def search_knowledge_base(question: str) -> str:
    """Search the customer service knowledge base for relevant information."""
    documents = retriever.invoke(question)

    if not documents:
        return "No relevant information was found in the knowledge base."

    return "\n\n".join(
        document.page_content
        for document in documents
    )

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt="""
You are a very useful and helpful customer service assistant.

Use the search_knowledge_base tool to find information from the company's knowledge base.

Answer the user's question using the information returned by the knowledge base.

If you cannot find the answer in the knowledge base, say:
"I do not have enough information to answer the question."

Do not invent or guess information.
"""
)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    print("data", data)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    question = data.get('message', '').strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
            )
        
        answer = result["messages"][-1].content

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        return jsonify({"Response": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)
