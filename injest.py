import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY not set in the environment."
    )

os.environ["OPENAI_API_KEY"] = openai_api_key

DOCUMENTS_PATH = "documents"

CHROMA_DB_PATH = "rag_db"

if not os.path.exists(DOCUMENTS_PATH):
    raise FileNotFoundError(
        f"Documents folder '{DOCUMENTS_PATH}' not found"
    )

documents = []

for filename in os.listdir(DOCUMENTS_PATH):

    file_path = os.path.join(DOCUMENTS_PATH, filename)

    if filename.lower().endswith(".txt"):

        print(f"Loading: {filename}")

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename
                }
            )
        )

if not documents:
    raise ValueError(
        "No .txt documents found in the documents folder."
    )

print(f"Loaded {len(documents)} document(s).")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} document chunks")

embedding = OpenAIEmbeddings()

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_DB_PATH
)

print()
print("=======================================")
print("Documents successfully added to Chroma.")
print("=======================================")
print(f"Database location: {CHROMA_DB_PATH}")
print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")