from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# Create embeddings
embedding = OllamaEmbeddings(model="llama3")

# Store in ChromaDB
db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="./chroma_db"
)

db.persist()

print("✅ Data stored in ChromaDB")