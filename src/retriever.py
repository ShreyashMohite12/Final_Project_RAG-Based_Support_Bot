from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

embedding = OllamaEmbeddings(model="llama3")

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={"k": 3})