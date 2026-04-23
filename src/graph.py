from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Setup DB
embedding = OllamaEmbeddings(model="llama3")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama3")

# State
class State(dict):
    pass

# Processing Node
def process_node(state):
    query = state["query"]

    docs = retriever.get_relevant_documents(query)

    if not docs:
        return {
            "answer": "Escalating to human...",
            "escalate": True
        }

    context = "\n".join([d.page_content for d in docs])

    answer = llm.invoke(f"""
    Answer only from context:
    {context}

    Question: {query}
    """)

    return {
        "answer": answer,
        "escalate": False
    }

# Output Node
def output_node(state):
    return state

# Graph
graph = StateGraph(State)

graph.add_node("process", process_node)
graph.add_node("output", output_node)

graph.set_entry_point("process")
graph.add_edge("process", "output")

app = graph.compile()