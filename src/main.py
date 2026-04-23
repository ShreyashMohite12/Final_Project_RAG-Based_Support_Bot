from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from langgraph.graph import StateGraph

# ---------------- LOAD PDF ----------------
print("🔹 Loading PDF...")

reader = PdfReader("support.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

print("✅ PDF Loaded")

# ---------------- CHUNKING ----------------
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print("✅ Text Chunked")

# ---------------- EMBEDDINGS ----------------
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
embeddings = model.encode(chunks)

print("✅ Embeddings Created")

# ---------------- LLM ----------------
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

print("✅ Model Ready")

# ---------------- GRAPH STATE ----------------
class State(dict):
    pass

# ---------------- PROCESS NODE ----------------
def process_node(state):
    query = state["query"]

    query_embedding = model.encode([query])[0]

    scores = np.dot(embeddings, query_embedding)
    top_idx = np.argsort(scores)[-3:]

    context = "\n".join([chunks[i] for i in top_idx])

    # CONDITION: If low similarity → escalate
    if np.max(scores) < 0.3:
        return {
            "answer": "Low confidence. Escalating to human...",
            "escalate": True
        }

    prompt = f"""
    Answer ONLY from context:
    {context}

    Question: {query}
    """

    result = generator(prompt, max_length=150, do_sample=True)

    return {
        "answer": result[0]["generated_text"],
        "escalate": False
    }

# ---------------- OUTPUT NODE ----------------
def output_node(state):
    return state

# ---------------- BUILD GRAPH ----------------
graph = StateGraph(State)

graph.add_node("process", process_node)
graph.add_node("output", output_node)

graph.set_entry_point("process")
graph.add_edge("process", "output")

app = graph.compile()

# ---------------- RUN LOOP ----------------
while True:
    q = input("\nAsk: ")

    result = app.invoke({"query": q})

    if result["escalate"]:
        print("⚠️ Escalation triggered")
        human = input("👨‍💼 Human Agent Response: ")
        print("Final Answer:", human)
    else:
        print("🤖 Bot:", result["answer"])