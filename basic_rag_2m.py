import os
import time
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
_client = None
_chroma_client = None


def get_groq_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    _client = Groq(api_key=api_key)
    return _client


def get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb

    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client


def get_collection():
    return get_chroma_client().get_or_create_collection("quantum_papers")

embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def index_papers():
    """Index only if empty."""
    collection = get_collection()
    if collection.count() > 0:
        print(f"✅ Already have {collection.count()} papers indexed. Skipping.")
        return
    files = [f for f in os.listdir("quantum_papers_text") if f.endswith(".txt")]
    docs = []
    for f in files:
        with open(f"quantum_papers_text/{f}", encoding='utf-8') as file:
            docs.append(file.read())
    print(f"Indexing {len(docs)} papers (first time only)...")
    embeddings = list(embed_model.embed(docs))
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        collection.upsert(ids=[str(i)], embeddings=[emb.tolist()], documents=[doc])
    print("Done indexing.")

def retrieve(query, top_k=3):
    collection = get_collection()
    q_emb = list(embed_model.embed([query]))[0].tolist()
    return collection.query(query_embeddings=[q_emb], n_results=top_k)["documents"][0]

def basic_rag(query):
    start = time.time()
    context = "\n\n".join(retrieve(query))
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - start) * 1000
    return {
        "answer": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "latency_ms": round(latency, 2)
    }

if __name__ == "__main__":
    index_papers()
    query = "What is quantum error correction?"
    res = basic_rag(query)
    print(res["answer"])
    print(f"Tokens: {res['tokens']}\nLatency: {res['latency_ms']} ms")