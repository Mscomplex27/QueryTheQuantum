import os
import time
from groq import Groq
from dotenv import load_dotenv
from pyTigerGraph import TigerGraphConnection
from entity_extractor import extract_entities

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TG_HOST = os.getenv("TG_HOST")
TG_USERNAME = os.getenv("TG_USERNAME")
TG_PASSWORD = os.getenv("TG_PASSWORD")
TG_GRAPH = os.getenv("TG_GRAPH", "GraphRAG_Hackathon")

conn = TigerGraphConnection(
    host=TG_HOST,
    graphname=TG_GRAPH,
    username=TG_USERNAME,
    password=TG_PASSWORD,
    useCert=False
)

# Auto token
conn.apiToken = conn.getToken(conn.createSecret())

client = Groq(api_key=GROQ_API_KEY)

def parse_chunks(result):
    """
    Robust parser for TigerGraph GSQL output
    """

    chunks = []

    try:
        if isinstance(result, str):
            return []

        if isinstance(result, dict):
            result = [result]

        if isinstance(result, list):
            for item in result:
                # SAFE fallback extraction
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, str) and len(v) > 20:
                            chunks.append(v)

    except Exception as e:
        print("Parse error:", e)

    return chunks

def multi_hop_retrieve(seed_entities, top_k=5):
    """
    2-hop GraphRAG retrieval:
    Entity → Chunk → Entity → Chunk
    """

    if not seed_entities:
        return "No entities found."

    all_chunks = set()

    for entity in seed_entities:

        q1 = f"""
        USE GRAPH {TG_GRAPH}

        INTERPRET QUERY () FOR GRAPH {TG_GRAPH} {{
            start = {{Entity.* WHERE name == "{entity}"}};

            result = SELECT c
                     FROM start:e -(has_entity:he)- Chunk:c;

            PRINT result;
        }}
        """

        try:
            res1 = conn.gsql(q1)
            all_chunks.update(parse_chunks(res1))
        except Exception as e:
            print("1-hop error:", e)

        q2 = f"""
        USE GRAPH {TG_GRAPH}

        INTERPRET QUERY () FOR GRAPH {TG_GRAPH} {{
            start = {{Entity.* WHERE name == "{entity}"}};

            hop1 = SELECT c
                   FROM start:e -(has_entity:he)- Chunk:c;

            hop2 = SELECT c2
                   FROM hop1:c -(has_entity:he2)- Entity:e2
                        -(has_entity:he3)- Chunk:c2;

            PRINT hop1, hop2;
        }}
        """

        try:
            res2 = conn.gsql(q2)
            all_chunks.update(parse_chunks(res2))
        except Exception as e:
            print("2-hop error:", e)

    return "\n\n".join(list(all_chunks)[:top_k])

def graphrag_query(query: str):
    start_time = time.time()
    entities = extract_entities(query)
    context = multi_hop_retrieve(entities)
    prompt = f"""
You are a quantum computing expert.
Use ONLY the following graph context:
{context}
Question:
{query}
Explain clearly and technically.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )

    latency = (time.time() - start_time) * 1000

    tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
    cost_usd = 0.0

    return {
        "answer": response.choices[0].message.content,
        "entities": entities,
        "latency_ms": round(latency, 2),
        "context_length": len(context),
        "tokens_used": response.usage.total_tokens,  # add this
        "cost_usd": 0.0,
        "reasoning_path": ""
    }
if __name__ == "__main__":

    test_query = "How does IBM Eagle improve quantum error correction using surface codes?"

    result = graphrag_query(test_query)

    print("\n🔹 ANSWER:\n")
    print(result["answer"])
    print("\n🔹 ENTITIES:", result["entities"])
    print("🔹 LATENCY:", result["latency_ms"], "ms")
    print("🔹 CONTEXT SIZE:", result["context_length"])