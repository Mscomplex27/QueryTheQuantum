from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_groq_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    _client = Groq(api_key=api_key)
    return _client

def extract_entities(query):
    prompt = f"""
Extract key scientific entities from this quantum computing query.

Return ONLY a Python list.

Query:
{query}

Focus on:
- quantum concepts
- hardware (IBM, Google, etc.)
- algorithms
- error correction methods

Example output:
["IBM Eagle", "Quantum Error Correction", "Surface Codes"]
"""

    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return eval(response.choices[0].message.content)
    except:
        return [query]