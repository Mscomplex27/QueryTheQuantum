from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return eval(response.choices[0].message.content)
    except:
        return [query]