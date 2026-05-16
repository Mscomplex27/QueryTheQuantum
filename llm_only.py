import os 
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_llm_only(query):
    """Run query through LLM-only pipeline and return answer + metrics"""
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=1024
    )
    
    latency_ms = (time.time() - start_time) * 1000
    
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost_usd = 0.0
    
    return {
        "pipeline": "LLM-Only",
        "answer": answer,
        "tokens_used": tokens_used,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "cost_usd": cost_usd
    }

if __name__ == "__main__":
    test_query = "What is quantum error correction?"
    print(f"Query: {test_query}\n")
    result = run_llm_only(test_query)
    print(f"Answer: {result['answer'][:500]}...")
    print(f"\nTokens used: {result['tokens_used']}")
    print(f"Latency: {result['latency_ms']} ms")