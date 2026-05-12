"""
Run all 50 ground truth queries through LLM-only, Basic RAG, and GraphRAG pipelines.
Saves answers and metrics to a CSV file for later accuracy evaluation.
"""

import json
import csv
import time
from datetime import datetime
from tqdm import tqdm
import sys
import os

# ========== FIXED IMPORT PATHS ==========
# Add current directory for local modules (llm_only, basic_rag_2m)
sys.path.insert(0, os.path.dirname(__file__))
# Add the subfolder so that load_data.py can find entity_extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tigergraph_graphrag'))

# Import your pipeline functions
from llm_only import run_llm_only
from basic_rag_2m import basic_rag
from load_data import graphrag_query   # now works because tigergraph_graphrag is in path
# ========================================

def load_ground_truth(json_path="queries.json"):
    """Load the ground truth queries and reference answers."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['queries']

def run_pipeline_safe(pipeline_func, query, pipeline_name):
    """Execute a pipeline with error handling and return result dict."""
    try:
        result = pipeline_func(query)
        # Ensure result has required fields
        if pipeline_name == "LLM-only":
            return {
                "answer": result.get("answer", ""),
                "tokens": result.get("tokens_used", 0),
                "latency_ms": result.get("latency_ms", 0),
                "cost_usd": result.get("cost_usd", 0),
                "error": None
            }
        elif pipeline_name == "Basic RAG":
            return {
                "answer": result.get("answer", ""),
                "tokens": result.get("tokens", 0),
                "latency_ms": result.get("latency_ms", 0),
                "cost_usd": result.get("cost_usd", 0.0),
                "error": None
            }
        elif pipeline_name == "GraphRAG":
            return {
                "answer": result.get("answer", ""),
                "tokens": result.get("tokens_used", 0),
                "latency_ms": result.get("latency_ms", 0),
                "cost_usd": result.get("cost_usd", 0.0),
                "entities": result.get("entities", []),
                "reasoning_path": result.get("reasoning_path", ""),
                "error": None
            }
    except Exception as e:
        return {
            "answer": "",
            "tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0,
            "error": str(e)
        }

def main():
    # Load queries
    queries = load_ground_truth()
    print(f"Loaded {len(queries)} ground truth queries.")

    # Prepare CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "query_id", "question", "reference_answer",
            "llm_only_answer", "llm_only_tokens", "llm_only_latency_ms", "llm_only_cost_usd", "llm_only_error",
            "basic_rag_answer", "basic_rag_tokens", "basic_rag_latency_ms", "basic_rag_cost_usd", "basic_rag_error",
            "graphrag_answer", "graphrag_tokens", "graphrag_latency_ms", "graphrag_cost_usd", "graphrag_entities", "graphrag_reasoning_path", "graphrag_error"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for q in tqdm(queries, desc="Running pipelines"):
            qid = q['id']
            question = q['question']
            ref_answer = q['reference_answer']

            llm_res = run_pipeline_safe(run_llm_only, question, "LLM-only")
            rag_res = run_pipeline_safe(basic_rag, question, "Basic RAG")
            graph_res = run_pipeline_safe(graphrag_query, question, "GraphRAG")

            row = {
                "query_id": qid,
                "question": question,
                "reference_answer": ref_answer,
                "llm_only_answer": llm_res["answer"],
                "llm_only_tokens": llm_res["tokens"],
                "llm_only_latency_ms": llm_res["latency_ms"],
                "llm_only_cost_usd": llm_res["cost_usd"],
                "llm_only_error": llm_res.get("error", ""),
                "basic_rag_answer": rag_res["answer"],
                "basic_rag_tokens": rag_res["tokens"],
                "basic_rag_latency_ms": rag_res["latency_ms"],
                "basic_rag_cost_usd": rag_res["cost_usd"],
                "basic_rag_error": rag_res.get("error", ""),
                "graphrag_answer": graph_res["answer"],
                "graphrag_tokens": graph_res["tokens"],
                "graphrag_latency_ms": graph_res["latency_ms"],
                "graphrag_cost_usd": graph_res["cost_usd"],
                "graphrag_entities": json.dumps(graph_res.get("entities", [])),
                "graphrag_reasoning_path": graph_res.get("reasoning_path", ""),
                "graphrag_error": graph_res.get("error", "")
            }
            writer.writerow(row)
            csvfile.flush()
            time.sleep(0.5)

    print(f"\n✅ Benchmark results saved to {csv_filename}")

if __name__ == "__main__":
    main()