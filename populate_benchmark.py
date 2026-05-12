import json
import sys
from llm_only import run_llm_only
from basic_rag_2m import basic_rag
sys.path.insert(0, '/Users/mac/Documents/Projects/GraphRAGHackathon/tigergraph_graphrag')
from load_data import graphrag_query
# from tigergraph_graphrag.load_data import graphrag_query
N = 10   # number of queries to process (change as needed)

with open("benchmark_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

queries = data["queries"][:N]
for idx, q in enumerate(queries, start=1):
    question = q["question"]
    print(f"\nProcessing query {idx}: {question[:60]}...")
    
    # Get pipeline answers
    llm_res = run_llm_only(question)
    rag_res = basic_rag(question)
    graph_res = graphrag_query(question)
    
    # Update JSON fields
    q["llm_only_answer"] = llm_res.get("answer", "")
    q["basic_rag_answer"] = rag_res.get("answer", "")
    q["graphrag_answer"] = graph_res.get("answer", "")
    
    # Save after each query (to avoid losing progress)
    with open("benchmark_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Updated query {idx}.")

print("\n✅ Population complete")