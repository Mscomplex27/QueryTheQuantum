import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to(DEVICE)
model.eval()
print("Model loaded.")

bertscorer = load("bertscore")

def judge_pass(question: str, reference: str, answer: str) -> bool:
    prompt = f"Question: {question}\nCorrect Answer: {reference}\nCandidate Answer: {answer}\n\nIs the candidate answer correct? Reply with only 'PASS' or 'FAIL'.\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip().upper()
    return "PASS" in result

# Load JSON data
with open("benchmark_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

queries = data["queries"]

# For each pipeline, collect answers that are non‑empty
pipelines = [
    ("LLM-only", "llm_only_answer"),
    ("Basic RAG", "basic_rag_answer"),
    ("GraphRAG", "graphrag_answer")
]

print("\n=== Benchmark Evaluation ===\n")
for name, field in pipelines:
    # Gather valid entries (answer not empty)
    valid = []
    for q in queries:
        ans = q[field]
        if ans and ans.strip():
            valid.append({
                "question": q["question"],
                "reference": q["reference_answer"],
                "answer": ans
            })
    if not valid:
        print(f"{name}: No answers provided. Skipping.")
        continue

    print(f"Evaluating {name} on {len(valid)} queries...")
    pass_count = 0
    refs = []
    preds = []
    for item in tqdm(valid):
        refs.append(item["reference"])
        preds.append(item["answer"])
        if judge_pass(item["question"], item["reference"], item["answer"]):
            pass_count += 1
    pass_rate = pass_count / len(valid) * 100

    # BERTScore
    bert_scores = bertscorer.compute(predictions=preds, references=refs, lang="en", rescale_with_baseline=True)
    avg_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])

    print(f"\n{name} Results:")
    print(f"  Queries evaluated: {len(valid)}")
    print(f"  LLM-as-Judge Pass Rate: {pass_rate:.1f}%")
    print(f"  Avg BERTScore F1 (rescaled): {avg_f1:.4f}")
    print()

print("=== Done ===")