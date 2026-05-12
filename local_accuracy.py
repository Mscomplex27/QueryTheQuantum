import csv
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

# ---- MANUAL DATA ENTRY ----
# Create lists of your collected data
questions = []   # e.g., ["What is quantum error correction?", ...]
ref_answers = [] # reference answers from ground_truth_queries.json
llm_answers = []
rag_answers = []
graph_answers = []

# Fill them here (you can copy-paste from your notes)

# ---- Evaluate ----
pipelines = [
    ("LLM-only", llm_answers),
    ("Basic RAG", rag_answers),
    ("GraphRAG", graph_answers)
]

for name, answers in pipelines:
    pass_count = 0
    total = 0
    for i, ans in enumerate(answers):
        if not ans:
            continue
        if judge_pass(questions[i], ref_answers[i], ans):
            pass_count += 1
        total += 1
    pass_rate = pass_count / total * 100 if total else 0
    # BERTScore
    bert_scores = bertscorer.compute(predictions=answers, references=ref_answers, lang="en", rescale_with_baseline=True)
    avg_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
    print(f"{name}: Pass rate = {pass_rate:.1f}%, BERTScore F1 = {avg_f1:.4f}")