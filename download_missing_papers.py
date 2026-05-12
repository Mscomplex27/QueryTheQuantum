import os
import arxiv
from tqdm import tqdm
import time

def fetch_missing_papers(max_results=4500, output_dir="quantum_papers_text"):
    os.makedirs(output_dir, exist_ok=True)
    existing = {f.replace(".txt", "") for f in os.listdir(output_dir) if f.endswith(".txt")}
    print(f"Found {len(existing)} existing papers. Will skip them.")
    
    search = arxiv.Search(
        query="cat:quant-ph OR cat:cond-mat OR cat:physics",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    papers = list(client.results(search))
    
    new_count = 0
    total_new_tokens = 0
    
    for paper in tqdm(papers, desc="Checking papers"):
        paper_id = paper.get_short_id().replace('/', '_')
        if paper_id in existing:
            continue
        content = f"Title: {paper.title}\n"
        content += f"Authors: {', '.join(author.name for author in paper.authors)}\n"
        content += f"Categories: {', '.join(paper.categories)}\n"
        content += f"Published: {paper.published}\n"
        content += f"Abstract: {paper.summary}\n"
        
        total_new_tokens += len(content) // 3
        with open(os.path.join(output_dir, f"{paper_id}.txt"), 'w', encoding='utf-8') as f:
            f.write(content)
        
        new_count += 1
        time.sleep(3)
        if new_count >= 500: 
            break
    
    print(f"\nAdded {new_count} new papers.")
    print(f"Estimated new tokens: ~{total_new_tokens}")
    print(f"Total tokens now: ~{1924565 + total_new_tokens}")

if __name__ == "__main__":
    fetch_missing_papers(max_results=4500)