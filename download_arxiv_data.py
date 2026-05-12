import os
import arxiv
from tqdm import tqdm
import time

def fetch_and_save_papers(max_results=4600, output_dir="quantum_papers_text"):
    """
    Fetches papers from arXiv using the modern Client API and saves their text content.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    search = arxiv.Search(
        query="cat:quant-ph OR cat:cond-mat OR cat:physics",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print(f"Fetching up to {max_results} papers... This may take a few minutes.")
    client = arxiv.Client()
    papers = list(client.results(search))
    
    print(f"Found {len(papers)} papers. Downloading abstracts and authors...")

    total_tokens = 0
    successful_downloads = 0

    for paper in tqdm(papers, desc="Saving Papers"):
        try:
            content = f"Title: {paper.title}\n"
            content += f"Authors: {', '.join(author.name for author in paper.authors)}\n"
            content += f"Categories: {', '.join(paper.categories)}\n"
            content += f"Published: {paper.published}\n"
            content += f"Abstract: {paper.summary}\n"

            total_tokens += len(content) // 3
            successful_downloads += 1

            paper_id = paper.get_short_id().replace('/', '_')
            with open(os.path.join(output_dir, f"{paper_id}.txt"), 'w', encoding='utf-8') as f:
                f.write(content)
            time.sleep(3)

        except Exception as e:
            print(f"Failed to process paper {paper.title}: {e}")

    print(f"\nSuccessfully saved {successful_downloads} papers.")
    print(f"Estimated total tokens: ~{total_tokens}")

if __name__ == "__main__":
    fetch_and_save_papers(max_results=4000)