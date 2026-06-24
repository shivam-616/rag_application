import os
import urllib.request
import time
import arxiv

# --- CONFIGURATION ---
TARGET_SIZE_MB = 500  # Target size in Megabytes
OUTPUT_DIR = "arxiv_dataset"


# ---------------------

def download_arxiv_subset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    current_size_mb = 0
    print(f"🚀 Starting download. Target size: {TARGET_SIZE_MB} MB")

    # Initialize modern arxiv client
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query="cat:cs.AI",
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = client.results(search)

    for paper in results:
        if current_size_mb >= TARGET_SIZE_MB:
            break

        # Extract metadata identifiers safely
        paper_id = paper.get_short_id()
        filename = f"{paper_id}.pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            continue

        try:
            print(f"Downloading: {filename}...", end="", flush=True)

            # Fetch using standard urllib from the direct PDF address
            req = urllib.request.Request(paper.pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())

            # Update metrics tracker
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            current_size_mb += file_size
            print(f" Done! ({file_size:.2f} MB) | Total: {current_size_mb:.2f} MB")

            # Rate limiting sleep protocol
            time.sleep(2)

        except Exception as e:
            print(f" Failed: {e}")
            time.sleep(2)

    print(f"\n✅ Success! Your {current_size_mb:.2f} MB PDF dataset is ready in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    download_arxiv_subset()
