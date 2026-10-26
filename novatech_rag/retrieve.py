
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

load_dotenv(project_folder.parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "baai/bge-large-en-v1.5"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

# --------------------------------------------------
# 2. Load chunks and embeddings
# --------------------------------------------------

with open(data_folder / "text_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open(data_folder / "embedding_vectors.json", encoding="utf-8") as f:
    saved_vectors = json.load(f)

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}

if len(chunks) != len(saved_vectors):
    raise ValueError(
        "Chunk and embedding counts differ. Regenerate embeddings."
    )

# --------------------------------------------------
# 3. Ask a question
# --------------------------------------------------

question = input("Ask a question about NovaTech: ").strip()

if not question:
    raise ValueError("Please enter a question")

# --------------------------------------------------
# 4. Generate question embedding via OpenRouter
# --------------------------------------------------

url = "https://openrouter.ai/api/v1/embeddings"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "input": [question]
}

for attempt in range(3):
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )

        response.raise_for_status()
        break

    except requests.RequestException as error:
        print(f"Attempt {attempt + 1} failed: {error}")

        if attempt == 2:
            raise

        print("Retrying...")
        time.sleep(3 * (attempt + 1))

question_vector = np.array(
    response.json()["data"][0]["embedding"],
    dtype=float
)

# --------------------------------------------------
# 5. Calculate cosine similarity
# --------------------------------------------------

results = []

question_norm = np.linalg.norm(question_vector)

if question_norm == 0:
    raise ValueError("Question embedding has zero magnitude")

for item in saved_vectors:

    chunk_id = item["chunk_id"]

    if chunk_id not in chunk_lookup:
        raise ValueError(
            f"Missing chunk ID {chunk_id}. Regenerate embeddings."
        )

    chunk_vector = np.array(
        item["embedding"],
        dtype=float
    )

    if chunk_vector.shape != question_vector.shape:
        raise ValueError(
            "Embedding dimensions differ. Check your embedding model."
        )

    chunk_norm = np.linalg.norm(chunk_vector)

    if chunk_norm == 0:
        continue

    similarity = np.dot(
        question_vector,
        chunk_vector
    ) / (question_norm * chunk_norm)

    chunk = chunk_lookup[chunk_id]

    results.append({
        "chunk_id": chunk_id,
        "page": chunk["page"],
        "similarity": float(similarity),
        "text": chunk["text"]
    })

# --------------------------------------------------
# 6. Identify the requested product
# --------------------------------------------------

products = {
    "NovaBook Air 14": "NT-LAP-101",
    "NovaBook Pro 16": "NT-LAP-102",
    "NovaBook Study 15": "NT-LAP-103",
    "NovaPhone S": "NT-PHN-201",
    "NovaPhone Plus": "NT-PHN-202",
    "NovaPhone Lite": "NT-PHN-203",
    "NovaSound Headphones": "NT-AUD-302"
}

matched_product = None
matched_id = None

for name, product_id in sorted(
    products.items(),
    key=lambda item: len(item[0]),
    reverse=True
):

    pattern = rf"(?<!\w){re.escape(name)}(?!\w)"

    if re.search(pattern, question, re.IGNORECASE):
        matched_product = name
        matched_id = product_id
        break

# --------------------------------------------------
# 7. Improve retrieval ranking
# --------------------------------------------------

for result in results:

    text = result["text"]
    score = result["similarity"]

    if matched_product:

        # Bonus for mentioning the exact product
        product_pattern = (
            rf"(?<!\w){re.escape(matched_product)}(?!\w)"
        )

        if re.search(
            product_pattern,
            text,
            re.IGNORECASE
        ):
            score += 0.02

        # Strong bonus for the product's dedicated page
        profile_pattern = (
            rf"(?:Laptop|Smartphone|Audio) Product Profile:"
            rf"\s*{re.escape(matched_product)}"
        )

        if re.search(
            profile_pattern,
            text,
            re.IGNORECASE
        ):
            score += 0.10

        # Small bonus for matching product ID
        elif matched_id in text:
            score += 0.01

    result["ranking_score"] = float(score)

# --------------------------------------------------
# 8. Sort and retrieve top 3
# --------------------------------------------------

results.sort(
    key=lambda x: x["ranking_score"],
    reverse=True
)

top_results = results[:3]

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nTOP 3 RELEVANT CHUNKS\n")

if matched_product:
    print(f"Detected product: {matched_product}")
    print(f"Product ID: {matched_id}\n")

for rank, result in enumerate(top_results, start=1):

    print(f"RESULT {rank}")
    print(f"Page: {result['page']}")
    print(f"Cosine similarity: {result['similarity']:.4f}")
    print(f"Ranking score: {result['ranking_score']:.4f}")
    print(f"Text: {result['text']}")
    print("-" * 60)

# --------------------------------------------------
# 10. Save readable retrieval results
# --------------------------------------------------

output_path = data_folder / "retrieval_results.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "question": question,
            "detected_product": matched_product,
            "results": top_results
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nResults saved to:", output_path)
