
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

project_folder = Path(__file__).resolve().parent

# Load the existing .env from AI_STS
load_dotenv(project_folder.parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "baai/bge-large-en-v1.5"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

input_path = project_folder / "data" / "text_chunks.json"
output_path = project_folder / "data" / "embedding_vectors.json"

with open(input_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

url = "https://openrouter.ai/api/v1/embeddings"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

vectors = []
batch_size = 10

for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]

    payload = {
        "model": MODEL,
        "input": [chunk["text"] for chunk in batch]
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
            result = response.json()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

    embeddings = sorted(
        result["data"],
        key=lambda item: item["index"]
    )

    if len(embeddings) != len(batch):
        raise ValueError("Embedding count does not match batch size")

    for chunk, item in zip(batch, embeddings):
        vectors.append({
            "chunk_id": chunk["chunk_id"],
            "embedding": item["embedding"]
        })

    print(f"Embedded {len(vectors)}/{len(chunks)} chunks")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(vectors, f)

print("Embeddings saved to:", output_path)
print("Total embeddings:", len(vectors))
