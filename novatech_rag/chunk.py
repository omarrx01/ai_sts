
import json
import re
from pathlib import Path

project_folder = Path(__file__).resolve().parent
input_path = project_folder / "data" / "extracted_text.json"
output_path = project_folder / "data" / "text_chunks.json"

CHUNK_SIZE = 900
OVERLAP = 150

with open(input_path, "r", encoding="utf-8") as f:
    pages = json.load(f)

chunks = []
chunk_id = 1

for page in pages:
    page_number = page["page"]
    text = page["text"].strip()

    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Split unusually long sentences at word boundaries
    units = []

    for sentence in sentences:
        if len(sentence) <= CHUNK_SIZE:
            units.append(sentence)
        else:
            words = sentence.split()
            current = ""

            for word in words:
                if current and len(current) + len(word) + 1 > CHUNK_SIZE:
                    units.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()

            if current:
                units.append(current)

    current_units = []
    current_length = 0

    for unit in units:
        extra_length = len(unit) + (1 if current_units else 0)

        if current_units and current_length + extra_length > CHUNK_SIZE:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": " ".join(current_units)
            })
            chunk_id += 1

            # Keep complete trailing units for overlap
            overlap_units = []
            overlap_length = 0

            for previous in reversed(current_units):
                if overlap_length + len(previous) > OVERLAP:
                    break

                overlap_units.insert(0, previous)
                overlap_length += len(previous) + 1

            current_units = overlap_units
            current_length = len(" ".join(current_units))

        current_units.append(unit)
        current_length = len(" ".join(current_units))

    if current_units:
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_number,
            "text": " ".join(current_units)
        })
        chunk_id += 1

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print("Total chunks created:", len(chunks))
print("First chunk preview:")
print(chunks[0]["text"][:300] if chunks else "No chunks")
