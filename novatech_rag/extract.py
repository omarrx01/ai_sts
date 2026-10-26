
import json
from pathlib import Path
import pymupdf

# Step 1: Define file paths
project_folder = Path(__file__).resolve().parent

pdf_path = project_folder / "data" / "NovaTech_Electronics_Knowledge_Base_v3.0_25pages_Detailed.pdf"
output_path = project_folder / "data" / "extracted_text.json"

# Step 2: Open the PDF
pages_data = []

with pymupdf.open(pdf_path) as pdf:
    print("Total PDF pages:", len(pdf))

    # Step 3: Extract text from every page
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text")

        # Clean extra spaces and line breaks
        cleaned_text = " ".join(text.split())

        if cleaned_text:
            pages_data.append({
                "page": page_number,
                "text": cleaned_text
            })

# Step 4: Save extracted text
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pages_data, f, indent=2, ensure_ascii=False)

print("Pages extracted:", len(pages_data))
print("Text saved to:", output_path)
