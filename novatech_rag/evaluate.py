
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from rag_engine import retrieve_chunks, generate_answer, CHAT_MODEL

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"

DATASET_PATH = DATA_FOLDER / "novatech_golden_dataset.json"
RESULTS_PATH = DATA_FOLDER / "evaluation_results.json"
SUMMARY_PATH = DATA_FOLDER / "evaluation_summary.json"

load_dotenv(PROJECT_FOLDER.parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# --------------------------------------------------
# 2. Load golden dataset
# --------------------------------------------------

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Golden dataset must be a JSON list.")

    for item in dataset:
        if "question" not in item or "ground_truth" not in item:
            raise ValueError(
                "Each dataset item needs question and ground_truth."
            )

    return dataset


# --------------------------------------------------
# 3. LLM-as-a-Judge
# --------------------------------------------------

def judge_answer(question, ground_truth, generated_answer, contexts):
    """
    Evaluate:
    - Answer correctness
    - Faithfulness
    - Context relevance

    Scores range from 0 to 1.
    """

    context_text = "\n\n".join(contexts)

    system_prompt = """
You are an impartial evaluator of a Retrieval-Augmented
Generation (RAG) system.

Evaluate the answer using three metrics:

1. correctness:
How well does the generated answer match the ground truth?

2. faithfulness:
Are the factual claims in the generated answer supported
by the retrieved context?

3. context_relevance:
How relevant is the retrieved context to the question?

Use scores between 0 and 1:
0 = poor
0.5 = partially correct or supported
1 = fully correct or supported

Important:
- Do not reward unsupported factual claims.
- Correctness must use the ground truth.
- Faithfulness must use the retrieved context.
- Context relevance must use the question and context.
- For unanswerable questions, correctly refusing to invent
  information should receive a high correctness score.
- Return only valid JSON.

Expected format:
{
  "correctness": 0.0,
  "faithfulness": 0.0,
  "context_relevance": 0.0,
  "reason": "Brief explanation"
}
"""

    user_prompt = f"""
QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{context_text}
"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 350,
            "response_format": {"type": "json_object"}
        },
        timeout=90
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    scores = json.loads(content)

    for metric in ["correctness", "faithfulness", "context_relevance"]:
        value = float(scores[metric])
        scores[metric] = max(0.0, min(1.0, value))

    return scores


# --------------------------------------------------
# 4. Calculate retrieval page hit
# --------------------------------------------------

def calculate_page_hit(item, retrieved_chunks):
    """
    Checks whether at least one retrieved chunk comes
    from an expected source page.

    This is a diagnostic metric, not full retrieval recall.
    """

    pages = item.get("pages", item.get("page"))

    if pages is None:
        return None

    if not isinstance(pages, list):
        pages = [pages]

    retrieved_pages = {
        str(chunk["page"]) for chunk in retrieved_chunks
    }

    expected_pages = {str(page) for page in pages}

    return int(bool(retrieved_pages & expected_pages))


# --------------------------------------------------
# 5. Evaluate a single question
# --------------------------------------------------

def evaluate_question(item):
    question = item["question"]
    ground_truth = item["ground_truth"]

    start_time = time.perf_counter()

    # Retrieval
    retrieved_chunks = retrieve_chunks(question, top_k=3)

    retrieval_time = time.perf_counter() - start_time

    # Generation
    generated_answer = generate_answer(question, retrieved_chunks)

    total_time = time.perf_counter() - start_time

    contexts = [chunk["text"] for chunk in retrieved_chunks]

    # LLM judge (not included in RAG latency)
    scores = judge_answer(
        question,
        ground_truth,
        generated_answer,
        contexts
    )

    page_hit = calculate_page_hit(item, retrieved_chunks)

    result = {
        "id": item.get("id"),
        "question": question,
        "type": item.get("type", "single_hop"),
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "retrieved_pages": [
            chunk["page"] for chunk in retrieved_chunks
        ],
        "retrieved_chunks": retrieved_chunks,
        "retrieval_latency_seconds": round(retrieval_time, 3),
        "total_latency_seconds": round(total_time, 3),
        "page_hit_at_3": page_hit,
        "correctness": scores["correctness"],
        "faithfulness": scores["faithfulness"],
        "context_relevance": scores["context_relevance"],
        "judge_reason": scores.get("reason", "")
    }

    return result


# --------------------------------------------------
# 6. Calculate average scores
# --------------------------------------------------

def average(values):
    return round(sum(values) / len(values), 3) if values else None


def summarize(results):
    summary = {
        "total_questions": len(results),
        "average_correctness": average([
            r["correctness"] for r in results
        ]),
        "average_faithfulness": average([
            r["faithfulness"] for r in results
        ]),
        "average_context_relevance": average([
            r["context_relevance"] for r in results
        ]),
        "average_latency_seconds": average([
            r["total_latency_seconds"] for r in results
        ]),
        "page_hit_at_3": average([
            r["page_hit_at_3"]
            for r in results
            if r["page_hit_at_3"] is not None
        ])
    }

    # Results grouped by question type
    summary["by_type"] = {}

    for question_type in sorted(set(r["type"] for r in results)):
        group = [r for r in results if r["type"] == question_type]

        summary["by_type"][question_type] = {
            "count": len(group),
            "average_correctness": average([
                r["correctness"] for r in group
            ]),
            "average_faithfulness": average([
                r["faithfulness"] for r in group
            ]),
            "average_context_relevance": average([
                r["context_relevance"] for r in group
            ]),
            "average_latency_seconds": average([
                r["total_latency_seconds"] for r in group
            ])
        }

    return summary


# --------------------------------------------------
# 7. Run evaluation
# --------------------------------------------------

def main():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    results = []

    print(f"\nStarting NovaTech RAG evaluation: {len(dataset)} questions\n")

    for index, item in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {item['question']}")

        try:
            result = evaluate_question(item)
            results.append(result)

            print(
                f"  Correctness: {result['correctness']:.2f} | "
                f"Faithfulness: {result['faithfulness']:.2f} | "
                f"Context: {result['context_relevance']:.2f} | "
                f"Latency: {result['total_latency_seconds']:.2f}s"
            )

        except Exception as error:
            print(f"  ERROR: {error}")

        # Save progress after every question
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if not results:
        print("No questions were evaluated successfully.")
        return

    summary = summarize(results)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("\nResults saved to:", RESULTS_PATH)
    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
