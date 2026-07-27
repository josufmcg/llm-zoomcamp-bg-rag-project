"""Generate ground truth question-answer pairs for evaluation.

For each document in the knowledge base, uses OpenAI to generate
5 natural-language questions that the document should answer.

Usage:
    uv run python scripts/generate_ground_truth.py

Output:
    data/ground_truth.csv
"""

import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from bg_rag.db import compute_doc_hash


class Questions(BaseModel):
    """Structured output: list of generated questions."""
    questions: list[str]


GENERATION_INSTRUCTIONS = """
You emulate a player who is looking for help with Baldur's Gate II character creation.
Given a character class FAQ record, formulate 5 questions that a player might ask
that this record would answer.

Requirements:
- Questions should be natural and varied (how people ask on gaming forums).
- Questions should be complete sentences, not too short.
- Try to use as few words as possible from the record itself.
- Each question should be answerable from the record's content.
- Questions should cover different aspects: class overview, kits, abilities,
  restrictions, best strategies, comparisons.
- Do NOT make questions that are too generic (e.g., "What classes are there?").
  Each question should specifically relate to the given class.
""".strip()


def load_documents() -> list[dict]:
    """Load documents from the dataset file."""
    path = Path("data/bg_characters.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'uv run python scripts/generate_dataset.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def generate_questions(doc: dict, client: OpenAI) -> list[str]:
    """Generate test questions for a single document.

    Args:
        doc: Document dict with id, category, subcategory, question, text.
        client: OpenAI client.

    Returns:
        List of 5 generated questions.
    """
    user_prompt = json.dumps(doc, indent=2)

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {"role": "developer", "content": GENERATION_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        text_format=Questions,
    )

    return response.output_parsed.questions


def main() -> None:
    load_dotenv()
    client = OpenAI()

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    output_path = Path("data/ground_truth.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for doc in documents:
        print(f"  Generating questions for {doc['id']}...")
        questions = generate_questions(doc, client)
        # Use the same MD5 hash the DB stores as doc_id so evaluation
        # can match search results (which return doc_id = the hash).
        doc_hash = compute_doc_hash(doc["category"], doc["subcategory"], doc["text"])
        for q in questions:
            rows.append({
                "doc_id": doc_hash,
                "question": q,
                "doc_category": doc["category"],
                "doc_subcategory": doc["subcategory"],
            })

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "question", "doc_category", "doc_subcategory"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} ground truth Q&A pairs → {output_path}")


if __name__ == "__main__":
    main()
