"""Generate the BG2 character dataset using LLM-assisted extraction.

Fetches the GameFAQs BG2 Character FAQ and uses OpenAI to extract
structured character class records.

Usage:
    uv run python scripts/generate_dataset.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


FAQ_PATH = Path("data/bg_faq.txt")

CHARACTER_CLASSES = [
    "Fighter",
    "Ranger",
    "Paladin",
    "Barbarian",
    "Cleric",
    "Druid",
    "Monk",
    "Thief",
    "Bard",
    "Mage",
    "Sorcerer",
]

EXTRACTION_INSTRUCTIONS = """
You are extracting character class information from a Baldur's Gate II FAQ guide.

For each character class, create a structured record with:
1. "id": lowercase class name (e.g., "fighter")
2. "category": always "Character Creation"
3. "subcategory": class name in PascalCase (e.g., "Fighter")
4. "question": a natural-language question that a player might ask about this class.
   Examples:
   - "What is the Fighter class and how should I play it?"
   - "What are the Ranger's abilities, kits, and best strategies?"
   - "Should I pick a Paladin and which kit is best?"
   Make each question unique, natural, and specific to the class.
5. "text": the COMPLETE description of the class from the FAQ, including:
   - Class description and overview
   - Special abilities and restrictions
   - General comments and recommendations
   - ALL kit/subclass variants with their abilities, restrictions, and comments
   - Any recommended ability points, races, or strategies mentioned

IMPORTANT:
- Include ALL kit information for each class (e.g., Fighter has Berserker, Wizard Slayer, Kensai).
- Keep the original text's meaning but clean up formatting issues (remove line breaks mid-sentence).
- Do NOT invent information. Only use what's in the FAQ text.
- Each "text" field should be a coherent, readable paragraph (or paragraphs) — not bullet points.
""".strip()


class CharacterRecord(BaseModel):
    id: str
    category: str
    subcategory: str
    question: str
    text: str


class CharacterDataset(BaseModel):
    records: list[CharacterRecord]


def fetch_faq_text() -> str:
    """Read the FAQ text from the local file."""
    return FAQ_PATH.read_text(encoding="utf-8")


def extract_records(faq_text: str, client: OpenAI) -> list[dict]:
    """Use OpenAI to extract structured character records from FAQ text."""
    user_prompt = f"""
Here is the full text of the Baldur's Gate II Character FAQ.
Extract one record for each of these 11 classes: {', '.join(CHARACTER_CLASSES)}.

FAQ TEXT:
{faq_text[:50000]}
"""

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {"role": "developer", "content": EXTRACTION_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        text_format=CharacterDataset,
    )

    dataset = response.output_parsed
    return [record.model_dump() for record in dataset.records]


def main() -> None:
    load_dotenv()
    client = OpenAI()

    print(f"Reading FAQ text from {FAQ_PATH}...")
    faq_text = fetch_faq_text()
    print(f"  Read {len(faq_text)} characters")

    print("Extracting character records with LLM...")
    records = extract_records(faq_text, client)
    print(f"  Extracted {len(records)} records")

    # Validate we got all 11 classes
    extracted_ids = {r["id"] for r in records}
    expected_ids = {c.lower() for c in CHARACTER_CLASSES}
    missing = expected_ids - extracted_ids
    if missing:
        print(f"  WARNING: Missing classes: {missing}")

    # Save to data/bg_characters.json
    output_path = Path("data/bg_characters.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"  Saved to {output_path}")

    # Print summary
    for record in records:
        text_len = len(record["text"])
        print(f"  {record['id']:12s} | {record['subcategory']:12s} | {text_len:5d} chars | Q: {record['question'][:60]}...")


if __name__ == "__main__":
    main()
