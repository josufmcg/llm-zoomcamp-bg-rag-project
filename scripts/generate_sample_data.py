"""Generate sample conversation and feedback data for dashboard testing.

Creates fake conversation records and feedback entries to populate
the Grafana dashboard during development.

Usage:
    uv run python scripts/generate_sample_data.py
"""

import random
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from bg_rag.db import get_db_connection

SAMPLE_QUESTIONS = [
    "What is the best Fighter kit?",
    "Should I play a Mage or Sorcerer?",
    "What class can use stealth?",
    "Which Paladin kit is most powerful?",
    "How does the Monk class work?",
    "What are the Ranger's abilities?",
    "Is the Barbarian good for beginners?",
    "What spells does the Druid get?",
    "How do I build a good Thief?",
    "What is the Bard class good for?",
    "What's the difference between Cleric kits?",
]

SAMPLE_ANSWERS = [
    "The Kensai is considered the most powerful Fighter kit for pure damage output.",
    "Both are excellent spellcasters. The Mage has more spell variety, while the Sorcerer can cast more often.",
    "The Thief class specializes in stealth with Hide in Shadows and Move Silently skills.",
    "The Cavalier is a fan favorite with immunity to fear and poison, plus bonus damage vs demons.",
    "The Monk is a versatile fighter with martial arts, magic resistance, and thief abilities.",
    "Rangers get Weapon Specialization, a Racial Enemy bonus, Stealth, and Charm Person/Mammal.",
    "The Barbarian is great for beginners with high hit points, fast movement, and berserker rage.",
    "Druids get nature-themed spells including Shape Change, summoning, and elemental protections.",
    "Focus on Open Locks and Find Traps first, then decide between backstabbing or trap-setting.",
    "The Bard has high Lore for item identification, Bard Song buffs, and access to mage spells.",
    "Priest of Talos gets Lightning Bolt, Priest of Helm gets True Sight, Priest of Lathander gets Boon.",
]

RELEVANCE_OPTIONS = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
SEARCH_METHODS = ["vector", "keyword", "hybrid"]


def generate_one() -> None:
    """Generate a single fake conversation with optional feedback."""
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)
    search_method = random.choice(SEARCH_METHODS)
    timestamp = datetime.now(timezone.utc)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, model, search_method, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    question,
                    answer,
                    "gpt-4.1-mini",
                    search_method,
                    f"QUESTION: {question}\nCONTEXT: ...",
                    random.randint(200, 800),
                    random.randint(100, 400),
                    random.randint(300, 1200),
                    random.uniform(0.5, 4.0),
                    random.uniform(0.0001, 0.005),
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()["id"]

            # Judge feedback (70% chance)
            if random.random() < 0.7:
                relevance = random.choice(RELEVANCE_OPTIONS)
                cur.execute(
                    """
                    INSERT INTO feedback (
                        conversation_id, source, relevance, explanation, timestamp
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        "judge",
                        relevance,
                        f"The answer is {relevance.lower().replace('_', ' ')}.",
                        timestamp,
                    ),
                )

            # User feedback (50% chance)
            if random.random() < 0.5:
                score = random.choice([1, 1, 1, 1, -1])  # 80% positive
                cur.execute(
                    """
                    INSERT INTO feedback (
                        conversation_id, source, score, timestamp
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (conversation_id, "user", score, timestamp),
                )

        conn.commit()
    finally:
        conn.close()


def main() -> None:
    load_dotenv()

    print("Generating sample data (Ctrl+C to stop)...")
    count = 0
    try:
        while True:
            generate_one()
            count += 1
            if count % 10 == 0:
                print(f"  Generated {count} conversations...")
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopped after {count} conversations.")


if __name__ == "__main__":
    main()
