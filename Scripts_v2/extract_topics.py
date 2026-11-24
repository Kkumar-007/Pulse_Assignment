"""Stage 3: extract short issue phrases from each actionable review.

v1 stored the model's reply as free text, so preambles ("Here are the distinct
issues...") and "N/A plus an explanation" were later counted as topics (262 v1
replies ran over 12 words). Now the reply is a JSON list, cleaned and length-capped,
and stored as JSON so no ";" splitting is needed. A failed call is recorded as
failed instead of becoming the topic "Error".
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import EXTRACT_WORKERS, MAX_TOPIC_WORDS, MAX_TOPICS_PER_REVIEW, OUT_DIR, day_strings
from llm import chat_json

SYSTEM = """You extract the specific issues and requests from one review of a food-delivery app.
Write each as a short noun phrase of 2 to 5 words in Title Case, for example
"Late Delivery", "Refund Not Processed", "Rude Delivery Partner", "App Crashes At Checkout".
List each distinct issue once. If the review names no specific issue or request, return an empty list.
Reviews may be in English, Hindi or Hinglish; always write the phrases in English."""

SCHEMA = {"type": "object",
          "properties": {"topics": {"type": "array", "maxItems": MAX_TOPICS_PER_REVIEW,
                                    "items": {"type": "string"}}},
          "required": ["topics"]}

NOT_A_TOPIC = {"N/A", "NA", "NONE", "NO ISSUE", "NO ISSUES"}


def clean(phrases):
    kept = []
    for phrase in phrases:
        phrase = re.sub(r"\s+", " ", str(phrase)).strip().strip("'\"`.;:,").strip()
        if (phrase and phrase.upper() not in NOT_A_TOPIC
                and len(phrase.split()) <= MAX_TOPIC_WORDS and phrase not in kept):
            kept.append(phrase)
    return kept


def extract(text):
    """List of phrases ([] = no issue found), or None if the call failed."""
    if not text.strip():
        return []
    reply = chat_json(SYSTEM, f'Review: "{text}"', SCHEMA, num_predict=160)
    if not isinstance(reply, dict) or not isinstance(reply.get("topics"), list):
        return None
    return clean(reply["topics"])


def extract_day(day):
    path = os.path.join(OUT_DIR, "actionable", f"actionable_{day}.csv")
    if not os.path.isfile(path):
        print(f"[{day}] no actionable file, skipped")
        return
    df = pd.read_csv(path, keep_default_na=False)   # v1 relied on pandas silently reading "N/A" as NaN
    texts = df["content"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        results = list(pool.map(extract, texts))

    df["extract_ok"] = [r is not None for r in results]
    df["topics"] = [json.dumps(r) if r is not None else "" for r in results]
    os.makedirs(os.path.join(OUT_DIR, "topics"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "topics", f"topics_{day}.csv"), index=False)

    empty = sum(1 for r in results if r == [])
    failed = sum(1 for r in results if r is None)
    print(f"[{day}] {len(df)} reviews: {len(df) - empty - failed} with topics, "
          f"{empty} with no issue (triage false positives), {failed} failed")


if __name__ == "__main__":
    for day in day_strings():
        extract_day(day)
