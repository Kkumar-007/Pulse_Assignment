"""Stage 3 (v3): extract the specific issues a review states, each grounded in its own words.

What went wrong in v2: the prompt's example phrases ("Late Delivery", "Refund Not
Processed", "Rude Delivery Partner", "App Crashes At Checkout") leaked into the
output. The model padded long, angry reviews with them (none of 10 sampled
"App Crashes At Checkout" reviews mentioned a crash), and allowing 5 phrases per
review invited the padding (3.0 per review vs 1.75 in v1).

v3: no domain example phrases in the prompt, at most 3 issues, and every issue
must come with evidence copied from the review. Issues whose evidence is not in
the review are dropped and counted, and so are generic sentiment phrases.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import (EVIDENCE_MIN_OVERLAP, EXTRACT_WORKERS, GENERIC_ISSUES, MAX_TOPIC_WORDS,
                    MAX_TOPICS_PER_REVIEW, OUT_DIR, TRIAGE_DIR, day_strings)
from llm import chat_json

SYSTEM = """You list the specific problems and requests that a customer states in one review of a food-delivery app.
Rules:
- Include only issues the review itself states. Never add an issue just because it is common in food delivery.
- Name each issue in English, in Title Case, as a short noun phrase of 2 to 5 words: what went wrong, and with what.
- For each issue, copy the words from the review that state it into "evidence", exactly as written.
- Skip general sentiment that names no specific problem, such as "bad service", "worst app" or "very disappointed".
- Most reviews state one or two issues. If the review states none, return an empty list.
Reviews may be in English, Hindi or Hinglish. Write the issue in English, but copy the evidence exactly as written."""

SCHEMA = {"type": "object",
          "properties": {"issues": {"type": "array", "maxItems": MAX_TOPICS_PER_REVIEW,
                                    "items": {"type": "object",
                                              "properties": {"issue": {"type": "string"},
                                                             "evidence": {"type": "string"}},
                                              "required": ["issue", "evidence"]}}},
          "required": ["issues"]}


def words(text):
    return re.findall(r"\w+", str(text).casefold())


def grounded(evidence, review_words):
    ev = words(evidence)
    return bool(ev) and sum(w in review_words for w in ev) / len(ev) >= EVIDENCE_MIN_OVERLAP


def generic_key(issue):
    return " ".join(words(issue))


def extract(text):
    """(kept issues, audit records), or (None, None) if the call failed."""
    if not text.strip():
        return [], []
    reply = chat_json(SYSTEM, f'Review: "{text}"', SCHEMA, num_predict=320)
    if not isinstance(reply, dict) or not isinstance(reply.get("issues"), list):
        return None, None
    review_words = set(words(text))
    kept, audit = [], []
    for item in reply["issues"][:MAX_TOPICS_PER_REVIEW]:
        if not isinstance(item, dict):
            continue
        issue = re.sub(r"\s+", " ", str(item.get("issue", ""))).strip().strip("'\"`.;:,").strip()
        evidence = str(item.get("evidence", ""))
        if not issue or len(issue.split()) > MAX_TOPIC_WORDS:
            status = "dropped: malformed"
        elif generic_key(issue) in GENERIC_ISSUES:
            status = "dropped: generic"
        elif not grounded(evidence, review_words):
            status = "dropped: evidence not in review"
        elif issue in kept:
            status = "dropped: duplicate"
        else:
            status = "kept"
            kept.append(issue)
        audit.append({"issue": issue, "evidence": evidence, "status": status})
    return kept, audit


def extract_day(day):
    path = os.path.join(TRIAGE_DIR, "actionable", f"actionable_{day}.csv")
    if not os.path.isfile(path):
        print(f"[{day}] no actionable file, skipped")
        return
    df = pd.read_csv(path, keep_default_na=False)
    texts = df["content"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        results = list(pool.map(extract, texts))

    df["extract_ok"] = [kept is not None for kept, _ in results]
    df["topics"] = [json.dumps(kept, ensure_ascii=False) if kept is not None else "" for kept, _ in results]
    df["extraction_audit"] = [json.dumps(a, ensure_ascii=False) if a is not None else "" for _, a in results]
    os.makedirs(os.path.join(OUT_DIR, "topics"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "topics", f"topics_{day}.csv"), index=False)

    statuses = [r["status"] for _, a in results if a for r in a]
    empty = sum(1 for kept, _ in results if kept == [])
    failed = sum(1 for kept, _ in results if kept is None)
    print(f"[{day}] {len(df)} reviews: {len(df) - empty - failed} with issues, {empty} with none, "
          f"{failed} failed | issues kept {statuses.count('kept')}, ungrounded "
          f"{statuses.count('dropped: evidence not in review')}, generic {statuses.count('dropped: generic')}")


if __name__ == "__main__":
    for day in day_strings():
        extract_day(day)
