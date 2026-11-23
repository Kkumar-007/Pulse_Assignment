"""Stage 2: label each review ACTIONABLE or NOT_ACTIONABLE.

v1 bugs fixed here:
  1. `"ACTIONABLE" in line` was also true for "NOT_ACTIONABLE", so every label
     the model wrote counted as actionable.
  2. A 10-token output cap cut off later labels in each batch of 4; missing labels
     were filled in as NOT_ACTIONABLE, so position in the batch decided the label
     (v1 outputs: 98% actionable in position 1, 0% in position 4).
  3. Failed calls silently became NOT_ACTIONABLE.
Now: JSON output restricted to the two labels, a count check with a one-by-one
fallback, failures recorded as UNKNOWN, and the keyword rescue kept in its own column.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import (KEYWORD_RESCUE, OUT_DIR, RAW_DIR, RESCUE_KEYWORDS,
                    TRIAGE_BATCH_SIZE, TRIAGE_WORKERS, day_strings)
from llm import chat_json

LABELS = ["ACTIONABLE", "NOT_ACTIONABLE"]

SYSTEM = """You are a review analyst for a food-delivery app. Label every review.
ACTIONABLE: names a specific problem, bug, complaint or feature request that a product team could act on.
NOT_ACTIONABLE: praise, thanks, a rating with no detail, an insult with no detail, spam, or no specific issue.
Examples:
"The delivery driver was very rude and threw my package." -> ACTIONABLE
"Refund not credited for my cancelled order" -> ACTIONABLE
"Love the app, use it daily!" -> NOT_ACTIONABLE
"nice app" -> NOT_ACTIONABLE
Return exactly one label per review, in the same order as the reviews."""


def schema_for(n):
    return {"type": "object",
            "properties": {"labels": {"type": "array", "minItems": n, "maxItems": n,
                                      "items": {"type": "string", "enum": LABELS}}},
            "required": ["labels"]}


def label_batch(texts):
    """Labels for `texts`, in order. Falls back to one call per review on a count mismatch."""
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    reply = chat_json(SYSTEM, f"Label these {len(texts)} reviews:\n{numbered}",
                      schema_for(len(texts)), num_predict=64 + 16 * len(texts))
    labels = reply.get("labels") if isinstance(reply, dict) else None
    if isinstance(labels, list) and len(labels) == len(texts) and all(l in LABELS for l in labels):
        return labels
    if len(texts) > 1:
        return [label_batch([t])[0] for t in texts]
    return ["UNKNOWN"]


def triage_day(day):
    path = os.path.join(RAW_DIR, f"reviews_{day}.csv")
    if not os.path.isfile(path):
        print(f"[{day}] no input file, skipped")
        return None
    df = pd.read_csv(path)
    texts = (df["content"].fillna("").astype(str)
             .str.replace(r"\s+", " ", regex=True).str.strip().tolist())

    starts = list(range(0, len(texts), TRIAGE_BATCH_SIZE))
    labels = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=TRIAGE_WORKERS) as pool:
        results = pool.map(lambda s: label_batch(texts[s:s + TRIAGE_BATCH_SIZE]), starts)
        for start, batch_labels in zip(starts, results):   # map keeps input order
            labels[start:start + len(batch_labels)] = batch_labels

    df["llm_label"] = labels
    has_keyword = pd.Series([any(k in t.lower() for k in RESCUE_KEYWORDS) for t in texts], index=df.index)
    df["rescued"] = KEYWORD_RESCUE & df["llm_label"].eq("NOT_ACTIONABLE") & has_keyword
    df["classification"] = df["llm_label"].mask(df["rescued"], "ACTIONABLE")

    os.makedirs(os.path.join(OUT_DIR, "triage"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "actionable"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "triage", f"triage_{day}.csv"), index=False)
    df[df["classification"] == "ACTIONABLE"].to_csv(
        os.path.join(OUT_DIR, "actionable", f"actionable_{day}.csv"), index=False)

    summary = {"day": day, "reviews": len(df),
               "actionable": int((df["classification"] == "ACTIONABLE").sum()),
               "rescued_by_keyword": int(df["rescued"].sum()),
               "unknown": int((df["llm_label"] == "UNKNOWN").sum())}
    print(f"[{day}] {summary['reviews']} reviews, {summary['actionable']} actionable "
          f"({summary['rescued_by_keyword']} by keyword), {summary['unknown']} unknown")
    return summary


if __name__ == "__main__":
    rows = [r for r in (triage_day(d) for d in day_strings()) if r]
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "daily_totals.csv"), index=False)
    print(f"Saved {OUT_DIR}/daily_totals.csv")
