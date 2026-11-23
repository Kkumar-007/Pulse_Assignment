"""Stage 4: merge raw phrases into canonical topics and count them per day.

v1 bugs fixed here:
  - The LLM's raw reply became the topic name, so quoted copies ("'Poor Customer
    Service'"), the literal "NEW_TOPIC" and the model's notes became rows
    (322 of 1,051 v1 rows were not canonical topics). Now the model answers with a
    candidate NUMBER, and anything outside the list counts as "no match".
  - Candidates were sent as a Python list repr, which invited copied quote marks.
  - Phrases were processed in set order, which Python randomises each run, so
    reruns gave different canonical lists and the first phrasing seen became the
    name. Now phrases go from most to least frequent: deterministic, and the most
    common phrasing names each topic.
  - A lock made the "parallel" loop sequential; it is now an honest plain loop.
Also new: trivial variants (case, quotes, punctuation) merge without an LLM call,
and every decision is written to topic_mapping.csv so merges can be audited.
"""
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, OUT_DIR, SHORTLIST_K, day_strings
from llm import chat_json

SYSTEM = """You decide whether a new customer-issue phrase means the same thing as one of a few existing issue names.
Be strict: match only if a product team would treat them as the same issue.
"Rude Delivery Person" and "Delivery Partner Rude" are the same issue.
"Late Delivery" and "Delivery Issue" are different issues.
Answer with the number of the matching candidate, or 0 if none of them matches."""

SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}


def key(phrase):
    """Normalised form: case, quotes and punctuation don't create new topics.
    Only punctuation and symbols are removed, so Hindi vowel signs survive."""
    kept = "".join(" " if unicodedata.category(ch)[0] in "PS" and ch != "/" else ch
                   for ch in phrase.casefold())
    return re.sub(r"\s+", " ", kept).strip()


def load_phrases(days):
    counts, surface, per_day = Counter(), defaultdict(Counter), {d: Counter() for d in days}
    for day in days:
        path = os.path.join(OUT_DIR, "topics", f"topics_{day}.csv")
        if not os.path.isfile(path):
            print(f"[{day}] no topics file, skipped")
            continue
        df = pd.read_csv(path, keep_default_na=False)
        for cell in df.loc[df["extract_ok"].astype(str) == "True", "topics"]:
            for phrase in json.loads(cell or "[]"):
                k = key(phrase)
                if k:
                    counts[k] += 1
                    surface[k][phrase] += 1
                    per_day[day][k] += 1
    display = {k: surface[k].most_common(1)[0][0] for k in counts}
    return counts, display, per_day


def choose(phrase, candidates):
    """Index into `candidates` of the LLM's match, or None for no match / failed call."""
    options = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, 1))
    reply = chat_json(SYSTEM, f'New phrase: "{phrase}"\nCandidates:\n{options}', SCHEMA, num_predict=16)
    choice = reply.get("choice") if isinstance(reply, dict) else None
    if type(choice) is int and 1 <= choice <= len(candidates):
        return choice - 1
    return None


if __name__ == "__main__":
    days = day_strings()
    counts, display, per_day = load_phrases(days)
    order = sorted(counts, key=lambda k: (-counts[k], k))       # most frequent first
    print(f"{sum(counts.values())} mentions, {len(order)} distinct phrases after normalising.")
    if not order:
        raise SystemExit("No topics found. Run extract_topics.py first.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([display[k] for k in order], normalize_embeddings=True,
                           batch_size=256, show_progress_bar=False)
    canon, canon_vecs = [], np.empty_like(vectors)
    mapping, audit = {}, []
    for i, k in enumerate(order):
        target, decision, best = k, "new", None
        if canon:
            sims = canon_vecs[:len(canon)] @ vectors[i]
            shortlist = list(np.argsort(-sims)[:SHORTLIST_K])
            best = float(sims[shortlist[0]])
            picked = choose(display[k], [display[canon[j]] for j in shortlist])
            if picked is not None:
                target, decision = canon[shortlist[picked]], "merged"
        if decision == "new":
            canon_vecs[len(canon)] = vectors[i]
            canon.append(k)
        mapping[k] = target
        audit.append({"phrase": display[k], "mentions": counts[k], "canonical": display[target],
                      "decision": decision, "top_similarity": None if best is None else round(best, 3)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(order)} phrases, {len(canon)} topics so far")

    table = defaultdict(lambda: [0] * len(days))
    for d, day in enumerate(days):
        for k, c in per_day[day].items():
            table[display[mapping[k]]][d] += c
    trends = (pd.DataFrame.from_dict(table, orient="index", columns=days)
              .assign(total=lambda t: t.sum(axis=1)).sort_values("total", ascending=False))
    trends.index.name = "topic"

    os.makedirs(OUT_DIR, exist_ok=True)
    trends.to_csv(os.path.join(OUT_DIR, "topic_trends.csv"))
    pd.DataFrame(audit).to_csv(os.path.join(OUT_DIR, "topic_mapping.csv"), index=False)
    with open(os.path.join(OUT_DIR, "canonical_topics.json"), "w", encoding="utf-8") as f:
        json.dump({"topics": list(trends.index)}, f, indent=2, ensure_ascii=False)
    singletons = int((trends["total"] == 1).sum())
    print(f"{len(trends)} topics ({singletons} seen only once). "
          f"Saved topic_trends.csv, topic_mapping.csv, canonical_topics.json in {OUT_DIR}/")
