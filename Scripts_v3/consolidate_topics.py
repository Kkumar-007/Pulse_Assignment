"""Stage 4 (v3): merge phrases into topics with a similarity floor and pairwise checks.

What went wrong in v2: given a numbered list with "0 = none", the 8B model almost
always picked a number. 1,842 of 1,888 phrases were merged into 46 topics, and a
third of all mentions were merged at cosine similarity below 0.4 ("Cold Food" ->
"Late Delivery", "Missing Items" -> "Rude Delivery Partner").

v3:
  - A phrase whose best candidate is below SIM_FLOOR becomes a new topic, no LLM call.
  - Otherwise each candidate above the floor is checked one at a time, most similar
    first, with a yes/no question ("same specific problem?"). The first yes wins.
  - Kept from v2: most-frequent-first order, normalised keys, and an audit row per phrase.

Speed (same decisions as the one-at-a-time version, checked on test data):
  - Phrases go in blocks. For each block, the yes/no checks the loop will need are
    sent in parallel first (round 1: every phrase's most similar candidate; round 2:
    the next candidate, only for phrases that got "no"; and so on). The loop then
    walks the block in order and finds almost every answer already waiting.
  - Every answer is cached in output_v3/merge_cache_<hash>.jsonl, so an interrupted
    or repeated run resumes instead of starting over. Changing the model or the
    prompt changes the hash, which starts a fresh cache.
"""
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (CONSOLIDATE_BLOCK, CONSOLIDATE_WORKERS, EMBEDDING_MODEL, LLM_MODEL, OUT_DIR,
                    SHORTLIST_K, SIM_FLOOR, day_strings)
from llm import chat_json

SYSTEM = """You check whether two short customer-issue phrases from food-delivery app reviews describe the same specific problem: one a product team would investigate and fix as a single issue.
Answer false if they concern different things (for example food quality versus delivery time, or support staff versus delivery partners), if one is a cause or a consequence of the other, or if one is much broader than the other.
Examples:
"Delayed Delivery" / "Late Delivery" -> true
"Refund Not Credited" / "Refund Not Processed" -> true
"Cold Food" / "Late Delivery" -> false
"Rude Customer Support" / "Rude Delivery Partner" -> false
"Order Issues" / "Wrong Item Delivered" -> false"""

SCHEMA = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}


def key(phrase):
    """Normalised form: case, quotes and punctuation don't create new topics."""
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


def same_issue(a, b):
    """True / False from the LLM; None if the call failed (treated as not the same)."""
    reply = chat_json(SYSTEM, f'Phrase A: "{a}"\nPhrase B: "{b}"\nSame specific problem?',
                      SCHEMA, num_predict=16)
    value = reply.get("same") if isinstance(reply, dict) else None
    return value if isinstance(value, bool) else None


CACHE_PATH = os.path.join(
    OUT_DIR, f"merge_cache_{hashlib.sha1((LLM_MODEL + SYSTEM).encode()).hexdigest()[:8]}.jsonl")


class Judge:
    """Yes/no merge answers: cached on disk, and fetched in parallel ahead of need."""

    def __init__(self, workers):
        self.cache, self.calls = {}, 0
        if os.path.isfile(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self.cache[(row["a"], row["b"])] = row["same"]
                    except (ValueError, KeyError):
                        pass                       # a line cut off by an interrupted run
        self.cached_at_start = len(self.cache)
        os.makedirs(OUT_DIR, exist_ok=True)
        self.file = open(CACHE_PATH, "a", encoding="utf-8")
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def _store(self, pair, value):
        self.calls += 1
        if value is not None:                      # failed calls aren't cached, so they get retried
            self.cache[pair] = value
            self.file.write(json.dumps({"a": pair[0], "b": pair[1], "same": value},
                                       ensure_ascii=False) + "\n")

    def prefetch(self, pairs):
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        for pair, value in zip(todo, self.pool.map(lambda p: same_issue(*p), todo)):
            self._store(pair, value)
        self.file.flush()

    def same(self, a, b):
        if (a, b) not in self.cache:               # only topics created inside the current block
            self._store((a, b), same_issue(a, b))
        return self.cache.get((a, b))

    def close(self):
        self.pool.shutdown()
        self.file.close()


if __name__ == "__main__":
    days = day_strings()
    counts, display, per_day = load_phrases(days)
    order = sorted(counts, key=lambda k: (-counts[k], k))       # most frequent first
    print(f"{sum(counts.values())} mentions, {len(order)} distinct phrases after normalising.")
    if not order:
        raise SystemExit("No topics found. Run extract_topics.py first.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding {len(order)} phrases on {model.device} (seconds either way; the LLM calls are the slow part).")
    vectors = model.encode([display[k] for k in order], normalize_embeddings=True,
                           batch_size=256, show_progress_bar=False)
    judge = Judge(CONSOLIDATE_WORKERS)
    print(f"{judge.cached_at_start} answers already cached in {CACHE_PATH}; "
          f"{CONSOLIDATE_WORKERS} parallel LLM calls.")
    canon, canon_vecs = [], np.empty_like(vectors)
    mapping, audit = {}, []
    started = time.time()
    for start in range(0, len(order), CONSOLIDATE_BLOCK):
        block = range(start, min(start + CONSOLIDATE_BLOCK, len(order)))

        # 1. Ask ahead, in parallel, using the topics that exist at the start of the block.
        #    Round r asks about candidate r only if candidates 0..r-1 all answered "no":
        #    the same questions, in the same order, that step 2 will ask.
        n0 = len(canon)
        if n0:
            ahead = {}
            for i in block:
                sims = canon_vecs[:n0] @ vectors[i]
                ahead[i] = [j for j in np.argsort(-sims)[:SHORTLIST_K] if sims[j] >= SIM_FLOOR]
            for r in range(SHORTLIST_K):
                judge.prefetch([(display[order[i]], display[canon[c[r]]]) for i, c in ahead.items()
                                if len(c) > r and all(judge.cache.get(
                                    (display[order[i]], display[canon[j]])) is False for j in c[:r])])

        # 2. Decide in order, exactly as the one-at-a-time version did.
        for i in block:
            k = order[i]
            target, decision, best, merged_sim = k, "new: first topic", None, None
            if canon:
                sims = canon_vecs[:len(canon)] @ vectors[i]
                shortlist = [j for j in np.argsort(-sims)[:SHORTLIST_K] if sims[j] >= SIM_FLOOR]
                best = float(sims.max())
                decision = "new: below floor" if not shortlist else "new: rejected by LLM"
                for j in shortlist:
                    if judge.same(display[k], display[canon[j]]):
                        target, decision, merged_sim = canon[j], "merged", float(sims[j])
                        break
            if decision.startswith("new"):
                canon_vecs[len(canon)] = vectors[i]
                canon.append(k)
            mapping[k] = target
            audit.append({"phrase": display[k], "mentions": counts[k], "canonical": display[target],
                          "decision": decision,
                          "top_similarity": None if best is None else round(best, 3),
                          "merged_similarity": None if merged_sim is None else round(merged_sim, 3)})

        done, elapsed = block.stop, time.time() - started
        print(f"  {done}/{len(order)} phrases, {len(canon)} topics, {judge.calls} LLM calls, "
              f"{elapsed / 60:.1f} min elapsed, ~{elapsed / done * (len(order) - done) / 60:.1f} min left")
    judge.close()
    calls = judge.calls

    table = defaultdict(lambda: [0] * len(days))
    for d, day in enumerate(days):
        for k, c in per_day[day].items():
            table[display[mapping[k]]][d] += c
    trends = (pd.DataFrame.from_dict(table, orient="index", columns=days)
              .assign(total=lambda t: t.sum(axis=1)).sort_values("total", ascending=False))
    trends.index.name = "topic"

    os.makedirs(OUT_DIR, exist_ok=True)
    trends.to_csv(os.path.join(OUT_DIR, "topic_trends.csv"))
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(os.path.join(OUT_DIR, "topic_mapping.csv"), index=False)
    with open(os.path.join(OUT_DIR, "canonical_topics.json"), "w", encoding="utf-8") as f:
        json.dump({"topics": list(trends.index)}, f, indent=2, ensure_ascii=False)
    print(f"{len(trends)} topics ({int((trends['total'] == 1).sum())} seen only once), {calls} LLM calls. "
          f"Decisions: {audit_df['decision'].value_counts().to_dict()}")
    print(f"Saved topic_trends.csv, topic_mapping.csv, canonical_topics.json in {OUT_DIR}/")
