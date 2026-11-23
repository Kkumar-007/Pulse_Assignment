"""Measure merge precision: what share of consolidation's merges are right?

Step 1 draws a random sample of merges to label:
    python analysis/merge_sample.py output_v3/topic_mapping.csv
  -> writes analysis/merge_sample_output_v3.csv. Put Y (same issue) or N in `correct`.
Step 2, run the same command again: it scores your labels overall and by similarity band.
Do the same with output_v2/topic_mapping.csv to measure v2's over-merging.
"""
import math
import os
import sys

import pandas as pd

mapping_path = sys.argv[1] if len(sys.argv) > 1 else "output_v3/topic_mapping.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
tag = os.path.basename(os.path.dirname(mapping_path)) or "mapping"
sample_path = os.path.join("analysis", f"merge_sample_{tag}.csv")


def wilson(k, n, z=1.96):
    if n == 0:
        return "n/a"
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.0%} ({max(0.0, centre - half):.0%}-{min(1.0, centre + half):.0%}, n={n})"


if not os.path.exists(sample_path):
    m = pd.read_csv(mapping_path)
    merged = m[m["decision"] == "merged"]
    sim_col = "merged_similarity" if "merged_similarity" in merged.columns else "top_similarity"
    s = merged.sample(n=min(N, len(merged)), random_state=0)[["phrase", "canonical", "mentions", sim_col]]
    s = s.rename(columns={sim_col: "similarity"})
    s["correct"] = ""
    s.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(s)} merges to {sample_path}. Mark each Y or N in `correct`, then run this again.")
else:
    s = pd.read_csv(sample_path, keep_default_na=False)
    s["ok"] = s["correct"].str.strip().str.upper().str[0].map({"Y": True, "N": False})
    s = s.dropna(subset=["ok"])
    if s.empty:
        raise SystemExit(f"No Y/N labels in {sample_path} yet.")
    print(f"{mapping_path}: merge precision {wilson(int(s['ok'].sum()), len(s))}")
    w = (s["ok"].astype(bool) * s["mentions"]).sum() / s["mentions"].sum()
    print(f"Weighted by mentions: {w:.0%}")
    s["band"] = pd.cut(pd.to_numeric(s["similarity"], errors="coerce"),
                       [0, .4, .5, .6, .7, .8, 1.01], right=False)
    for band, g in s.groupby("band", observed=True):
        print(f"  similarity {band}: {wilson(int(g['ok'].sum()), len(g))}")
