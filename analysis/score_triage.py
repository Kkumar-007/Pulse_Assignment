"""Score v1 and v2 triage against your hand labels.

Needs analysis/label_sample.csv with my_label filled in (A or N). Run from the project root:
    python analysis/score_triage.py
Reports precision and recall for "actionable", each with a 95% Wilson interval,
because 200 reviews give wide intervals and you should quote them.
"""
import glob
import math
import os

import pandas as pd

SAMPLE = os.path.join("analysis", "label_sample.csv")
SYSTEMS = {
    "v1 (original)": ("output_data/Actionable_Reviews/intermediate_actionable_reviews_*.csv", "classification"),
    "v2 model only": ("output_v2/triage/triage_*.csv", "llm_label"),
    "v2 + keyword rescue": ("output_v2/triage/triage_*.csv", "classification"),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.0%} ({lo:.0%}-{hi:.0%}, n={n})" if n else "n/a"


labels = pd.read_csv(SAMPLE, keep_default_na=False)
labels["truth"] = labels["my_label"].str.strip().str.upper().str[0].map({"A": True, "N": False})
labels = labels.dropna(subset=["truth"])
print(f"{len(labels)} hand-labelled reviews, {labels['truth'].mean():.0%} actionable by your labels.\n")

for name, (pattern, col) in SYSTEMS.items():
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"{name}: no files yet ({pattern})")
        continue
    preds = pd.concat([pd.read_csv(p, usecols=["reviewId", col]) for p in files]).drop_duplicates("reviewId")
    merged = labels.merge(preds, on="reviewId", how="inner")
    merged = merged[merged[col].isin(["ACTIONABLE", "NOT_ACTIONABLE"])]
    pred = merged[col].eq("ACTIONABLE")
    truth = merged["truth"].astype(bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    print(f"{name}:  precision {fmt(tp, tp + fp)}   recall {fmt(tp, tp + fn)}   "
          f"accuracy {fmt(tp + tn, len(merged))}")
    print(f"    confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")
