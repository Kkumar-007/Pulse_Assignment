"""Export aggregate triage results for the dashboard (no review text).

The hand labels and per-review outputs stay private because they contain review
text and reviewers' names. This writes only counts and rates:
  analysis/summary_triage_scores.csv    precision / recall / accuracy per system
  analysis/summary_triage_position.csv  actionable rate by position in batch
  analysis/summary_triage_stars.csv     actionable rate by star rating
Run from the project root after labelling analysis/label_sample.csv:
    python analysis/export_triage_summary.py
"""
import glob
import math
import os

import pandas as pd

SYSTEMS = {
    "v1": ("output_data/Actionable_Reviews/intermediate_actionable_reviews_*.csv", "classification"),
    "v2 (model only)": ("output_v2/triage/triage_*.csv", "llm_label"),
    "v2 (with keyword rescue)": ("output_v2/triage/triage_*.csv", "classification"),
}
KEYWORDS = ["not working", "error", "failed", "crash", "bug", "issue",
            "refund", "delay", "rude", "cancelled"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load(pattern, col):
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        df["position"] = df.index % 4 + 1
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["actionable"] = df[col].eq("ACTIONABLE")
    df["keyword"] = df["content"].astype(str).str.lower().apply(lambda t: any(k in t for k in KEYWORDS))
    return df


labels = pd.read_csv(os.path.join("analysis", "label_sample.csv"), keep_default_na=False, encoding="utf-8-sig")
labels["truth"] = labels["my_label"].str.strip().str.upper().str[0].map({"A": True, "N": False})
labels = labels.dropna(subset=["truth"])

scores, position, stars = [], [], []
for name, (pattern, col) in SYSTEMS.items():
    df = load(pattern, col)
    merged = labels.merge(df[["reviewId", col]].drop_duplicates("reviewId"), on="reviewId")
    merged = merged[merged[col].isin(["ACTIONABLE", "NOT_ACTIONABLE"])]
    pred, truth = merged[col].eq("ACTIONABLE"), merged["truth"].astype(bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    p, plo, phi = wilson(tp, tp + fp)
    r, rlo, rhi = wilson(tp, tp + fn)
    scores.append({"system": name, "labelled_reviews": len(merged), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "precision": p, "precision_lo": plo, "precision_hi": phi,
                   "recall": r, "recall_lo": rlo, "recall_hi": rhi, "accuracy": (tp + tn) / len(merged)})
    if name == "v2 (with keyword rescue)":
        continue                                   # position and stars: the model's own labels
    nk = df[~df["keyword"]]
    version = name.split(" ")[0]
    for pos, g in nk.groupby("position"):
        position.append({"version": version, "position": int(pos),
                         "actionable_rate": g["actionable"].mean(), "reviews": len(g)})
    for score, g in nk.groupby("score"):
        stars.append({"version": version, "stars": int(score),
                      "actionable_rate": g["actionable"].mean(), "reviews": len(g)})

pd.DataFrame(scores).round(4).to_csv(os.path.join("analysis", "summary_triage_scores.csv"), index=False)
pd.DataFrame(position).round(4).to_csv(os.path.join("analysis", "summary_triage_position.csv"), index=False)
pd.DataFrame(stars).round(4).to_csv(os.path.join("analysis", "summary_triage_stars.csv"), index=False)
print(pd.DataFrame(scores)[["system", "precision", "recall", "accuracy"]].round(3).to_string(index=False))
print("Saved analysis/summary_triage_scores.csv, summary_triage_position.csv, summary_triage_stars.csv")
