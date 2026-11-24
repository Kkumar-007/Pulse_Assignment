"""Check whether triage labels depend on a review's position inside its batch.

If triage judges content, the actionable rate should be about the same in every
position (row index % batch size). Run from the project root:

  v1:  python analysis/triage_position_check.py
  v2:  python analysis/triage_position_check.py "output_v2/triage/triage_*.csv" llm_label
       (llm_label = the model's own label, before the keyword rescue)
"""
import glob
import os
import sys

import pandas as pd

pattern = sys.argv[1] if len(sys.argv) > 1 else "output_data/Actionable_Reviews/intermediate_actionable_reviews_*.csv"
col = sys.argv[2] if len(sys.argv) > 2 else "classification"
BATCH = 4
KEYWORDS = ["not working", "error", "failed", "crash", "bug", "issue",
            "refund", "delay", "rude", "cancelled"]   # v1's keyword rescue list

frames = []
for path in sorted(glob.glob(pattern)):
    df = pd.read_csv(path)
    df["position"] = df.index % BATCH + 1
    df["day"] = os.path.basename(path)[-14:-4]
    frames.append(df)
if not frames:
    raise SystemExit(f"No files match {pattern}")
df = pd.concat(frames, ignore_index=True)
df["actionable"] = df[col].eq("ACTIONABLE")
text = df["content"].astype(str).str.lower()
df["keyword"] = text.apply(lambda t: any(k in t for k in KEYWORDS))
df["words"] = text.str.split().str.len()
nk = df[~df["keyword"]]

print(f"{pattern}  [{col}]")
print(f"Reviews: {len(df):,}   actionable: {df['actionable'].mean():.1%}   "
      f"not labelled: {(~df[col].isin(['ACTIONABLE', 'NOT_ACTIONABLE'])).sum()}")
print("\nActionable rate by position in batch, reviews without a rescue keyword:")
print(nk.groupby("position")["actionable"].agg(["mean", "size"]).round(3).to_string())
print("\nActionable rate by star rating, reviews without a rescue keyword:")
print(nk.groupby("score")["actionable"].agg(["mean", "size"]).round(3).to_string())
print("\nActionable rate by review length (words), reviews without a rescue keyword:")
bins = pd.cut(nk["words"], [0, 3, 8, 20, 10_000])
print(nk.groupby(bins, observed=True)["actionable"].agg(["mean", "size"]).round(3).to_string())
praise = df[df["actionable"] & (df["score"] == 5) & (df["words"] <= 6) & ~df["keyword"]]
print(f"\n5-star reviews of 6 words or fewer labelled ACTIONABLE: {len(praise)}")
for t in praise["content"].head(8):
    print("  -", str(t)[:80])
