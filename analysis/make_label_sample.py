"""Draw a random sample of raw reviews to label by hand.

Label them BEFORE you look at any model output, so the model can't sway you.
Fill the `my_label` column with A (actionable) or N (not actionable), save the
file, then run analysis/score_triage.py. Run from the project root:
    python analysis/make_label_sample.py [sample_size]
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, "Scripts_v2")
from config import RAW_DIR, day_strings  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = os.path.join("analysis", "label_sample.csv")

if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists; delete or rename it first so you don't overwrite your labels.")
frames = [pd.read_csv(p).assign(day=os.path.basename(p)[8:18])
          for p in sorted(glob.glob(os.path.join(RAW_DIR, "reviews_*.csv")))
          if os.path.basename(p)[8:18] in day_strings()]
reviews = pd.concat(frames, ignore_index=True)
sample = reviews.sample(n=min(N, len(reviews)), random_state=42)[["reviewId", "day", "score", "content"]]
sample["my_label"] = ""
sample.to_csv(OUT, index=False, encoding="utf-8-sig")   # utf-8-sig so Excel shows Hindi text correctly
print(f"Wrote {len(sample)} reviews to {OUT}. Label each A or N in the my_label column.")
