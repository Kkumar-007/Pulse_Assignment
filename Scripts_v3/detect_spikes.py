"""Stage 5 (v3): flag days when an issue rises well above its recent baseline.

Each day's count is compared with what the issue's recent share of actionable
reviews predicts for that day's volume, so a rise in review volume alone is not
flagged. Same logic as v2, with a higher minimum count, and it reports how many
comparisons were made: with many topics and days, a few flags at z >= 3 can
appear by chance.
"""
import os

import numpy as np
import pandas as pd

from config import BASELINE_DAYS, MIN_SPIKE_COUNT, OUT_DIR, SPIKE_Z, TRIAGE_DIR

if __name__ == "__main__":
    trends = pd.read_csv(os.path.join(OUT_DIR, "topic_trends.csv"), index_col="topic")
    days = [c for c in trends.columns if c != "total"]
    volume = (pd.read_csv(os.path.join(TRIAGE_DIR, "daily_totals.csv"), dtype={"day": str})
              .set_index("day")["actionable"].reindex(days))
    counts = trends[days]
    shares = counts.div(volume, axis=1)

    flags, tests = [], 0
    for t in range(BASELINE_DAYS, len(days)):
        day = days[t]
        expected = shares[days[t - BASELINE_DAYS:t]].mean(axis=1) * volume[day]
        observed = counts[day]
        z = (observed - expected) / np.sqrt(expected.clip(lower=0.5))   # Poisson-style
        tests += int((observed >= MIN_SPIKE_COUNT).sum())
        hit = (observed >= MIN_SPIKE_COUNT) & (z >= SPIKE_Z)
        for topic in trends.index[hit]:
            flags.append({"day": day, "topic": topic, "observed": int(observed[topic]),
                          "expected": round(float(expected[topic]), 1), "z": round(float(z[topic]), 1)})

    out = pd.DataFrame(flags, columns=["day", "topic", "observed", "expected", "z"])
    out.sort_values(["day", "z"], ascending=[True, False]).to_csv(
        os.path.join(OUT_DIR, "spikes.csv"), index=False)
    print(f"{len(out)} spikes flagged from {tests} topic-days checked "
          f"(first {BASELINE_DAYS} days have no baseline). Saved {OUT_DIR}/spikes.csv")
    if len(out):
        print(out.to_string(index=False))
