"""Stage 5 (new): flag days when an issue rises well above its recent baseline.

A raw count is not a trend: if twice as many reviews arrive after an app release,
every issue doubles. So each day's count is compared with what the issue's recent
SHARE of actionable reviews predicts for that day's volume.
"""
import os

import numpy as np
import pandas as pd

from config import BASELINE_DAYS, MIN_SPIKE_COUNT, OUT_DIR, SPIKE_Z

if __name__ == "__main__":
    trends = pd.read_csv(os.path.join(OUT_DIR, "topic_trends.csv"), index_col="topic")
    days = [c for c in trends.columns if c != "total"]
    volume = (pd.read_csv(os.path.join(OUT_DIR, "daily_totals.csv"), dtype={"day": str})
              .set_index("day")["actionable"].reindex(days))
    counts = trends[days]
    shares = counts.div(volume, axis=1)

    flags = []
    for t in range(BASELINE_DAYS, len(days)):
        day = days[t]
        baseline_share = shares[days[t - BASELINE_DAYS:t]].mean(axis=1)
        expected = baseline_share * volume[day]
        observed = counts[day]
        z = (observed - expected) / np.sqrt(expected.clip(lower=0.5))   # Poisson-style
        hit = (observed >= MIN_SPIKE_COUNT) & (z >= SPIKE_Z)
        for topic in trends.index[hit]:
            flags.append({"day": day, "topic": topic, "observed": int(observed[topic]),
                          "expected": round(float(expected[topic]), 1), "z": round(float(z[topic]), 1)})

    out = pd.DataFrame(flags, columns=["day", "topic", "observed", "expected", "z"])
    out.sort_values(["day", "z"], ascending=[True, False]).to_csv(
        os.path.join(OUT_DIR, "spikes.csv"), index=False)
    print(f"{len(out)} spikes flagged (needs {BASELINE_DAYS} days of history, so the first "
          f"{BASELINE_DAYS} days are never checked). Saved {OUT_DIR}/spikes.csv")
    if len(out):
        print(out.to_string(index=False))
