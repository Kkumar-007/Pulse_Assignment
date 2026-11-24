"""Settings for v3. v3 replaces stages 3-5 only (extract, consolidate, spikes);
it reads the v2 triage output, which checked out (no position effect, p = 0.09).

Run from the project root, for example:
    python Scripts_v3/extract_topics.py
"""
from datetime import date, timedelta

# --- Data -------------------------------------------------------------------
START_DATE = date(2025, 10, 1)
END_DATE = date(2025, 10, 9)          # last complete day

# --- Folders ----------------------------------------------------------------
TRIAGE_DIR = "output_v2"              # v2 triage output: actionable/ and daily_totals.csv
OUT_DIR = "output_v3"

# --- Models -----------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"   # not "localhost": on Windows that tries IPv6 first,
                                                  # is refused, and costs ~2 s per call (v2 log: 0.47 s
                                                  # of GPU work per call, one call every 2.5 s)
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED = 0
REQUEST_TIMEOUT = 120
RETRIES = 3

# --- Stage 3: extraction ----------------------------------------------------
EXTRACT_WORKERS = 8
MAX_TOPICS_PER_REVIEW = 3             # v2 allowed 5 and the model padded lists to fill them
MAX_TOPIC_WORDS = 6
EVIDENCE_MIN_OVERLAP = 0.8            # share of the evidence's words that must appear in the review
GENERIC_ISSUES = {                    # sentiment with no specific problem; dropped if extracted anyway
    "bad service", "poor service", "worst service", "bad experience", "worst experience",
    "poor experience", "bad app", "worst app", "app not recommended", "not recommended",
    "waste of time", "very disappointed", "disappointing service", "unreliable service",
}

# --- Stage 4: consolidation -------------------------------------------------
SHORTLIST_K = 3
CONSOLIDATE_WORKERS = 4               # parallel LLM calls; match OLLAMA_NUM_PARALLEL (see UPDATES.md)
CONSOLIDATE_BLOCK = 64                # phrases whose LLM checks are sent together
SIM_FLOOR = 0.60                      # below this cosine similarity a phrase is a new topic, no LLM call.
                                      # Tested 0.70 (23 Nov): on fresh hand-labelled samples of 100 merges,
                                      # 0.60 scored 60% precision (526 topics) and 0.70 scored 59% (762 topics).
                                      # No gain, more fragments, so 0.60 stays. Runs: output_v3_floor060/, _floor070/.

# --- Stage 5: spike detection -----------------------------------------------
BASELINE_DAYS = 5
MIN_SPIKE_COUNT = 5                   # v2 used 3; "3 observed vs 0.5 expected" is noise
SPIKE_Z = 3.0


def day_strings():
    n = (END_DATE - START_DATE).days + 1
    return [(START_DATE + timedelta(days=i)).isoformat() for i in range(n)]
