"""Settings shared by every v2 script.

Run every script from the project root, for example:
    python Scripts_v2/process_day.py
"""
from datetime import date, timedelta

# --- Data -------------------------------------------------------------------
APP_ID = "in.swiggy.android"
LANG, COUNTRY = "en", "in"
START_DATE = date(2025, 10, 1)
END_DATE = date(2025, 10, 9)   # last COMPLETE day. Never the day you scrape:
                               # a partial day looks like a collapse in every issue.

# --- Folders ----------------------------------------------------------------
RAW_DIR = "input_data"         # scraper output, reused by v2 so v1 and v2 see identical input
OUT_DIR = "output_v2"          # v2 writes here, so the v1 outputs stay available for comparison

# --- Models -----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED = 0
REQUEST_TIMEOUT = 120          # seconds per LLM call
RETRIES = 3

# --- Stage 2: triage --------------------------------------------------------
TRIAGE_BATCH_SIZE = 4
TRIAGE_WORKERS = 4
KEYWORD_RESCUE = True          # kept from v1, but recorded in its own column so it can be measured
RESCUE_KEYWORDS = ["not working", "error", "failed", "crash", "bug", "issue",
                   "refund", "delay", "rude", "cancelled"]

# --- Stage 3: extraction ----------------------------------------------------
EXTRACT_WORKERS = 8
MAX_TOPICS_PER_REVIEW = 5
MAX_TOPIC_WORDS = 6

# --- Stage 4: consolidation -------------------------------------------------
SHORTLIST_K = 3                # canonical topics shown to the LLM for each new phrase

# --- Stage 5: spike detection -----------------------------------------------
BASELINE_DAYS = 5              # days of history each day is compared against
MIN_SPIKE_COUNT = 3            # ignore spikes smaller than this many mentions
SPIKE_Z = 3.0                  # how far above expected a day must be to be flagged


def day_strings():
    """Every day from START_DATE to END_DATE inclusive, as YYYY-MM-DD strings."""
    n = (END_DATE - START_DATE).days + 1
    return [(START_DATE + timedelta(days=i)).isoformat() for i in range(n)]
