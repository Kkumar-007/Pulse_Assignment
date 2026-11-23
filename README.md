# Pulse: from app reviews to a daily table of customer issues

Pulse turns raw Google Play reviews into a table of specific customer issues counted per day, so a product manager can see what is going wrong and whether it is getting worse. It started as a take-home assignment for a product company whose client was a food-delivery app. It runs on Swiggy reviews (1–9 October 2025), with a local Llama 3.1 8B served by Ollama and `all-MiniLM-L6-v2` sentence embeddings.

The first version (v1, `Scripts/`) ran end to end but had never been measured. When I evaluated it, two stages turned out not to work. This repository keeps v1 for reference, adds two rounds of fixes (v2, v3), and includes the hand-labelled evaluation behind every number below.

## Results

All figures come from hand labels: 200 randomly sampled reviews for triage, and 100 randomly sampled merge decisions per version for consolidation. The figures in brackets are 95% Wilson intervals.

| Measure | v1 | v2 | v3 (final) |
|---|---|---|---|
| Triage precision | 41% (30–52) | 90% (80–95) | same as v2 |
| Triage recall | 42% (31–54) | 92% (83–97) | same as v2 |
| Merge precision (share of merges that are correct) | not measured | 27% (19–36) | 60% (50–69) |
| Merge precision, weighted by mentions | not measured | 21% | 75% |
| Topics in the final table | 1,051 | 46 | 526 |
| Topics seen only once | 70% | 0% | 54% |

- **v1 triage was worse than doing nothing.** Its accuracy was 60%. Labelling every review "not actionable" would have scored 67%.
- **v1 merge precision wasn't measured.** In v1's output, 322 of the 1,051 rows were raw model replies (quoted copies, the literal `NEW_TOPIC`, notes), not topics.

## Dashboard

`dashboard/app.py` is a Streamlit app with three tabs:
- **Issues:** the topic table with a daily trend for each topic, and the flagged rises.
- **How phrases were merged:** every merge decision behind each topic.
- **How well it works:** the evaluation above, with charts.

It reads only the aggregated files in this repository.

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Pipeline (final version)

| Stage | Script | What it does |
|---|---|---|
| 1. Scrape | `Scripts_v2/scraper.py` | Pulls reviews into one CSV per day. Refuses to treat the scrape day, which is incomplete, as a full day. |
| 2. Triage | `Scripts_v2/process_day.py` | Labels each review ACTIONABLE or NOT_ACTIONABLE, in batches of 4, with JSON output limited to the two labels and a check that the label count matches. A small keyword rescue catches complaints the model misses; it is recorded in its own column so it can be scored. |
| 3. Extract | `Scripts_v3/extract_topics.py` | Lists up to 3 specific issues per review. Each issue must quote evidence from the review, and an issue whose evidence isn't in the review is dropped. Generic sentiment ("bad service") is dropped too. |
| 4. Consolidate | `Scripts_v3/consolidate_topics.py` | Merges phrases into topics, most frequent first. A phrase whose closest existing topic has cosine similarity below 0.60 becomes a new topic. Otherwise the LLM answers a yes/no "same specific problem?" question for up to 3 candidates. Every decision is logged in `topic_mapping.csv`. |
| 5. Spikes | `Scripts_v3/detect_spikes.py` | Flags days when an issue's count is well above what its recent share of reviews predicts, so a rise in review volume alone isn't flagged. |

All LLM calls go through `llm.py`, using Ollama's `/api/chat` with a JSON schema, temperature 0 and a fixed seed.

## What changed, and why

### v1 → v2: the triage step didn't read the reviews
- **The label check always passed.** `"ACTIONABLE" in line` is also true for `"NOT_ACTIONABLE"`, so every label the model wrote counted as actionable.
- **The output cap truncated batches.** A 10-token limit cut off later labels in each batch of 4, and missing labels defaulted to NOT_ACTIONABLE.
- **So position decided the label.** Among reviews without a rescue keyword, the actionable rate by position in the batch was 98%, 29%, 9% and 0% (5,501 reviews). In v2 it is 21%, 20%, 24% and 23% (chi-square p = 0.09). The actionable rate for 1-star vs 5-star reviews went from 32% vs 35% (no relationship) to 67% vs 1.4%.
- **Fixed in v2:** structured output, a label-count check with a one-by-one fallback, failures recorded instead of defaulted, `/api/chat` in place of a hand-written chat template, and the partial scrape day excluded.

### v2 → v3: my fix over-corrected
- **Example phrases leaked.** The four example phrases in the v2 extraction prompt became four of the top seven topics. None of 10 sampled reviews tagged "App Crashes At Checkout" mentioned a crash, and reviews averaged 3.0 phrases (v1: 1.75).
- **The merge step merged almost everything.** It asked the model to pick a candidate number or 0, and the 8B model almost always picked a number: 1,842 of 1,888 phrases were merged into 46 topics. Hand labels put merge precision at 27%.
- **Fixed in v3:** no domain example phrases, evidence-grounded extraction (2.0 issues per review, 0.2% dropped as ungrounded), a similarity floor, and pairwise yes/no checks.

### A threshold that didn't hold up
Labelling the v3 merges by similarity band showed 29% precision at 0.6–0.7, 66% at 0.7–0.8 and 89% above 0.8, which suggested raising the floor to 0.70 (predicted precision about 76%). A fresh sample from the 0.70 run scored 59% (49–68). Raising the floor split phrases into finer topics (762 instead of 526), and the merges left between them were the hard cases ("Delayed Refund" into "No Refund"). There was no gain and more fragmentation, so the floor stays at 0.60. Both runs are kept: `output_v3/` (0.60, final) and `output_v3_floor070/`.

### Speed
Ollama's server log showed 0.47 s of GPU work per consolidation call, but only one call every 2.5 s. On Windows, `localhost` tries IPv6 first and retries before falling back to IPv4. Changing to `127.0.0.1`, reusing HTTP connections, and sending the checks in parallel (with `OLLAMA_NUM_PARALLEL=4`) took consolidation from about 80 minutes to 4.7. The decisions are identical to the one-at-a-time version: on the same 1,551 phrases with a deterministic mock model, both versions wrote byte-identical outputs.

## Running it

Requires Ollama 0.5 or later (for JSON-schema output) and Python 3.11 or later (the pinned NumPy 2.3 needs it).

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b

python Scripts_v2/scraper.py            # dates in Scripts_v2/config.py
python Scripts_v2/process_day.py        # triage
python Scripts_v3/extract_topics.py
python Scripts_v3/consolidate_topics.py
python Scripts_v3/detect_spikes.py
```

Run every script from the repository root. Optional Ollama settings for parallel calls on an 8 GB GPU: `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_CONTEXT_LENGTH=2048`.

## Evaluating it

```bash
python analysis/triage_position_check.py "output_v2/triage/triage_*.csv" llm_label  # position effect
python analysis/merge_sample.py output_v3/topic_mapping.csv   # scores the labelled sample in analysis/
python analysis/make_label_sample.py                          # draw 200 reviews to label
python analysis/score_triage.py                               # v1 vs v2 precision and recall
python analysis/export_triage_summary.py                      # aggregate triage results for the dashboard
```

`merge_sample.py` draws a sample to label the first time it's run on a mapping file and scores it the next time. The three labelled merge samples are included, so the merge figures above can be reproduced from this repository. The triage checks need the review-level files, which aren't published (see below).

## Repository layout

```
Scripts/              v1 as originally submitted (kept for reference)
output/               v1 output
Scripts_v2/           scraper and triage (final), plus v2 extraction and consolidation
Scripts_v3/           final extraction, consolidation and spike detection
analysis/             evaluation scripts, the labelled merge samples and aggregate triage results
dashboard/            Streamlit dashboard (its own requirements.txt)
output_v2/            v2 topic table, merge log, daily totals, spikes
output_v3/            final topic table, merge log, spikes
output_v3_floor070/   the 0.70-floor experiment
```

Raw reviews and per-review outputs are excluded (`.gitignore`) because they contain reviewers' names and profile links.

## Limitations and next steps

- **One annotator.** All labels come from a single annotator using a strict rule: a merge is right only if a product team would fix both phrases as one issue, and borderline counts as wrong. Six pairs labelled twice got the same label both times, but they were easy cases.
- **Merge precision is 60%.** The remaining errors are fine distinctions (a delayed refund versus no refund, unhelpful versus unresponsive support) that neither cosine similarity nor the 8B judge separates. The next step is a stronger judge, given few-shot examples from the 300 labelled merges and scored on held-out labels.
- **The table needs a theme layer.** 54% of topics appear once. A product manager needs 20–30 themes with topics underneath, not 526 rows.
- **App-store reviews are a biased sample.** They over-represent angry and very happy customers, so the table shows what is changing more reliably than how common a problem is.
- **Spike detection is a demonstration.** With nine days of data and dozens of topics, a few flags at z ≥ 3 can occur by chance.
