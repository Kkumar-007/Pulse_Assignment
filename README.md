# Pulse_Assignment

## Project Overview
This project implements a modular, multi-stage data processing pipeline designed to transform unstructured, raw app store reviews into structured, time-series trend data. This is achieved through the orchestration of several specialized AI Agents, each powered by a Local Large Language Model (LLM) via Ollama.

By delegating specific, high-stakes reasoning tasks (triage, extraction, and semantic consolidation) to distinct LLM agents, the system accurately filters noise and unifies disparate user feedback into a canonical, time-stamped table (topic_trends.csv) of emerging customer issues, enabling rapid product and operational trend analysis.

## Architecture: Multi-Agent Data Processing Pipeline
The system is a sequential pipeline consisting of five distinct stages, three of which utilize a dedicated AI agent to perform a complex, intelligent task:

# StageScriptAI Agent / Component Pipeline

| Stage | Script / Component | Technology | Core Function | Input |
|-------|-------------------|------------|---------------|-------|
| **1. Ingestion** | `scraper.py` | `google-play-scraper` | Data Scraper (Non-Agent)<br>Fetches raw reviews from the app store. | App ID |
| **2. Triage & Filtering** | `process_day.py` | Ollama (`llama3.1:8b`) | Review Triage Agent<br>Filters raw reviews to identify and isolate Actionable feedback (bug reports, specific feature requests). | Raw Reviews |
| **3. Topic Extraction** | `extract_topics.py` | Ollama (`llama3.1:8b`) | Topic Extraction Agent<br>Extracts raw, unstructured topic phrases (e.g., "delivery person rude") from the actionable review text. | Actionable Reviews |
| **4. Consolidation & Aggregation** | `consolidate_topics.py` | Ollama (`llama3.1:8b`) & Sentence-Transformers | Semantic Consolidation Agent<br>Maps noisy raw topics to a unified set of Canonical Topics using semantic similarity and LLM validation. | Raw Topics, Canonical List |
| **5. Formatting** | `json_to_csv.py` | Pandas | Data Formatter (Non-Agent)<br>Converts the JSON time-series output into a final, human-readable CSV matrix. | `topic_trends.json` |

## Installation and Setup
This project requires a running **Ollama** server for LLM inference.

### Prerequisites

**Install Ollama**: Follow the [official instructions](https://ollama.ai) to install and start the Ollama server on your machine.

**Pull LLM Model**: This pipeline is configured to use the `llama3.1:8b` model for all LLM tasks. Pull the model before running the scripts:
```bash
ollama pull llama3.1:8b
```

### Python Dependancies
Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

## Usage Guide

### Quick Start (Full Pipeline Run)

Execute the scripts sequentially from the project root directory. Each script builds upon the output of the previous one.
```bash
# 1. Scrape the reviews
python scraper.py

# 2. Filter for actionable reviews
python process_day.py

# 3. Extract raw topics from actionable reviews
python extract_topics.py

# 4. Consolidate raw topics into canonical list and aggregate counts
python consolidate_topics.py

# 5. Convert final JSON output to CSV format
python json_to_csv.py
```
### Individual Component Use
- To scrape data without re-running the analysis, just run python scraper.py.
- To update the canonical topic list with new data without re-scraping, ensure stages 2 and 3 have run for the new data, and then run python consolidate_topics.py.

## Output Format

The final output file, `output_data/topic_trends.csv`, is a matrix that allows for easy analysis of how issue frequency changes over time.

| Column | Description |
|--------|-------------|
| **Topic** | The Canonical Topic name (e.g., "Inaccurate Delivery Estimate", "Unresponsive Customer Support"). This is the result of the LLM/Embedding consolidation. |
| **YYYY-MM-DD** | A column for each day in the analysis range (e.g., `2025-10-01`). |
| **Cell Value** | The count (integer) of actionable reviews that were mapped to that specific Topic on that specific Date. |

### Example Snippet (Conceptual)

| Topic | 2025-10-08 | 2025-10-09 | 2025-10-10 |
|-------|------------|------------|------------|
| Delivery issue | 2 | 2 | 0 |
| Delivery partner rude | 1 | 3 | 1 |
| Lack of Accountability | 1 | 0 | 0 |
| Order Cancellation Problems | 0 | 1 | 0 |

## Technical Details

### Technologies

- **LLM Inference**: Ollama (Local/Self-hosted LLM)
- **LLM Model**: `llama3.1:8b` (Used for Triage, Topic Extraction, and Topic Consolidation)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) (Used for semantic similarity in consolidation)
- **Data Handling**: Pandas
- **Scraping**: `google-play-scraper`

### Performance Considerations

- **Concurrency**: All LLM-dependent stages (`process_day.py`, `extract_topics.py`, `consolidate_topics.py`) utilize Python's `ThreadPoolExecutor` to parallelize multiple requests to the Ollama server, significantly reducing total processing time.
- **Batching**: `process_day.py` implements prompt batching for review triage, maximizing the efficiency of the LLM calls.

### Data Quality & Features

- **Triage for Actionability**: The pipeline discards over 50% of raw data (simple praise, single-word complaints, etc.) early on. This ensures that all downstream LLM costs and processing time are focused exclusively on feedback that requires a product or operational response.

- **Standardized Extraction**: The extraction stage ensures raw topics adhere to a strict noun-phrase format, improving the quality of the input for the final consolidation phase.

- **Semantic Consolidation**: `consolidate_topics.py` uses a hybrid approach:
  - **LLM Verification**: The LLM is used to confirm semantic identity between a new raw topic and existing candidates.
  - **Embedding Similarity**: Sentence-BERT embeddings are used to cluster raw topics that are similar in meaning (e.g., "delivery boy was rude" and "delivery partner's poor behavior") under a single Canonical Topic ("Delivery partner rude"). This prevents fragmentation and ensures trends are accurate.
