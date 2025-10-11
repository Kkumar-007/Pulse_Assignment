import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---- Config ----
INPUT_FOLDER = 'output_data/topics'
CANONICAL_TOPICS_FILE = 'canonical_topics.json'
FINAL_COUNTS_TABLE_FILE = 'output_data/topic_trends.json'
OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
CONSOLIDATION_MODEL_NAME = 'llama3.1:8b'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
START_DATE = "2025-10-01"
END_DATE = "2025-10-10"
MAX_WORKERS = 8  # Use what your system can handle

# ---- Load Model ----
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
lock = threading.Lock()  # For thread-safe updates

CONSOLIDATION_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a topic consolidation agent. Your task is to match a new user-generated topic to an existing canonical topic from a provided list of candidates.
- If the new topic is semantically identical to one of the candidates, respond with that exact canonical topic.
- If no candidate is a suitable match, you MUST respond with the exact phrase 'NEW_TOPIC'.
- Be strict. Only match if the meaning is truly the same. "Late Delivery" and "Delivery Issue" are different. "Rude Delivery Person" and "Delivery partner rude" are the same.<|eot_id|><|start_header_id|>user<|end_header_id|>
New Topic: "{new_topic}"
Canonical Candidates: {candidate_list}
Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def get_canonical_topic(raw_topic, candidates, candidate_embeddings):
    if not candidates:
        return "NEW_TOPIC"
    new_topic_embedding = embedding_model.encode(raw_topic, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(new_topic_embedding, candidate_embeddings)
    top_k = min(3, len(candidates))
    top_results = cosine_scores[0].topk(top_k)
    selection = [candidates[i] for i in top_results.indices]
    try:
        prompt = CONSOLIDATION_PROMPT_TEMPLATE.format(
            new_topic=raw_topic, 
            candidate_list=str(selection)
        )
        payload = {"model": CONSOLIDATION_MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        answer = response.json()['response'].strip()
        return answer
    except Exception as e:
        print(f"LLM error for topic '{raw_topic}': {e}")
        return "Error"

def consolidate_one_topic(raw_topic):
    global canonical_topics_list, canonical_embeddings
    # Lock because list and embedding must be updated atomically
    with lock:
        decision = get_canonical_topic(raw_topic, canonical_topics_list, canonical_embeddings)
        if decision == "NEW_TOPIC":
            canonical_topics_list.append(raw_topic)
            canonical_embeddings = embedding_model.encode(canonical_topics_list, convert_to_tensor=True)
            final_topic = raw_topic
        else:
            final_topic = decision
    return raw_topic, final_topic

if __name__ == "__main__":
    # Load canonical topics
    if os.path.exists(CANONICAL_TOPICS_FILE):
        with open(CANONICAL_TOPICS_FILE) as f:
            canonical_topics_list = json.load(f)['topics']
    else:
        canonical_topics_list = []
    canonical_embeddings = embedding_model.encode(canonical_topics_list, convert_to_tensor=True) if canonical_topics_list else None
    print(f"Starting with {len(canonical_topics_list)} canonical topics.")

    # Step 1. Collect all unique raw topics and per-day raw counts
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    all_dates = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range((end_date - start_date).days + 1)]
    unique_raw_topics = set()
    per_day_raw_counts = {date: {} for date in all_dates}

    for date in all_dates:
        fname = os.path.join(INPUT_FOLDER, f"topics_{date}.csv")
        if not os.path.isfile(fname):
            print(f"[skip] {fname}")
            continue
        df = pd.read_csv(fname)
        for topics_str in df['extracted_topics'].dropna():
            for raw_topic in [t.strip() for t in topics_str.split(';')]:
                if not raw_topic or raw_topic == "N/A":
                    continue
                unique_raw_topics.add(raw_topic)
                per_day_raw_counts[date][raw_topic] = per_day_raw_counts[date].get(raw_topic, 0) + 1

    print(f"Found {len(unique_raw_topics)} unique raw topics across all days.")

    # Step 2. Consolidate all unique raw topics using parallel threads
    consolidation_cache = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(consolidate_one_topic, topic): topic for topic in unique_raw_topics}
        for i, future in enumerate(as_completed(futures)):
            raw, final = future.result()
            consolidation_cache[raw] = final
            if (i+1) % 10 == 0 or (i+1) == len(unique_raw_topics):
                print(f"\rConsolidation progress: {i+1}/{len(unique_raw_topics)}", end="")
    print("\nConsolidation complete.")

    print(f"Final canonical topics: {len(canonical_topics_list)}")

    # Step 3. Tabulate canonical topic counts per day
    topic_table = {topic: [0]*len(all_dates) for topic in canonical_topics_list}
    for d_idx, date in enumerate(all_dates):
        day_raw_counts = per_day_raw_counts[date]
        for raw_topic, count in day_raw_counts.items():
            canon_topic = consolidation_cache.get(raw_topic, raw_topic)
            if canon_topic not in topic_table:
                topic_table[canon_topic] = [0]*len(all_dates)
            topic_table[canon_topic][d_idx] += count

    # Save canonical topics
    with open(CANONICAL_TOPICS_FILE, 'w') as f:
        json.dump({"topics": sorted(canonical_topics_list)}, f, indent=2)
    print("Saved updated canonical topics.")

    # Save final trend table
    trend_table = {"dates": all_dates, "topics": topic_table}
    with open(FINAL_COUNTS_TABLE_FILE, 'w') as f:
        json.dump(trend_table, f, indent=2)
    print("Saved topic trend table.")

    # Print for reporting
    print("\n# Topic Trend Table\n")
    print("| Topic | " + " | ".join(all_dates) + " |")
    print("|-------" + "|".join(["------"]*len(all_dates))+"|")
    for topic, counts in trend_table["topics"].items():
        print(f"| {topic} | " + " | ".join(str(c) for c in counts) + " |")
