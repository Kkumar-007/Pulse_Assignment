import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime, timedelta

# -- Configuration --
INPUT_FOLDER = 'input_data'
OUTPUT_FOLDER = 'output_data'
OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.1:8b'
BATCH_SIZE = 4
MAX_WORKERS = 4
RETRY_LIMIT = 3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -- Prompt Template (same as your current code) --
TRIAGE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a review analyst. Your task is to determine if each user review contains actionable feedback.
Actionable feedback includes specific issues, bug reports, feature requests, or detailed complaints.
Simple praise, insults without detail, or spam are NOT actionable.

Respond ONLY with 'ACTIONABLE' or 'NOT_ACTIONABLE' for each review below.
Separate your responses line-by-line matching each review.

Examples:
Review: "The delivery driver was very rude and threw my package."
Response: ACTIONABLE
Review: "Love the app, use it daily!"
Response: NOT_ACTIONABLE

Now classify these reviews:
{batched_reviews}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def classify_batch_with_ollama(reviews, batch_index=0):
    formatted_reviews = "\n".join([f'Review {i+1}: "{r}"' for i, r in enumerate(reviews)])
    prompt = TRIAGE_PROMPT_TEMPLATE.format(batched_reviews=formatted_reviews)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 10}
    }

    for attempt in range(RETRY_LIMIT):
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=90)
            response.raise_for_status()
            raw_output = response.json().get("response", "").strip()
            lines = [l.strip().upper() for l in raw_output.splitlines() if l.strip()]
            predictions = []
            for l in lines:
                if "ACTIONABLE" in l:
                    predictions.append("ACTIONABLE")
                else:
                    predictions.append("NOT_ACTIONABLE")
            if len(predictions) < len(reviews):
                predictions.extend(["NOT_ACTIONABLE"] * (len(reviews) - len(predictions)))
            return predictions[:len(reviews)]

        except requests.exceptions.RequestException as e:
            print(f"[Retry {attempt+1}/{RETRY_LIMIT}] Error on batch {batch_index}: {e}")
            time.sleep(3)
        except Exception as e:
            print(f"Unexpected error on batch {batch_index}: {e}")
            return ["NOT_ACTIONABLE"] * len(reviews)

    print(f"Failed after {RETRY_LIMIT} retries for batch {batch_index}. Defaulting to NOT_ACTIONABLE.")
    return ["NOT_ACTIONABLE"] * len(reviews)

def simple_postprocessing(text, classification):
    actionable_keywords = ['not working', 'error', 'failed', 'crash', 'bug', 'issue', 'refund', 'delay', 'rude', 'cancelled']
    if classification == "NOT_ACTIONABLE":
        text_lower = text.lower()
        if any(k in text_lower for k in actionable_keywords):
            return "ACTIONABLE"
    return classification

def two_stage_triage(df):
    reviews = df['content'].astype(str).tolist()
    classifications = [None] * len(reviews)
    batches = [(i, reviews[i:i+BATCH_SIZE]) for i in range(0, len(reviews), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_batch_with_ollama, batch, idx // BATCH_SIZE): idx for idx, batch in batches}
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                results = future.result()
                start = batch_index
                classifications[start:start+len(results)] = results
            except Exception as e:
                print(f"Batch {batch_index} failed: {e}")

    df['classification'] = classifications

    borderline_reviews = df[df['classification'] == 'NOT_ACTIONABLE'].copy()
    borderline_reviews.reset_index(inplace=True)
    reclassified = []

    for _, row in borderline_reviews.iterrows():
        review = row['content']
        new_class = simple_postprocessing(review, "NOT_ACTIONABLE")
        reclassified.append((row['index'], new_class))

    for idx, new_class in reclassified:
        df.at[idx, 'classification'] = new_class

    return df

if __name__ == "__main__":
    start_date = datetime.strptime("2025-10-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-10-10", "%Y-%m-%d")

    current_date = end_date
    while current_date >= start_date:
        date_str = current_date.strftime("%Y-%m-%d")
        input_file = os.path.join(INPUT_FOLDER, f"reviews_{date_str}.csv")
        if not os.path.isfile(input_file):
            print(f"Input file {input_file} not found, skipping.")
            current_date -= timedelta(days=1)
            continue

        print(f"Processing reviews for {date_str}...")
        df = pd.read_csv(input_file)

        if 'content' not in df.columns:
            print(f"File {input_file} missing 'content' column, skipping.")
            current_date -= timedelta(days=1)
            continue

        start_time = time.time()
        df = two_stage_triage(df)
        elapsed = time.time() - start_time
        print(f"Triage for {date_str} complete in {elapsed:.2f}s")

        actionable_df = df[df['classification'] == 'ACTIONABLE'].copy()
        print(f"Total reviews processed: {len(df)}")
        print(f"Actionable reviews found: {len(actionable_df)}")

        intermediate_file = os.path.join(OUTPUT_FOLDER, f"intermediate_actionable_reviews_{date_str}.csv")
        output_file = os.path.join(OUTPUT_FOLDER, f"final_actionable_reviews_{date_str}.csv")

        df.to_csv(intermediate_file, index=False)
        actionable_df.to_csv(output_file, index=False)
        print(f"Saved intermediate classification to {intermediate_file}")
        print(f"Saved actionable reviews to {output_file}")

        current_date -= timedelta(days=1)
