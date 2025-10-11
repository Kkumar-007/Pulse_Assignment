import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
INPUT_FOLDER = 'output_data/Actionable_Reviews'
OUTPUT_FOLDER = 'output_data/topics'
OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
EXTRACTION_MODEL_NAME = 'llama3.1:8b'
MAX_WORKERS = 8  # Increase concurrency to 8 threads or as per hardware


EXTRACTION_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that extracts key topics from user reviews. From the review provided, identify and list all distinct issues, requests, or feedback points.
- Each topic must be a short, descriptive noun phrase (3-5 words).
- If there are multiple distinct topics, separate them with a semicolon.
- If there are no clear topics, respond with "N/A".<|eot_id|><|start_header_id|>user<|end_header_id|>
Review: "{review_text}"
Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def extract_topics_with_ollama(review_text):
    if not isinstance(review_text, str) or not review_text.strip():
        return "N/A"

    try:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(review_text=review_text)

        payload = {
            "model": EXTRACTION_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }

        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()

        extracted_topics = response.json()['response'].strip()
        return extracted_topics

    except Exception as e:
        print(f"Error during topic extraction for review: {e}")
        return "Error"


def process_file(date_str):
    input_file = os.path.join(INPUT_FOLDER, f"final_actionable_reviews_{date_str}.csv")
    output_file = os.path.join(OUTPUT_FOLDER, f"topics_{date_str}.csv")

    if not os.path.isfile(input_file):
        print(f"Input file {input_file} not found, skipping.")
        return

    print(f"Processing actionable reviews for {date_str}...")

    df = pd.read_csv(input_file)

    if 'content' not in df.columns:
        print(f"File {input_file} missing 'content' column, skipping.")
        return

    all_topics = [None] * len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_topics_with_ollama, row['content']): idx for idx, row in df.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                all_topics[idx] = result
            except Exception as e:
                print(f"Error processing review index {idx}: {e}")
                all_topics[idx] = "Error"

    df['extracted_topics'] = all_topics
    df.to_csv(output_file, index=False)
    print(f"Saved extracted topics to {output_file}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    start_date = datetime.strptime("2025-10-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-10-10", "%Y-%m-%d")

    current_date = end_date
    while current_date >= start_date:
        date_str = current_date.strftime("%Y-%m-%d")
        process_file(date_str)
        current_date -= timedelta(days=1)
