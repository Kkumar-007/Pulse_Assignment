import os
import time
from datetime import datetime, timedelta
import pandas as pd
from google_play_scraper import reviews, Sort

# --- Configuration ---
APP_ID = 'in.swiggy.android'
OUTPUT_DIR = 'input_data'
TARGET_DATE = datetime(2025, 10, 10)
NUM_DAYS_TO_SCRAPE = 10

# --- Robust Scraping Logic with Pagination ---

if __name__ == "__main__":
    print(f"Starting robust review scrape for {APP_ID} using pagination.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine the oldest date we are interested in
    start_date = TARGET_DATE - timedelta(days=NUM_DAYS_TO_SCRAPE)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"Will fetch all reviews posted on or after: {start_date.strftime('%Y-%m-%d')}")

    all_reviews_list = []
    continuation_token = None
    
    # This loop will continue fetching reviews in batches until it finds reviews
    # older than our target start_date.
    while True:
        try:
            # Fetch a batch of reviews. The count per batch can be smaller.
            result, new_token = reviews(
                APP_ID,
                lang='en',
                country='in',
                sort=Sort.NEWEST,
                count=2000, # Fetch in batches of 2000
                continuation_token=continuation_token
            )

            if not result:
                print("No more reviews to fetch.")
                break

            all_reviews_list.extend(result)
            continuation_token = new_token
            
            # Check the date of the last review in the entire list we've built
            last_review_date = pd.to_datetime(all_reviews_list[-1]['at']).tz_localize(None)

            print(f"Fetched {len(result)} reviews. Total so far: {len(all_reviews_list)}. Oldest review in batch is from {last_review_date.strftime('%Y-%m-%d')}")

            # If the oldest review we have is already older than our start date,
            # we can stop fetching. We have all the data we need.
            if last_review_date < start_date:
                print("Oldest review is past the target date range. Halting scrape.")
                break
                
            # A small sleep to be polite to the server between batch requests
            time.sleep(2)

        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            break

    print(f"\nTotal reviews fetched: {len(all_reviews_list)}. Now processing into daily files.")
    
    # --- Processing and Saving ---
    
    if not all_reviews_list:
        print("No reviews were found to process.")
        exit()

    df = pd.DataFrame(all_reviews_list)
    df['at'] = pd.to_datetime(df['at']).dt.tz_localize(None)

    # Loop through the last 31 days and save files
    for i in range(NUM_DAYS_TO_SCRAPE):
        target_date_filter = TARGET_DATE - timedelta(days=i)
        date_str = target_date_filter.strftime('%Y-%m-%d')
        file_path = os.path.join(OUTPUT_DIR, f'reviews_{date_str}.csv')

        start_of_day = target_date_filter.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        daily_reviews_df = df[(df['at'] >= start_of_day) & (df['at'] < end_of_day)]

        if not daily_reviews_df.empty:
            daily_reviews_df.to_csv(file_path, index=False)
            print(f"[{date_str}] Saved {len(daily_reviews_df)} reviews.")
        else:
            print(f"[{date_str}] No reviews found in the fetched data for this day.")

    print("\n--- Daily data processing complete. ---")