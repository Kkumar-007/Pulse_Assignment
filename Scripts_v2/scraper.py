"""Stage 1: scrape Google Play reviews into one CSV per day.

Changes from v1: dates come from config.py, and the script refuses to treat the
scrape day as a complete day (v1 saved a partial final day, which looked like a
collapse in every issue).
"""
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
from google_play_scraper import Sort, reviews

from config import APP_ID, COUNTRY, END_DATE, LANG, RAW_DIR, START_DATE, day_strings

if __name__ == "__main__":
    if END_DATE >= date.today():
        raise SystemExit(f"END_DATE {END_DATE} is today or later, so that day is incomplete. "
                         "Set END_DATE in config.py to yesterday or earlier.")
    os.makedirs(RAW_DIR, exist_ok=True)
    start = datetime.combine(START_DATE, datetime.min.time())
    print(f"Scraping {APP_ID} reviews from {START_DATE} to {END_DATE}.")

    fetched, token = [], None
    while True:
        try:
            batch, token = reviews(APP_ID, lang=LANG, country=COUNTRY, sort=Sort.NEWEST,
                                   count=2000, continuation_token=token)
        except Exception as error:   # keep what we have, but say so loudly
            print(f"Scrape stopped early: {error}. Data before this point may be incomplete.")
            break
        if not batch:
            break
        fetched.extend(batch)
        oldest = pd.to_datetime(fetched[-1]["at"]).tz_localize(None)
        print(f"Fetched {len(batch)} (total {len(fetched)}); oldest so far {oldest:%Y-%m-%d}")
        if oldest < start:
            break
        time.sleep(2)

    if not fetched:
        raise SystemExit("No reviews fetched.")
    df = pd.DataFrame(fetched)
    df["at"] = pd.to_datetime(df["at"]).dt.tz_localize(None)
    for day in day_strings():
        day_start = datetime.fromisoformat(day)
        daily = df[(df["at"] >= day_start) & (df["at"] < day_start + timedelta(days=1))]
        daily.to_csv(os.path.join(RAW_DIR, f"reviews_{day}.csv"), index=False)
        print(f"[{day}] {len(daily)} reviews")
