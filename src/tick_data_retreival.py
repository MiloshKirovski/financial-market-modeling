import os
import requests
import datetime
import pandas as pd
from pathlib import Path

API_KEY = os.getenv("API_KEY")
BASE_PATH = Path("../data/library")
BASE_PATH.mkdir(exist_ok=True)


def fetch_day(symbol: str, date: datetime.date):
    date_str = date.strftime("%Y-%m-%d")
    print(f"\n=== Fetching {symbol} ticks for {date_str} ===")

    url = (
        f"https://api.polygon.io/v3/trades/{symbol}"
        f"?timestamp.gte={date_str}T00:00:00Z"
        f"&timestamp.lt={date_str}T23:59:59Z"
        f"&limit=50000"
        f"&apiKey={API_KEY}"
    )

    all_results = []

    while True:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}. Skipping day.")
            return pd.DataFrame()

        data = r.json()

        if "results" not in data or len(data["results"]) == 0:
            print("  No trades.")
            break

        all_results.extend(data["results"])
        print(f"  Got batch: {len(data['results'])}, total: {len(all_results)}")

        next_url = data.get("next_url")
        if not next_url:
            break

        url = next_url + f"&apiKey={API_KEY}"

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    timestamp_col = None
    for col in ["sip_timestamp", "participant_timestamp", "trf_timestamp"]:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        print("  No timestamp column found! Returning empty.")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df[timestamp_col], unit="ns")

    return df


def save_daily_ticks(symbol="BAC", days_back=5):
    symbol_folder = BASE_PATH / symbol
    symbol_folder.mkdir(exist_ok=True)

    today = datetime.datetime.now(datetime.UTC)

    for i in range(1, days_back + 1):
        day = today - datetime.timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        file_path = symbol_folder / f"{day_str}.parquet"

        if file_path.exists():
            print(f"{day_str} already saved. Skipping.")
            continue

        df = fetch_day(symbol, day)

        if df.empty:
            print(f"  -> No data for {day_str}.")
            continue

        df.to_parquet(file_path, index=False)
        print(f"  Saved {len(df)} trades to {file_path}")


if __name__ == "__main__":
    save_daily_ticks("BAC", days_back=300)
