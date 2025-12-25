import pandas as pd
import numpy as np


INPUT_FILE = "../data/dollar_bars_labeled.parquet"
OUTPUT_FILE = "../data/dollar_bars_weighted.parquet"


def get_concurrency(events, time_index):
    concurrency = pd.Series(0, index=time_index)

    for _, row in events.iterrows():
        concurrency.loc[row['start_time']:row['t1']] += 1

    return concurrency


def get_average_uniqueness(events, concurrency):
    avg_u = []

    for _, row in events.iterrows():
        c = concurrency.loc[row['start_time']:row['t1']]
        avg_u.append((1.0/c).mean())

    return pd.Series(avg_u, index=events.index)


def apply_time_decay(weights, half_life=5000):
    n = len(weights)
    ranks = np.arange(n)[::-1]

    decay = np.exp(-np.log(2) * ranks / half_life)
    return weights * decay


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    df = df.dropna(subset=["t1", "label"]).sort_values(by=["start_time"]).reset_index(drop=True)

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['t1'] = pd.to_datetime(df['t1'])

    time_index = pd.Index(sorted(set(df['start_time']).union(set(df["t1"]))))

    concurrency = get_concurrency(df, time_index)

    df['uniqueness'] = get_average_uniqueness(df, concurrency)

    df['sample_weight'] = apply_time_decay(df['uniqueness'], half_life=5000)

    df['sample_weight'] = df['sample_weight'] / df['sample_weight'].sum()

    df.to_parquet(OUTPUT_FILE, index=False)
