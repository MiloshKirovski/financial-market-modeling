import pandas as pd
import numpy as np


INPUT_FILE = "../data/dollar_bars_labeled.parquet"
OUTPUT_FILE = "../data/dollar_bars_weighted.parquet"


def apply_time_decay(weights, half_life=5000):
    n = len(weights)
    ranks = np.arange(n)[::-1]
    decay = np.exp(-np.log(2) * ranks / half_life)
    return weights * decay


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    df = df.dropna(subset=["start_time", "t1", "label"]).sort_values("start_time").reset_index(drop=True)

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["t1"] = pd.to_datetime(df["t1"])

    bad = df["t1"] < df["start_time"]
    if bad.any():
        df.loc[bad, "t1"] = df.loc[bad, "start_time"]

    time_values = np.sort(
        np.unique(
            np.concatenate(
                [df["start_time"].values.astype("datetime64[ns]"),
                 df["t1"].values.astype("datetime64[ns]")]
            )
        )
    )

    start_pos = np.searchsorted(time_values, df["start_time"].values.astype("datetime64[ns]"), side="left")
    end_pos = np.searchsorted(time_values, df["t1"].values.astype("datetime64[ns]"), side="left")

    diff = np.zeros(len(time_values) + 1, dtype=np.int32)
    np.add.at(diff, start_pos, 1)
    np.add.at(diff, end_pos + 1, -1)

    concurrency = np.cumsum(diff[:-1]).astype(float)

    uniq = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        c = concurrency[start_pos[i] : end_pos[i] + 1]
        uniq[i] = np.mean(1.0 / c)

    df["uniqueness"] = uniq

    w = apply_time_decay(df["uniqueness"].values, half_life=5000)
    s = w.sum()
    df["sample_weight"] = w / s if s > 0 else np.ones_like(w) / len(w)

    df.to_parquet(OUTPUT_FILE, index=False)
