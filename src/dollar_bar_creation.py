import os
import glob
import pandas as pd
import numpy as np


TICKS_DIR = "../data/library/BAC"
OUTPUT_FILE = "../data/dollar_bars.parquet"

EMA_ALPHA_PRIOR = 0.05
EMA_ALPHA_INTRADAY = 0.01
K = 10.0
LAMBDA = 0.85

MARKET_OPEN = "09:30:00"
MARKET_CLOSE = "16:00:00"


def generate_adaptive_dollar_bars(trades, dollar_thresholds):
    times = trades[:, 0]
    prices = trades[:, 1].astype(float)
    sizes = trades[:, 2].astype(float)

    bars = []

    dollar_acc = 0.0
    size_acc = 0.0

    open_price = prices[0]
    high_price = prices[0]
    low_price = prices[0]
    start_time = times[0]

    for i in range(len(prices)):
        td = prices[i] * sizes[i]
        if td <= 0:
            continue

        dollar_acc += td
        size_acc += sizes[i]

        if prices[i] > high_price:
            high_price = prices[i]
        if prices[i] < low_price:
            low_price = prices[i]

        theta = float(dollar_thresholds[i])

        if dollar_acc >= theta and size_acc > 0:
            bars.append({
                "start_time": start_time,
                "end_time": times[i],
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(prices[i]),
                "size": float(size_acc),
                "dollar_volume": float(dollar_acc),
                "threshold": float(theta)
            })

            dollar_acc = 0.0
            size_acc = 0.0
            open_price = prices[i]
            high_price = prices[i]
            low_price = prices[i]
            start_time = times[i]

    out = pd.DataFrame(bars)
    if not out.empty:
        out = out[(out["size"] > 0) & (out["dollar_volume"] > 0)]
        out = out[out["end_time"] >= out["start_time"]]
    return out


if __name__ == "__main__":
    all_bars = []

    files = sorted(glob.glob(os.path.join(TICKS_DIR, "*.parquet")))

    for i in range(1, len(files)):
        prev_file = files[i - 1]
        file = files[i]

        prev_df = pd.read_parquet(prev_file)
        prev_df["timestamp"] = pd.to_datetime(prev_df["timestamp"])
        prev_df = prev_df.sort_values("timestamp")
        prev_df = (
            prev_df.set_index("timestamp")
            .between_time(MARKET_OPEN, MARKET_CLOSE)
            .reset_index()
        )

        df = pd.read_parquet(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = (
            df.set_index("timestamp")
            .between_time(MARKET_OPEN, MARKET_CLOSE)
            .reset_index()
        )

        if df.empty:
            continue

        minute = df["timestamp"].dt.floor("1min")

        if not prev_df.empty:
            prev_dollar_per_min = (
                (prev_df["price"] * prev_df["size"])
                .groupby(prev_df["timestamp"].dt.floor("1min"))
                .sum()
            )
            ema_prior = prev_dollar_per_min.ewm(
                alpha=EMA_ALPHA_PRIOR, adjust=False
            ).mean()
            thr_prior = minute.map(ema_prior).ffill().bfill()
        else:
            thr_prior = pd.Series(index=df.index, dtype=float)

        curr_dollar_per_min = (
            (df["price"] * df["size"])
            .groupby(minute)
            .sum()
        )
        ema_intraday = curr_dollar_per_min.ewm(
            alpha=EMA_ALPHA_INTRADAY, adjust=False
        ).mean()
        thr_intraday = minute.map(ema_intraday).ffill()

        if thr_prior.isna().all():
            dollar_threshold = K * thr_intraday
        else:
            dollar_threshold = K * (
                LAMBDA * thr_prior + (1.0 - LAMBDA) * thr_intraday
            )

        dollar_threshold = dollar_threshold.replace([np.inf, -np.inf], np.nan)
        if dollar_threshold.isna().any():
            med = dollar_threshold.median()
            dollar_threshold = dollar_threshold.fillna(med)

        bars = generate_adaptive_dollar_bars(
            df[["timestamp", "price", "size"]].values,
            dollar_threshold.values
        )

        if not bars.empty:
            bars["date"] = df["timestamp"].dt.date.iloc[0]
            all_bars.append(bars)

    final_bars = pd.concat(all_bars, ignore_index=True)
    final_bars.to_parquet(OUTPUT_FILE, index=False)
