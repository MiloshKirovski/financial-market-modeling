import pandas as pd
import numpy as np

BARS_FILE = "../data/dollar_bars.parquet"
OUTPUT_FILE = "../data/dollar_bars_labeled.parquet"

VOL_WINDOW = 50
PT_MULT = 1.0
SL_MULT = 0.5
MAX_HOLD_MINUTES = 15

FAST_SPAN = 20
SLOW_SPAN = 60

def compute_volatility(returns, window):
    return returns.ewm(span=window, adjust=False).std()


def compute_side_from_ema(close, fast_span, slow_span):
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    raw = np.sign(fast - slow)
    return raw.shift(1)


def compute_side_momo(close, vol, lookback=10, z_enter=0.40, z_exit=0.05, min_hold=3):
    close = close.astype(float)

    mom = close.pct_change(lookback)
    z = mom / (vol.replace(0, np.nan))

    raw = pd.Series(0.0, index=close.index)

    raw[z >= z_enter] = 1.0
    raw[z <= -z_enter] = -1.0

    side = np.zeros(len(close), dtype=float)
    curr = 0.0
    hold = 0

    z_vals = z.to_numpy(dtype=float)
    raw_vals = raw.to_numpy(dtype=float)

    for i in range(len(close)):
        if np.isnan(z_vals[i]):
            side[i] = 0.0
            continue

        if curr == 0.0:
            if raw_vals[i] != 0.0:
                curr = raw_vals[i]
                hold = min_hold
        else:
            if hold > 0:
                hold -= 1
            else:
                if abs(z_vals[i]) <= z_exit:
                    curr = 0.0
                elif curr == 1.0 and z_vals[i] <= -z_enter:
                    curr = -1.0
                    hold = min_hold
                elif curr == -1.0 and z_vals[i] >= z_enter:
                    curr = 1.0
                    hold = min_hold

        side[i] = curr

    return pd.Series(side, index=close.index).shift(1)


def triple_barrier_labeling_with_meta_labeling(df):
    df = df.copy()

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    df["return"] = df["close"].pct_change()
    df["vol"] = compute_volatility(df["return"], VOL_WINDOW)

    # df["side"] = compute_side_from_ema(df["close"], FAST_SPAN, SLOW_SPAN)
    df["side"] = compute_side_momo(df["close"], df["vol"], lookback=10, z_enter=0.75, z_exit=0.05, min_hold=3)

    labels = np.full(len(df), np.nan, dtype=float)
    t1_time = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    t1_idx = np.full(len(df), np.nan, dtype=float)

    closes = df["close"].to_numpy(dtype=float)
    vols = df["vol"].to_numpy(dtype=float)
    sides = df["side"].to_numpy(dtype=float)
    end_times = df["end_time"].to_numpy(dtype="datetime64[ns]")

    n = len(df)

    for i in range(n):
        if np.isnan(vols[i]) or np.isnan(sides[i]) or sides[i] == 0.0:
            continue

        pt = PT_MULT * vols[i]
        sl = SL_MULT * vols[i]

        entry = closes[i]
        t_end = df["end_time"].iloc[i] + pd.Timedelta(minutes=MAX_HOLD_MINUTES)
        end_i = int(np.searchsorted(end_times, np.datetime64(t_end), side="left"))
        end_i = min(max(end_i, i), n - 1)

        label = 0
        hit_idx = end_i

        for j in range(i + 1, end_i + 1):
            ret = (closes[j] - entry) / entry
            ret_side = sides[i] * ret

            if ret_side >= pt:
                label = 1
                hit_idx = j
                break

            if ret_side <= -sl:
                label = -1
                hit_idx = j
                break

        labels[i] = label
        t1_time[i] = end_times[hit_idx]
        t1_idx[i] = hit_idx


    df["label"] = labels
    df["t1"] = pd.to_datetime(t1_time)
    df["t1_idx"] = pd.Series(t1_idx).astype("Int64")

    df["meta_label"] = np.nan
    m = df["label"].notna() & df["side"].notna() & (df["side"] != 0)
    df.loc[m, "meta_label"] = (df.loc[m, "label"] == 1).astype(int)

    return df


if __name__ == "__main__":
    bars = pd.read_parquet(BARS_FILE)
    bars = bars.sort_values("start_time").reset_index(drop=True)

    labeled = triple_barrier_labeling_with_meta_labeling(bars)
    labeled.to_parquet(OUTPUT_FILE, index=False)
