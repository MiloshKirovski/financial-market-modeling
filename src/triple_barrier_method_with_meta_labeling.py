import pandas as pd
import numpy as np

BARS_FILE = "../data/dollar_bars.parquet"
OUTPUT_FILE = "../data/dollar_bars_labeled.parquet"

VOL_WINDOW = 50
PT_MULT = 1.0
SL_MULT = 1.0
MAX_HOLD_BARS = 20


def compute_volatility(returns, window):
    return returns.ewm(span=window, adjust=False).std()


def triple_barrier_labeling_with_meta_labeling(df):
    df = df.copy()

    df['return'] = df['close'].pct_change()
    df['vol'] = compute_volatility(df['return'], VOL_WINDOW)
    df['side'] = np.sign(df['return'].shift(1))

    labels = []
    t1_list = []

    closes = df['close'].values
    vols = df['vol'].values

    for i in range(len(df)):
        if np.isnan(vols[i]):
            labels.append(np.nan)
            t1_list.append(np.nan)
            continue

        pt = PT_MULT * vols[i]
        sl = SL_MULT * vols[i]

        entry_price = closes[i]
        label = 0
        t1 = min(i + MAX_HOLD_BARS, len(df) -1)

        for j in range(i + 1, t1 + 1):
            ret = (closes[j] - entry_price) / entry_price

            if ret >= pt:
                label = 1
                t1 = j
                break
            if ret <= -sl:
                label = -1
                t1 = j
                break

        labels.append(label)
        t1_list.append(df.index[t1])

    df['label'] = labels
    df['t1'] = t1_list

    df['meta_label'] = 0
    mask = (df['side'].notna()) & (df['label'] != 0)
    df.loc[mask, "meta_label"] = (df.loc[mask, "side"] == df.loc[mask, "label"]).astype(int)

    return df


if __name__ == "__main__":
    bars = pd.read_parquet(BARS_FILE)
    bars = bars.sort_values("start_time").reset_index(drop=True)

    labeled = triple_barrier_labeling_with_meta_labeling(bars)
    labeled.to_parquet(OUTPUT_FILE, index=False)
