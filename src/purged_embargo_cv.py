import numpy as np
import pandas as pd

INPUT_FILE = "../data/dollar_bars_labeled.parquet"
OUTPUT_FILE = "../data/purged_cv_folds.npz"

N_SPLITS = 5
EMBARGO_BARS = 10

def make_contiguous_folds(n, n_splits):
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    bounds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        bounds.append((start, end))
        start = end
    return bounds


def purged_embargo_indices(starts, ends, test_start, test_end, embargo_bars):
    test_idx = np.arange(test_start, test_end, dtype=int)

    embargo_start = test_end
    embargo_end = min(len(starts) - 1, test_end + embargo_bars - 1)

    overlap = (starts < test_end) & (ends >= test_start)
    embargo = (starts >= embargo_start) & (starts <= embargo_end)

    train_mask = ~(overlap | embargo)
    train_idx = np.where(train_mask)[0]

    return train_idx.astype(int), test_idx.astype(int)


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    df = df.dropna(subset=["start_time", "t1_idx"]).sort_values("start_time").reset_index(drop=True)

    n = len(df)

    starts = np.arange(n, dtype=int)
    ends = df["t1_idx"].astype(int).to_numpy()
    ends = np.clip(ends, 0, n - 1)
    ends = np.maximum(ends, starts)

    fold_bounds = make_contiguous_folds(n, N_SPLITS)

    out = {}
    for k, (ts, te) in enumerate(fold_bounds):
        train_idx, test_idx = purged_embargo_indices(starts, ends, ts, te, EMBARGO_BARS)
        out[f'train_{k}'] = train_idx
        out[f'test_{k}'] = test_idx

    np.savez_compressed(OUTPUT_FILE, **out)