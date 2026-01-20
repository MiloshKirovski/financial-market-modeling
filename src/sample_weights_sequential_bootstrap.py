import numpy as np
import pandas as pd

INPUT_FILE = "../data/dollar_bars_labeled.parquet"
OUTPUT_FILE = "../data/sequential_bootstrap_indices.npy"


class Fenwick:
    def __init__(self, values):
        self.n = len(values)
        self.bit = np.zeros(self.n + 1, dtype=float)
        for i, v in enumerate(values, start=1):
            self.bit[i] += v
            j = i + (i & -i)
            if j <= self.n:
                self.bit[j] += self.bit[i]

    def add(self, idx0, delta):
        i = idx0 + 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum(self):
        s = 0.0
        i = self.n
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def find_prefix(self, target):
        idx = 0
        bitmask = 1 << (self.n.bit_length() - 1)
        while bitmask:
            t = idx + bitmask
            if t <= self.n and self.bit[t] < target:
                idx = t
                target -= self.bit[t]
            bitmask >>= 1
        return idx


def build_events_per_bar(n, ends):
    events_per_bar = [[] for _ in range(n)]
    for i in range(n):
        e = ends[i]
        for t in range(i, e + 1):
            events_per_bar[t].append(i)
    return events_per_bar


def sequential_bootstrap_interval(ends, sample_length=None, seed=42):
    rng = np.random.default_rng(seed)

    n = len(ends)
    if sample_length is None:
        sample_length = n

    lengths = (ends - np.arange(n) + 1).astype(int)
    c = np.zeros(n, dtype=int)

    sum_u = lengths.astype(float)
    w = sum_u / lengths
    fw = Fenwick(w)

    events_per_bar = build_events_per_bar(n, ends)

    selected = np.empty(sample_length, dtype=int)

    for k in range(sample_length):
        total = fw.sum()
        if total <= 0:
            selected[k] = int(rng.integers(0, n))
        else:
            u = rng.random() * total
            idx = fw.find_prefix(u)
            selected[k] = idx

        i = selected[k]
        e = ends[i]

        for t in range(i, e + 1):
            old = 1.0 / (c[t] + 1.0)
            c[t] += 1
            new = 1.0 / (c[t] + 1.0)
            delta = new - old

            for j in events_per_bar[t]:
                old_w = w[j]
                sum_u[j] += delta
                new_w = sum_u[j] / lengths[j]
                w[j] = new_w
                fw.add(j, new_w - old_w)

    return selected


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    df = df.dropna(subset=["t1_idx"]).sort_values("start_time").reset_index(drop=True)

    ends = df["t1_idx"].astype(int).to_numpy()
    n = len(ends)

    ends = np.clip(ends, 0, n - 1)
    ends = np.maximum(ends, np.arange(n))

    boot_indices = sequential_bootstrap_interval(ends, sample_length=n, seed=42)
    np.save(OUTPUT_FILE, boot_indices)
