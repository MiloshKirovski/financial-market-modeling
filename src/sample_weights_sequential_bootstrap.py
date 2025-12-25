import numpy as np
import pandas as pd

INPUT_FILE = "../data/dollar_bars_weighted.parquet"
OUTPUT_FILE = "../data/sequential_bootstrap_indices.npy"


def get_indicator_matrix(events, time_index):
    indicator = pd.DataFrame(
        0,
        index=time_index,
        columns=events.index,
        dtype=np.int8,
    )

    for i, row in events.iterrows():
        indicator.loc[row['start_time'] : row['t1'], i] = 1

    return indicator


def sequential_boostrap(indicator, sample_length=None):
    if sample_length is None:
        sample_length = indicator.shape[1]

    selected = []
    avg_uniqueness = pd.Series(0.0, index=indicator.index)

    while len(selected) < sample_length:
        if len(selected) == 0:
            probs = np.ones(len(avg_uniqueness))
        else:
            concurrency = indicator[selected].sum(axis=1)
            concurrency = concurrency.replace(0, np.nan)

            for i in avg_uniqueness.index:
                overlap = indicator[i]
                avg_uniqueness[i] = (overlap / concurrency).mean()

            probs = avg_uniqueness.fillna(0).values

        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / probs_sum

        choice = np.random.choice(avg_uniqueness.index, p=probs)
        selected.append(choice)

    return np.array(selected, dtype=int)


if __name__ == '__main__':
    df = pd.read_parquet(INPUT_FILE)

    df = df.dropna(subset=['start_time', 't1']).sort_values('start_time').reset_index(drop=True)

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['t1'] = pd.to_datetime(df['t1'])

    time_index = pd.Index(sorted(set(df['start_time']).union(set(df['t1']))))

    indicator = get_indicator_matrix(df, time_index)

    boot_indices = sequential_boostrap(
        indicator,
        sample_length=len(df)
    )

    np.save(OUTPUT_FILE, boot_indices)