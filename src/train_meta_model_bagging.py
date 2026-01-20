import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


DATA_FILE = "../data/dollar_bars_weighted.parquet"
CV_FILE = "../data/purged_cv_folds.npz"
BOOT_FILE = "../data/sequential_bootstrap_indices.npy"
OUTPUT_OOF_FILE = "../data/meta_model_oof_predictions.parquet"
OUTPUT_FOLD_FILE = "../data/meta_model_cv_metrics.parquet"

N_SPLITS = 5
N_ESTIMATORS = 200
MAX_DEPTH = 3
MIN_SAMPLE_LEAFS = 50
RANDOM_STATE = 42
PROBA_THRESHOLD = 0.5


def build_features(df):
    dt = pd.to_datetime(df["start_time"])
    minute_of_day = (dt.dt.hour * 60 + dt.dt.minute).astype(float)

    X = pd.DataFrame({
        "ret": df["return"].astype(float),
        "vol": df["vol"].astype(float),
        "range": ((df["high"] - df["low"]) / df["close"]).astype(float),
        "log_size": np.log1p(df["size"].astype(float)),
        "log_dollar_value": np.log1p(df["dollar_volume"].astype(float)),
        "minute_of_day": minute_of_day,
    }, index=df.index)

    y = df["meta_label"].astype(float)
    w = df["sample_weight"].astype(float)

    return X, y, w


def get_fold_arrays(cv_npz, k):
    train_idx = cv_npz[f"train_{k}"].astype(int)
    test_idx = cv_npz[f"test_{k}"].astype(int)
    return train_idx, test_idx


def make_bootstrap_sample(train_idx, boot_indices, n_draws, rng):
    train_set = set(train_idx.tolist())

    start = int(rng.integers(0, max(1, len(boot_indices))))
    ordered = np.concatenate([boot_indices[start:], boot_indices[:start]])

    picked = []
    for ix in ordered:
        if int(ix) in train_set:
            picked.append(int(ix))
            if len(picked) >= n_draws:
                break

    if len(picked) < n_draws:
        extra = rng.choice(train_idx, size=n_draws - len(picked), replace=True).astype(int).tolist()
        picked.extend(extra)

    return np.array(picked, dtype=int)


def bagged_predict_proba(X_train, y_train, w_train, X_test, train_idx, boot_indices, rng):
    proba_sum = np.zeros(len(X_test), dtype=float)

    for _ in range(N_ESTIMATORS):
        sample_idx = make_bootstrap_sample(train_idx, boot_indices, n_draws=len(train_idx), rng=rng)

        tree = DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLE_LEAFS,
            random_state=int(rng.integers(0, 2**31 -1)),
        )

        tree.fit(
            X_train.loc[sample_idx].values,
            y_train.loc[sample_idx].values.astype(int),
            sample_weight=w_train.loc[sample_idx].values,
        )

        proba_sum += tree.predict_proba(X_test.values)[:, 1]

    return proba_sum / float(N_ESTIMATORS)


if __name__ == '__main__':
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("start_time").reset_index(drop=True)

    cv = np.load(CV_FILE)
    boot = np.load(BOOT_FILE).astype(int)

    max_idx_in_folds = max(int(cv[f"test_{k}"].max()) for k in range(N_SPLITS))
    if max_idx_in_folds > len(df):
        raise RuntimeError(f"Fold indices go up to {max_idx_in_folds}")

    X, y, w = build_features(df)

    valid = (
        X.notna().all(axis=1)
        & y.notna()
        & df["side"].notna()
        & (df["side"].astype(float) != 0)
    )
    valid_idx = np.where(valid.values)[0]

    oof = pd.DataFrame({
        "start_time": pd.to_datetime(df["start_time"]),
        "t1": pd.to_datetime(df["t1"]),
        "t1_idx": df["t1_idx"].astype(float),
        "side": df["side"].astype(float),
        "label": df["label"].astype(float),
        "meta_label": df["meta_label"].astype(float),
        "sample_weight": w.astype(float),
        "proba": np.nan,
        "pred": np.nan,
        "fold": np.nan,
    })

    fold_rows = []
    rng = np.random.default_rng(RANDOM_STATE)

    for k in range(N_SPLITS):
        train_idx, test_idx = get_fold_arrays(cv, k)

        train_idx = np.intersect1d(train_idx, valid_idx, assume_unique=False)
        test_idx = np.intersect1d(test_idx, valid_idx, assume_unique=False)

        if len(train_idx) == 0 or len(test_idx) == 0:
            fold_rows.append({
                "fold": k,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "accuracy": np.nan,
                "auc": np.nan,
                "selection_rate": np.nan,
            })
            continue

        X_train, y_train, w_train = X.loc[train_idx], y.loc[train_idx], w.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx].astype(int)

        proba = bagged_predict_proba(X_train, y_train.astype(int), w_train, X_test, train_idx, boot, rng)
        pred = (proba >= PROBA_THRESHOLD).astype(int)

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)
        sel = float(pred.mean())

        oof.loc[test_idx, "proba"] = proba
        oof.loc[test_idx, "pred"] = pred
        oof.loc[test_idx, "fold"] = k

        fold_rows.append({
            "fold": k,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
            "auc": float(auc),
            "selection_rate": float(sel),
        })

    fold_metrics = pd.DataFrame(fold_rows)
    oof.to_parquet(OUTPUT_OOF_FILE, index=False)
    fold_metrics.to_parquet(OUTPUT_FOLD_FILE, index=False)

    print(fold_metrics)