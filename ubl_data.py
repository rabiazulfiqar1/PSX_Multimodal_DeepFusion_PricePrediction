"""
UBL data access layer for the FYP dashboard.

Two sources:
  RAW        Kaggle dataset raheenz/ubl-news-price (UBL.csv, fundamentals, news)
  ARTIFACTS  CSV/NPY outputs the notebook writes to /kaggle/working
             (fused dataset, indicators, news features, model results)

All public functions return plain dicts / lists / numbers, so the API layer
can serialise them directly.
"""
from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DATASET = "raheenz/ubl-news-price"
DATA_DIR = Path(os.environ.get("UBL_DATA_DIR", "data/raw"))
ARTIFACT_DIR = Path(os.environ.get("UBL_ARTIFACT_DIR", "data/artifacts"))

# Split boundaries from the notebook (section 2/16)
TRAIN_END, VAL_END = "2025-01-23", "2025-10-10"

MODELS = ["Price", "Technical", "Fundamentals", "News",
          "Price+Tech", "Price+Tech+Fund", "All-4 fused"]
DEFAULT_MODEL = "All-4 fused"
METRICS = ["Dir. acc.", "MAE", "RMSE", "MAPE", "R²"]

RAW_FILES = {
    "price": "UBL.csv",
    "fundamentals": "UBL_fundamental_features.csv",
    "news": "ubl_news_clean.csv",
}

# artifact key -> (file name, read_csv kwargs)
ARTIFACT_FILES = {
    "fused":          ("ubl_fused_dataset.csv", {}),
    "dictionary":     ("ubl_fused_dataset_DICTIONARY.csv", {}),
    "technical":      ("technical_indicators.csv", {}),
    "news_daily":     ("news_features.csv", {}),
    "fundamentals":   ("fundamentals_imputed.csv", {}),
    "articles":       ("article_scored.csv", {}),
    "ablation_raw":   ("ablation_raw_per_seed.csv", {}),
    "ablation_table": ("ablation_table_test.csv", {"index_col": 0}),
    "wf_predictions": ("wf_rolling_predictions.csv", {}),
    "wf_table":       ("wf_rolling_table.csv", {"index_col": 0}),
    "wf_folds":       ("wf_rolling_per_fold.csv", {}),
}

DATE_COLS = {"DATE", "tdate", "fund_asof", "pub_date", "feature_date"}


# ---------------------------------------------------------------- import ----

def download_raw(dest: str | Path = DATA_DIR) -> Path:
    """Download the Kaggle dataset and copy its CSVs into `dest`.
    Requires `pip install kagglehub` and Kaggle credentials
    (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY env vars)."""
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*.csv"):
        shutil.copy(f, dest / f.name)
    return dest


def _path(key: str, raw: bool) -> Path:
    if raw:
        p, hint = DATA_DIR / RAW_FILES[key], "run download_raw()"
    else:
        p = ARTIFACT_DIR / ARTIFACT_FILES[key][0]
        hint = f"copy the notebook outputs into {ARTIFACT_DIR}"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. To fix: {hint}.")
    return p


def _read(path: Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, **kw)
    for c in DATE_COLS & set(df.columns):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_raw(key: str) -> pd.DataFrame:
    """Raw Kaggle file. key: 'price' | 'fundamentals' | 'news'."""
    return _read(_path(key, raw=True))


@lru_cache(maxsize=None)
def _artifact(key: str) -> pd.DataFrame:
    return _read(_path(key, raw=False), **ARTIFACT_FILES[key][1])


# --------------------------------------------------------------- helpers ----

def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN -> null, dates -> YYYY-MM-DD)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))


def _between(df: pd.DataFrame, col: str, start=None, end=None) -> pd.DataFrame:
    if start:
        df = df[df[col] >= pd.Timestamp(start)]
    if end:
        df = df[df[col] <= pd.Timestamp(end)]
    return df


def _check_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {MODELS}")


def _metrics(actual, pred, prev) -> dict:
    a, p, q = (np.asarray(x, dtype=float) for x in (actual, pred, prev))
    err = a - p
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / a)) * 100),
        "R²": float(1 - np.sum(err ** 2) / np.sum((a - a.mean()) ** 2)),
        "Dir. acc.": float(np.mean(np.sign(a - q) == np.sign(p - q)) * 100),
    }


# ------------------------------------------------------------- dataset ------

def dataset_info() -> dict:
    """Headline numbers for the dashboard header."""
    fused = _artifact("fused")
    return {
        "ticker": "UBL",
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "date_min": fused["DATE"].min().strftime("%Y-%m-%d"),
        "date_max": fused["DATE"].max().strftime("%Y-%m-%d"),
        "rows": len(fused),
        "unified_samples": int(fused["in_unified"].sum()),
        "splits": {k: int(v) for k, v in fused["split"].value_counts().items()},
        "split_boundaries": {"train_end": TRAIN_END, "val_end": VAL_END},
        "models": MODELS,
    }


def get_dictionary() -> list[dict]:
    return _records(_artifact("dictionary"))


def get_price(start=None, end=None) -> list[dict]:
    df = _between(_artifact("fused"), "DATE", start, end)
    return _records(df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]])


def get_technical(start=None, end=None) -> list[dict]:
    return _records(_between(_artifact("technical"), "DATE", start, end))


def get_fundamentals() -> list[dict]:
    """One row per quarterly filing (imputed)."""
    return _records(_artifact("fundamentals"))


def get_news_daily(start=None, end=None) -> list[dict]:
    return _records(_between(_artifact("news_daily"), "DATE", start, end))


def get_news_articles(date=None, limit: int = 20) -> list[dict]:
    """Scored articles (text omitted). Optionally only those mapped to `date`."""
    df = _artifact("articles").drop(columns=["text"], errors="ignore")
    if date:
        df = df[df["feature_date"] == pd.Timestamp(date)]
    df = df.sort_values("pub_date", ascending=False).head(int(limit))
    return _records(df)


def get_day(date: str) -> dict:
    """Every fused feature for one trading day."""
    fused = _artifact("fused")
    row = fused[fused["DATE"] == pd.Timestamp(date)]
    if row.empty:
        raise ValueError(f"No trading day {date} in the dataset")
    return _records(row)[0]


# -------------------------------------------------------------- models ------

def get_models() -> list[str]:
    return MODELS


def get_model_comparison() -> dict:
    """Fixed-split test results (mean ± std over seeds) and walk-forward table."""
    raw = _artifact("ablation_raw")
    test = raw[raw["Split"] == "Test"]
    agg = test.groupby("Model")[METRICS].agg(["mean", "std"])

    fixed = []
    for m in agg.index:
        row = {"model": m, "seeds": int(test.loc[test["Model"] == m, "Seed"].nunique())}
        for met in METRICS:
            row[met] = {"mean": float(agg.loc[m, (met, "mean")]),
                        "std": float(agg.loc[m, (met, "std")])}
        fixed.append(row)
    fixed.sort(key=lambda r: -r["Dir. acc."]["mean"])

    return {
        "fixed_split_test": fixed,
        "fixed_split_formatted": _records(
            _artifact("ablation_table").rename_axis("model").reset_index()),
        "walk_forward": _records(
            _artifact("wf_table").rename_axis("model").reset_index()),
    }


def get_walkforward_predictions(model: str = DEFAULT_MODEL, start=None, end=None) -> list[dict]:
    """Out-of-sample rolling walk-forward predictions (backtest, not live)."""
    _check_model(model)
    p = _artifact("wf_predictions")
    p = _between(p[p["Model"] == model], "DATE", start, end).sort_values("DATE")
    p = p.assign(
        actual_change_pct=(p["actual"] / p["prev_close"] - 1) * 100,
        pred_change_pct=(p["pred"] / p["prev_close"] - 1) * 100,
    )
    return _records(p[["fold", "DATE", "prev_close", "actual", "pred",
                       "actual_change_pct", "pred_change_pct"]])


def get_latest_prediction(model: str = DEFAULT_MODEL) -> dict:
    _check_model(model)
    p = _artifact("wf_predictions")
    p = p[p["Model"] == model].sort_values("DATE")
    if p.empty:
        raise ValueError(f"No predictions for '{model}'")
    last = _records(p.tail(1))[0]
    last["note"] = "Out-of-sample walk-forward backtest value, not a live forecast."
    return last


def get_walkforward_folds(model: str = DEFAULT_MODEL) -> list[dict]:
    _check_model(model)
    f = _artifact("wf_folds")
    return _records(f[f["Model"] == model].drop(columns=["Model"]))


def get_walkforward_summary(model: str = DEFAULT_MODEL) -> dict:
    """Headline metrics for one model vs the naive 'tomorrow = today' baseline."""
    _check_model(model)
    wf = _artifact("wf_predictions")
    p = wf[wf["Model"] == model].sort_values("DATE")
    if p.empty:
        raise ValueError(f"No predictions for '{model}'")

    naive = _metrics(p["actual"], p["prev_close"], p["prev_close"])
    naive.pop("Dir. acc.")  # undefined for a no-change forecast

    f = _artifact("wf_folds")
    f = f[f["Model"] == model]
    return {
        "model": model,
        "period": {"start": p["DATE"].min().strftime("%Y-%m-%d"),
                   "end": p["DATE"].max().strftime("%Y-%m-%d")},
        "n_predictions": len(p),
        "metrics": _metrics(p["actual"], p["pred"], p["prev_close"]),
        "naive_baseline": naive,
        "always_up_dir_acc": float(np.mean(p["actual"] > p["prev_close"]) * 100),
        "folds_total": len(f),
        "folds_beating_naive_mae": int((f["MAE"] < f["Naive_MAE"]).sum()),
        "fold_dir_acc_mean": float(f["Dir. acc."].mean()),
        "fold_dir_acc_std": float(f["Dir. acc."].std()),
    }
