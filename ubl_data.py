"""Data access layer for the UBL and MARI FYP dashboard."""
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
ARTIFACT_ROOT = Path(os.environ.get("PSX_ARTIFACT_ROOT", "data/artifacts"))
STOCK_ARTIFACT_DIRS = {
    "UBL": Path(os.environ.get("UBL_ARTIFACT_DIR", str(ARTIFACT_ROOT / "UBL"))),
    "MARI": Path(os.environ.get("MARI_ARTIFACT_DIR", str(ARTIFACT_ROOT / "MARI"))),
}
ARTIFACT_DIR = STOCK_ARTIFACT_DIRS["UBL"]

# Split boundaries from the notebook (section 2/16)
TRAIN_END, VAL_END = "2025-01-23", "2025-10-10"

MODELS = ["Price", "Technical", "Fundamentals", "News",
          "Price+Tech", "Price+Tech+Fund", "All-4 fused"]
DEFAULT_MODEL = "All-4 fused"
METRICS = ["Dir. acc.", "MAE", "RMSE", "MAPE", "R²"]

RAW_FILES = {
    "price": "UBL.csv",
    "mari_price": "MARI.csv",
    "fundamentals": "UBL_fundamental_features.csv",
    "mari_fundamentals": "MARI_fundamental_features.csv",
    "news": "ubl_news_clean.csv",
    "mari_news": "MARI_news_clean.csv",
}

# UBL artifact key -> (file name, read_csv kwargs)
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

MARI_ARTIFACT_FILES = {
    "articles":       ("article_scored.csv", {}),
    "ablation_raw":   ("ablation_raw_per_seed.csv", {}),
    "ablation_table": ("ablation_table_test.csv", {}),
    "wf_predictions": ("wf_rolling_predictions.csv", {}),
    "wf_raw":         ("wf_rolling_raw.csv", {}),
    "wf_table":       ("wf_rolling_table.csv", {}),
    "wf_folds":       ("wf_rolling_per_fold.csv", {}),
}

DATE_COLS = {"DATE", "tdate", "fund_asof", "pub_date", "pub_ts", "feature_date"}


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


@lru_cache(maxsize=None)
def _stock_artifact(stock: str, key: str) -> pd.DataFrame:
    stock = stock.upper()
    if stock == "UBL":
        return _artifact(key)
    if stock != "MARI":
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    if key not in MARI_ARTIFACT_FILES:
        raise FileNotFoundError(f"No MARI artifact is registered for '{key}'.")
    filename, kwargs = MARI_ARTIFACT_FILES[key]
    path = STOCK_ARTIFACT_DIRS[stock] / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Add the MARI artifact file to continue.")
    return _read(path, **kwargs)


def _read(path: Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, **kw)
    for c in DATE_COLS & set(df.columns):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_raw(key: str) -> pd.DataFrame:
    """Load a checked-in source CSV by its key in RAW_FILES."""
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


def get_price(start=None, end=None, stock: str = "UBL") -> list[dict]:
    stock = stock.upper()
    if stock == "UBL":
        df = _artifact("fused")
    elif stock == "MARI":
        df = load_raw("mari_price")
    else:
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    df = _between(df, "DATE", start, end)
    return _records(df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]])


def get_technical(start=None, end=None, stock: str = "UBL") -> list[dict]:
    if stock.upper() == "MARI":
        return []
    if stock.upper() != "UBL":
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    return _records(_between(_artifact("technical"), "DATE", start, end))


def get_fundamentals(stock: str = "UBL") -> list[dict]:
    """Return the source or imputed fundamentals available for a stock."""
    stock = stock.upper()
    if stock == "UBL":
        df = _artifact("fundamentals")
    elif stock == "MARI":
        df = load_raw("mari_fundamentals")
    else:
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    return _records(df)


def get_news_daily(start=None, end=None, stock: str = "UBL") -> list[dict]:
    if stock.upper() == "UBL":
        df = _artifact("news_daily")
    elif stock.upper() == "MARI":
        articles = _stock_artifact("MARI", "articles")
        df = (
            articles.groupby("feature_date", as_index=False)
            .agg(sent_mean=("sentiment", "mean"), n_articles=("article_id", "size"))
            .rename(columns={"feature_date": "DATE"})
        )
        df["has_news"] = 1
    else:
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    return _records(_between(df, "DATE", start, end))


def get_news_articles(date=None, limit: int = 20, stock: str = "UBL") -> list[dict]:
    """Scored articles (text omitted). Optionally only those mapped to `date`."""
    if stock.upper() not in {"UBL", "MARI"}:
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
    df = _stock_artifact(stock, "articles").drop(columns=["text"], errors="ignore")
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


# Dataset Explorer reads only from this fixed, local allowlist.
DATASET_FILES = {
    "ubl-price": ("UBL price history", "raw", "price"),
    "mari-price": ("MARI price history", "raw", "mari_price"),
    "ubl-fundamentals": ("UBL fundamentals (imputed)", "artifact", "fundamentals"),
    "mari-fundamentals": ("MARI fundamentals (source)", "raw", "mari_fundamentals"),
    "ubl-news": ("UBL cleaned news articles", "raw", "news"),
    "mari-news": ("MARI cleaned news articles", "raw", "mari_news"),
    "ubl-fused": ("UBL fused dataset", "artifact", "fused"),
    "ubl-technical": ("UBL technical indicators", "artifact", "technical"),
    "ubl-daily-news": ("UBL daily news features", "artifact", "news_daily"),
    "ubl-articles-scored": ("UBL scored news articles", "artifact", "articles"),
    "ubl-ablation-seeds": ("UBL ablation results by seed", "artifact", "ablation_raw"),
    "ubl-ablation-test": ("UBL ablation test summary", "artifact", "ablation_table"),
    "ubl-walk-forward-predictions": ("UBL walk-forward predictions", "artifact", "wf_predictions"),
    "ubl-walk-forward-folds": ("UBL walk-forward fold metrics", "artifact", "wf_folds"),
    "ubl-walk-forward-summary": ("UBL walk-forward model summary", "artifact", "wf_table"),
    "mari-articles-scored": ("MARI scored news articles", "mari_artifact", "articles"),
    "mari-ablation-seeds": ("MARI ablation results by seed", "mari_artifact", "ablation_raw"),
    "mari-ablation-test": ("MARI ablation test summary", "mari_artifact", "ablation_table"),
    "mari-walk-forward-predictions": ("MARI walk-forward predictions", "mari_artifact", "wf_predictions"),
    "mari-walk-forward-raw": ("MARI walk-forward raw metrics", "mari_artifact", "wf_raw"),
    "mari-walk-forward-folds": ("MARI walk-forward fold metrics", "mari_artifact", "wf_folds"),
    "mari-walk-forward-summary": ("MARI walk-forward summary", "mari_artifact", "wf_table"),
}


@lru_cache(maxsize=None)
def _dataset_frame(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{dataset_id}'.")
    _, kind, key = DATASET_FILES[dataset_id]
    if kind == "raw":
        return load_raw(key)
    if kind == "mari_artifact":
        return _stock_artifact("MARI", key)
    df = _artifact(key)
    if dataset_id in {"ubl-ablation-test", "ubl-walk-forward-summary"}:
        return df.rename_axis("Model").reset_index()
    return df


def _dataset_date_values(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if column.lower().startswith("unnamed"):
        trailing_date = series.astype(str).str.extract(
            r"([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\s*$", expand=False
        )
        return pd.to_datetime(trailing_date, format="%b %d, %Y", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _dataset_column_types(df: pd.DataFrame) -> tuple[list[dict], list[str], dict]:
    columns, date_columns, date_coverage = [], [], {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            kind = "number"
        elif pd.api.types.is_datetime64_any_dtype(series) or (
            "date" in str(column).lower() or str(column).lower().startswith("unnamed")
        ):
            dates = _dataset_date_values(series, str(column))
            if dates.notna().any():
                kind = "date"
                date_columns.append(str(column))
                date_coverage[str(column)] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                }
            else:
                kind = "text"
        else:
            kind = "text"
        columns.append({"name": str(column), "type": kind})
    return columns, date_columns, date_coverage


def list_datasets() -> list[dict]:
    available = []
    for dataset_id, metadata in DATASET_FILES.items():
        label, kind = metadata[:2]
        try:
            df = _dataset_frame(dataset_id)
        except FileNotFoundError:
            continue
        available.append({
            "id": dataset_id,
            "label": label,
            "rows": len(df),
            "columns": len(df.columns),
            "source": str(
                DATA_DIR if kind == "raw"
                else STOCK_ARTIFACT_DIRS["MARI"] if kind == "mari_artifact"
                else ARTIFACT_DIR
            ),
            "ticker": dataset_id.partition("-")[0].upper(),
        })
    return available


def _filtered_dataset(
    dataset_id: str,
    search: str | None = None,
    search_column: str | None = None,
    date_column: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    numeric_column: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[str], dict]:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("Minimum value must not exceed maximum value.")
    source = _dataset_frame(dataset_id)
    df = source
    columns, date_columns, date_coverage = _dataset_column_types(source)
    names = set(source.columns)

    if search:
        if search_column:
            if search_column not in names:
                raise ValueError(f"Unknown search column '{search_column}'.")
            search_columns = [search_column]
        else:
            search_columns = [c for c in source.columns if not pd.api.types.is_numeric_dtype(source[c])]
        mask = pd.Series(False, index=df.index)
        for column in search_columns:
            mask |= df[column].astype(str).str.contains(search, case=False, regex=False, na=False)
        df = df[mask]

    if date_from or date_to:
        if not date_column:
            date_column = date_columns[0] if date_columns else None
        if date_column not in date_columns:
            raise ValueError("Choose a valid date column before applying a date range.")
        values = _dataset_date_values(df[date_column], date_column)
        if date_from:
            values_from = pd.to_datetime(date_from, errors="coerce")
            if pd.isna(values_from):
                raise ValueError("Invalid start date.")
            df = df[values >= values_from]
            values = values.loc[df.index]
        if date_to:
            values_to = pd.to_datetime(date_to, errors="coerce")
            if pd.isna(values_to):
                raise ValueError("Invalid end date.")
            df = df[values <= values_to]

    if min_value is not None or max_value is not None:
        if numeric_column not in names or not pd.api.types.is_numeric_dtype(source[numeric_column]):
            raise ValueError("Choose a numeric column before applying a numeric range.")
        if min_value is not None:
            df = df[df[numeric_column] >= min_value]
        if max_value is not None:
            df = df[df[numeric_column] <= max_value]

    return source, df, columns, date_columns, date_coverage


def get_dataset_page(
    dataset_id: str,
    page: int = 1,
    page_size: int = 25,
    search: str | None = None,
    search_column: str | None = None,
    date_column: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    numeric_column: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> dict:
    if page < 1:
        raise ValueError("Page must be 1 or greater.")
    if page_size < 1 or page_size > 100:
        raise ValueError("Page size must be between 1 and 100.")
    if sort_order not in {"asc", "desc"}:
        raise ValueError("Sort order must be 'asc' or 'desc'.")
    source, filtered, columns, date_columns, date_coverage = _filtered_dataset(
        dataset_id, search, search_column, date_column, date_from, date_to,
        numeric_column, min_value, max_value,
    )
    if sort_by:
        if sort_by not in source.columns:
            raise ValueError(f"Unknown sort column '{sort_by}'.")
        filtered = filtered.sort_values(sort_by, ascending=sort_order == "asc", na_position="last", kind="mergesort")
    filtered_count = len(filtered)
    start = (page - 1) * page_size
    rows = _records(filtered.iloc[start:start + page_size])
    return {
        "dataset": dataset_id,
        "row_count": len(source),
        "column_count": len(source.columns),
        "filtered_count": filtered_count,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, int(np.ceil(filtered_count / page_size))),
        "columns": columns,
        "date_columns": date_columns,
        "missing_values": {str(c): int(source[c].isna().sum()) for c in source.columns},
        "date_coverage": date_coverage,
        "rows": rows,
    }


def get_dataset_csv(dataset_id: str, **filters) -> str:
    filters = dict(filters)
    sort_by = filters.pop("sort_by", None)
    sort_order = filters.pop("sort_order", "asc")
    if sort_order not in {"asc", "desc"}:
        raise ValueError("Sort order must be 'asc' or 'desc'.")
    _, filtered, _, _, _ = _filtered_dataset(dataset_id, **filters)
    if sort_by:
        if sort_by not in filtered.columns:
            raise ValueError(f"Unknown sort column '{sort_by}'.")
        filtered = filtered.sort_values(sort_by, ascending=sort_order == "asc", na_position="last", kind="mergesort")
    return filtered.to_csv(index=False)


# -------------------------------------------------------------- models ------

def get_models(stock: str = "UBL") -> list[str]:
    stock = stock.upper()
    if stock == "UBL":
        return MODELS
    if stock == "MARI":
        return _stock_artifact(stock, "wf_predictions")["Model"].drop_duplicates().tolist()
    raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")


def get_model_comparison(stock: str = "UBL") -> dict:
    """Fixed-split test results (mean ± std over seeds) and walk-forward table."""
    stock = stock.upper()
    if stock == "UBL":
        raw = _artifact("ablation_raw")
        ablation_table = _artifact("ablation_table").rename_axis("model").reset_index()
        walk_forward_table = _artifact("wf_table").rename_axis("model").reset_index()
    elif stock == "MARI":
        raw = _stock_artifact(stock, "ablation_raw")
        ablation_table = _stock_artifact(stock, "ablation_table")
        walk_forward_table = _stock_artifact(stock, "wf_table")
    else:
        raise ValueError(f"Unknown stock '{stock}'. Choose UBL or MARI.")
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
        "fixed_split_formatted": _records(ablation_table),
        "walk_forward": _records(walk_forward_table),
    }


def get_walkforward_predictions(
    model: str = DEFAULT_MODEL,
    start=None,
    end=None,
    stock: str = "UBL",
) -> list[dict]:
    """Out-of-sample rolling walk-forward predictions (backtest, not live)."""
    stock = stock.upper()
    if model not in get_models(stock):
        raise ValueError(f"Unknown {stock} model '{model}'.")
    p = _stock_artifact(stock, "wf_predictions")
    p = _between(p[p["Model"] == model], "DATE", start, end).sort_values("DATE")
    p = p.assign(
        actual_change_pct=(p["actual"] / p["prev_close"] - 1) * 100,
        pred_change_pct=(p["pred"] / p["prev_close"] - 1) * 100,
    )
    return _records(p[["fold", "DATE", "prev_close", "actual", "pred",
                       "actual_change_pct", "pred_change_pct"]])


def get_latest_prediction(model: str = DEFAULT_MODEL, stock: str = "UBL") -> dict:
    stock = stock.upper()
    if model not in get_models(stock):
        raise ValueError(f"Unknown {stock} model '{model}'.")
    p = _stock_artifact(stock, "wf_predictions")
    p = p[p["Model"] == model].sort_values("DATE")
    if p.empty:
        raise ValueError(f"No predictions for '{model}'")
    last = _records(p.tail(1))[0]
    last["note"] = "Out-of-sample walk-forward backtest value, not a live forecast."
    return last


def get_walkforward_folds(model: str = DEFAULT_MODEL, stock: str = "UBL") -> list[dict]:
    stock = stock.upper()
    if model not in get_models(stock):
        raise ValueError(f"Unknown {stock} model '{model}'.")
    f = _stock_artifact(stock, "wf_folds")
    return _records(f[f["Model"] == model].drop(columns=["Model"]))


def get_walkforward_summary(model: str = DEFAULT_MODEL, stock: str = "UBL") -> dict:
    """Headline metrics for one model vs the naive 'tomorrow = today' baseline."""
    stock = stock.upper()
    if model not in get_models(stock):
        raise ValueError(f"Unknown {stock} model '{model}'.")
    wf = _stock_artifact(stock, "wf_predictions")
    p = wf[wf["Model"] == model].sort_values("DATE")
    if p.empty:
        raise ValueError(f"No predictions for '{model}'")

    naive = _metrics(p["actual"], p["prev_close"], p["prev_close"])
    naive.pop("Dir. acc.")  # undefined for a no-change forecast

    f = _stock_artifact(stock, "wf_folds")
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
