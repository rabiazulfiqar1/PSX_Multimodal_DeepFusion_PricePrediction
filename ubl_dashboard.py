"""Build dashboard and project-findings payloads from the available data."""
import json
import math
from pathlib import Path

import pandas as pd

import ubl_data as d

PROJECT_EXPERIMENT_RESULTS = Path(__file__).with_name("project_experiment_results.json")


def _get_mari_dashboard(model: str | None = None) -> dict:
    models = d.get_models("MARI")
    if not models:
        raise FileNotFoundError("No MARI walk-forward predictions are available.")
    model = model or (d.DEFAULT_MODEL if d.DEFAULT_MODEL in models else models[0])
    if model not in models:
        raise ValueError(f"Unknown MARI model '{model}'. Choose from {models}.")
    predictions_frame = d._stock_artifact("MARI", "wf_predictions")
    model_predictions = predictions_frame[predictions_frame["Model"] == model].sort_values("DATE")
    if model_predictions.empty:
        raise ValueError(f"No MARI walk-forward predictions are available for '{model}'.")

    ablation_raw = d._stock_artifact("MARI", "ablation_raw")
    test_metrics = (
        ablation_raw[ablation_raw["Split"].str.casefold() == "test"]
        .groupby("Model")[["MAE", "RMSE", "MAPE", "Dir. acc.", "R²"]]
        .agg(["mean", "std"])
    )
    ablation_table = d._stock_artifact("MARI", "ablation_table").set_index("Model")
    walk_forward = d._stock_artifact("MARI", "wf_raw").set_index("Model")
    walk_forward_table = d._stock_artifact("MARI", "wf_table")
    feature_names = {
        "price": "Price",
        "tech": "Technical indicators",
        "fund": "Fundamental features",
        "news": "News features",
    }

    ablation = []
    for name in models:
        blocks = str(ablation_table.loc[name, "Blocks"])
        def standard_deviation(metric):
            value = test_metrics.loc[name, (metric, "std")]
            return float(value) if pd.notna(value) else None

        features = " + ".join(
            feature_names.get(part.strip(), part.strip())
            for part in blocks.split("+")
        )
        ablation.append({
            "name": name,
            "features": features,
            "fixed_split_mae": float(test_metrics.loc[name, ("MAE", "mean")]),
            "fixed_split_mae_std": standard_deviation("MAE"),
            "fixed_split_rmse": float(test_metrics.loc[name, ("RMSE", "mean")]),
            "fixed_split_rmse_std": standard_deviation("RMSE"),
            "fixed_split_mape": float(test_metrics.loc[name, ("MAPE", "mean")]),
            "fixed_split_mape_std": standard_deviation("MAPE"),
            "fixed_split_directional_accuracy": float(
                test_metrics.loc[name, ("Dir. acc.", "mean")]
            ),
            "fixed_split_directional_accuracy_std": standard_deviation("Dir. acc."),
            "fixed_split_r2": float(test_metrics.loc[name, ("R²", "mean")]),
            "fixed_split_r2_std": standard_deviation("R²"),
            "walk_forward_mae": float(walk_forward.loc[name, "MAE"]),
            "walk_forward_rmse": float(walk_forward.loc[name, "RMSE"]),
            "walk_forward_mape": float(walk_forward.loc[name, "MAPE"]),
            "directional_accuracy": float(walk_forward.loc[name, "Dir. acc."]),
        })

    folds = d._stock_artifact("MARI", "wf_folds")
    folds = folds[folds["Model"] == model].sort_values("fold")
    iterations = []
    sample_index = 0
    for _, fold in folds.iterrows():
        fold_predictions = model_predictions[model_predictions["fold"] == fold["fold"]]
        size = len(fold_predictions)
        if not size:
            continue
        iterations.append({
            "fold": int(fold["fold"]),
            "train_start": None,
            "train_end": None,
            "gap_start": None,
            "validation_start": None,
            "validation_end": None,
            "test_start": sample_index,
            "test_end": sample_index + size - 1,
            "train_start_date": None,
            "train_end_date": None,
            "validation_start_date": None,
            "validation_end_date": None,
            "test_start_date": fold_predictions["DATE"].min().strftime("%Y-%m-%d"),
            "test_end_date": fold_predictions["DATE"].max().strftime("%Y-%m-%d"),
            "train_days": None,
            "validation_days": None,
            "test_days": size,
            "MAE": float(fold["MAE"]),
            "Naive_MAE": float(fold["Naive_MAE"]),
            "Dir. acc.": float(fold["Dir. acc."]),
        })
        sample_index += size

    summary = d.get_walkforward_summary(model, "MARI")
    prediction_rows = [
        {
            "date": row.DATE.strftime("%Y-%m-%d"),
            "actual": float(row.actual),
            "model": float(row.pred),
            "naive": float(row.prev_close),
        }
        for row in model_predictions.itertuples()
    ]
    missing_outputs = [
        filename for filename in ("article_emb.npy", "wf_rolling_predictions.png")
        if not (d.STOCK_ARTIFACT_DIRS["MARI"] / filename).exists()
    ]
    return {
        "stock": "MARI",
        "data_status": "REAL_EXPERIMENT_ARTIFACTS",
        "note": "MARI price, fundamental, cleaned-news, scored-news, ablation, and walk-forward CSVs are available. The listed embedding and prediction PNG files are absent.",
        "predictions": prediction_rows,
        "models": models,
        "selected_model": model,
        "prediction_horizon_days": 1,
        "walk_forward": {
            "total_days": len(model_predictions),
            "iterations": iterations,
            "window_type": None,
            "retrain_every": None,
            "validation_fraction": None,
            "method": "The prediction CSV identifies test folds, dates, actuals, and predictions. It does not contain training or validation window boundaries.",
            "model_summary": d._records(walk_forward_table),
        },
        "ablation": ablation,
        "naive_mae": summary["naive_baseline"]["MAE"],
        "naive_rmse": summary["naive_baseline"]["RMSE"],
        "naive_mape": summary["naive_baseline"]["MAPE"],
        "multi_horizon": [{
            "days": 1,
            "n_predictions": summary["n_predictions"],
            "metrics": summary["metrics"],
        }],
        "missing_outputs": missing_outputs,
    }


def get_dashboard(stock: str = "UBL", model: str | None = None) -> dict:
    stock = stock.upper()
    if stock == "MARI":
        return _get_mari_dashboard(model)
    if stock != "UBL":
        raise ValueError(f"No data for '{stock}' yet. UBL and MARI are available.")

    # Samples that passed the notebook's quality filters, in time order.
    fused = d._artifact("fused")
    unified = fused[fused["in_unified"] == 1].sort_values("tdate").reset_index(drop=True)
    pos = {pd.Timestamp(t): i for i, t in enumerate(unified["tdate"])}
    total = len(unified)

    wf = d._artifact("wf_predictions")
    main = wf[wf["Model"] == d.DEFAULT_MODEL].sort_values("DATE")

    predictions = [
        {
            "date": r.DATE.strftime("%Y-%m-%d"),
            "actual": float(r.actual),
            "model": float(r.pred),
            "naive": float(r.prev_close),
        }
        for r in main.itertuples()
    ]

    # Match the notebook's fixed-size rolling window, 15% inner validation,
    # one-row fit/validation gap, and 21-sample test folds.
    first_test = pos[pd.Timestamp(main["DATE"].min())]
    rolling_window = first_test
    step = 21
    sample_dates = pd.to_datetime(unified["tdate"])
    iterations = []
    fold_metrics = d._artifact("wf_folds")
    fold_metrics = fold_metrics[fold_metrics["Model"] == d.DEFAULT_MODEL].set_index("fold")
    for k in sorted(main["fold"].unique()):
        f = main[main["fold"] == k]
        test_start = pos[pd.Timestamp(f["DATE"].min())]
        test_end = pos[pd.Timestamp(f["DATE"].max())]
        pool_start = max(0, test_start - rolling_window)
        pool_size = test_start - pool_start
        validation_size = int(math.ceil(pool_size * 0.15))
        fit_size = pool_size - validation_size - 1
        gap = pool_start + fit_size
        fold = fold_metrics.loc[k]
        iterations.append({
            "fold": int(k),
            "train_start": pool_start,
            "train_end": pool_start + fit_size - 1,
            "gap_start": gap,
            "validation_start": gap + 1,
            "validation_end": test_start - 1,
            "test_start": test_start,
            "test_end": test_end,
            "train_start_date": sample_dates.iloc[pool_start].strftime("%Y-%m-%d"),
            "train_end_date": sample_dates.iloc[pool_start + fit_size - 1].strftime("%Y-%m-%d"),
            "validation_start_date": sample_dates.iloc[gap + 1].strftime("%Y-%m-%d"),
            "validation_end_date": sample_dates.iloc[test_start - 1].strftime("%Y-%m-%d"),
            "test_start_date": sample_dates.iloc[test_start].strftime("%Y-%m-%d"),
            "test_end_date": sample_dates.iloc[test_end].strftime("%Y-%m-%d"),
            "train_days": fit_size,
            "validation_days": validation_size,
            "test_days": int(fold["n"]),
            "MAE": float(fold["MAE"]),
            "Naive_MAE": float(fold["Naive_MAE"]),
            "Dir. acc.": float(fold["Dir. acc."]),
        })

    # Per-model comparison uses the checked-in held-out and rolling artifacts.
    raw = d._artifact("ablation_raw")
    fixed = (
        raw[raw["Split"] == "Test"]
        .groupby("Model")[d.METRICS]
        .agg(["mean", "std"])
    )
    blocks = d._artifact("ablation_table")["Blocks"].to_dict()
    feature_names = {
        "price": "Price",
        "tech": "Technical indicators",
        "fund": "Fundamental features",
        "news": "News features",
    }

    ablation = []
    for m in d.MODELS:
        p = wf[wf["Model"] == m]
        mm = d._metrics(p["actual"], p["pred"], p["prev_close"])
        features = " + ".join(feature_names.get(part.strip(), part.strip()) for part in blocks[m].split("+"))
        def fixed_mean(metric):
            return float(fixed.loc[m, (metric, "mean")])

        def fixed_std(metric):
            value = fixed.loc[m, (metric, "std")]
            return float(value) if pd.notna(value) else None

        ablation.append({
            "name": m,
            "features": features,
            "fixed_split_mae": fixed_mean("MAE"),
            "fixed_split_mae_std": fixed_std("MAE"),
            "fixed_split_rmse": fixed_mean("RMSE"),
            "fixed_split_rmse_std": fixed_std("RMSE"),
            "fixed_split_mape": fixed_mean("MAPE"),
            "fixed_split_mape_std": fixed_std("MAPE"),
            "fixed_split_directional_accuracy": fixed_mean("Dir. acc."),
            "fixed_split_directional_accuracy_std": fixed_std("Dir. acc."),
            "fixed_split_r2": fixed_mean("R²"),
            "fixed_split_r2_std": fixed_std("R²"),
            "walk_forward_mae": mm["MAE"],
            "walk_forward_mape": mm["MAPE"],
            "walk_forward_rmse": mm["RMSE"],
            "directional_accuracy": mm["Dir. acc."],
        })

    naive = d._metrics(main["actual"], main["prev_close"], main["prev_close"])

    return {
        "stock": "UBL",
        "data_status": "REAL_EXPERIMENT_ARTIFACTS",
        "predictions": predictions,
        "walk_forward": {
            "total_days": total,
            "iterations": iterations,
            "window_type": "fixed-size rolling",
            "retrain_every": step,
            "validation_fraction": 0.15,
            "method": "Each 21-sample test fold follows a fixed-size history window. The newest 15% of that window is used for early-stopping validation, with a one-sample gap before validation.",
        },
        "ablation": ablation,
        "naive_mae": naive["MAE"],
        "naive_rmse": naive["RMSE"],
        "naive_mape": naive["MAPE"],
        # Multi-day horizons are not present in the checked-in fused-model artifacts.
        "multi_horizon": [],
    }


def get_project_findings() -> dict:
    """Combine UBL project results with the available MARI experiment outputs."""
    project_results = json.loads(PROJECT_EXPERIMENT_RESULTS.read_text(encoding="utf-8"))
    data_sources = {
        "UBL": [
            ("Price", "ubl-price", "DATE"),
            ("Fundamentals", "ubl-fundamentals", "DATE"),
            ("Cleaned news", "ubl-news", "pub_date"),
            ("Technical indicators", "ubl-technical", "DATE"),
        ],
        "MARI": [
            ("Price", "mari-price", "DATE"),
            ("Fundamentals", "mari-fundamentals", "Unnamed: 1"),
            ("Cleaned news", "mari-news", "pub_date"),
            ("Scored news", "mari-articles-scored", "feature_date"),
            ("Ablation test summary", "mari-ablation-test", None),
            ("Walk-forward predictions", "mari-walk-forward-predictions", "DATE"),
        ],
    }
    coverage = []
    for stock, sources in data_sources.items():
        datasets = []
        for label, dataset_id, date_column in sources:
            page = d.get_dataset_page(dataset_id, page_size=1)
            date_range = page["date_coverage"].get(date_column, {}) if date_column else {}
            datasets.append({
                "name": label,
                "rows": page["row_count"],
                "date_start": date_range.get("start"),
                "date_end": date_range.get("end"),
            })
        coverage.append({
            "stock": stock,
            "datasets": datasets,
            "model_results_available": True,
            "model_results_note": (
                "Saved fused-model and modality experiment results are available."
                if stock == "UBL"
                else "MARI walk-forward predictions and test/walk-forward evaluation results are available. No multi-day forecast series is present."
            ),
        })

    dashboard = get_dashboard("UBL")
    models = dashboard["ablation"]
    best = min(models, key=lambda row: row["walk_forward_mape"])
    configs_by_name = {row["name"]: row for row in models}
    walk_forward_models = []
    for name in d.MODELS:
        folds = d.get_walkforward_folds(name)
        if not folds:
            raise ValueError(f"No walk-forward folds found for '{name}'.")
        row = configs_by_name[name]
        walk_forward_models.append({
            "name": name,
            "features": row["features"],
            "folds_total": len(folds),
            "folds_beating_naive_mae": sum(fold["MAE"] < fold["Naive_MAE"] for fold in folds),
            "mean_fold_mae": float(sum(fold["MAE"] for fold in folds) / len(folds)),
            "walk_forward_mae": row["walk_forward_mae"],
            "walk_forward_rmse": row["walk_forward_rmse"],
            "walk_forward_mape": row["walk_forward_mape"],
        })

    mari = get_dashboard("MARI")
    mari_models = mari["ablation"]
    mari_test_best = min(mari_models, key=lambda row: row["fixed_split_mape"])
    mari_walk_forward_best = min(mari_models, key=lambda row: row["walk_forward_mape"])
    mari_walk_forward_models = []
    for row in mari_models:
        folds = d.get_walkforward_folds(row["name"], "MARI")
        mari_walk_forward_models.append({
            "name": row["name"],
            "features": row["features"],
            "fixed_split_mae": row["fixed_split_mae"],
            "fixed_split_rmse": row["fixed_split_rmse"],
            "fixed_split_mape": row["fixed_split_mape"],
            "walk_forward_mae": row["walk_forward_mae"],
            "walk_forward_rmse": row["walk_forward_rmse"],
            "walk_forward_mape": row["walk_forward_mape"],
            "folds_total": len(folds),
            "folds_beating_naive_mae": sum(
                fold["MAE"] < fold["Naive_MAE"] for fold in folds
            ),
        })

    return {
        "data_coverage": coverage,
        "horizon_evaluation": project_results["evaluation"],
        "horizon_sources": project_results["horizon_sources"],
        "horizons": project_results["horizons"],
        "final_ubl": {
            "best_trained_model": best,
            "fused_model": configs_by_name["All-4 fused"],
            "naive_baseline": {
                "MAE": dashboard["naive_mae"],
                "RMSE": dashboard["naive_rmse"],
                "MAPE": dashboard["naive_mape"],
            },
            "walk_forward_models": walk_forward_models,
            "fold_count": len(dashboard["walk_forward"]["iterations"]),
        },
        "final_mari": {
            "best_test_model": mari_test_best,
            "best_walk_forward_model": mari_walk_forward_best,
            "walk_forward_models": mari_walk_forward_models,
            "fold_count": len(mari["walk_forward"]["iterations"]),
            "selected_model": mari["selected_model"],
            "single_step": mari["multi_horizon"][0],
            "supported_horizons": [1],
            "missing_outputs": mari["missing_outputs"],
        },
        "limitations": [
            "Horizon-specific branch results are UBL modality-only models and use a direct chronological holdout; they are not results for the final fused model.",
            "The main branch fused-model artifacts contain one-day walk-forward predictions only.",
            "MARI prediction outputs support one-step walk-forward evaluation only; no 5-, 10-, or 21-day prediction series is available.",
            "MARI walk-forward files do not describe the training or validation windows, so those spans are not shown.",
            f"MARI optional output files are absent: {', '.join(mari['missing_outputs']) or 'none'}.",
            "Feature importance and causal modality contributions have not been computed.",
        ],
    }