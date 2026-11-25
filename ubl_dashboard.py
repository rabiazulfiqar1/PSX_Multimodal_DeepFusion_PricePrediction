"""
Builds the single payload that index.html loads from /api/dashboard/{stock}.
Uses the same artifacts as ubl_data.py, reshaped for the page.
"""
import pandas as pd

import ubl_data as d


def get_dashboard(stock: str = "UBL") -> dict:
    if stock.upper() != "UBL":
        raise ValueError(f"No data for '{stock}' yet. Only UBL is available.")

    # Samples that passed the notebook's quality filters, in time order.
    fused = d._artifact("fused")
    unified = fused[fused["in_unified"] == 1].reset_index(drop=True)
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

    # Walk-forward windows, as positions in the sample sequence.
    # The notebook's rolling window size equals the number of samples before
    # the first test day, so the training window ends just before each test fold.
    first_test = pos[pd.Timestamp(main["DATE"].min())]
    iterations = []
    for k in sorted(main["fold"].unique()):
        f = main[main["fold"] == k]
        s = pos[pd.Timestamp(f["DATE"].min())]
        e = pos[pd.Timestamp(f["DATE"].max())]
        iterations.append({
            "train_start": max(0, s - first_test),
            "train_end": s - 1,
            "test_start": s,
            "test_end": e,
        })

    # Per-model comparison: fixed-split MAPE from the seed runs,
    # walk-forward metrics from the rolling predictions.
    raw = d._artifact("ablation_raw")
    fixed_mape = raw[raw["Split"] == "Test"].groupby("Model")["MAPE"].mean()

    ablation = []
    for m in d.MODELS:
        p = wf[wf["Model"] == m]
        mm = d._metrics(p["actual"], p["pred"], p["prev_close"])
        ablation.append({
            "name": m,
            "fixed_split_mape": float(fixed_mape[m]),
            "walk_forward_mape": mm["MAPE"],
            "walk_forward_rmse": mm["RMSE"],
            "directional_accuracy": mm["Dir. acc."],
        })

    naive = d._metrics(main["actual"], main["prev_close"], main["prev_close"])

    return {
        "stock": "UBL",
        "predictions": predictions,
        "walk_forward": {"total_days": total, "iterations": iterations},
        "ablation": ablation,
        "naive_rmse": naive["RMSE"],
        "naive_mape": naive["MAPE"],
        # Multi-day horizons are not computed yet; the page handles an empty list.
        "multi_horizon": [],
    }