import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ubl_dashboard
import ubl_data


def test_project_findings_report_ubl_and_mari_results_separately(monkeypatch):
    row_counts = {
        "ubl-price": 100,
        "ubl-fundamentals": 10,
        "ubl-news": 20,
        "ubl-technical": 100,
        "mari-price": 200,
        "mari-fundamentals": 12,
        "mari-news": 30,
        "mari-articles-scored": 30,
        "mari-ablation-test": 8,
        "mari-walk-forward-predictions": 160,
    }

    def dataset_page(dataset_id, page_size):
        assert page_size == 1
        return {
            "row_count": row_counts[dataset_id],
            "date_coverage": {},
        }

    def dashboard(stock, model=None):
        if stock == "UBL":
            return {
                "ablation": [
                    {
                        "name": "Technical",
                        "features": "Technical indicators",
                        "walk_forward_mae": 8.0,
                        "walk_forward_rmse": 11.0,
                        "walk_forward_mape": 1.99,
                    },
                    {
                        "name": "All-4 fused",
                        "features": "All modalities",
                        "walk_forward_mae": 10.0,
                        "walk_forward_rmse": 13.0,
                        "walk_forward_mape": 2.50,
                    },
                ],
                "naive_mae": 7.8,
                "naive_rmse": 11.1,
                "naive_mape": 1.94,
                "walk_forward": {"iterations": [{}, {}]},
            }
        return {
            "ablation": [
                {
                    "name": "Technical",
                    "features": "Technical indicators",
                    "fixed_split_mae": 8.0,
                    "fixed_split_rmse": 12.0,
                    "fixed_split_mape": 1.27,
                    "walk_forward_mae": 8.7,
                    "walk_forward_rmse": 12.8,
                    "walk_forward_mape": 1.31,
                    "directional_accuracy": 49.0,
                },
                {
                    "name": "Fundamentals",
                    "features": "Fundamental features",
                    "fixed_split_mae": 9.6,
                    "fixed_split_rmse": 13.1,
                    "fixed_split_mape": 1.44,
                    "walk_forward_mae": 8.5,
                    "walk_forward_rmse": 12.2,
                    "walk_forward_mape": 1.28,
                    "directional_accuracy": 51.4,
                },
            ],
            "naive_mae": 7.8,
            "naive_rmse": 11.7,
            "naive_mape": 1.18,
            "walk_forward": {"iterations": [{}, {}]},
            "multi_horizon": [{"days": 1, "n_predictions": 160, "metrics": {
                "MAE": 8.5, "RMSE": 12.2, "MAPE": 1.28,
            }}],
            "selected_model": model or "All-4 fused",
            "missing_outputs": ["article_emb.npy"],
        }

    def folds(model, stock="UBL"):
        if stock == "UBL":
            return (
                [{"MAE": 7.0, "Naive_MAE": 8.0}, {"MAE": 9.0, "Naive_MAE": 8.0}]
                if model == "Technical"
                else [{"MAE": 10.0, "Naive_MAE": 8.0}]
            )
        return [{"MAE": 7.0, "Naive_MAE": 8.0}, {"MAE": 9.0, "Naive_MAE": 8.0}]

    monkeypatch.setattr(ubl_dashboard.d, "get_dataset_page", dataset_page)
    monkeypatch.setattr(ubl_dashboard.d, "MODELS", ["Technical", "All-4 fused"])
    monkeypatch.setattr(ubl_dashboard, "get_dashboard", dashboard)
    monkeypatch.setattr(ubl_dashboard.d, "get_walkforward_folds", folds)

    report = ubl_dashboard.get_project_findings()

    assert [row["stock"] for row in report["data_coverage"]] == ["UBL", "MARI"]
    assert all(row["model_results_available"] for row in report["data_coverage"])
    assert report["final_ubl"]["best_trained_model"]["name"] == "Technical"
    assert report["final_ubl"]["walk_forward_models"][0]["folds_beating_naive_mae"] == 1
    assert report["final_mari"]["best_test_model"]["name"] == "Technical"
    assert report["final_mari"]["best_walk_forward_model"]["name"] == "Fundamentals"
    assert report["final_mari"]["supported_horizons"] == [1]
    assert report["final_mari"]["single_step"]["n_predictions"] == 160
    assert "article_emb.npy" in report["final_mari"]["missing_outputs"]


def test_dataset_catalog_includes_mari_sources_and_artifacts(monkeypatch):
    monkeypatch.setattr(
        ubl_data,
        "_dataset_frame",
        lambda _: pd.DataFrame({"DATE": ["2026-01-01"], "CLOSE": [1.0]}),
    )

    datasets = ubl_data.list_datasets()

    assert {
        "mari-price",
        "mari-fundamentals",
        "mari-news",
        "mari-articles-scored",
        "mari-ablation-seeds",
        "mari-ablation-test",
        "mari-walk-forward-predictions",
        "mari-walk-forward-folds",
        "mari-walk-forward-raw",
        "mari-walk-forward-summary",
    }.issubset({dataset["id"] for dataset in datasets})
    mari = next(dataset for dataset in datasets if dataset["id"] == "mari-price")
    assert mari["ticker"] == "MARI"
    assert mari["source"] == str(ubl_data.DATA_DIR)
