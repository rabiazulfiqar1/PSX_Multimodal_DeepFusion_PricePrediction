import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ubl_dashboard
import ubl_data


@pytest.fixture
def sample_dataset(monkeypatch):
    frame = pd.DataFrame({
        "DATE": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "CLOSE": [10.0, 20.0, 30.0, 40.0],
        "COMPANY": ["Alpha Bank", "Beta Bank", "Gamma", "Alpha Bank"],
        "VOLUME": [100, None, 300, 400],
    })
    monkeypatch.setattr(ubl_data, "_dataset_frame", lambda _: frame)
    return frame


def test_dataset_page_filters_sorts_and_paginates(sample_dataset):
    result = ubl_data.get_dataset_page(
        "fixture",
        page=2,
        page_size=1,
        search="bank",
        search_column="COMPANY",
        date_column="DATE",
        date_from="2025-01-02",
        date_to="2025-01-04",
        numeric_column="CLOSE",
        min_value=20,
        sort_by="CLOSE",
        sort_order="desc",
    )

    assert result["row_count"] == 4
    assert result["filtered_count"] == 2
    assert result["rows"][0]["CLOSE"] == 20.0
    assert result["missing_values"]["VOLUME"] == 1
    assert result["date_coverage"]["DATE"] == {
        "start": "2025-01-01",
        "end": "2025-01-04",
    }


def test_dataset_csv_exports_all_filtered_rows_in_sort_order(sample_dataset):
    content = ubl_data.get_dataset_csv(
        "fixture",
        search="bank",
        search_column="COMPANY",
        date_column="DATE",
        date_from="2025-01-02",
        date_to="2025-01-04",
        numeric_column="CLOSE",
        min_value=20,
        sort_by="CLOSE",
        sort_order="desc",
    )
    exported = pd.read_csv(io.StringIO(content))

    assert exported["CLOSE"].tolist() == [40.0, 20.0]
    assert len(exported) == 2


def test_dataset_page_rejects_reversed_numeric_range(sample_dataset):
    with pytest.raises(ValueError, match="Minimum value"):
        ubl_data.get_dataset_page(
            "fixture",
            numeric_column="CLOSE",
            min_value=40,
            max_value=10,
        )


def test_mari_price_reads_the_mari_source_file_only(monkeypatch):
    source = pd.DataFrame({
        "DATE": pd.to_datetime(["2026-01-01"]),
        "OPEN": [50.0],
        "HIGH": [55.0],
        "LOW": [49.0],
        "CLOSE": [53.0],
        "VOLUME": [1000],
        "CHANGE": [3.0],
    })

    def load_raw(key):
        assert key == "mari_price"
        return source

    monkeypatch.setattr(ubl_data, "load_raw", load_raw)
    monkeypatch.setattr(
        ubl_data,
        "_artifact",
        lambda _: pytest.fail("MARI price must not read UBL artifacts"),
    )

    result = ubl_data.get_price(stock="MARI")

    assert result == [{
        "DATE": "2026-01-01",
        "OPEN": 50.0,
        "HIGH": 55.0,
        "LOW": 49.0,
        "CLOSE": 53.0,
        "VOLUME": 1000,
    }]


def test_mari_dashboard_uses_its_own_walk_forward_and_ablation_artifacts(monkeypatch, tmp_path):
    models = ["All-4 fused"]
    artifacts = {
        "wf_predictions": pd.DataFrame({
            "fold": [1, 1, 2],
            "Model": models * 3,
            "DATE": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            "prev_close": [10.0, 11.0, 12.0],
            "actual": [11.0, 12.0, 13.0],
            "pred": [10.5, 11.5, 12.5],
        }),
        "ablation_raw": pd.DataFrame({
            "Model": models,
            "Split": ["Test"],
            "MAE": [0.5],
            "RMSE": [0.5],
            "MAPE": [4.0],
            "R²": [0.99],
            "Dir. acc.": [100.0],
        }),
        "ablation_table": pd.DataFrame({
            "Model": models,
            "Blocks": ["price + tech + fund + news"],
        }),
        "wf_raw": pd.DataFrame({
            "Model": models,
            "MAE": [0.5],
            "RMSE": [0.5],
            "MAPE": [4.0],
            "Dir. acc.": [100.0],
        }),
        "wf_table": pd.DataFrame({
            "Model": models,
            "Blocks": ["price + tech + fund + news"],
            "Dir. acc.": ["100.0%"],
            "MAE": ["0.50"],
            "RMSE": ["0.50"],
            "MAPE": ["4.00%"],
            "R²": ["0.99"],
        }),
        "wf_folds": pd.DataFrame({
            "Model": models * 2,
            "fold": [1, 2],
            "n": [2, 1],
            "MAE": [0.5, 0.5],
            "Dir. acc.": [100.0, 100.0],
            "Naive_MAE": [1.0, 1.0],
        }),
    }

    monkeypatch.setattr(ubl_dashboard.d, "get_models", lambda stock: models)
    monkeypatch.setattr(
        ubl_dashboard.d,
        "_stock_artifact",
        lambda stock, key: artifacts[key] if stock == "MARI" else pytest.fail(stock),
    )
    monkeypatch.setitem(ubl_dashboard.d.STOCK_ARTIFACT_DIRS, "MARI", tmp_path)

    result = ubl_dashboard.get_dashboard("MARI")

    assert result["data_status"] == "REAL_EXPERIMENT_ARTIFACTS"
    assert result["selected_model"] == "All-4 fused"
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["actual"] == 11.0
    assert result["ablation"][0]["features"] == (
        "Price + Technical indicators + Fundamental features + News features"
    )
    assert len(result["walk_forward"]["iterations"]) == 2
    assert result["walk_forward"]["iterations"][0]["train_start"] is None
    assert result["multi_horizon"][0]["days"] == 1
    assert result["missing_outputs"] == ["article_emb.npy", "wf_rolling_predictions.png"]


def test_hbl_dashboard_filler_is_removed():
    with pytest.raises(ValueError, match="UBL and MARI"):
        ubl_dashboard.get_dashboard("HBL")
