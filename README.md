# PSX Multimodal Deep-Fusion Price Prediction

This project predicts the next trading-day return of United Bank Limited (UBL) on the Pakistan Stock Exchange (PSX). It combines price history, technical indicators, quarterly fundamental information, and financial-news features in a multimodal machine-learning pipeline.

## Project structure

- [`multimodal-mlp-window.ipynb`](./multimodal-mlp-window.ipynb): final early-fusion multimodal MLP with a 30-trading-day price window and modality ablations.
- [`eda.py`](./eda.py): exploratory data analysis and data-quality checks for price, fundamental, and news data.
- [`requirements.txt`](./requirements.txt): minimal Python dependencies currently recorded for the analysis scripts.
- [`CONFIG.json`](./CONFIG.json): project configuration placeholder.

## Final dataset

The exported Kaggle dataset is version 3 and contains the following files:

| File | Description |
| --- | --- |
| `UBL.csv` | Source UBL OHLCV price data. |
| `UBL_fundamental_features.csv` | Source quarterly UBL fundamental features. |
| `technical_indicators.csv` | Calculated TA-Lib indicators keyed by trading date. |
| `news_features.csv` | Daily news sentiment, recency, and article-count features keyed by trading date. |
| `ubl_fused_dataset.csv` | Final fused price, technical, fundamental, and news dataset with split and target columns. |
| `ubl_fused_dataset_DICTIONARY.csv` | Column descriptions and modality metadata for the fused dataset. |
| `ubl_news_clean.csv` | Cleaned and prepared UBL news articles used as the news-data source. |

The final notebook exports the fused dataset and its data dictionary, along with the technical and daily news feature tables. The complete final dataset is also published through the Kaggle Data Explorer.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Final integrated multimodal MLP and the project’s current consolidated implementation. |
| `feature/price-pipeline` | Price-only experiments, including an MLP, walk-forward evaluation, and multi-horizon prediction. |
| `feature/news-pipeline` | News cleaning and news-only MLP experiments, including walk-forward and multi-horizon evaluation. |
| `feat/fundamental-pipeline` | Fundamental-feature preparation and fundamental-only MLP experiments, including walk-forward and multi-horizon analysis. |

The feature branches provide modality-specific baselines and development history; `main` combines the modalities into the final model.

## Final multimodal MLP window

The final notebook performs the following steps:

1. Cleans UBL OHLCV data and restricts it to the configured date range.
2. Creates log returns, price-change, moving-average, and volatility features.
3. Computes technical indicators with TA-Lib, including EMA, MACD, Parabolic SAR, Bollinger Bands, and stochastic oscillators.
4. Imputes selected fundamental values and merges the latest available filing backward onto each trading day to avoid look-ahead leakage.
5. Scores financial news with FinBERT, aggregates sentiment and recency features by trading day, and creates daily 768-dimensional news embeddings.
6. Builds a unified sample containing:
   - a 30-day price-feature window;
   - same-day technical features;
   - same-day fundamentals;
   - daily news and FinBERT features.
7. Predicts the next-day log return and evaluates chronological train, validation, and test splits without shuffling.
8. Exports intermediate feature files, the fused dataset, and a data dictionary to the configured working directory.

The checked-in notebook reports 1,176 cleaned trading days, 1,127 unified samples, and chronological splits of 774 training, 176 validation, and 177 test samples. It also supports modality ablation experiments so the contribution of price, technical, fundamental, and news inputs can be compared on identical rows.

## Data and execution

The notebook is configured for the Kaggle input paths shown in its configuration cell. Provide the source and cleaned files listed in the final dataset, or update those paths before running:

- `UBL.csv`
- `UBL_fundamental_features.csv`
- `ubl_news_clean.csv`

Running the notebook exports `technical_indicators.csv`, `news_features.csv`, `ubl_fused_dataset.csv`, and `ubl_fused_dataset_DICTIONARY.csv` to the configured working directory. These exported tables make up the reusable final dataset.

Run the notebook in Jupyter or Kaggle after installing the required modeling and feature-extraction packages, including TensorFlow, scikit-learn, TA-Lib, PyTorch, Transformers, pandas, and NumPy. The notebook caches FinBERT article scores and embeddings so subsequent runs can reuse them.

## Notes

- The target is the next trading day’s log return, not the next closing price directly.
- The chronological split is based on the date being predicted.
- Fundamental and news joins are causal: only information available on or before the relevant trading day is used.
- Input datasets and generated feature caches are not committed to this repository.
