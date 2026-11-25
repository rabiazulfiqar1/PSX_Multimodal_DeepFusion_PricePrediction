# Stock-specific artifacts

Keep generated model and evaluation CSVs in a folder named for their ticker:

- `UBL/` contains the UBL fused dataset, feature outputs, ablation results, and walk-forward artifacts.
- `MARI/` contains MARI scored-news, ablation, and walk-forward artifacts.

The dashboard reads only each stock's own folder. Do not copy one ticker's results into another ticker's directory. MARI embeddings and the walk-forward PNG are optional; they are not currently present or used by the dashboard.
