from datetime import date, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import math
import random

app = FastAPI(title="PSX Forecast Lab Filler API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STOCKS = {
    "UBL": {"name": "United Bank Limited", "start": 375.0},
    "HBL": {"name": "Habib Bank Limited", "start": 165.0},
}
START = date(2025, 10, 13)
END = date(2026, 6, 30)

def trading_dates():
    out, d = [], START
    while d <= END:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out

def make_predictions(stock):
    cfg = STOCKS[stock]
    rng = random.Random(42 if stock == "UBL" else 84)
    price = cfg["start"]
    rows = []
    for i, d in enumerate(trading_dates()):
        prev = price
        price *= 1 + (rng.random() - 0.49) * 0.025
        naive = prev
        model = prev * (1 + (rng.random() - 0.5) * 0.009)
        rows.append({"date": d.isoformat(), "actual": round(price, 2),
                     "naive": round(naive, 2), "model": round(model, 2)})
    return rows

def build_dashboard(stock):
    predictions = make_predictions(stock)
    horizons = [
        {"days": 1, "label": "1 day", "model_mape": 1.9, "naive_mape": 2.0, "forecast_price": predictions[-1]["actual"] * 1.002},
        {"days": 5, "label": "5 days", "model_mape": 2.8, "naive_mape": 3.0, "forecast_price": predictions[-1]["actual"] * 1.006},
        {"days": 10, "label": "10 days", "model_mape": 4.1, "naive_mape": 4.4, "forecast_price": predictions[-1]["actual"] * 1.009},
        {"days": 15, "label": "15 days", "model_mape": 5.3, "naive_mape": 5.7, "forecast_price": predictions[-1]["actual"] * 1.012},
        {"days": 20, "label": "20 days", "model_mape": 6.4, "naive_mape": 6.9, "forecast_price": predictions[-1]["actual"] * 1.015},
    ]
    total = 1127
    iterations = []
    for i in range(9):
        train_start = i * 21
        train_end = train_start + 950
        test_start = train_end
        test_end = min(test_start + 21, total)
        iterations.append({"train_start": train_start, "train_end": train_end,
                           "test_start": test_start, "test_end": test_end})
    names = ["Price only", "Technical only", "Fundamentals only", "News only",
             "Price + technical", "Price + fundamentals", "Price + news",
             "All features"]
    ablation = []
    for i, name in enumerate(names):
        ablation.append({
            "name": name,
            "fixed_split_mape": round(1.8 + i * 0.13, 2),
            "walk_forward_mape": round(1.9 + i * 0.18, 2),
            "walk_forward_rmse": round(8.9 + i * 0.42, 2),
            "directional_accuracy": round(53.0 - i * 1.1, 1)
        })
    return {
        "stock": stock, "company": STOCKS[stock]["name"],
        "data_status": "ILLUSTRATIVE_FILLER",
        "evaluation_start": START.isoformat(), "evaluation_end": END.isoformat(),
        "predictions": predictions, "multi_horizon": horizons,
        "walk_forward": {"total_days": total, "train_days": 950, "test_days": 21, "iterations": iterations},
        "naive_rmse": 9.2, "naive_mape": 2.0, "ablation": ablation,
        "notice": "All numeric values are synthetic filler data. Do not present as experimental findings."
    }

@app.get("/")
def root():
    return {"name": "PSX Forecast Lab API", "status": "running", "warning": "Filler data only"}

@app.get("/api/stocks")
def stocks():
    return [{"symbol": k, "name": v["name"], "enabled": True} for k, v in STOCKS.items()]

@app.get("/api/dashboard/{stock}")
def dashboard(stock: str):
    stock = stock.upper()
    if stock not in STOCKS:
        raise HTTPException(status_code=404, detail="Stock not enabled in this starter backend")
    return build_dashboard(stock)
