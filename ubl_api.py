"""
HTTP API for the UBL FYP dashboard.

Run:   uvicorn ubl_api:app --reload --port 8000
Docs:  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import ubl_dashboard as dash
import ubl_data as d

app = FastAPI(title="UBL FYP API", version="1.0.0")

# Open for local development. Restrict allow_origins before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
async def missing_data(_, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(ValueError)
async def bad_request(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/api/health")
def health():
    return {"status": "ok"}


# dataset
@app.get("/api/dataset/info")
def dataset_info():
    return d.dataset_info()


@app.get("/api/dataset/dictionary")
def dictionary():
    return d.get_dictionary()


# market data
@app.get("/api/price")
def price(start: str | None = None, end: str | None = None):
    return d.get_price(start, end)


@app.get("/api/technical")
def technical(start: str | None = None, end: str | None = None):
    return d.get_technical(start, end)


@app.get("/api/fundamentals")
def fundamentals():
    return d.get_fundamentals()


@app.get("/api/day/{date}")
def day(date: str):
    return d.get_day(date)


# news
@app.get("/api/news/daily")
def news_daily(start: str | None = None, end: str | None = None):
    return d.get_news_daily(start, end)


@app.get("/api/news/articles")
def news_articles(date: str | None = None, limit: int = 20):
    return d.get_news_articles(date, limit)


# models
@app.get("/api/models")
def models():
    return d.get_models()


@app.get("/api/models/comparison")
def comparison():
    return d.get_model_comparison()


@app.get("/api/models/summary")
def summary(model: str = d.DEFAULT_MODEL):
    return d.get_walkforward_summary(model)


@app.get("/api/models/predictions")
def predictions(model: str = d.DEFAULT_MODEL, start: str | None = None, end: str | None = None):
    return d.get_walkforward_predictions(model, start, end)


@app.get("/api/models/latest")
def latest(model: str = d.DEFAULT_MODEL):
    return d.get_latest_prediction(model)


@app.get("/api/models/folds")
def folds(model: str = d.DEFAULT_MODEL):
    return d.get_walkforward_folds(model)

@app.get("/api/dashboard/{stock}")
def dashboard(stock: str):
    return dash.get_dashboard(stock)