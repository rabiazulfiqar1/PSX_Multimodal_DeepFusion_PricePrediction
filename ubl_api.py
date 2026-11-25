"""
HTTP API for the PSX Forecast Lab dashboard.

Run:   uvicorn ubl_api:app --reload --port 8000
Docs:  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import ubl_dashboard as dash
import ubl_data as d

app = FastAPI(title="PSX Forecast Lab API", version="1.0.0")

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


@app.get("/api/datasets")
def datasets():
    return d.list_datasets()


@app.get("/api/datasets/{dataset_id}")
def dataset_page(
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
):
    return d.get_dataset_page(
        dataset_id, page, page_size, search, search_column, date_column,
        date_from, date_to, numeric_column, min_value, max_value, sort_by, sort_order,
    )


@app.get("/api/datasets/{dataset_id}/download")
def dataset_csv(
    dataset_id: str,
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
):
    content = d.get_dataset_csv(
        dataset_id, search=search, search_column=search_column,
        date_column=date_column, date_from=date_from, date_to=date_to,
        numeric_column=numeric_column, min_value=min_value, max_value=max_value,
        sort_by=sort_by, sort_order=sort_order,
    )
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}-filtered.csv"'},
    )


# market data
@app.get("/api/price")
def price(start: str | None = None, end: str | None = None, stock: str = "UBL"):
    return d.get_price(start, end, stock)


@app.get("/api/technical")
def technical(start: str | None = None, end: str | None = None, stock: str = "UBL"):
    return d.get_technical(start, end, stock)


@app.get("/api/fundamentals")
def fundamentals(stock: str = "UBL"):
    return d.get_fundamentals(stock)


@app.get("/api/day/{date}")
def day(date: str):
    return d.get_day(date)


# news
@app.get("/api/news/daily")
def news_daily(start: str | None = None, end: str | None = None, stock: str = "UBL"):
    return d.get_news_daily(start, end, stock)


@app.get("/api/news/articles")
def news_articles(date: str | None = None, limit: int = 20, stock: str = "UBL"):
    return d.get_news_articles(date, limit, stock)


# models
@app.get("/api/models")
def models(stock: str = "UBL"):
    return d.get_models(stock)


@app.get("/api/models/comparison")
def comparison(stock: str = "UBL"):
    return d.get_model_comparison(stock)


@app.get("/api/models/summary")
def summary(model: str = d.DEFAULT_MODEL, stock: str = "UBL"):
    return d.get_walkforward_summary(model, stock)


@app.get("/api/models/predictions")
def predictions(
    model: str = d.DEFAULT_MODEL,
    start: str | None = None,
    end: str | None = None,
    stock: str = "UBL",
):
    return d.get_walkforward_predictions(model, start, end, stock)


@app.get("/api/models/latest")
def latest(model: str = d.DEFAULT_MODEL, stock: str = "UBL"):
    return d.get_latest_prediction(model, stock)


@app.get("/api/models/folds")
def folds(model: str = d.DEFAULT_MODEL, stock: str = "UBL"):
    return d.get_walkforward_folds(model, stock)

@app.get("/api/dashboard/{stock}")
def dashboard(stock: str, model: str | None = None):
    return dash.get_dashboard(stock, model)


@app.get("/api/findings")
def project_findings():
    return dash.get_project_findings()
