// UBL API client for the FYP dashboard.
// Set window.UBL_API_BASE before importing if the API is not on localhost:8000.

const BASE = window.UBL_API_BASE || "http://127.0.0.1:8000/api";

async function get(path, params = {}) {
  const url = new URL(BASE + path);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  }
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || body.detail || res.statusText);
  return body;
}

export const ubl = {
  health: () => get("/health"),

  // dataset
  info: () => get("/dataset/info"),
  dictionary: () => get("/dataset/dictionary"),

  // market data
  price: (start, end) => get("/price", { start, end }),
  technical: (start, end) => get("/technical", { start, end }),
  fundamentals: () => get("/fundamentals"),
  day: (date) => get(`/day/${encodeURIComponent(date)}`),

  // news
  newsDaily: (start, end) => get("/news/daily", { start, end }),
  newsArticles: (date, limit = 20) => get("/news/articles", { date, limit }),

  // models
  models: () => get("/models"),
  comparison: () => get("/models/comparison"),
  summary: (model) => get("/models/summary", { model }),
  predictions: (model, start, end) => get("/models/predictions", { model, start, end }),
  latest: (model) => get("/models/latest", { model }),
  folds: (model) => get("/models/folds", { model }),
};
