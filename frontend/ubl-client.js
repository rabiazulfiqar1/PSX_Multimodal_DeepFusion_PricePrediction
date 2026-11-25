// API client for the FYP dashboard. UBL_API_BASE may be the API root or its /api path.

const API_ROOT = (window.UBL_API_BASE || "http://127.0.0.1:8000").replace(/\/api\/?$/, "").replace(/\/$/, "");
const BASE = API_ROOT + "/api";

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
  datasets: () => get("/datasets"),
  datasetPage: (id, params = {}) => get(`/datasets/${encodeURIComponent(id)}`, params),
  datasetCsvUrl: (id, params = {}) => {
    const url = new URL(BASE + `/datasets/${encodeURIComponent(id)}/download`);
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    }
    return url.toString();
  },

  // market data
  price: (start, end, stock = "UBL") => get("/price", { start, end, stock }),
  technical: (start, end, stock = "UBL") => get("/technical", { start, end, stock }),
  fundamentals: (stock = "UBL") => get("/fundamentals", { stock }),
  day: (date) => get(`/day/${encodeURIComponent(date)}`),

  // news
  newsDaily: (start, end, stock = "UBL") => get("/news/daily", { start, end, stock }),
  newsArticles: (date, limit = 20, stock = "UBL") => get("/news/articles", { date, limit, stock }),

  // models
  findings: () => get("/findings"),
  models: (stock = "UBL") => get("/models", { stock }),
  comparison: (stock = "UBL") => get("/models/comparison", { stock }),
  summary: (model, stock = "UBL") => get("/models/summary", { model, stock }),
  predictions: (model, start, end, stock = "UBL") => get("/models/predictions", { model, start, end, stock }),
  latest: (model, stock = "UBL") => get("/models/latest", { model, stock }),
  folds: (model, stock = "UBL") => get("/models/folds", { model, stock }),
};
