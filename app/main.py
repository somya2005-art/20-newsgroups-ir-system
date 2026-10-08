"""
FastAPI application — main entry point.
Provides:
  GET  /           → search interface (HTML)
  POST /search     → execute query, return results JSON
  GET  /evaluate   → run evaluation, return metrics + PR-curve
  GET  /status     → index build status
"""

import base64
import io
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .dataset import load_corpus
from .retrieval import IRIndex, SearchEngine
from .evaluation import (
    compare_all_rankers,
    get_relevant_docs,
    precision_recall_curve,
)

# --------------------------------------------------------------------------- #
#  Global state (index + engine built once at startup)                          #
# --------------------------------------------------------------------------- #

_index: IRIndex | None = None
_engine: SearchEngine | None = None
_corpus = None
_build_status = {"ready": False, "message": "Initializing ..."}

BASE_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build index in background thread so server starts immediately."""
    global _index, _engine, _corpus, _build_status
    loop = asyncio.get_event_loop()

    def _build():
        global _index, _engine, _corpus, _build_status
        try:
            _build_status["message"] = "Loading corpus ..."
            _corpus = load_corpus()
            _build_status["message"] = "Building / loading IR indices ..."
            _index = IRIndex()
            _index.build(_corpus)
            _engine = SearchEngine(_index)
            _build_status["ready"] = True
            _build_status["message"] = f"Ready - {len(_corpus)} documents indexed."
        except Exception as exc:
            _build_status["message"] = f"Error: {exc}"

    await loop.run_in_executor(None, _build)
    yield


app = FastAPI(title="20 Newsgroups IR System", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# --------------------------------------------------------------------------- #
#  Routes                                                                       #
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"status": _build_status},
    )


@app.get("/status")
async def status():
    return JSONResponse(_build_status)


@app.post("/search")
async def search(
    query: str = Form(...),
    method: str = Form("rrf"),
    top_k: int = Form(10),
    category_filter: str = Form(""),
):
    if not _build_status["ready"]:
        return JSONResponse({"error": "Index not ready yet. " + _build_status["message"]}, status_code=503)

    top_k = min(max(top_k, 1), 50)

    # Search candidates (if category filter is active, fetch from full corpus so we get top_k within category)
    candidate_k = len(_corpus.documents) if category_filter else max(top_k, 20)

    raw_results = {
        "tfidf": _engine.search(query, "tfidf", top_k=candidate_k),
        "bm25":  _engine.search(query, "bm25",  top_k=candidate_k),
        "sbert": _engine.search(query, "sbert", top_k=candidate_k),
        "rrf":   _engine.search(query, "rrf",   top_k=candidate_k),
    }

    results_by_method = {}
    for m, res in raw_results.items():
        filtered = []
        for doc_id, score in res:
            doc = _corpus.documents[doc_id]
            if category_filter and doc.category != category_filter:
                continue
            filtered.append((doc_id, score))
            if len(filtered) == top_k:
                break
        results_by_method[m] = filtered

    # Build the result list for the requested method
    selected = results_by_method[method]
    result_docs = []
    for doc_id, score in selected:
        doc = _corpus.documents[doc_id]
        snippet = doc.text.strip()[:300].replace("\n", " ")
        result_docs.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "category": doc.category,
            "snippet": snippet,
        })

    # Evaluation
    eval_results = []
    pr_curves = {}
    if selected:
        target_category = category_filter if category_filter else _corpus.documents[selected[0][0]].category
        relevant = {
            doc.doc_id for doc in _corpus.documents if doc.category == target_category
        }
        eval_list = compare_all_rankers(results_by_method, relevant, k=top_k)
        eval_results = [
            {
                "method": e.method.upper(),
                "precision_at_k": round(e.precision_at_k, 4),
                "recall_at_k": round(e.recall_at_k, 4),
                "map": round(e.map_score, 4),
                "ndcg": round(e.ndcg_score, 4),
            }
            for e in eval_list
        ]
        # PR curve for selected method
        sel_ids = [d for d, _ in selected]
        prec, rec = precision_recall_curve(sel_ids, relevant, max_k=top_k)
        pr_curves[method] = {"precisions": prec, "recalls": rec}

    return JSONResponse({
        "query": query,
        "method": method,
        "top_k": top_k,
        "results": result_docs,
        "evaluation": eval_results,
        "pr_curves": pr_curves,
    })


@app.get("/categories")
async def categories():
    if not _build_status["ready"]:
        return JSONResponse({"categories": []})
    return JSONResponse({"categories": _corpus.categories})
