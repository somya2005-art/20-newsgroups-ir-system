# 20 Newsgroups IR System
> **Project 17** — Information Retrieval System for Newsgroup Postings

## 🎯 Goal
Build an IR system for 18,000 newsgroup posts across 20 discussion topics with multiple ranking algorithms and a hybrid fusion model.

## 📁 Project Structure
```
newsgroups_ir/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── dataset.py        # 20 Newsgroups loader + caching
│   ├── preprocessing.py  # Header removal, tokenization, stopwords, stemming
│   ├── retrieval.py      # TF-IDF, BM25, SBERT, RRF indexing & search
│   └── evaluation.py     # Precision@K, Recall, MAP, nDCG, PR-curve
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html        # Search interface
├── models/               # Cached indices (auto-generated)
├── data/                 # Cached dataset (auto-generated)
├── requirements.txt
└── run.py
```

## ⚙️ Setup & Run

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
python run.py
```
Then open **http://localhost:8000** in your browser.

> **First run**: The app will auto-download the 20 Newsgroups dataset (~15 MB) and build all three indices. SBERT embeddings take ~5–10 minutes the first time. Subsequent runs load from cache instantly.

---

## 🔧 Standardized Tasks

| Task | Implementation |
|------|---------------|
| **Header Removal** | Regex strips `From:`, `Subject:`, quotes, emails, URLs |
| **Tokenization** | NLTK `word_tokenize` + lowercase |
| **Stop-word Removal** | NLTK stopwords + custom newsgroup-specific extras |
| **Stemming** | Porter Stemmer |
| **TF-IDF Indexing** | `sklearn.TfidfVectorizer` (50k features, sublinear TF) |
| **BM25** | `rank_bm25.BM25Okapi` |
| **Sentence Transformer** | `all-MiniLM-L6-v2` via `sentence-transformers` |
| **Hybrid RRF** | Reciprocal Rank Fusion over all 3 rankers (k=60) |

## 📊 Evaluation Metrics
- **Precision@K** — fraction of top-K results that are relevant
- **Recall@K** — fraction of relevant docs found in top-K
- **MAP** — Mean Average Precision
- **nDCG** — Normalized Discounted Cumulative Gain

*Ground truth*: Documents from the same newsgroup category as the top result.

## 🖥️ Expected Output
- ✅ Search Interface (web UI)
- ✅ Performance Comparison Table (all 4 methods side-by-side)
- ✅ Precision-Recall Graph (interactive Chart.js)
