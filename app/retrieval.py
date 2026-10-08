"""
Indexing & ranking algorithms:
  1. TF-IDF  (sklearn TfidfVectorizer)
  2. BM25    (rank_bm25)
  3. Sentence Transformer (all-MiniLM-L6-v2 dense embeddings)
  4. Hybrid RRF — Reciprocal Rank Fusion of all three
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .preprocessing import preprocess, preprocess_to_string
from .dataset import Corpus, Document

_MODEL_DIR = Path(__file__).parent.parent / "models"
_TFIDF_PATH = _MODEL_DIR / "tfidf_index.pkl"
_BM25_PATH  = _MODEL_DIR / "bm25_index.pkl"
_SBERT_PATH = _MODEL_DIR / "sbert_embeddings.npy"
_DOCS_PATH  = _MODEL_DIR / "doc_ids.pkl"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- #
#  Index builder                                                                #
# --------------------------------------------------------------------------- #

class IRIndex:
    """Builds and persists all three IR indices for the corpus."""

    def __init__(self):
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.bm25: Optional[BM25Okapi] = None
        self.sbert_model: Optional[SentenceTransformer] = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.doc_ids: list[int] = []
        self.corpus: Optional[Corpus] = None
        self._ready = False

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self, corpus: Corpus, force_rebuild: bool = False):
        """Build (or load from cache) all indices."""
        self.corpus = corpus
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if not force_rebuild and self._all_cached():
            print("Loading cached indices ...")
            self._load_all()
        else:
            print(f"Building indices for {len(corpus)} documents ...")
            self._build_tfidf(corpus)
            self._build_bm25(corpus)
            self._build_sbert(corpus)
            self._save_all()

        self._ready = True
        print("[OK] All indices ready.")

    def _all_cached(self) -> bool:
        return all(p.exists() for p in [_TFIDF_PATH, _BM25_PATH, _SBERT_PATH, _DOCS_PATH])

    # ------------------------------------------------------------------ #
    #  TF-IDF                                                              #
    # ------------------------------------------------------------------ #

    def _build_tfidf(self, corpus: Corpus):
        print("  -> Building TF-IDF index ...")
        processed = [preprocess_to_string(doc.text) for doc in corpus.documents]
        self.doc_ids = [doc.doc_id for doc in corpus.documents]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50_000,
            sublinear_tf=True,
            min_df=2,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)

    # ------------------------------------------------------------------ #
    #  BM25                                                                #
    # ------------------------------------------------------------------ #

    def _build_bm25(self, corpus: Corpus):
        print("  -> Building BM25 index ...")
        tokenized = [preprocess(doc.text) for doc in corpus.documents]
        self.bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------ #
    #  Sentence Transformer                                                #
    # ------------------------------------------------------------------ #

    def _build_sbert(self, corpus: Corpus):
        print("  -> Building SBERT embeddings (this may take a few minutes) ...")
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        # Use a short snippet (first 256 chars) per doc to save time/memory
        snippets = [doc.text[:256] for doc in corpus.documents]
        self.sbert_embeddings = self.sbert_model.encode(
            snippets,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save_all(self):
        with open(_TFIDF_PATH, "wb") as f:
            pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix, self.doc_ids), f)
        with open(_BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(_SBERT_PATH, self.sbert_embeddings)
        with open(_DOCS_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)

    def _load_all(self):
        with open(_TFIDF_PATH, "rb") as f:
            self.tfidf_vectorizer, self.tfidf_matrix, self.doc_ids = pickle.load(f)
        with open(_BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        self.sbert_embeddings = np.load(_SBERT_PATH)
        with open(_DOCS_PATH, "rb") as f:
            self.doc_ids = pickle.load(f)
        # Load model for query encoding
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


# --------------------------------------------------------------------------- #
#  Search / Ranking                                                             #
# --------------------------------------------------------------------------- #

class SearchEngine:
    """
    Wraps IRIndex and exposes search methods returning ranked doc IDs + scores.
    """

    RRF_K = 60  # standard RRF constant

    def __init__(self, index: IRIndex):
        self.index = index

    # ------------------------------------------------------------------ #
    #  Individual rankers                                                  #
    # ------------------------------------------------------------------ #

    def search_tfidf(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) list sorted by TF-IDF cosine similarity."""
        q_vec = self.index.tfidf_vectorizer.transform([preprocess_to_string(query)])
        scores = cosine_similarity(q_vec, self.index.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.index.doc_ids[i], float(scores[i])) for i in top_idx]

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) list sorted by BM25 score."""
        tokens = preprocess(query)
        scores = self.index.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.index.doc_ids[i], float(scores[i])) for i in top_idx]

    def search_sbert(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) list sorted by SBERT cosine similarity."""
        q_emb = self.index.sbert_model.encode(
            [query], normalize_embeddings=True
        )
        scores = (self.index.sbert_embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.index.doc_ids[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------ #
    #  Reciprocal Rank Fusion                                              #
    # ------------------------------------------------------------------ #

    def search_rrf(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Hybrid Reciprocal Rank Fusion over TF-IDF, BM25, SBERT.
        RRF score = Σ 1 / (k + rank_i)  for each ranker i.
        """
        # Get larger candidate sets for fusion
        n = max(top_k * 5, 100)
        lists = [
            self.search_tfidf(query, top_k=n),
            self.search_bm25(query, top_k=n),
            self.search_sbert(query, top_k=n),
        ]

        rrf_scores: dict[int, float] = {}
        for ranked_list in lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    # ------------------------------------------------------------------ #
    #  Unified search dispatcher                                           #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        method: str = "rrf",
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        method: "tfidf" | "bm25" | "sbert" | "rrf"
        Returns list of (doc_id, score) tuples.
        """
        dispatch = {
            "tfidf": self.search_tfidf,
            "bm25":  self.search_bm25,
            "sbert": self.search_sbert,
            "rrf":   self.search_rrf,
        }
        fn = dispatch.get(method, self.search_rrf)
        return fn(query, top_k=top_k)
