"""
Dataset loader for the 20 Newsgroups corpus.
Downloads and caches the dataset, returns structured documents.
"""

import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.datasets import fetch_20newsgroups

_CACHE_PATH = Path(__file__).parent.parent / "data" / "newsgroups_cache.pkl"

# All 20 categories
ALL_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


@dataclass
class Document:
    doc_id: int
    text: str           # raw text
    category: str       # newsgroup category label
    target: int         # numeric label


@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.documents)


def load_corpus(use_cache: bool = True) -> Corpus:
    """
    Load the 20 Newsgroups dataset.
    Uses a local pickle cache to avoid re-downloading.
    """
    if use_cache and _CACHE_PATH.exists():
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("Downloading 20 Newsgroups dataset ...")
    dataset = fetch_20newsgroups(
        subset="all",
        remove=(),          # keep everything; we strip in preprocessing
        categories=None,    # all 20 categories
    )

    corpus = Corpus(categories=list(dataset.target_names))
    for idx, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        corpus.documents.append(
            Document(
                doc_id=idx,
                text=text,
                category=dataset.target_names[target],
                target=int(target),
            )
        )

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(corpus, f)

    print(f"Loaded {len(corpus)} documents across {len(corpus.categories)} categories.")
    return corpus
