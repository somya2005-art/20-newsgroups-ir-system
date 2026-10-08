"""
Text preprocessing pipeline for 20 Newsgroups dataset.
Steps: Header Removal → Tokenization → Stop-word Removal → Stemming/Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only once)
def download_nltk_data():
    resources = ["punkt", "stopwords", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_data()

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))

# Extra stopwords common in newsgroup headers / email threads
_EXTRA_STOPS = {
    "subject", "from", "reply", "wrote", "organization", "lines",
    "article", "writes", "nntp", "posting", "host", "http", "www",
    "com", "edu", "gov", "net", "org", "re", "wa", "ha", "said",
}
_stop_words.update(_EXTRA_STOPS)

_HEADER_PATTERN = re.compile(
    r"^(From|Subject|Newsgroups|Date|Organization|Lines|Message-ID|"
    r"References|NNTP-Posting-Host|Mime-Version|Content-Type|"
    r"Content-Transfer-Encoding|X-Newsreader|X-Mailer|"
    r"Summary|Keywords|Cc|Reply-To):.*$",
    re.MULTILINE | re.IGNORECASE,
)
_QUOTE_PATTERN = re.compile(r"^>.*$", re.MULTILINE)
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def remove_headers(text: str) -> str:
    """Remove email-style headers, quoted lines, emails, URLs."""
    text = _HEADER_PATTERN.sub("", text)
    text = _QUOTE_PATTERN.sub("", text)
    text = _EMAIL_PATTERN.sub("", text)
    text = _URL_PATTERN.sub("", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize the text."""
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stop words and short tokens."""
    return [t for t in tokens if t not in _stop_words and len(t) > 2]


def stem(tokens: list[str]) -> list[str]:
    """Apply Porter stemming."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str, apply_stemming: bool = True) -> list[str]:
    """
    Full preprocessing pipeline.
    Returns a list of processed tokens.
    """
    text = remove_headers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if apply_stemming:
        tokens = stem(tokens)
    return tokens


def preprocess_to_string(text: str, apply_stemming: bool = True) -> str:
    """Returns preprocessed text as a single joined string (for BM25 / TF-IDF)."""
    return " ".join(preprocess(text, apply_stemming))
