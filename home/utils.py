import os
import json
from collections import Counter
from typing import List, Dict, Any, Optional

import pdfplumber
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import difflib
import numpy as np
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------
# Ensure required NLTK resources (idempotent)
# ----------------------------------------------------------------------
RESOURCES = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

for key, path in RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(key)

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def get_model() -> SentenceTransformer:
    """Lazy-load sentence transformer model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def tokenize(text: str) -> List[str]:
    """Tokenize with basic normalization, stopword removal, and lemmatization."""
    if not text:
        return []
    tokens = nltk.word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w.isalnum() and w not in stop_words
    ]


# ----------------------------------------------------------------------
# PDF Processing
# ----------------------------------------------------------------------

def process_pdf(file_path: str) -> Optional[Dict[str, Any]]:
    """Extract text and metadata from PDF, preprocess, generate embeddings.

    Returns a dict with: author, title, abstract, summary, keywords, text, embedding
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_text = [(page.extract_text() or "") for page in pdf.pages]
            text = " ".join(pages_text)
            metadata = pdf.metadata or {}
            author = (metadata.get("Author") or "Unknown").strip() or "Unknown"
            title = (metadata.get("Title") or "Untitled").strip() or "Untitled"

            # Detect abstract safely (first page that contains the substring 'abstract')
            abstract = next(
                (
                    txt[:800]
                    for txt in pages_text
                    if txt and "abstract" in txt.lower()
                ),
                "",
            )

        summary = text[:1500] if text else ""

        # Preprocess
        cleaned_tokens = tokenize(text)
        # Add author and title tokens so embeddings capture these fields too
        enriched_tokens = cleaned_tokens + ["title", *tokenize(title), "author", *tokenize(author)]
        processed_text = " ".join(enriched_tokens)

        # Embeddings (L2-normalized for cosine via dot product)
        model = get_model()
        embedding = model.encode(processed_text, normalize_embeddings=True).tolist()

        # Keywords: top-40 frequent tokens (excluding single-char to reduce noise)
        freq = Counter([t for t in enriched_tokens if len(t) > 1])
        keywords = " ".join([w for w, _ in freq.most_common(40)])

        return {
            "author": author,
            "title": title,
            "abstract": abstract,
            "summary": summary,
            "keywords": keywords,
            "text": processed_text,
            "embedding": embedding,
        }

    except Exception as e:
        print(f"[ERROR] PDF processing failed for {file_path}: {e}")
        return None


# ----------------------------------------------------------------------
# Query helpers
# ----------------------------------------------------------------------

def correct_spell(query: str, corpus_words: set[str]) -> str:
    """Correct spelling in query using nearest words from a corpus."""
    if not query:
        return query
    words = query.split()
    corrected: List[str] = []
    for w in words:
        matches = difflib.get_close_matches(w, corpus_words, n=1, cutoff=0.82)
        corrected.append(matches[0] if matches else w)
    return " ".join(corrected)


def expand_query(query: str) -> str:
    """Expand query with WordNet synonyms (conservative set)."""
    if not query:
        return query
    expanded = set(query.split())
    for word in query.split():
        # Avoid exploding for very short tokens or numbers
        if len(word) < 3 or word.isdigit():
            continue
        syns = wordnet.synsets(word)
        for syn in syns[:5]:  # cap for efficiency
            for lemma in syn.lemma_names()[:3]:  # cap per synset
                if lemma.isascii() and lemma.isalnum():
                    expanded.add(lemma.lower())
    return " ".join(expanded)


def to_numpy(vec: Any) -> Optional[np.ndarray]:
    """Ensure embeddings are numpy float32 arrays."""
    if vec is None:
        return None
    if isinstance(vec, str):  # stored as JSON string
        try:
            vec = json.loads(vec)
        except Exception:
            return None
    try:
        arr = np.asarray(vec, dtype=np.float32)
        return arr
    except Exception:
        return None


# ----------------------------------------------------------------------
# Hybrid Search (Author-Boosted)
# ----------------------------------------------------------------------

def _build_bm25_corpus(docs: List[Dict[str, Any]], field_boosts: Dict[str, float]) -> List[List[str]]:
    """Construct token lists per document with field-aware boosting by repetition."""
    base_mult = 3  # overall boost intensity (tune if needed)
    tokenized_docs: List[List[str]] = []

    for doc in docs:
        tokens: List[str] = []
        # author
        a = tokenize(str(doc.get("author", "")))
        tokens += a * max(1, int(round(field_boosts.get("author", 1.0) * base_mult)))
        # title
        t = tokenize(str(doc.get("title", "")))
        tokens += t * max(1, int(round(field_boosts.get("title", 1.0) * base_mult)))
        # keywords
        k = tokenize(str(doc.get("keywords", "")))
        tokens += k * max(1, int(round(field_boosts.get("keywords", 1.0) * base_mult)))
        # abstract
        ab = tokenize(str(doc.get("abstract", "")))
        tokens += ab * max(1, int(round(field_boosts.get("abstract", 1.0) * base_mult)))
        # summary
        sm = tokenize(str(doc.get("summary", "")))
        tokens += sm * max(1, int(round(field_boosts.get("summary", 1.0) * base_mult)))

        # fallback to full text if nothing else
        if not tokens:
            tokens = tokenize(str(doc.get("text", "")))
        tokenized_docs.append(tokens)

    return tokenized_docs


def _author_overlap_bonus(query_tokens: List[str], author_text: str) -> float:
    """Compute a small additive bonus if query tokens overlap with author tokens."""
    author_tokens = set(tokenize(author_text))
    if not author_tokens:
        return 0.0
    overlap = author_tokens.intersection(set(query_tokens))
    if not overlap:
        return 0.0
    # bonus proportional to overlap size but capped
    return min(0.25 + 0.05 * len(overlap), 0.6)


def hybrid_search(
    query: str,
    docs: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    limit: int = 10,
    bm25_weight: float = 0.45,
    sem_weight: float = 0.55,
    field_boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid BM25 + semantic search over document dicts with AUTHOR BOOSTING.

    Each doc is expected to contain: {id, title, author, summary, keywords, embedding} at minimum.
    """
    if not docs:
        return []

    # Defaults: give author and title extra weight
    if field_boosts is None:
        field_boosts = {
            "author": 3.0,
            "title": 2.2,
            "keywords": 1.6,
            "abstract": 1.3,
            "summary": 1.0,
        }

    # Build a richer correction corpus: authors, titles, keywords
    correction_vocab = set()
    for d in docs:
        correction_vocab.update(tokenize(str(d.get("author", ""))))
        correction_vocab.update(tokenize(str(d.get("title", ""))))
        correction_vocab.update(tokenize(str(d.get("keywords", ""))))
    # Also a slice of main text for coverage
    for d in docs[:2000]:  # cap for efficiency on large collections
        correction_vocab.update(tokenize(" ".join(str(d.get("text", ""))[:2000].split())))

    # Query pipeline: correct → expand
    q = correct_spell(query.strip(), correction_vocab)
    q = expand_query(q)
    q_tokens = tokenize(q)

    # BM25 with field-aware boosting
    tokenized_docs = _build_bm25_corpus(docs, field_boosts)
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(q_tokens)

    # Semantic similarity (dot product of normalized embeddings)
    model = get_model()
    q_emb = model.encode(q, normalize_embeddings=True).reshape(1, -1)

    embeddings: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i, doc in enumerate(docs):
        emb = to_numpy(doc.get("embedding"))
        if emb is not None and emb.size > 0:
            embeddings.append(emb)
            valid_idx.append(i)

    if not embeddings:
        # If no embeddings in collection, degrade gracefully to BM25 only
        scores = np.asarray(bm25_scores, dtype=np.float32)
    else:
        doc_embs = np.vstack(embeddings)  # (N, D)
        sem_scores = np.dot(q_emb, doc_embs.T).flatten()
        scores = np.zeros(len(docs), dtype=np.float32)
        scores[valid_idx] = bm25_weight * bm25_scores[valid_idx] + sem_weight * sem_scores

    # Add small bonus for direct author token overlap
    for i, doc in enumerate(docs):
        scores[i] += _author_overlap_bonus(q_tokens, str(doc.get("author", "")))

    # Strong boost if the full query is exactly the author name
    q_norm = query.strip().lower()
    for i, doc in enumerate(docs):
        a_norm = str(doc.get("author", "")).strip().lower()
        if a_norm and (q_norm == a_norm or (q_norm in a_norm and len(q_norm) > 2)):
            scores[i] += 0.8  # prominent lift for exact/contained author match

    # Rank & slice
    top_indices = np.argsort(scores)[::-1][: max(1, limit)]

    results = [
        {
            "id": docs[i].get("id"),
            "title": docs[i].get("title", "Untitled"),
            "author": docs[i].get("author", "Unknown"),
            "summary": (docs[i].get("summary") or "")[:360],
            "file": docs[i].get("file"),  # URL or path
            "score": float(scores[i]),
        }
        for i in top_indices
        if scores[i] > 0
    ]

    return results


__all__ = [
    "process_pdf",
    "hybrid_search",
    "get_model",
    "tokenize",
    "correct_spell",
    "expand_query",
    "to_numpy",
]
