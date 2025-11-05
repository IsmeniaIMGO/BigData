from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import scipy.sparse as sp


# Vectorizadores
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Falta scikit-learn. Instala con: pip install scikit-learn"
    ) from e


def vectorize_bow(
    corpus: Iterable[str],
    max_features: int | None = None,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """
    Bag-of-Words con CountVectorizer.
    Returns: (X, vocab, vectorizer)
      - X: scipy.sparse CSR matrix (n_docs x n_features)
      - vocab: dict[str,int]
      - vectorizer: CountVectorizer fitted
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(list(corpus))
    vocab = vectorizer.vocabulary_
    return X, vocab, vectorizer


def vectorize_tfidf(
    corpus: Iterable[str],
    max_features: int | None = None,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """
    TF-IDF con TfidfVectorizer.
    Returns: (X, vocab, vectorizer)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(list(corpus))
    vocab = vectorizer.vocabulary_
    return X, vocab, vectorizer


def embed_texts(texts: Iterable[str], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeddings con Sentence-Transformers. Requiere 'sentence-transformers'.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Falta sentence-transformers. Instala con: pip install sentence-transformers"
        ) from e

    st_model = SentenceTransformer(model)
    embeddings = st_model.encode(list(texts), show_progress_bar=False, normalize_embeddings=False)
    return np.asarray(embeddings)


def reduce_dimensionality(
    X: np.ndarray | sp.spmatrix,
    method: str = "svd",
    dim: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce dimensionalidad con 'svd' (TruncatedSVD), 'umap' (umap-learn) o 'tsne' (sklearn).
    Devuelve un array denso (n_samples x dim).
    """
    method = method.lower()

    if method == "svd":
        svd = TruncatedSVD(n_components=dim, random_state=random_state)
        X_reduced = svd.fit_transform(X)
        return np.asarray(X_reduced)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta umap-learn. Instala con: pip install umap-learn") from e
        # UMAP requiere denso en la mayoría de casos
        X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
        reducer = umap.UMAP(n_components=dim, random_state=random_state)
        return reducer.fit_transform(X_dense)

    if method == "tsne":
        X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
        tsne = TSNE(n_components=dim, random_state=random_state, init="pca")
        return tsne.fit_transform(X_dense)

    raise ValueError("method debe ser 'svd', 'umap' o 'tsne'")


def save_features(X: np.ndarray | sp.spmatrix, path: str) -> None:
    """
    Guarda matriz de características en .npz.
    - Si es sparse CSR: guarda data/indices/indptr/shape para reconstrucción.
    - Si es denso: guarda como 'X'.
    """
    if sp.issparse(X):
        X_csr = X.tocsr()
        np.savez_compressed(
            path,
            data=X_csr.data,
            indices=X_csr.indices,
            indptr=X_csr.indptr,
            shape=X_csr.shape,
            sparse=True,
        )
    else:
        np.savez_compressed(path, X=np.asarray(X), sparse=False)


def load_features(path: str) -> np.ndarray | sp.csr_matrix:
    """
    Carga matriz de características desde .npz y reconstruye sparse si aplica.
    """
    with np.load(path, allow_pickle=False) as f:
        sparse = bool(f["sparse"]) if "sparse" in f else False
        if sparse:
            data = f["data"]
            indices = f["indices"]
            indptr = f["indptr"]
            shape = tuple(f["shape"])  # type: ignore
            return sp.csr_matrix((data, indices, indptr), shape=shape)
        return f["X"]


def cosine_sim_matrix(X: np.ndarray | sp.spmatrix, Y: np.ndarray | sp.spmatrix | None = None) -> np.ndarray:
    """
    Devuelve la matriz de similitud coseno entre X y Y (o X vs X si Y=None).
    Soporta matrices densas o sparse.
    """
    return cosine_similarity(X, Y)


def top_k_cosine(query_vec: np.ndarray | sp.spmatrix, X: np.ndarray | sp.spmatrix, k: int = 10):
    """
    Devuelve índices y puntuaciones top-k más similares por coseno de una consulta contra X.
    query_vec debe ser de forma (1, n_features) o (n_features,). Si es 1D se convierte a 2D.
    """
    if query_vec.ndim == 1:  # type: ignore[attr-defined]
        query_vec = np.asarray(query_vec)[None, :]
    sims = cosine_similarity(query_vec, X).ravel()
    if k <= 0:
        k = len(sims)
    top_idx = np.argpartition(-sims, kth=min(k - 1, len(sims) - 1))[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return top_sorted, sims[top_sorted]
