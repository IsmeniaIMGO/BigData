from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np


Token = str
DocId = int


def _ensure_tokens(doc: Union[str, Iterable[str]]) -> List[Token]:
    if isinstance(doc, str):
        return [t for t in re.split(r"\W+", doc.lower()) if t]
    return [str(t).lower() for t in doc]


def build_inverted_index(corpus: Iterable[Union[str, Iterable[str]]]) -> Dict:
    """
    Construye un índice invertido simple con conteos tf y metadatos para BM25.
    Estructura devuelta:
    {
      'type': 'inverted',
      'postings': { term: {doc_id: tf, ...}, ... },
      'df': { term: df, ... },
      'doc_len': [len_tokens_por_doc],
      'avgdl': float,
      'N': int
    }
    """
    postings: Dict[str, Dict[DocId, int]] = {}
    doc_len: List[int] = []

    for i, doc in enumerate(corpus):
        tokens = _ensure_tokens(doc)
        doc_len.append(len(tokens))
        freqs: Dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        for term, tf in freqs.items():
            if term not in postings:
                postings[term] = {i: tf}
            else:
                postings[term][i] = tf

    N = len(doc_len)
    avgdl = (sum(doc_len) / N) if N > 0 else 0.0
    df = {term: len(docs) for term, docs in postings.items()}

    return {
        "type": "inverted",
        "postings": postings,
        "df": df,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "N": N,
    }


def _bm25_idf(N: int, df: int) -> float:
    # IDF estilo BM25
    # idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0


def _bm25_scores(query_tokens: List[str], index: Dict, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    N = index["N"]
    avgdl = index["avgdl"]
    doc_len = index["doc_len"]
    postings = index["postings"]
    df = index["df"]

    scores = np.zeros(N, dtype=np.float64)
    if N == 0:
        return scores

    # Considerar solo docs que contienen al menos un término de la consulta
    candidate_docs: set[int] = set()
    for t in query_tokens:
        if t in postings:
            candidate_docs.update(postings[t].keys())

    for d in candidate_docs:
        dl = doc_len[d]
        denom_norm = k1 * (1 - b + b * (dl / (avgdl or 1.0)))
        s = 0.0
        for t in query_tokens:
            if t not in postings:
                continue
            tf = postings[t].get(d, 0)
            if tf == 0:
                continue
            idf = _bm25_idf(N, df[t])
            s += idf * ((tf * (k1 + 1)) / (tf + denom_norm))
        scores[d] = s
    return scores


def build_ann_index(
    embeddings: np.ndarray,
    method: str = "brute",
    metric: str = "cosine",
):
    """
    Crea un índice ANN. Métodos soportados:
    - 'brute': coseno por fuerza bruta (sin dependencias adicionales)
    - 'annoy': requiere 'annoy'
    - 'faiss': requiere 'faiss-cpu'
    Devuelve un dict con los datos necesarios para búsqueda.
    """
    method = method.lower()
    emb = np.asarray(embeddings)
    dim = emb.shape[1]

    if method == "brute":
        # Normalizamos si métrica es coseno para acelarar dot products
        if metric == "cosine":
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb_norm = emb / norms
        else:
            emb_norm = emb
        return {"type": "ann", "method": "brute", "metric": metric, "emb": emb_norm}

    if method == "annoy":
        try:
            from annoy import AnnoyIndex  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta 'annoy'. Instala con: pip install annoy") from e
        # Annoy usa 'angular' para coseno
        ann_metric = "angular" if metric == "cosine" else "euclidean"
        t = AnnoyIndex(dim, ann_metric)
        for i, v in enumerate(emb):
            t.add_item(i, v.tolist())
        t.build(10)  # n_trees razonable por defecto
        return {"type": "ann", "method": "annoy", "index": t, "metric": metric}

    if method == "faiss":
        try:
            import faiss  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta 'faiss-cpu'. Instala con: pip install faiss-cpu") from e
        if metric == "cosine":
            # Normalizar y usar índice L2 equivale a coseno sobre unit vectors
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            index = faiss.IndexFlatIP(dim)
            index.add(emb_norm.astype(np.float32))
            return {"type": "ann", "method": "faiss", "index": index, "metric": metric, "unit": True}
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(emb.astype(np.float32))
            return {"type": "ann", "method": "faiss", "index": index, "metric": metric, "unit": False}

    raise ValueError("method debe ser 'brute', 'annoy' o 'faiss'")


def search_query(query, index: Dict, top_k: int = 10):
    """
    Busca según el tipo de índice.
    - Índice invertido: query es str o lista de tokens. Devuelve lista (idx, score_bm25).
    - Índice ANN: query es vector (np.ndarray) 1D. Devuelve lista (idx, score_similaridad).
    """
    if index.get("type") == "inverted":
        q_tokens = _ensure_tokens(query)
        scores = _bm25_scores(q_tokens, index)
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    if index.get("type") == "ann":
        method = index.get("method")
        metric = index.get("metric", "cosine")
        q = np.asarray(query)
        if q.ndim == 1:
            q = q[None, :]

        if method == "brute":
            emb = index["emb"]
            if metric == "cosine":
                qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
                sims = (qn @ emb.T).ravel()
            else:
                # Euclid: convertir a dist y luego a score negativo
                d2 = np.sum((emb[None, :, :] - q[:, None, :]) ** 2, axis=2).ravel()
                sims = -d2
            top = np.argpartition(-sims, kth=min(top_k - 1, sims.size - 1))[:top_k]
            top = top[np.argsort(-sims[top])]
            return [(int(i), float(sims[i])) for i in top]

        if method == "annoy":
            t = index["index"]
            ids, dists = t.get_nns_by_vector(q.ravel().tolist(), top_k, include_distances=True)
            # Annoy retorna distancia angular; convertir a similitud aproximada
            sims = [1.0 - (d / 2.0) for d in dists] if metric == "cosine" else [-d for d in dists]
            return list(zip(ids, sims))

        if method == "faiss":
            faiss_index = index["index"]
            unit = index.get("unit", False)
            import faiss  # type: ignore

            qv = q.astype(np.float32)
            if metric == "cosine" and unit:
                qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
                sims, ids = faiss_index.search(qv, top_k)
                return list(zip(ids.ravel().tolist(), sims.ravel().astype(float).tolist()))
            else:
                # Para L2, faiss devuelve distancias; convertimos a score negativo
                dists, ids = faiss_index.search(qv, top_k)
                sims = (-dists).ravel().astype(float).tolist()
                return list(zip(ids.ravel().tolist(), sims))

    raise ValueError("Índice no reconocido o consulta/formato inválido")


def rank_results(scores: np.ndarray, strategy: str = "cosine") -> np.ndarray:
    """
    Ordena índices por score descendente. 'cosine' o 'bm25' comportan igual (descendente).
    """
    return np.argsort(-np.asarray(scores))
