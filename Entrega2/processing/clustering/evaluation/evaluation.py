from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import scipy.sparse as sp


try:
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("Falta scikit-learn. Instala con: pip install scikit-learn") from e


def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def evaluate_internal_metrics(
    X,
    labels: Sequence[int],
    metric: str = "cosine",
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Calcula métricas internas de clustering (sin verdad-terreno):
      - silhouette (mejor alto): usa 'metric' (recomendado 'cosine' para TF‑IDF/embeddings)
      - calinski_harabasz (mejor alto)
      - davies_bouldin (mejor bajo)

    Nota: calinski/davies operan mejor con datos densos/reducidos (p.ej., SVD); si X es sparse se densifica.
    """
    y = np.asarray(labels)
    res = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

    # Silhouette (puede trabajar con sparse y varias métricas)
    try:
        # Necesita al menos 2 clusters distintos
        if len(np.unique(y)) >= 2 and X is not None and len(y) == (X.shape[0] if hasattr(X, "shape") else len(X)):
            res["silhouette"] = float(
                silhouette_score(X, y, metric=metric, sample_size=sample_size, random_state=random_state)
            )
    except Exception:
        pass

    # CH y DB (convertir a denso si es necesario)
    try:
        Xd = _to_dense(X)
        if len(np.unique(y)) >= 2:
            res["calinski_harabasz"] = float(calinski_harabasz_score(Xd, y))
            res["davies_bouldin"] = float(davies_bouldin_score(Xd, y))
    except Exception:
        pass

    return res


def compare_labels(labels_a: Sequence[int], labels_b: Sequence[int]) -> dict:
    """Compara dos etiquetados con ARI y NMI."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    return {
        "ARI": float(adjusted_rand_score(a, b)),
        "NMI": float(normalized_mutual_info_score(a, b)),
    }


def cluster_profiles(
    X,
    labels: Sequence[int],
    feature_names: Iterable[str],
    top_n: int = 20,
    agg: str = "sum",
) -> dict:
    """
    Genera perfiles de clusters a partir de una matriz de características (TF‑IDF/BOW/embeddings) y etiquetas.
    Para TF‑IDF/BOW, 'sum' produce términos representativos por cluster.

    Devuelve: {cluster_id: [(feature, score), ...]}
    """
    y = np.asarray(labels)
    feats = list(feature_names)
    n_clusters = int(np.max(y)) + 1 if len(y) else 0
    out: dict[int, list[tuple[str, float]]] = {}

    for c in range(n_clusters):
        mask = (y == c)
        if not np.any(mask):
            out[c] = []
            continue
        Xc = X[mask]
        # Agregación por cluster
        if agg == "mean":
            vec = Xc.mean(axis=0)
        else:
            vec = Xc.sum(axis=0)

        # Convertir vector a 1D np.array
        if sp.issparse(vec):
            vec = np.asarray(vec.A1)
        else:
            vec = np.asarray(vec).ravel()

        top_idx = np.argsort(-vec)[: top_n if top_n else len(vec)]
        out[c] = [(feats[i] if i < len(feats) else str(i), float(vec[i])) for i in top_idx if vec[i] > 0]

    return out


def sweep_kmeans_k(
    X,
    ks: Iterable[int],
    scaler: str = "standard",
    metric: str = "cosine",
    sample_size: int | None = None,
    random_state: int = 42,
):
    """
    Ejecuta KMeans sobre un rango de k y calcula métricas internas.
    Devuelve una lista de dicts (puedes convertir a DataFrame).
    """
    # Preparación de features (escala si aplica)
    if scaler == "standard":
        if sp.issparse(X):
            Xproc = StandardScaler(with_mean=False).fit_transform(X)
        else:
            Xproc = StandardScaler().fit_transform(_to_dense(X))
    else:
        Xproc = X

    results: List[dict] = []
    for k in ks:
        try:
            km = KMeans(n_clusters=int(k), random_state=random_state, n_init="auto")
            labels = km.fit_predict(_to_dense(Xproc))
            metrics = evaluate_internal_metrics(Xproc, labels, metric=metric, sample_size=sample_size, random_state=random_state)
            row = {"k": int(k), **metrics}
            results.append(row)
        except Exception:
            results.append({"k": int(k), "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan})

    return results
