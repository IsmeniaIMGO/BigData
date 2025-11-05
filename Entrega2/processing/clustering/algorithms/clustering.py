from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp


def _to_dense(x):
    """Convierte a denso si es matriz dispersa."""
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def prepare_features_for_clustering(
    X: np.ndarray | sp.spmatrix,
    scaler: Literal["none", "standard"] = "none",
):
    """
    Prepara características para clustering.
    - scaler='none': devuelve X tal cual
    - scaler='standard': StandardScaler (with_mean=False si es sparse)
    """
    if scaler == "none":
        return X

    if scaler == "standard":
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta scikit-learn. Instala con: pip install scikit-learn") from e

        if sp.issparse(X):
            ss = StandardScaler(with_mean=False)
            return ss.fit_transform(X)
        else:
            ss = StandardScaler()
            return ss.fit_transform(np.asarray(X))

    raise ValueError("scaler debe ser 'none' o 'standard'")


def run_kmeans(X: np.ndarray | sp.spmatrix, k: int, seed: int = 42):
    """
    Ejecuta KMeans y devuelve etiquetas. Convierte a denso si es necesario.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta scikit-learn. Instala con: pip install scikit-learn") from e

    Xfit = _to_dense(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(Xfit)


def run_dbscan(X: np.ndarray | sp.spmatrix, eps: float = 0.5, min_samples: int = 5):
    """
    Ejecuta DBSCAN (euclidean) y devuelve etiquetas. Convierte a denso por compatibilidad.
    """
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta scikit-learn. Instala con: pip install scikit-learn") from e

    Xfit = _to_dense(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return db.fit_predict(Xfit)


def run_hdbscan(X: np.ndarray | sp.spmatrix, min_cluster_size: int = 15):
    """
    Ejecuta HDBSCAN y devuelve etiquetas. Convierte a denso por compatibilidad.
    Requiere 'hdbscan'.
    """
    try:
        import hdbscan  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta hdbscan. Instala con: pip install hdbscan") from e

    Xfit = _to_dense(X)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(Xfit)


def save_labels(labels, path: str) -> None:
    """
    Guarda etiquetas en CSV (.csv) o Numpy (.npy/.npz) según extensión.
    """
    import os
    import csv

    arr = np.asarray(labels)
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if ext == ".csv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label"])
            for v in arr:
                w.writerow([int(v)])
        return

    if ext in {".npy", ".npz"}:
        np.save(path, arr)
        return

    # Por defecto CSV
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"])
        for v in arr:
            w.writerow([int(v)])


def _as_condensed(distance_matrix: np.ndarray) -> np.ndarray:
    """Convierte una matriz de distancias NxN a vector condensado para scipy.linkage."""
    try:
        from scipy.spatial.distance import squareform  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta SciPy. Instala con: pip install scipy") from e

    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix debe ser cuadrada NxN")
    # Asegurar simetría y diagonal cero
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def hierarchical_clustering_ward(distance_matrix: np.ndarray | None):
    """
    Aplica clustering jerárquico (Ward) desde matriz de distancias.
    - Convierte a forma condensada y llama scipy.cluster.hierarchy.linkage(method='ward').
    - En caso de error, hace fallback a 'average'.
    Devuelve la matriz de enlace Z (n-1 x 4).
    """
    if distance_matrix is None:
        raise ValueError("distance_matrix no puede ser None")

    try:
        from scipy.cluster.hierarchy import linkage  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta SciPy. Instala con: pip install scipy") from e

    try:
        y = _as_condensed(distance_matrix)
        Z = linkage(y, method="ward")
        return Z
    except Exception:
        # Fallback a average
        y = _as_condensed(distance_matrix)
        Z = linkage(y, method="average")
        return Z


def hierarchical_clustering_agnes(distance_matrix: np.ndarray | None):
    """
    Clustering jerárquico AGNES (aglomerativo, average linkage).
    - Si n > 1000: usa scipy.linkage(method='average') por eficiencia.
    - Si n <= 1000: implementación manual (O(n^3)) con promedio entre clusters, devuelve Z.
    - En caso de error, fallback a scipy.linkage(method='average').
    """
    if distance_matrix is None:
        raise ValueError("distance_matrix no puede ser None")

    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix debe ser cuadrada NxN")
    n = D.shape[0]

    # Camino rápido para n grande
    if n > 1000:
        try:
            from scipy.cluster.hierarchy import linkage  # type: ignore
            y = _as_condensed(D)
            return linkage(y, method="average")
        except Exception:
            from scipy.cluster.hierarchy import linkage  # type: ignore
            y = _as_condensed(D)
            return linkage(y, method="average")

    # Implementación manual (average linkage)
    try:
        # Asegurar simetría y diagonal cero
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)

        # Clusters iniciales: cada punto es su propio cluster
        clusters = {i: [i] for i in range(n)}
        sizes = {i: 1 for i in range(n)}
        active = set(range(n))
        next_id = n
        Z = []  # filas [idx1, idx2, dist, tamaño]

        # Distancias actuales entre clusters (usar matriz mutable)
        D_work = D.copy()

        for _ in range(n - 1):
            # Encontrar par (i,j) activo con menor distancia i<j
            best = (None, None, np.inf)
            active_list = sorted(active)
            for ii, i in enumerate(active_list):
                row = D_work[i, :]
                for j in active_list[ii + 1 :]:
                    d = row[j]
                    if d < best[2]:
                        best = (i, j, d)

            i, j, dmin = best
            if i is None or j is None:
                break

            # Registrar fusión: ids absolutos deben ser enteros; usamos los ids actuales
            size_i = sizes[i]
            size_j = sizes[j]
            new_size = size_i + size_j
            Z.append([float(i), float(j), float(dmin), float(new_size)])

            # Crear nuevo cluster con id next_id
            clusters[next_id] = clusters[i] + clusters[j]
            sizes[next_id] = new_size

            # Actualizar distancias promedio a otros clusters activos
            for k in list(active):
                if k in (i, j):
                    continue
                # Average linkage: d(Cu, Ck) = (|Ci|*d(i,k) + |Cj|*d(j,k)) / (|Ci|+|Cj|)
                dik = D_work[i, k]
                djk = D_work[j, k]
                new_d = (size_i * dik + size_j * djk) / new_size
                # Expandir D_work si hace falta
                if next_id >= D_work.shape[0]:
                    # Expandir con filas/columnas nuevas
                    pad = next_id - D_work.shape[0] + 1
                    D_work = np.pad(D_work, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)
                D_work[next_id, k] = new_d
                D_work[k, next_id] = new_d

            # Desactivar i y j, activar new
            active.remove(i)
            active.remove(j)
            active.add(next_id)

            # Para evitar reusar i y j: poner sus distancias altas
            D_work[i, :] = np.inf
            D_work[:, i] = np.inf
            D_work[j, :] = np.inf
            D_work[:, j] = np.inf

            next_id += 1

        Z = np.asarray(Z, dtype=float)

        # Si algo salió raro (por ejemplo, datos degenerados), fallback
        if Z.shape[0] != n - 1:
            from scipy.cluster.hierarchy import linkage  # type: ignore
            y = _as_condensed(distance_matrix)
            return linkage(y, method="average")
        return Z
    except Exception:
        # Fallback robusto
        from scipy.cluster.hierarchy import linkage  # type: ignore
        y = _as_condensed(distance_matrix)
        return linkage(y, method="average")

