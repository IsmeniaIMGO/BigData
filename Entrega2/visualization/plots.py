from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _out_path(name: str, subdir: str = "visualization") -> Path:
    from processing.utils.rutas import resolve_paths

    paths = resolve_paths()
    out_dir = paths["outputs"] / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def plot_publications_per_year(df: pd.DataFrame, year_col: str = "year", out_name: str = "pubs_per_year.png") -> str:
    import matplotlib.pyplot as plt

    counts = (
        df[year_col].dropna().astype(int).value_counts().sort_index()
        if year_col in df.columns
        else pd.Series(dtype=int)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Publicaciones por año")
    ax.set_xlabel("Año")
    ax.set_ylabel("Conteo")
    ax.grid(axis="y", alpha=0.3)
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_top_bars(counts: pd.Series, title: str, out_name: str = "top_bars.png", top_n: int = 20) -> str:
    import matplotlib.pyplot as plt

    counts = counts.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    counts.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Conteo")
    ax.grid(axis="x", alpha=0.3)
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_cluster_sizes(
    labels: Sequence[int],
    out_name: str = "cluster_sizes.png",
    sort_by: str = "id",  # "id" | "count"
) -> str:
    import matplotlib.pyplot as plt

    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    # Serie con índice numérico de cluster para poder ordenar por id o por tamaño
    series = pd.Series(counts, index=unique)
    if str(sort_by).lower() == "count":
        series = series.sort_values(ascending=False)
    else:
        series = series.sort_index()
    # Mostrar etiquetas como string para mejor formato en eje X
    series.index = [str(i) for i in series.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title("Tamaño de clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Nº documentos")
    ax.grid(axis="y", alpha=0.3)
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_embeddings_scatter(
    X2d: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    out_name: str = "embeddings_scatter.png",
    title: str = "Proyección 2D",
) -> str:
    import matplotlib.pyplot as plt

    X2d = np.asarray(X2d)
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is None:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.6)
    else:
        labels = np.asarray(labels)
        scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=labels, s=10, cmap="tab20", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_elbow(k_list: Sequence[int], inertias: Sequence[float], out_name: str = "kmeans_elbow.png") -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_list, inertias, marker="o")
    ax.set_title("KMeans Elbow")
    ax.set_xlabel("k")
    ax.set_ylabel("Inercia")
    ax.grid(True, alpha=0.3)
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_silhouette(X: np.ndarray, labels: Sequence[int], metric: str = "euclidean", out_name: str = "silhouette.png") -> str:
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples

    labels = np.asarray(labels)
    svals = silhouette_samples(X, labels, metric=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(svals, bins=30, alpha=0.8)
    ax.set_title("Distribución de Silhouette")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frecuencia")
    p = _out_path(out_name)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


# =============================
# Visualizaciones faltantes (otros metodos)
# =============================


def graficar_ranking(
    serie: pd.Series,
    titulo: str,
    nombre_archivo: str,
    xlabel: str = "Cantidad",
    ylabel: str = "Elemento",
) -> str:
    """Equivalente a ranking.graficar_ranking: barras horizontales ordenadas."""
    import matplotlib.pyplot as plt

    s = serie.sort_values(ascending=True)  # para que queden de abajo a arriba
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s.index, s.values)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.3)
    p = _out_path(nombre_archivo)
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_sorting_benchmarks(
    resultados: dict,
    categorias: dict,
    out_prefix: str = "sorting_benchmark",
) -> list[str]:
    """Replica create_graphs: una barra por categoría y una comparativa global."""
    import matplotlib.pyplot as plt
    paths: list[str] = []

    # Gráfica por categoría
    for categoria, algs in resultados.items():
        # filtrar algoritmos que no fallaron
        alg_ok = {k: v for k, v in algs.items() if (v and v.get("mean", None) not in (None, 0)) and v.get("error") is None}
        if not alg_ok:
            continue
        ordered = sorted(alg_ok.items(), key=lambda kv: kv[1]["mean"])  # menor tiempo primero
        nombres = [k for k, _ in ordered]
        tiempos = [float(v["mean"]) for _, v in ordered]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(nombres, tiempos, color="skyblue")
        ax.set_title(f"Tiempo de ejecución - {categoria} (n={len(categorias.get(categoria, []))})")
        ax.set_xlabel("Algoritmo")
        ax.set_ylabel("Tiempo (ms)")
        # Rotar etiquetas sin forzar un locator fijo para evitar UserWarning de set_ticklabels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for b, t in zip(bars, tiempos):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{t:.2f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
        p = _out_path(f"{out_prefix}_{categoria}.png", subdir="visualization/sorting")
        fig.tight_layout()
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(str(p))

    # Comparativa global
    import numpy as np

    alg_names = sorted({alg for cat in resultados.values() for alg in cat if cat[alg].get("error") is None})
    cats = list(resultados.keys())
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(alg_names))

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, alg in enumerate(alg_names):
        vals = []
        for cat in cats:
            r = resultados[cat].get(alg)
            vals.append(float(r["mean"]) if r and r.get("error") is None else 0.0)
        offset = (i - len(alg_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=alg)

    ax.set_title("Comparación de rendimiento de algoritmos de ordenamiento")
    ax.set_xlabel("Tipo de datos")
    ax.set_ylabel("Tiempo (ms)")
    ax.set_xticks(x, [f"{c}\n(n={len(categorias.get(c, []))})" for c in cats])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="y", alpha=0.3)
    p = _out_path(f"{out_prefix}_global.png", subdir="visualization/sorting")
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))
    return paths


def plot_counted_words(resultados: dict, out_dir_name: str = "visualization/text") -> list[str]:
    """Replica create_graphs_words: barras de frecuencias y anotación de tiempo por algoritmo."""
    import matplotlib.pyplot as plt
    paths: list[str] = []

    for nombre_alg, datos in resultados.items():
        tiempo = datos.get("tiempo")
        freq = datos.get("sorted_frequencies")
        words = datos.get("word")
        if tiempo is None or freq is None or words is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(words, freq, color="skyblue", edgecolor="darkblue", linewidth=0.8)
        ax.set_ylabel("Frecuencia de palabras")
        ax.set_xlabel("Palabras")
        ax.set_title(f"{nombre_alg} - Tiempo: {float(tiempo):.2f} µs")
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        p = _out_path(f"counted_words_{nombre_alg}.png", subdir=out_dir_name)
        fig.tight_layout()
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(str(p))
    return paths


def plot_dendrogram(
    linkage_matrix,
    labels: Optional[Sequence[str]] = None,
    method_name: str = "Ward",
    out_name: str = "dendrogram.png",
    max_d: float | None = None,
    color_threshold: float | None = None,
    truncate_mode: str | None = None,
    p: int = 30,
) -> str:
    """Dibuja y guarda un dendrograma (requiere SciPy)."""
    import matplotlib.pyplot as plt
    try:
        from scipy.cluster.hierarchy import dendrogram  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Se requiere SciPy para plot_dendrogram") from e

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(f"Dendrograma Jerárquico - Método {method_name}")
    dendrogram(
        linkage_matrix,
        labels=labels if labels and len(labels) <= 50 else None,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
        truncate_mode=truncate_mode if (labels is None or len(labels) > 50) else None,
        p=p if (labels is None or len(labels) > 50) else None,
        ax=ax,
    )
    if max_d is not None:
        ax.axhline(y=max_d, c="k", ls="--", lw=1)
    pth = _out_path(out_name, subdir="visualization/clustering")
    fig.tight_layout()
    fig.savefig(pth, dpi=300)
    plt.close(fig)
    return str(pth)


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    out_name: str = "similarity_heatmap.png",
    max_size: int = 100,
    random_seed: int = 42,
) -> str:
    """Heatmap de similitud con submuestreo opcional (usa seaborn)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import seaborn as sns

    M = np.asarray(similarity_matrix)
    if M.shape[0] > max_size:
        random.seed(random_seed)
        idx = sorted(random.sample(range(M.shape[0]), max_size))
        M = M[np.ix_(idx, idx)]
        labels = [labels[i] for i in idx] if labels else None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(M, annot=False, cmap="viridis", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Mapa de calor - Similitud entre documentos")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    p = _out_path(out_name, subdir="visualization/clustering")
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_category_distribution_heatmap(
    cluster_data: dict,
    categories: list[str],
    algorithm_key: str,
    algorithm_name: str,
    out_name: str | None = None,
) -> str:
    """Heatmap de distribución de categorías por cluster (valores en 0..1)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    cluster_ids = list(cluster_data.keys())
    data = np.zeros((len(categories), len(cluster_ids)))
    for j, cid in enumerate(cluster_ids):
        for i, cat in enumerate(categories):
            data[i, j] = float(cluster_data[cid].get(cat, 0.0))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[f"Cluster {c}" for c in cluster_ids],
        yticklabels=categories,
        ax=ax,
    )
    ax.set_title(f"Distribución de categorías - {algorithm_name}")
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Categorías")
    out_name = out_name or f"category_distribution_{algorithm_key}.png"
    p = _out_path(out_name, subdir="visualization/clustering")
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_category_distribution(cluster_data, categories, algorithm_key, algorithm_name):
    """Wrapper compatible con otros_metodos/visualizer.py."""
    return plot_category_distribution_heatmap(cluster_data, categories, algorithm_key, algorithm_name)


def plot_wordcloud(freqs: dict[str, int], out_name: str = "wordcloud.png", width: int = 800, height: int = 400) -> str:
    """Genera una nube de palabras (requiere 'wordcloud')."""
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Se requiere la librería 'wordcloud' para plot_wordcloud") from e

    from matplotlib import pyplot as plt

    wc = WordCloud(width=width, height=height, background_color="white").generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    p = _out_path(out_name, subdir="visualization/bibliometrics")
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_cooccurrence_network(edges: list[tuple[str, str, int]], out_name: str = "co_word_network.png", min_weight: int = 2) -> str:
    """Dibuja una red de co-ocurrencia (requiere networkx)."""
    try:
        import networkx as nx  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Se requiere 'networkx' para plot_cooccurrence_network") from e

    import matplotlib.pyplot as plt

    G = nx.Graph()
    for u, v, w in edges:
        if w >= min_weight:
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, k=0.3)
    fig, ax = plt.subplots(figsize=(12, 12))
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    ax.set_title("Co-word Network (min weight = {})".format(min_weight))
    ax.axis("off")
    p = _out_path(out_name, subdir="visualization/bibliometrics")
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_wordcloud_top_words(
    texts: Iterable[str] | pd.Series | pd.DataFrame,
    text_col: str | None = None,
    out_name: str = "wordcloud_top_words.png",
    top_n: int = 200,
    min_token_len: int = 2,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    languages: Sequence[str] = ("english", "spanish"),
    extra_stopwords: Optional[Iterable[str]] = None,
) -> str:
    """Genera una nube de palabras a partir de los textos ya preprocesados (sin stopwords).

    - texts puede ser:
      - Iterable[str] / Series: colección de textos preprocesados
      - DataFrame + text_col: se usará esa columna (e.g., 'abstract_clean')
    - Se cuentan tokens separando por espacios; se filtran tokens cortos (min_token_len)
    - Toma los top_n más frecuentes y delega en plot_wordcloud
    """
    from collections import Counter

    # Extraer secuencia de strings de entrada
    if isinstance(texts, pd.DataFrame):
        if not text_col:
            raise ValueError("Si 'texts' es un DataFrame, debes indicar 'text_col'.")
        series = texts[text_col].fillna("").astype(str)
    elif isinstance(texts, pd.Series):
        series = texts.fillna("").astype(str)
    else:
        series = pd.Series(list(texts), dtype=str)

    # Preparar stopwords (intenta NLTK; fallback básico)
    sw: set[str] = set()
    if remove_stopwords:
        try:
            from nltk.corpus import stopwords  # type: ignore

            for lang in languages:
                try:
                    sw.update(w.lower() for w in stopwords.words(lang))
                except Exception:
                    pass
        except Exception:
            # Fallback mínimo si NLTK no está disponible
            sw.update({
                "the","in","to","and","for","of","on","a","an","is","are","with","by","from","at","this","that",
                "de","la","el","y","en","para","que","los","las","con","por","del","un","una"})
    if extra_stopwords:
        sw.update([str(w).lower() for w in extra_stopwords])

    # Contar tokens (se asume preprocesamiento previo)
    cnt: Counter[str] = Counter()
    for t in series:
        s = t.lower() if lowercase else str(t)
        # separados por espacio; ignorar tokens muy cortos
        tokens = [w for w in s.split() if len(w) >= min_token_len and (not remove_stopwords or w.lower() not in sw)]
        cnt.update(tokens)

    if top_n and top_n > 0:
        freqs = dict(cnt.most_common(top_n))
    else:
        freqs = dict(cnt)

    return plot_wordcloud(freqs, out_name=out_name)


