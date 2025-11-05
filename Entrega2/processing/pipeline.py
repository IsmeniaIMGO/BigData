from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


# Asegurar que podamos importar 'processing.*' ejecutando este archivo directamente
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../Entrega2
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Utilidades y módulos del proyecto
from processing.utils.rutas import resolve_paths, read_csv, save_csv  # type: ignore
# Nota: para evitar importaciones pesadas al cargar el módulo, los imports
# de módulos con dependencias externas se realizan dentro de cada función.


# =============================
# Configs
# =============================


@dataclass
class UnifyCfg:
    export_csv: bool = True
    csv_fields: List[str] = field(default_factory=lambda: ["title", "author", "year", "doi", "abstract"])
    cleaned_name: str = "unified_cleaned.bib"
    dups_name: str = "duplicates.bib"
    csv_name: str = "unified_cleaned.csv"


@dataclass
class NLPCfg:
    columns: List[str] = field(default_factory=lambda: ["abstract"])  # columnas a procesar si existen
    lang: str = "es"
    lower: bool = True
    remove_accents: bool = False
    remove_punct: bool = True
    remove_digits: bool = False
    stopwords: bool = True
    lemmatization: bool = False
    ngrams: int | None = None
    output: str = "text"
    out_suffix: str = "_clean"


@dataclass
class FeaturesCfg:
    kind: str = "tfidf"  # 'tfidf' | 'bow' | 'embeddings'
    text_column: str = "abstract_clean"
    max_features: int | None = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    reduce: bool = True
    reduce_method: str = "svd"  # 'svd' | 'umap' | 'tsne'
    reduce_dim: int = 100
    out_name: str = "features.npz"


@dataclass
class SearchCfg:
    kind: str = "bm25"  # 'bm25' | 'ann'
    method: str = "brute"  # para ann: 'brute'|'annoy'|'faiss'
    text_column: str = "abstract_clean"  # para bm25
    out_name: str = "index.pkl"


@dataclass
class ClusteringCfg:
    algorithm: str = "kmeans"  # 'kmeans'|'dbscan'|'hdbscan'
    k: int = 10
    eps: float = 0.5
    min_samples: int = 5
    min_cluster_size: int = 15
    scaler: str = "standard"
    out_name: str = "labels.csv"


@dataclass
class EvalCfg:
    metric: str = "cosine"
    sample_size: int | None = None
    out_name: str = "metrics.json"


@dataclass
class PipelineConfig:
    unify: UnifyCfg = field(default_factory=UnifyCfg)
    nlp: NLPCfg = field(default_factory=NLPCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    clustering: ClusteringCfg = field(default_factory=ClusteringCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


# =============================
# Orquestación
# =============================


def run_unify(cfg: PipelineConfig) -> dict:
    paths = resolve_paths()
    raw_dir = paths["data_raw"]
    out_dir = paths["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports locales para evitar dependencias innecesarias al cargar el módulo
    from processing.unify.unifyBibtext import (
        load_bibtex_files,
        detect_duplicates,
        clean_entries,
        save_bibtex_file,
        extract_fields_to_csv,
    )

    # Carga de BibTeX
    all_entries = load_bibtex_files([str(raw_dir)])
    unique_entries, duplicate_entries = detect_duplicates(all_entries)

    # Limpieza y guardado .bib
    cleaned_entries = clean_entries(unique_entries)
    cleaned_duplicates = clean_entries(duplicate_entries)

    out_cleaned = out_dir / cfg.unify.cleaned_name
    out_dups = out_dir / cfg.unify.dups_name
    save_bibtex_file(cleaned_entries, out_cleaned)
    save_bibtex_file(cleaned_duplicates, out_dups)

    out_csv = None
    if cfg.unify.export_csv:
        out_csv = out_dir / cfg.unify.csv_name
        extract_fields_to_csv(cleaned_entries, out_csv, cfg.unify.csv_fields)

    return {
        "cleaned_bib": str(out_cleaned),
        "duplicates_bib": str(out_dups),
        "cleaned_csv": str(out_csv) if out_csv else None,
    }


def run_nlp_preprocess(cfg: PipelineConfig, input_csv: str | Path | None = None) -> str:
    paths = resolve_paths()
    csv_path = Path(input_csv) if input_csv else (paths["data_processed"] / cfg.unify.csv_name)
    df = read_csv(csv_path)

    from processing.nlp.preprocess import PreprocessConfig, preprocess_corpus  # type: ignore

    pcfg = PreprocessConfig(
        lang=cfg.nlp.lang,
        lower=cfg.nlp.lower,
        remove_accents=cfg.nlp.remove_accents,
        remove_punct=cfg.nlp.remove_punct,
        remove_digits=cfg.nlp.remove_digits,
        stopwords=cfg.nlp.stopwords,
        lemmatization=cfg.nlp.lemmatization,
        ngrams=cfg.nlp.ngrams,
        output=cfg.nlp.output,
    )

    for col in cfg.nlp.columns:
        if col in df.columns:
            df[f"{col}{cfg.nlp.out_suffix}"] = preprocess_corpus(df[col].fillna(""), pcfg)

    out_path = paths["data_processed"] / (csv_path.stem + "_nlp.csv")
    save_csv(df, out_path)
    return str(out_path)


def run_feature_engineering(cfg: PipelineConfig, processed_csv: str | Path | None = None) -> str:
    paths = resolve_paths()
    csv_path = Path(processed_csv) if processed_csv else (paths["data_processed"] / (cfg.unify.csv_name.replace(".csv", "_nlp.csv")))
    df = read_csv(csv_path)

    from processing.features.features import (
        vectorize_bow,
        vectorize_tfidf,
        embed_texts,
        reduce_dimensionality,
        save_features,
    )

    if cfg.features.text_column not in df.columns:
        raise ValueError(f"Columna de texto no encontrada: {cfg.features.text_column}")

    texts = df[cfg.features.text_column].fillna("")

    if cfg.features.kind == "bow":
        X, vocab, vec = vectorize_bow(texts, max_features=cfg.features.max_features, ngram_range=cfg.features.ngram_range)
    elif cfg.features.kind == "tfidf":
        X, vocab, vec = vectorize_tfidf(texts, max_features=cfg.features.max_features, ngram_range=cfg.features.ngram_range)
    elif cfg.features.kind == "embeddings":
        X = embed_texts(texts)
    else:
        raise ValueError("features.kind debe ser 'bow', 'tfidf' o 'embeddings'")

    # Reducción opcional
    if cfg.features.reduce:
        Xr = reduce_dimensionality(X, method=cfg.features.reduce_method, dim=cfg.features.reduce_dim)
    else:
        Xr = X

    out_path = paths["data_processed"] / cfg.features.out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_features(Xr, str(out_path))
    return str(out_path)


def run_search_build(cfg: PipelineConfig, features_path: str | Path | None = None, processed_csv: str | Path | None = None) -> str:
    paths = resolve_paths()
    out_dir = paths["outputs"] / "search"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.search.kind == "bm25":
        csv_path = Path(processed_csv) if processed_csv else (paths["data_processed"] / (cfg.unify.csv_name.replace(".csv", "_nlp.csv")))
        df = read_csv(csv_path)
        if cfg.search.text_column not in df.columns:
            raise ValueError(f"Columna no encontrada para BM25: {cfg.search.text_column}")
        from processing.sorting.search.search import build_inverted_index  # type: ignore

        index = build_inverted_index(df[cfg.search.text_column].fillna("").tolist())
        out_index = out_dir / cfg.search.out_name
        with open(out_index, "wb") as f:
            pickle.dump(index, f)
        return str(out_index)

    if cfg.search.kind == "ann":
        from processing.features.features import load_features  # type: ignore
        from processing.sorting.search.search import build_ann_index  # type: ignore

        fpath = Path(features_path) if features_path else (paths["data_processed"] / cfg.features.out_name)
        X = load_features(str(fpath))
        # Si es sparse, conviértelo a denso para ANN
        if hasattr(X, "toarray"):
            X = X.toarray()  # type: ignore
        index = build_ann_index(np.asarray(X), method=cfg.search.method, metric="cosine")
        out_index = out_dir / cfg.search.out_name
        with open(out_index, "wb") as f:
            pickle.dump(index, f)
        return str(out_index)

    raise ValueError("search.kind debe ser 'bm25' o 'ann'")


def run_clustering(cfg: PipelineConfig, features_path: str | Path | None = None) -> str:
    paths = resolve_paths()
    from processing.features.features import load_features  # type: ignore
    from processing.clustering.algorithms.clustering import (  # type: ignore
        prepare_features_for_clustering,
        run_kmeans,
        run_dbscan,
        run_hdbscan,
        save_labels,
    )

    fpath = Path(features_path) if features_path else (paths["data_processed"] / cfg.features.out_name)
    X = load_features(str(fpath))

    Xprep = prepare_features_for_clustering(X, scaler=cfg.clustering.scaler)

    algo = cfg.clustering.algorithm.lower()
    if algo == "kmeans":
        labels = run_kmeans(Xprep, k=cfg.clustering.k)
    elif algo == "dbscan":
        labels = run_dbscan(Xprep, eps=cfg.clustering.eps, min_samples=cfg.clustering.min_samples)
    elif algo == "hdbscan":
        labels = run_hdbscan(Xprep, min_cluster_size=cfg.clustering.min_cluster_size)
    else:
        raise ValueError("clustering.algorithm debe ser 'kmeans', 'dbscan' o 'hdbscan'")

    out_path = (paths["outputs"] / "clustering" / cfg.clustering.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_labels(labels, str(out_path))
    return str(out_path)


def run_evaluation(cfg: PipelineConfig, features_path: str | Path | None = None, labels_path: str | Path | None = None) -> str:
    paths = resolve_paths()
    from processing.features.features import load_features  # type: ignore
    from processing.clustering.evaluation.evaluation import evaluate_internal_metrics  # type: ignore

    fpath = Path(features_path) if features_path else (paths["data_processed"] / cfg.features.out_name)
    X = load_features(str(fpath))

    # Cargar labels
    lpath = Path(labels_path) if labels_path else (paths["outputs"] / "clustering" / cfg.clustering.out_name)
    # Admite CSV o NPY
    labels: np.ndarray
    if lpath.suffix.lower() == ".csv":
        import csv

        vals: List[int] = []
        with open(lpath, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if row:
                    vals.append(int(float(row[0])))
        labels = np.asarray(vals)
    else:
        labels = np.load(lpath)

    metrics = evaluate_internal_metrics(X, labels, metric=cfg.eval.metric, sample_size=cfg.eval.sample_size)

    out_path = (paths["outputs"] / "clustering" / cfg.eval.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return str(out_path)


def run_all(cfg: PipelineConfig) -> dict:
    unify_paths = run_unify(cfg)
    nlp_csv = run_nlp_preprocess(cfg, input_csv=unify_paths.get("cleaned_csv"))
    features_npz = run_feature_engineering(cfg, processed_csv=nlp_csv)
    index_path = run_search_build(cfg, features_path=features_npz, processed_csv=nlp_csv)
    labels_path = run_clustering(cfg, features_path=features_npz)
    metrics_path = run_evaluation(cfg, features_path=features_npz, labels_path=labels_path)

    return {
        "unify": unify_paths,
        "nlp_csv": nlp_csv,
        "features": features_npz,
        "index": index_path,
        "labels": labels_path,
        "metrics": metrics_path,
    }


if __name__ == "__main__":
    # Ejecución con config por defecto
    cfg = PipelineConfig()
    summary = run_all(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
