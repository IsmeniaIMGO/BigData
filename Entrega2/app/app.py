from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# Asegurar importaciones relativas al proyecto cuando se ejecuta como script
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[1]  # .../Entrega2
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _resolve_paths():
    from processing.utils.rutas import resolve_paths  # lazy import

    return resolve_paths()


def cmd_unify(args) -> dict:
    from processing.pipeline import PipelineConfig, run_unify

    cfg = PipelineConfig()
    return run_unify(cfg)


def cmd_nlp(args) -> str:
    from processing.pipeline import PipelineConfig, run_nlp_preprocess

    cfg = PipelineConfig()
    input_csv = args.input if args.input else None
    return run_nlp_preprocess(cfg, input_csv=input_csv)


def cmd_features(args) -> str:
    from processing.pipeline import PipelineConfig, run_feature_engineering

    cfg = PipelineConfig()
    processed_csv = args.input if args.input else None
    return run_feature_engineering(cfg, processed_csv=processed_csv)


def cmd_search(args) -> str:
    from processing.pipeline import PipelineConfig, run_search_build

    cfg = PipelineConfig()
    features_path = args.features if args.features else None
    processed_csv = args.input if args.input else None
    return run_search_build(cfg, features_path=features_path, processed_csv=processed_csv)


def cmd_clustering(args) -> str:
    from processing.pipeline import PipelineConfig, run_clustering

    cfg = PipelineConfig()
    features_path = args.features if args.features else None
    return run_clustering(cfg, features_path=features_path)


def cmd_evaluation(args) -> str:
    from processing.pipeline import PipelineConfig, run_evaluation

    cfg = PipelineConfig()
    features_path = args.features if args.features else None
    labels_path = args.labels if args.labels else None
    return run_evaluation(cfg, features_path=features_path, labels_path=labels_path)


def _safe_series(df, col: str):
    return df[col] if col in df.columns else None


def integrated_visualizations(processed_csv: Optional[str] = None, labels_path: Optional[str] = None) -> dict:
    """Genera un conjunto pequeño de visualizaciones robustas y guarda en outputs/visualization.

    Retorna dict con rutas de imágenes generadas (si existen las columnas requeridas).
    """
    from processing.utils.rutas import read_csv
    from visualization.plots import (
        plot_publications_per_year,
        plot_top_bars,
        plot_cluster_sizes,
        plot_wordcloud,
        plot_cooccurrence_network,
        plot_category_distribution_heatmap,
        plot_similarity_heatmap,
        plot_embeddings_scatter,
        plot_silhouette,
    )
    from analysis import eda
    from analysis.bibliometrics import count_frequencies, create_frequency_tables, build_cooccurrence_network
    from processing.clustering.evaluation.evaluation import evaluate_category_distribution

    paths = _resolve_paths()

    # Determinar CSV preprocesado por defecto
    if processed_csv is None:
        default_csv = paths["data_processed"] / "unified_cleaned_nlp.csv"
        if default_csv.exists():
            processed_csv = str(default_csv)
        else:
            # fallback: si no existe nlp, usar el limpio
            alt_csv = paths["data_processed"] / "unified_cleaned.csv"
            processed_csv = str(alt_csv) if alt_csv.exists() else None

    images: dict[str, str] = {}
    tables: dict[str, str] = {}
    if processed_csv and Path(processed_csv).exists():
        df = read_csv(processed_csv)

        # Publicaciones por año
        if "year" in df.columns:
            try:
                p = plot_publications_per_year(df, year_col="year", out_name="cli_publications_per_year.png")
                images["publications_per_year"] = p
                # tabla
                t = eda.publications_per_year(df, year_col="year", out_csv="pubs_per_year.csv")
                if t:
                    tables["pubs_per_year"] = t
            except Exception:
                pass

        # Top autores (si viene en una columna tipo 'author' separada por ; o ,)
        if "author" in df.columns:
            try:
                # Expandir autores por separadores comunes
                authors_series = df["author"].fillna("")
                authors_series = authors_series.str.replace(" and ", ";", regex=False)
                authors_series = authors_series.str.replace(",", ";", regex=False)
                exploded = authors_series.str.split(";").explode().str.strip()
                top_authors = exploded[exploded != ""].value_counts().head(20)
                p = plot_top_bars(top_authors, title="Top autores", out_name="cli_top_authors.png")
                images["top_authors"] = p
                # tabla
                df2 = df.copy()
                df2["author"] = authors_series
                t = eda.top_authors(df2, author_col="author", sep=";", top_n=100, out_csv="top_authors.csv")
                if t:
                    tables["top_authors"] = t
            except Exception:
                pass

        # N-grams (1 y 2)
        try:
            text_col = "abstract_clean" if "abstract_clean" in df.columns else ("abstract" if "abstract" in df.columns else None)
            if text_col:
                t1 = eda.text_ngrams(df, text_col=text_col, n=1, top_n=100, out_csv="top_1grams.csv")
                if t1:
                    tables["top_1grams"] = t1
                t2 = eda.text_ngrams(df, text_col=text_col, n=2, top_n=100, out_csv="top_2grams.csv")
                if t2:
                    tables["top_2grams"] = t2
        except Exception:
            pass

        # Categorías predefinidas (tablas)
        try:
            cc = eda.get_predefined_categories()
            tcat = eda.analyze_predefined_categories(df, text_col="abstract", categories=cc, out_csv="predefined_categories_counts.csv")
            if tcat:
                tables["predefined_categories_counts"] = tcat
            tflags = eda.add_category_flags(df, text_col="abstract", categories=cc, out_csv="predefined_categories_flags.csv")
            if tflags:
                tables["predefined_categories_flags"] = tflags
        except Exception:
            pass

    # Tamaños de cluster si hay labels
    if labels_path is None:
        default_labels = paths["outputs"] / "clustering" / "labels.csv"
        labels_path = str(default_labels) if default_labels.exists() else None

    # Clustering: tamaños, distribución por categorías, silueta; además heatmap similitud y proyección 2D si hay features
    if labels_path and Path(labels_path).exists():
        try:
            import csv

            labels = []
            with open(labels_path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if row:
                        labels.append(int(float(row[0])))
            p = plot_cluster_sizes(labels, out_name="cli_cluster_sizes.png")
            images["cluster_sizes"] = p

            # Distribución por categorías por cluster
            try:
                cats = eda.get_predefined_categories()
                text_col = "abstract" if "abstract" in df.columns else ("abstract_clean" if "abstract_clean" in df.columns else None)
                if text_col:
                    distrib = evaluate_category_distribution(df[text_col].fillna("").astype(str).tolist(), labels, cats)
                    cat_list = list(cats.keys())
                    p2 = plot_category_distribution_heatmap(distrib, cat_list, algorithm_key="cli", algorithm_name="Pipeline", out_name="cli_category_distribution.png")
                    images["category_distribution"] = p2
                    # guardar tabla distribución
                    from processing.utils.rutas import save_csv
                    import pandas as pd
                    rows = []
                    for cid, m in distrib.items():
                        for cat, val in m.items():
                            rows.append({"cluster": cid, "category": cat, "ratio": float(val)})
                    import pathlib
                    paths2 = _resolve_paths()
                    out_csv = paths2["outputs"] / "analysis" / "category_distribution.csv"
                    out_csv.parent.mkdir(parents=True, exist_ok=True)
                    save_csv(pd.DataFrame(rows), out_csv)
                    tables["category_distribution"] = str(out_csv)
            except Exception:
                pass

            # Silueta y similares (si hay features)
            try:
                from processing.features.features import load_features
                paths2 = _resolve_paths()
                fpath = paths2["data_processed"] / "features.npz"
                if fpath.exists():
                    X = load_features(str(fpath))
                    # densificar si es sparse
                    Xd = X.toarray() if hasattr(X, "toarray") else X
                    # Silhouette (submuestreo para estabilidad)
                    import numpy as np
                    n = len(labels)
                    idx = np.arange(n)
                    if n > 2000:
                        rng = np.random.default_rng(42)
                        idx = np.sort(rng.choice(n, size=2000, replace=False))
                    Xs = Xd[idx]
                    ls = np.asarray(labels)[idx]
                    p3 = plot_silhouette(Xs, ls, metric="cosine", out_name="cli_silhouette.png")
                    images["silhouette"] = p3
                    # Heatmap de similitud (submuestreo)
                    try:
                        from sklearn.metrics.pairwise import cosine_similarity
                        import numpy as np
                        m = len(idx)
                        if m > 300:
                            sel = np.linspace(0, m - 1, num=300, dtype=int)
                            Xh = Xs[sel]
                        else:
                            Xh = Xs
                        S = cosine_similarity(Xh)
                        p4 = plot_similarity_heatmap(S, labels=None, out_name="cli_similarity_heatmap.png", max_size=100)
                        images["similarity_heatmap"] = p4
                    except Exception:
                        pass
                    # Proyección 2D con UMAP si está disponible
                    try:
                        import umap
                        import numpy as np
                        m = Xd.shape[0]
                        if m > 2000:
                            rng = np.random.default_rng(42)
                            idx2 = np.sort(rng.choice(m, size=2000, replace=False))
                            Xu = Xd[idx2]
                            lu = np.asarray(labels)[idx2]
                        else:
                            Xu = Xd
                            lu = np.asarray(labels)
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        X2 = reducer.fit_transform(Xu)
                        p5 = plot_embeddings_scatter(X2, labels=lu, out_name="cli_embeddings_scatter.png", title="UMAP 2D")
                        images["embeddings_scatter"] = p5
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    # Bibliometrics: wordcloud y co-ocurrencias
    try:
        if processed_csv and Path(processed_csv).exists():
            df = read_csv(processed_csv)
            cats = eda.get_predefined_categories()
            # frecuencias
            freq_by_cat, total_counter = count_frequencies(df, categories=cats, text_col="abstract")
            # guardar tablas por categoría
            tables_by_cat = create_frequency_tables(freq_by_cat)
            from processing.utils.rutas import save_csv
            import pandas as pd
            out_base = _resolve_paths()["outputs"] / "analysis"
            out_base.mkdir(parents=True, exist_ok=True)
            for cat, dfcat in tables_by_cat.items():
                pth = out_base / f"freq_{cat.replace(' ', '_')}.csv"
                save_csv(pd.DataFrame(dfcat), pth)
                tables[f"freq_{cat}"] = str(pth)
            # wordcloud global
            if total_counter:
                freqs = dict(total_counter.most_common(200))
                p = plot_wordcloud(freqs, out_name="cli_wordcloud.png")
                images["wordcloud"] = p
            # wordcloud por tokens más frecuentes en abstract_clean (si existe)
            try:
                if "abstract_clean" in df.columns:
                    from visualization.plots import plot_wordcloud_top_words
                    p2 = plot_wordcloud_top_words(df, text_col="abstract_clean", top_n=200, out_name="cli_wordcloud_top_words.png")
                    images["wordcloud_top_words"] = p2
            except Exception:
                pass
            # co-ocurrencias
            edges = build_cooccurrence_network(df, categories=cats, text_col="abstract")
            if edges:
                p = plot_cooccurrence_network(edges, out_name="cli_co_word_network.png", min_weight=2)
                images["co_word_network"] = p
    except Exception:
        pass

    # Benchmarks de ordenamiento (sorting): datasets sintéticos + varias implementaciones
    try:
        from processing.sorting.benchmark import run_benchmarks
        from visualization.plots import plot_sorting_benchmarks
        # Algoritmos disponibles (subset para tiempo razonable)
        algs = {}
        try:
            from processing.sorting.algorithms.bubble_sort import bubble_sort
            algs["bubble_sort"] = bubble_sort
        except Exception:
            pass
        try:
            from processing.sorting.algorithms.selection_sort import selection_sort
            algs["selection_sort"] = selection_sort
        except Exception:
            pass
        try:
            from processing.sorting.algorithms.binary_insertion_sort import binary_insertion_sort
            algs["binary_insertion_sort"] = binary_insertion_sort
        except Exception:
            pass
        try:
            from processing.sorting.algorithms.heap_sort import heap_sort
            algs["heap_sort"] = heap_sort
        except Exception:
            pass
        try:
            from processing.sorting.algorithms.quick_sort import quick_sort
            algs["quick_sort"] = quick_sort
        except Exception:
            pass
        try:
            from processing.sorting.algorithms.tim_sort import tim_sort
            algs["tim_sort"] = tim_sort
        except Exception:
            pass

        # Si no hay ningún algoritmo importado, saltar
        if algs:
            import random
            random.seed(42)
            n = 800  # tamaño moderado para evitar tiempos excesivos con O(n^2)
            base = [random.randint(0, 10_000) for _ in range(n)]
            datasets = {
                "aleatorio": base,
                "ordenado": sorted(base),
                "invertido": sorted(base, reverse=True),
                "casi_ordenado": sorted(base)[:],
                "duplicados": [random.choice(range(100)) for _ in range(n)],
            }
            # introducir pequeñas perturbaciones en casi_ordenado
            arr = datasets["casi_ordenado"]
            for _ in range(max(1, n // 100)):
                i, j = random.randrange(n), random.randrange(n)
                arr[i], arr[j] = arr[j], arr[i]
            datasets["casi_ordenado"] = arr

            resultados = run_benchmarks(algs, datasets, unit="ms", repeat=3)

            # Guardar tabla resumen
            try:
                from processing.utils.rutas import save_csv
                import pandas as pd
                rows = []
                for cat, d in resultados.items():
                    for alg, r in d.items():
                        rows.append({
                            "dataset": cat,
                            "algorithm": alg,
                            "mean": float(r.get("mean", 0.0) or 0.0),
                            "std": float(r.get("std", 0.0) or 0.0),
                            "unit": r.get("unit", "ms"),
                            "error": r.get("error"),
                        })
                out_csv = _resolve_paths()["outputs"] / "analysis" / "sorting_benchmarks.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                save_csv(pd.DataFrame(rows), out_csv)
                tables["sorting_benchmarks"] = str(out_csv)
            except Exception:
                pass

            # Gráficas
            try:
                paths_sb = plot_sorting_benchmarks(resultados, categorias=datasets, out_prefix="sorting_benchmark")
                # añadir cada imagen individualmente para el reporte
                for i, pth in enumerate(paths_sb):
                    images[f"sorting_benchmark_{i+1}"] = pth
            except Exception:
                pass
    except Exception:
        pass

    return {"images": images, "tables": tables}


def generate_report(artifacts: dict, images: dict, tables: Optional[dict] = None, out_path: Optional[str] = None) -> str:
    from processing.utils.export import write_report_md

    paths = _resolve_paths()
    report_dir = paths["outputs"] / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = Path(out_path) if out_path else (report_dir / "pipeline_report.md")

    # Artefactos en formato lista
    artifacts_lines = []
    for k, v in artifacts.items():
        if v:
            artifacts_lines.append(f"- {k}: {v}")
    artifacts_text = "\n".join(artifacts_lines) if artifacts_lines else "(sin artefactos)"

    # Cargar métricas si existen
    metrics_text = ""
    metrics_path = artifacts.get("metrics")
    if metrics_path and Path(metrics_path).exists():
        try:
            data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            metrics_text = "\n".join([f"- {k}: {v}" for k, v in data.items()])
        except Exception:
            metrics_text = "(no se pudieron leer las métricas)"
    else:
        metrics_text = "(no disponibles)"

    # Sección de visualizaciones
    viz_lines: list[str] = []
    for name, path in images.items():
        if path and Path(path).exists():
            rel = Path(path).relative_to(paths["outputs"]).as_posix()
            viz_lines.append(f"### {name}\n\n![{name}](../{rel})\n")
    viz_text = "\n".join(viz_lines) if viz_lines else "(no generadas)"

    sections = {
        "Artefactos": artifacts_text,
        "Métricas de clustering": metrics_text,
        "Visualizaciones": viz_text,
    }
    if tables:
        tbl_lines = [f"- {k}: {v}" for k, v in tables.items()]
        sections["Tablas generadas"] = "\n".join(tbl_lines) if tbl_lines else "(no generadas)"

    write_report_md(sections, out_md, title="Reporte de Pipeline")
    return str(out_md)


def cmd_visualize(args) -> dict:
    res = integrated_visualizations(processed_csv=args.input, labels_path=args.labels)
    return res


def cmd_report(args) -> str:
    # Descubrir artefactos comunes si no son provistos
    paths = _resolve_paths()
    artifacts = {
        "unify_csv": str(paths["data_processed"] / "unified_cleaned.csv"),
        "nlp_csv": str(paths["data_processed"] / "unified_cleaned_nlp.csv"),
        "features": str(paths["data_processed"] / "features.npz"),
        "index": str(paths["outputs"] / "search" / "index.pkl"),
        "labels": str(paths["outputs"] / "clustering" / "labels.csv"),
        "metrics": str(paths["outputs"] / "clustering" / "metrics.json"),
    }
    res = integrated_visualizations(processed_csv=args.input, labels_path=args.labels)
    images = res.get("images", {})
    tables = res.get("tables", {})
    return generate_report(artifacts, images, tables=tables, out_path=args.out)


def cmd_all(args) -> dict:
    from processing.pipeline import PipelineConfig, run_all

    cfg = PipelineConfig()
    summary = run_all(cfg)
    # Visualizaciones y reporte
    res = integrated_visualizations(processed_csv=summary.get("nlp_csv"), labels_path=summary.get("labels"))
    report = generate_report(summary, res.get("images", {}), tables=res.get("tables", {}))
    summary["report"] = report
    summary.update(res)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI para orquestar el pipeline de Entrega2")
    sub = p.add_subparsers(dest="command", required=True)

    # unify
    sub.add_parser("unify", help="Unificar y limpiar BibTeX → CSV")

    # nlp
    p_nlp = sub.add_parser("nlp", help="Preprocesamiento NLP sobre CSV")
    p_nlp.add_argument("--input", type=str, help="Ruta al CSV de entrada (opcional)")

    # features
    p_feat = sub.add_parser("features", help="Ingeniería de características desde CSV preprocesado")
    p_feat.add_argument("--input", type=str, help="Ruta al CSV preprocesado (opcional)")

    # search
    p_search = sub.add_parser("search", help="Construcción de índice de búsqueda (BM25/ANN)")
    p_search.add_argument("--features", type=str, help="Ruta al .npz de features (opcional)")
    p_search.add_argument("--input", type=str, help="Ruta al CSV preprocesado para BM25 (opcional)")

    # clustering
    p_clu = sub.add_parser("clustering", help="Clustering desde features")
    p_clu.add_argument("--features", type=str, help="Ruta al .npz de features (opcional)")

    # evaluation
    p_eval = sub.add_parser("evaluation", help="Evaluación de clustering")
    p_eval.add_argument("--features", type=str, help="Ruta al .npz de features (opcional)")
    p_eval.add_argument("--labels", type=str, help="Ruta al archivo de etiquetas (CSV/NPY, opcional)")

    # visualize
    p_viz = sub.add_parser("visualize", help="Genera visualizaciones clave")
    p_viz.add_argument("--input", type=str, help="Ruta al CSV (preprocesado preferible)")
    p_viz.add_argument("--labels", type=str, help="Ruta a etiquetas para tamaños de cluster")

    # report
    p_rep = sub.add_parser("report", help="Genera reporte Markdown con artefactos y visualizaciones")
    p_rep.add_argument("--input", type=str, help="Ruta al CSV (para graficar)")
    p_rep.add_argument("--labels", type=str, help="Ruta a etiquetas (para graficar clusters)")
    p_rep.add_argument("--out", type=str, help="Ruta del archivo Markdown de salida")

    # all
    sub.add_parser("all", help="Ejecuta todo el pipeline + visualizaciones + reporte")

    return p


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "unify":
        res = cmd_unify(args)
    elif args.command == "nlp":
        res = cmd_nlp(args)
    elif args.command == "features":
        res = cmd_features(args)
    elif args.command == "search":
        res = cmd_search(args)
    elif args.command == "clustering":
        res = cmd_clustering(args)
    elif args.command == "evaluation":
        res = cmd_evaluation(args)
    elif args.command == "visualize":
        res = cmd_visualize(args)
    elif args.command == "report":
        res = cmd_report(args)
    elif args.command == "all":
        res = cmd_all(args)
    else:
        parser.error("Comando no soportado")
        return 1

    # Imprimir JSON del resultado para consumo programático
    try:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except TypeError:
        print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
