from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from processing.utils.rutas import resolve_paths, save_csv


def _out_path(name: str, subdir: str = "analysis") -> Path:
    paths = resolve_paths()
    out_dir = paths["outputs"] / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def overview(df: pd.DataFrame) -> Dict[str, object]:
    """Estadísticas rápidas del dataset."""
    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "nulls_total": int(df.isna().sum().sum()),
        "nulls_by_col": df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicates_rows": int(df.duplicated().sum()),
    }
    # Guardar resumen JSON
    p = _out_path("overview.json")
    p.write_text(pd.Series(info).to_json(ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def publications_per_year(df: pd.DataFrame, year_col: str = "year", out_csv: str = "pubs_per_year.csv") -> str:
    if year_col not in df.columns:
        return ""
    counts = df[year_col].dropna().astype(int).value_counts().sort_index()
    p = _out_path(out_csv)
    save_csv(counts.rename("count").reset_index(names=[year_col]), p)
    return str(p)


def top_authors(df: pd.DataFrame, author_col: str = "author", sep: str = ";", top_n: int = 50, out_csv: str = "top_authors.csv") -> str:
    if author_col not in df.columns:
        return ""
    authors: List[str] = []
    for a in df[author_col].dropna().astype(str):
        authors.extend([x.strip() for x in a.split(sep) if x.strip()])
    counts = pd.Series(Counter(authors)).sort_values(ascending=False).head(top_n)
    p = _out_path(out_csv)
    save_csv(counts.rename("count").reset_index(names=["author"]), p)
    return str(p)


def text_ngrams(
    df: pd.DataFrame,
    text_col: str = "abstract_clean",
    n: int = 1,
    top_n: int = 50,
    out_csv: Optional[str] = None,
) -> str:
    if text_col not in df.columns:
        return ""

    def _ngrams(tokens: List[str], n: int) -> Iterable[str]:
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])

    counter: Counter = Counter()
    for text in df[text_col].fillna(""):
        tokens = str(text).split()
        counter.update(_ngrams(tokens, n))

    counts = pd.Series(counter).sort_values(ascending=False).head(top_n)
    out_csv = out_csv or f"top_{n}grams.csv"
    p = _out_path(out_csv)
    save_csv(counts.rename("count").reset_index(names=["ngram"]), p)
    return str(p)


def cluster_summary(labels: Sequence[int], out_csv: str = "cluster_sizes.csv") -> str:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"cluster": unique, "count": counts}).sort_values("count", ascending=False)
    p = _out_path(out_csv)
    save_csv(df, p)
    return str(p)


def write_report(sections: Dict[str, str], out_name: str = "report.md", title: str = "Análisis") -> str:
    from processing.utils.export import write_report_md

    p = _out_path(out_name)
    write_report_md(sections, p, title=title)
    return str(p)


# =============================
# Categorías predefinidas
# =============================


def get_predefined_categories() -> Dict[str, list[str]]:
    """Devuelve categorías temáticas con sinónimos/variantes para búsqueda.

    Se usan términos en minúsculas y búsqueda con límites de palabra.
    """
    return {
        "IOT": ["iot", "internet of things"],
        "Big Data (BDT)": ["big data", "bdt", "big-data"],
        "Cybersecurity": ["cybersecurity", "cyber security", "information security", "infosec"],
        "Blockchain": ["blockchain", "block chain"],
        "AI": ["ai", "artificial intelligence"],
        "Machine Learning (ML)": ["machine learning", "ml"],
        "Intrusion Detection Systems (IDS)": ["intrusion detection system", "intrusion detection systems", "ids"],
        "Intrusion Prevention System (IPS)": ["intrusion prevention system", "ips"],
        "Algorithms": ["algorithm", "algorithms"],
    }


def _normalize_text_for_match(s: str) -> str:
    s = (s or "").lower()
    # normalizar espacios
    s = re.sub(r"\s+", " ", s)
    return s


def analyze_predefined_categories(
    df: pd.DataFrame,
    text_col: str = "abstract",
    categories: Dict[str, list[str]] | None = None,
    out_csv: str = "predefined_categories_counts.csv",
) -> str:
    """Cuenta ocurrencias y nº de documentos con match por categoría.

    Crea un CSV con columnas: category, total_hits, docs_with_hits.
    """
    if text_col not in df.columns:
        # crear columna vacía para no fallar
        texts = pd.Series(["" for _ in range(len(df))])
    else:
        texts = df[text_col].fillna("").astype(str)

    categories = categories or get_predefined_categories()
    patterns: Dict[str, list[re.Pattern]] = {}
    for cat, terms in categories.items():
        patterns[cat] = [re.compile(rf"\\b{re.escape(t.lower())}\\b", flags=re.IGNORECASE) for t in terms]

    total_hits: Dict[str, int] = {cat: 0 for cat in categories}
    docs_with_hits: Dict[str, int] = {cat: 0 for cat in categories}

    for text in texts:
        norm = _normalize_text_for_match(text)
        for cat, pats in patterns.items():
            cat_hits = 0
            for ptn in pats:
                cat_hits += len(ptn.findall(norm))
            total_hits[cat] += cat_hits
            if cat_hits > 0:
                docs_with_hits[cat] += 1

    out = pd.DataFrame(
        {
            "category": list(categories.keys()),
            "total_hits": [total_hits[c] for c in categories],
            "docs_with_hits": [docs_with_hits[c] for c in categories],
        }
    ).sort_values("total_hits", ascending=False)

    p = _out_path(out_csv)
    save_csv(out, p)
    return str(p)


def add_category_flags(
    df: pd.DataFrame,
    text_col: str = "abstract",
    categories: Dict[str, list[str]] | None = None,
    out_csv: str = "predefined_categories_flags.csv",
) -> str:
    """Crea columnas booleanas por categoría indicando presencia en cada documento.

    Devuelve la ruta al CSV guardado con las nuevas columnas.
    """
    categories = categories or get_predefined_categories()

    if text_col not in df.columns:
        df = df.copy()
        df[text_col] = ""

    patterns: Dict[str, list[re.Pattern]] = {
        cat: [re.compile(rf"\\b{re.escape(t.lower())}\\b", flags=re.IGNORECASE) for t in terms]
        for cat, terms in categories.items()
    }

    def has_cat(text: str, pats: list[re.Pattern]) -> bool:
        norm = _normalize_text_for_match(text)
        return any(p.search(norm) for p in pats)

    out_df = df.copy()
    for cat, pats in patterns.items():
        col = f"cat_{re.sub(r'[^a-z0-9]+', '_', cat.lower()).strip('_')}"
        out_df[col] = out_df[text_col].fillna("").astype(str).map(lambda s: has_cat(s, pats))

    p = _out_path(out_csv)
    save_csv(out_df, p)
    return str(p)
