from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from processing.utils.rutas import resolve_paths, save_csv
from processing.nlp.preprocess import normalize_text as _norm_text, clean_text as _clean_text


def _out_path(name: str, subdir: str = "analysis") -> Path:
    paths = resolve_paths()
    out_dir = paths["outputs"] / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def _normalize_synonyms(synonyms: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if not synonyms:
        return {}
    out: Dict[str, str] = {}
    for k, v in synonyms.items():
        nk = _norm_text(k or "", lower=True, remove_accents=True)
        nv = _norm_text(v or "", lower=True, remove_accents=True)
        out[nk] = nv
    return out


def normalize_term(term: str, synonyms: Optional[Mapping[str, str]] = None) -> str:
    base = _norm_text((term or ""), lower=True, remove_accents=True).strip()
    syn = _normalize_synonyms(synonyms)
    return syn.get(base, base)


def preprocess_text(text: str) -> str:
    """Normalización basada en processing.nlp.preprocess (sin stopwords/lemmatización).

    - lower=True
    - remove_accents=False (ajústalo si quieres)
    - remove_punct=True
    - remove_digits=False
    """
    t = _norm_text(text or "", lower=True, remove_accents=False)
    t = _clean_text(t, remove_punct=True, remove_digits=False)
    return t


def count_frequencies(
    df: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    text_col: str = "abstract",
    synonyms: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Counter], Counter]:
    """Cuenta ocurrencias de términos por categoría en el texto."""
    freq_by_category: Dict[str, Counter] = {cat: Counter() for cat in categories}
    total_counter: Counter = Counter()

    if text_col not in df.columns:
        texts = [""] * len(df)
    else:
        texts = df[text_col].fillna("").astype(str).tolist()

    for raw in texts:
        text = preprocess_text(raw)
        for cat, terms in categories.items():
            for term in terms:
                t = normalize_term(term, synonyms)
                # match exacto de palabra con límites
                pattern = r"\b" + re.escape(t) + r"\b"
                matches = re.findall(pattern, text)
                c = len(matches)
                if c > 0:
                    freq_by_category[cat][t] += c
                    total_counter[t] += c
    return freq_by_category, total_counter


def create_frequency_tables(freq_by_category: Mapping[str, Counter]) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for cat, counter in freq_by_category.items():
        df_cat = pd.DataFrame(counter.items(), columns=["Variable", "Frequency"])\
            .sort_values(by="Frequency", ascending=False).reset_index(drop=True)
        dfs[cat] = df_cat
    return dfs


def build_cooccurrence_network(
    df: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    text_col: str = "abstract",
    synonyms: Optional[Mapping[str, str]] = None,
) -> List[Tuple[str, str, int]]:
    """Construye lista de aristas (u, v, weight) de co-ocurrencias entre términos."""
    # vocab unificado
    vocab: List[str] = []
    for terms in categories.values():
        vocab.extend([normalize_term(t, synonyms) for t in terms])
    vocab = sorted(set(vocab))

    if text_col not in df.columns:
        texts = [""] * len(df)
    else:
        texts = df[text_col].fillna("").astype(str).tolist()

    # co-ocurrencia en documento
    co = defaultdict(lambda: defaultdict(int))
    for raw in texts:
        text = preprocess_text(raw)
        present: List[str] = []
        for t in vocab:
            if re.search(r"\b" + re.escape(t) + r"\b", text):
                present.append(t)
        present = sorted(set(present))
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                u, v = present[i], present[j]
                co[u][v] += 1
                co[v][u] += 1

    edges: List[Tuple[str, str, int]] = []
    for u, nbrs in co.items():
        for v, w in nbrs.items():
            if u < v and w > 0:
                edges.append((u, v, w))
    return edges
