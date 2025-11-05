from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def strip_accents(text: str) -> str:
    """Elimina acentos/diacríticos usando unicodedata (sin dependencias extra)."""
    if not text:
        return text
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_text(
    s: str,
    lower: bool = True,
    remove_accents: bool = False,
    strip_spaces: bool = True,
) -> str:
    """Normalización básica: minúsculas, quitar acentos y espacios laterales."""
    if s is None:
        return ""
    out = s
    if strip_spaces:
        out = out.strip()
    if lower:
        out = out.lower()
    if remove_accents:
        out = strip_accents(out)
    return out


_RE_DIGITS = re.compile(r"\d+")


def clean_text(
    s: str,
    remove_punct: bool = True,
    remove_digits: bool = False,
    extra_punct: str | None = None,
) -> str:
    """Limpieza opcional de puntuación y dígitos.

    Nota: re estándar no conoce \\p{P}; usamos clases aproximadas.
    """
    if not s:
        return ""
    text = s
    if remove_punct:
        # Aproximación: reemplazar todo lo que no sea letra/dígito/espacio por espacio
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        if extra_punct:
            text = re.sub("[" + re.escape(extra_punct) + "]", " ", text)
    if remove_digits:
        text = _RE_DIGITS.sub(" ", text)
    # Normaliza espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ensure_nltk(resources: Sequence[str]) -> bool:
    """Intenta asegurar recursos NLTK; devuelve True si disponibles."""
    try:
        import nltk  # type: ignore

        ok = True
        for res in resources:
            try:
                nltk.data.find(res)
            except LookupError:
                try:
                    nltk.download(res.split("/")[-1], quiet=True)
                except Exception:
                    ok = False
        return ok
    except Exception:
        return False


def tokenize(s: str, lang: str = "en") -> List[str]:
    """Tokeniza con NLTK si está disponible; de lo contrario, fallback regex."""
    text = s or ""
    if _ensure_nltk(["tokenizers/punkt"]):
        try:
            from nltk import word_tokenize  # type: ignore

            return word_tokenize(text, language="spanish" if lang.startswith("es") else "english")
        except Exception:
            pass
    # Fallback: separar por no-alfanumérico
    return [t for t in re.split(r"\W+", text) if t]


def remove_stopwords(tokens: Iterable[str], lang: str = "en", extra: Iterable[str] | None = None) -> List[str]:
    """Elimina stopwords usando NLTK si se puede; acepta extras personalizados."""
    toks = list(tokens)
    sw: set[str] = set()
    if _ensure_nltk(["corpora/stopwords"]):
        try:
            from nltk.corpus import stopwords  # type: ignore

            sw = set(stopwords.words("spanish" if lang.startswith("es") else "english"))
        except Exception:
            sw = set()
    if extra:
        sw.update(normalize_text(w) for w in extra if w)
    return [t for t in toks if normalize_text(t) not in sw]


def lemmatize(tokens: Iterable[str], lang: str = "en") -> List[str]:
    """Lematiza (en) con WordNet; para es usa stemmer Snowball como fallback."""
    toks = list(tokens)
    if lang.startswith("en") and _ensure_nltk(["corpora/wordnet", "taggers/averaged_perceptron_tagger"]):
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
            from nltk.stem import WordNetLemmatizer  # type: ignore
            from nltk import pos_tag  # type: ignore

            def to_wn_pos(tag: str):
                if tag.startswith("J"):
                    return wn.ADJ
                if tag.startswith("V"):
                    return wn.VERB
                if tag.startswith("N"):
                    return wn.NOUN
                if tag.startswith("R"):
                    return wn.ADV
                return wn.NOUN

            ltz = WordNetLemmatizer()
            tagged = pos_tag(toks)
            return [ltz.lemmatize(w, to_wn_pos(p)) for w, p in tagged]
        except Exception:
            pass

    # Fallback para español u otros: stemmer Snowball si está disponible
    if _ensure_nltk([]):
        try:
            from nltk.stem.snowball import SnowballStemmer  # type: ignore

            stemmer = SnowballStemmer("spanish" if lang.startswith("es") else "english")
            return [stemmer.stem(w) for w in toks]
        except Exception:
            pass
    return toks


def build_ngrams(tokens: Sequence[str], n: int = 2, join_with: str = " ") -> List[str]:
    """Construye n-gramas simples a partir de tokens."""
    if n <= 1:
        return list(tokens)
    return [join_with.join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


@dataclass
class PreprocessConfig:
    lang: str = "en"
    lower: bool = True
    remove_accents: bool = False
    remove_punct: bool = True
    remove_digits: bool = False
    stopwords: bool = True
    lemmatization: bool = False
    ngrams: int | None = None  # 1,2,3...
    output: str = "tokens"  # "tokens" | "text"
    extra_stopwords: Iterable[str] | None = None


def preprocess_text(text: str, cfg: PreprocessConfig) -> List[str] | str:
    """Pipeline compacto de preprocesamiento para un texto."""
    t = normalize_text(text or "", lower=cfg.lower, remove_accents=cfg.remove_accents)
    t = clean_text(t, remove_punct=cfg.remove_punct, remove_digits=cfg.remove_digits)
    toks = tokenize(t, lang=cfg.lang)
    if cfg.stopwords:
        toks = remove_stopwords(toks, lang=cfg.lang, extra=cfg.extra_stopwords)
    if cfg.lemmatization:
        toks = lemmatize(toks, lang=cfg.lang)
    if cfg.ngrams and cfg.ngrams > 1:
        toks = build_ngrams(toks, n=cfg.ngrams)
    if cfg.output == "text":
        return " ".join(toks)
    return toks


def preprocess_corpus(texts: Iterable[str], cfg: PreprocessConfig) -> List[List[str]] | List[str]:
    """Aplica preprocess_text a una colección de textos."""
    return [preprocess_text(t, cfg) for t in texts]
