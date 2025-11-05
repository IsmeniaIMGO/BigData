from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False
    word: Optional[str] = None  # guarda la palabra completa en nodos terminales


class PrefixTrie:
    """Trie de prefijos para buscar términos en texto bruto de forma eficiente.

    Incluye comprobación de límites de palabra (alfa-numéricos) para evitar matches parciales.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.word = word

    @classmethod
    def from_vocab(cls, words: Iterable[str]) -> "PrefixTrie":
        trie = cls()
        for w in words:
            w = w.strip().lower()
            if w:
                trie.insert(w)
        return trie

    def _is_word_boundary(self, text: str, pos: int) -> bool:
        # True si pos está fuera o el char no es alfanumérico
        if pos < 0 or pos >= len(text):
            return True
        return not text[pos].isalnum()

    def count_matches(self, text: str) -> Dict[str, int]:
        """Cuenta ocurrencias de todas las palabras del trie en el texto.

        Solo considera matches con límites de palabra en ambos extremos.
        """
        text = text.lower()
        counts: Dict[str, int] = defaultdict(int)
        n = len(text)
        for i in range(n):
            # inicio debe ser un límite de palabra
            if not self._is_word_boundary(text, i - 1):
                continue
            node = self.root
            j = i
            while j < n and (ch := text[j]) in node.children:
                node = node.children[ch]
                j += 1
                if node.is_end and self._is_word_boundary(text, j):
                    # match válido con límites de palabra
                    counts[node.word or ""] += 1
        return dict(counts)


_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def count_words_in_series(texts: Iterable[str], vocab: Optional[Iterable[str]] = None, use_trie: bool = True) -> Dict[str, int]:
    """Cuenta palabras en una colección de textos.

    - Si vocab es None: tokeniza y cuenta todas las palabras.
    - Si vocab se pasa: restringe el conteo a ese vocabulario (usando trie si use_trie=True).
    """
    if vocab is None:
        counts: Dict[str, int] = defaultdict(int)
        for t in texts:
            for tok in simple_tokenize(t):
                counts[tok] += 1
        return dict(counts)

    vocab_l = [v.strip().lower() for v in vocab if v and v.strip()]
    if use_trie:
        trie = PrefixTrie.from_vocab(vocab_l)
        counts: Dict[str, int] = defaultdict(int)
        for t in texts:
            cm = trie.count_matches(t or "")
            for w, c in cm.items():
                counts[w] += c
        return dict(counts)
    else:
        # conteo por tokens exactos
        vocab_set = set(vocab_l)
        counts: Dict[str, int] = defaultdict(int)
        for t in texts:
            for tok in simple_tokenize(t):
                if tok in vocab_set:
                    counts[tok] += 1
        return dict(counts)


def count_words_abstracts_from_bib(bib_path: str, stopwords: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """Cuenta palabras de abstracts en un archivo BibTeX (tokenización simple).

    Nota: usa bibtexparser; se ignoran stopwords si se proporcionan.
    """
    # Lazy import evitando advertencias del analizador estático
    bibtexparser = __import__("bibtexparser")

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)
    abstracts = []
    for e in bib_db.entries:
        abstracts.append(e.get("abstract", ""))

    sw = set(w.lower() for w in (stopwords or []))
    counts: Dict[str, int] = defaultdict(int)
    for text in abstracts:
        for tok in simple_tokenize(text):
            if tok not in sw:
                counts[tok] += 1
    return dict(counts)
