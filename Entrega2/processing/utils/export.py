from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union


def _ensure_parent(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_model(obj: Any, path: Union[str, Path]) -> str:
    """Guarda un objeto de modelo en disco.

    - Si el objeto tiene método `save` o `save_pretrained`, se intenta usarlo.
    - En caso contrario, se usa pickle.
    """
    p = _ensure_parent(path)

    # Algunos objetos (p.ej., AnnoyIndex) tienen su propio método save
    if hasattr(obj, "save") and callable(getattr(obj, "save")):
        obj.save(str(p))  # type: ignore[attr-defined]
        return str(p)

    # Algunos modelos (HuggingFace) tienen save_pretrained
    if hasattr(obj, "save_pretrained") and callable(getattr(obj, "save_pretrained")):
        obj.save_pretrained(str(p))  # type: ignore[attr-defined]
        return str(p)

    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(p)


def load_model(path: Union[str, Path]) -> Any:
    """Carga un objeto guardado con pickle.

    Nota: si el objeto se guardó con métodos específicos (p.ej., Annoy), debe
    reconstruirse con su API propia fuera de esta función.
    """
    p = Path(path)
    with open(p, "rb") as f:
        return pickle.load(f)


def save_plot(fig: Any, path: Union[str, Path], dpi: int = 300, transparent: bool = False) -> str:
    """Guarda una figura.

    - Matplotlib: usa figure.savefig
    - Plotly: usa figure.write_image si está disponible
    """
    p = _ensure_parent(path)

    # Matplotlib
    if hasattr(fig, "savefig"):
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight", transparent=transparent)  # type: ignore[attr-defined]
        return str(p)

    # Plotly
    if hasattr(fig, "write_image"):
        fig.write_image(str(p))  # type: ignore[attr-defined]
        return str(p)

    raise TypeError("Tipo de figura no soportado: se espera un objeto con savefig (matplotlib) o write_image (plotly)")


def write_report_md(
    sections: Union[Dict[str, str], Iterable[Tuple[str, str]]],
    path: Union[str, Path],
    title: str | None = None,
) -> str:
    """Escribe un informe en Markdown.

    sections puede ser:
    - dict: {titulo: contenido}
    - iterable de tuplas: [(titulo, contenido), ...]
    """
    p = _ensure_parent(path)

    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")

    items: Iterable[Tuple[str, str]]
    if isinstance(sections, dict):
        items = sections.items()
    else:
        items = sections

    for h, content in items:
        lines.append(f"\n## {h}\n")
        lines.append(str(content).rstrip() + "\n")

    p.write_text("".join(lines), encoding="utf-8")
    return str(p)
