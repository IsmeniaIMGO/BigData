from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Union

import pandas as pd


PathLike = Union[str, Path]


def _entrega2_root() -> Path:
	"""
	Devuelve la raíz del módulo Entrega2 asumiendo que este archivo vive en
	Entrega2/processing/utils/rutas.py
	"""
	return Path(__file__).resolve().parents[2]


def resolve_paths(base_dir: PathLike | None = None) -> Dict[str, Path]:
	"""
	Resuelve rutas canónicas del proyecto.

	Args:
		base_dir: Carpeta base del módulo Entrega2. Si no se indica, se infiere
				  a partir de la ubicación de este archivo.

	Returns:
		dict con claves:
			- base: raíz de Entrega2
			- data_raw: Entrega2/data/raw
			- data_processed: Entrega2/data/processed
			- outputs: Entrega2/outputs
	"""
	base = Path(base_dir) if base_dir else _entrega2_root()
	paths = {
		"base": base,
		"data_raw": base / "data" / "raw",
		"data_processed": base / "data" / "processed",
		"outputs": base / "outputs",
	}
	return paths


def read_csv(path: PathLike) -> pd.DataFrame:
	"""
	Lee un CSV intentando UTF-8 y, si falla, latin-1 para robustez con acentos.
	"""
	p = Path(path)
	try:
		return pd.read_csv(p, encoding="utf-8")
	except UnicodeDecodeError:
		return pd.read_csv(p, encoding="latin-1")


def read_parquet(path: PathLike) -> pd.DataFrame:
	"""
	Lee un Parquet con pandas. Requiere pyarrow o fastparquet instalado.
	"""
	p = Path(path)
	try:
		return pd.read_parquet(p)
	except ImportError as e:
		raise ImportError(
			"Para leer Parquet instala 'pyarrow' o 'fastparquet' (pip install pyarrow)."
		) from e


def save_csv(df: pd.DataFrame, path: PathLike) -> None:
	"""
	Guarda un DataFrame a CSV en UTF-8 creando la carpeta si no existe.
	"""
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(p, index=False, encoding="utf-8")


def timestamped_path(base: PathLike, suffix: str = "") -> str:
	"""
	Genera una ruta con timestamp basada en un path base (sin crear archivos).

	Ejemplos:
		timestamped_path("outputs/analysis/reporte", ".csv")
			-> outputs/analysis/reporte_20251104_153012.csv
		timestamped_path("outputs/features/tfidf", "-v1.csv")
			-> outputs/features/tfidf_20251104_153012-v1.csv
	"""
	base_path = Path(base)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	name = f"{base_path.name}_{ts}{suffix}" if suffix else f"{base_path.name}_{ts}"
	return str(base_path.with_name(name))

