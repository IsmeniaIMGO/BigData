from __future__ import annotations

import os
from pathlib import Path

# Asegurar resultados
RESULTS = Path("Entrega3/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def run_pandas():
    import pandas_metrics  # type: ignore
    pandas_metrics.main()


def run_pyspark():
    import pyspark_metrics  # type: ignore
    pyspark_metrics.main()


def aggregate():
    # Import local del agregador en el mismo directorio
    import agregarTiempos as agregarTiempos  # type: ignore
    return agregarTiempos.main()


def run_plots():
    # Usar backend no interactivo para no bloquear
    os.environ.setdefault("MPLBACKEND", "Agg")
    import diagramas  # type: ignore  # ejecuta al importar


def main():
    print("[1/4] Ejecutando métricas con Pandas...")
    run_pandas()
    print("[2/4] Ejecutando métricas con PySpark...")
    run_pyspark()
    print("[3/4] Agregando tiempos...")
    out_csv = aggregate()
    print(f"Tiempos combinados en: {out_csv}")
    print("[4/4] Generando diagramas...")
    run_plots()
    print("Listo. Revisa la carpeta 'Entrega3/results/'.")


if __name__ == "__main__":
    main()
