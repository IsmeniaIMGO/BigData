from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableSequence, Tuple


TimeResult = Dict[str, float | str | None]


def _to_list(data: Iterable[Any]) -> List[Any]:
    return list(data) if not isinstance(data, list) else data.copy()


def measure_algorithm_time(
    algorithm: Callable[[MutableSequence[Any]], MutableSequence[Any]] | Callable[[MutableSequence[Any]], None],
    data: Iterable[Any],
    name: str | None = None,
    unit: str = "ms",
    repeat: int = 3,
) -> TimeResult:
    """Mide el tiempo de ejecución de un algoritmo de ordenamiento.

    - Copia los datos antes de cada ejecución para evitar efectos colaterales.
    - Soporta funciones que devuelven la lista o que ordenan in-place (None).

    Retorna un dict: { 'name', 'mean', 'std', 'unit', 'error' }
    """
    assert repeat >= 1
    name = name or getattr(algorithm, "__name__", "algorithm")
    times: List[float] = []
    error: str | None = None

    multiplier = {"s": 1.0, "ms": 1e3, "us": 1e6}.get(unit, 1e3)

    for _ in range(repeat):
        try:
            arr = _to_list(data)
            t0 = time.perf_counter()
            res = algorithm(arr)
            # Algunas implementaciones devuelven None (in-place)
            _ = res if res is not None else arr
            dt = (time.perf_counter() - t0) * multiplier
            times.append(dt)
        except Exception as e:  # noqa: BLE001
            error = str(e)
            break

    if error is not None or not times:
        return {"name": name, "mean": 0.0, "std": 0.0, "unit": unit, "error": error}

    mean_t = statistics.fmean(times)
    std_t = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {"name": name, "mean": float(mean_t), "std": float(std_t), "unit": unit, "error": None}


def run_benchmarks(
    algorithms: Mapping[str, Callable[[MutableSequence[Any]], MutableSequence[Any]] | Callable[[MutableSequence[Any]], None]],
    datasets: Mapping[str, Iterable[Any]],
    unit: str = "ms",
    repeat: int = 3,
) -> Dict[str, Dict[str, TimeResult]]:
    """Ejecuta benchmarks para múltiples algoritmos y datasets.

    Retorna un dict anidado: resultados[categoria][algoritmo] -> TimeResult
    """
    results: Dict[str, Dict[str, TimeResult]] = {}
    for ds_name, data in datasets.items():
        ds_res: Dict[str, TimeResult] = {}
        for alg_name, alg in algorithms.items():
            ds_res[alg_name] = measure_algorithm_time(alg, data, name=alg_name, unit=unit, repeat=repeat)
        results[ds_name] = ds_res
    return results
