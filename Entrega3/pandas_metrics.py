import pandas as pd
import time
from datetime import datetime
from pathlib import Path

DATA_PATH = "Entrega3/data/accidentalidad.csv"
RESULTS_DIR = Path("Entrega3/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TIMINGS: list[tuple[str, float]] = []

def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        _TIMINGS.append((func.__name__, dt))
        print(f"{func.__name__} -> {dt:.3f} s")
        return res
    return wrapper

@timeit
def load():
    # Nota: especificar dtypes ayuda a reducir inferencia y memoria
    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        dtype={
            "HERIDOS": "Int64",
            "MUERTOS": "Int64",
        },
    )
    return df


def _find_date_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "FECHA", "fecha", "Fecha", "FECHA_HORA", "FECHA_OCURRENCIA", "FECHA_OCURRENCIA_HECHO",
        "FECHA_ACCIDENTE",
        "DATE", "Date", "datetime", "DATETIME",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

@timeit
def preprocess(df):
    # limpiar columnas, tipos
    # fecha flexible
    date_col = _find_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # asegurar columna DEPARTAMENTO si existe, sino crear vacía para no romper flujos
    if "DEPARTAMENTO" not in df.columns:
        # mapear desde nombre real si existe
        if "DEPARTAMENTO_ACCIDENTE" in df.columns:
            df["DEPARTAMENTO"] = df["DEPARTAMENTO_ACCIDENTE"]
        else:
            df["DEPARTAMENTO"] = pd.NA
    # descartar filas sin fecha y sin departamento (si tenemos fecha detectada)
    if date_col:
        df = df.dropna(subset=[date_col, "DEPARTAMENTO"])
    else:
        df = df.dropna(subset=["DEPARTAMENTO"])  # seguir con métricas que no usan fecha

    # columnas derivadas de fecha
    if date_col:
        df["ANIO"] = df[date_col].dt.year
        df["MES"] = df[date_col].dt.month
        df["HORA"] = df[date_col].dt.hour
        # pandas: Monday=0; ajustamos a 1..7 similar a Spark
        df["DIA_SEMANA"] = df[date_col].dt.dayofweek + 1
    # columnas con cardinalidad baja a category
    # columnas con cardinalidad baja a category
    for col in ["DEPARTAMENTO", "TIPO_VEHICULO", "CLASE_ACCIDENTE"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

@timeit
def total_por_departamento(df):
    return (
        df.groupby("DEPARTAMENTO", observed=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

@timeit
def heridos_muertos_por_departamento(df):
    # Asegurar columnas numéricas; crear si faltan
    for c in ["HERIDOS", "MUERTOS"]:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")
    return (
        df.groupby("DEPARTAMENTO", observed=False)[["HERIDOS", "MUERTOS"]]
        .sum()
        .sort_values(by="MUERTOS", ascending=False)
    )

@timeit
def top_por_tipo_vehiculo(df, n=10):
    if "TIPO_VEHICULO" not in df.columns:
        return pd.Series(dtype="int64", name="TIPO_VEHICULO")
    return df["TIPO_VEHICULO"].value_counts().head(n)

@timeit
def tendencias_mensuales(df):
    if not set(["ANIO", "MES"]).issubset(df.columns):
        return pd.DataFrame(columns=["ANIO","MES","count","YEAR_MONTH"])  # vacío
    g = df.groupby(["ANIO","MES"]).size().reset_index(name="count")
    if g.empty:
        g["YEAR_MONTH"] = pd.Series(dtype=str)
        return g
    g["YEAR_MONTH"] = g["ANIO"].astype(str) + "-" + g["MES"].astype(str).str.zfill(2)
    return g.sort_values(["ANIO","MES"])

@timeit
def accidentes_por_hora_dia(df):
    if "HORA" in df.columns:
        hour = df.groupby("HORA").size().reset_index(name="count").sort_values("HORA")
    else:
        hour = pd.DataFrame(columns=["HORA","count"])  # vacío
    if "DIA_SEMANA" in df.columns:
        day = df.groupby("DIA_SEMANA").size().reset_index(name="count").sort_values("DIA_SEMANA")
    else:
        day = pd.DataFrame(columns=["DIA_SEMANA","count"])  # vacío
    return hour, day

def main():
    df = load()
    df = preprocess(df)
    dep = total_por_departamento(df)
    hm = heridos_muertos_por_departamento(df)
    tv = top_por_tipo_vehiculo(df)
    tmonth = tendencias_mensuales(df)
    hour, day = accidentes_por_hora_dia(df)

    # exportar resúmenes
    dep.to_csv(RESULTS_DIR / "pandas_acc_por_departamento.csv", index=False)
    hm.to_csv(RESULTS_DIR / "pandas_heridos_muertos_por_departamento.csv")
    tmonth.to_csv(RESULTS_DIR / "pandas_tendencia_mensual.csv", index=False)
    # exportar faltantes
    tv.to_csv(RESULTS_DIR / "pandas_top_tipo_vehiculo.csv")
    hour.to_csv(RESULTS_DIR / "pandas_accidentes_por_hora.csv", index=False)
    day.to_csv(RESULTS_DIR / "pandas_accidentes_por_dia.csv", index=False)

    # guardar tiempos para comparación
    import csv
    timing_csv = RESULTS_DIR / "pandas_timing.csv"
    with open(timing_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "metric", "time_s"])
        for metric, secs in _TIMINGS:
            w.writerow(["pandas", metric, f"{secs:.6f}"])

if __name__ == "__main__":
    main()
