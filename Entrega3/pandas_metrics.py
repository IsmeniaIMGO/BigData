import pandas as pd
import time
from datetime import datetime
from pathlib import Path

DATA_PATH = "Entrega3/data/accidentalidad.csv"
RESULTS_DIR = Path("Entrega3/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TIMINGS: list[tuple[str, float]] = []

# Decorador para medir tiempo de ejecución
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


# Cargar datos con pandas
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


# Preprocesar datos
@timeit
def preprocess(df):
    # limpiar columnas, tipos
    # fecha flexible
    date_col = _find_date_column(df)
    if date_col:
        # Normalizar meses en español a abreviaturas en inglés antes de parsear
        s = df[date_col].astype(str).str.strip().str.lower()
        replacements = {
            r"\benero\b": "Jan", r"\bene\b": "Jan",
            r"\bfebrero\b": "Feb", r"\bfeb\b": "Feb",
            r"\bmarzo\b": "Mar", r"\bmar\b": "Mar",
            r"\babril\b": "Apr", r"\babr\b": "Apr",
            r"\bmayo\b": "May", r"\bmay\b": "May",
            r"\bjunio\b": "Jun", r"\bjun\b": "Jun",
            r"\bjulio\b": "Jul", r"\bjul\b": "Jul",
            r"\bagosto\b": "Aug", r"\bago\b": "Aug",
            r"\bseptiembre\b": "Sep", r"\bsetiembre\b": "Sep", r"\bsept\b": "Sep", r"\bsep\b": "Sep",
            r"\boctubre\b": "Oct", r"\boct\b": "Oct",
            r"\bnoviembre\b": "Nov", r"\bnov\b": "Nov",
            r"\bdiciembre\b": "Dec", r"\bdic\b": "Dec",
        }
        for pat, rep in replacements.items():
            s = s.str.replace(pat, rep, regex=True)
        df[date_col] = pd.to_datetime(s, errors="coerce")
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

# Detectar columna de fecha común
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

#Total de accidentes por departamento
@timeit
def total_por_departamento(df):
    return (
        df.groupby("DEPARTAMENTO", observed=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

# Heridos y muertos por departamento
@timeit
def heridos_muertos_por_departamento(df):
    # Si existen columnas numéricas útiles, se usan; de lo contrario, derivar desde texto "CON HERIDOS"/"CON MUERTOS" en cualquier columna string
    use_numeric = False
    if set(["HERIDOS", "MUERTOS"]).issubset(df.columns):
        tmp = df[["HERIDOS", "MUERTOS"]].apply(pd.to_numeric, errors="coerce").fillna(0)
        if (tmp.sum().sum() > 0):
            use_numeric = True
            df["HERIDOS"] = tmp["HERIDOS"].astype("Int64")
            df["MUERTOS"] = tmp["MUERTOS"].astype("Int64")

    if not use_numeric:
        # Buscar patrones por columna (más robusto que concatenar), case-insensitive y tolerante a espacios
        text_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category") or str(df[c].dtype) == "string"]
        if text_cols:
            her = pd.DataFrame({c: df[c].astype(str).str.lower().str.contains("con heridos", na=False) for c in text_cols})
            mue = pd.DataFrame({c: df[c].astype(str).str.lower().str.contains("con muertos", na=False) for c in text_cols})
            df["__HERIDOS_BIN__"] = her.any(axis=1).astype("Int64")
            df["__MUERTOS_BIN__"] = mue.any(axis=1).astype("Int64")
            return (
                df.groupby("DEPARTAMENTO", observed=False)[["__HERIDOS_BIN__", "__MUERTOS_BIN__"]]
                  .sum()
                  .rename(columns={"__HERIDOS_BIN__":"HERIDOS","__MUERTOS_BIN__":"MUERTOS"})
                  .sort_values(by="MUERTOS", ascending=False)
            )
        else:
            # sin texto ni numérico válido
            return pd.DataFrame(columns=["HERIDOS","MUERTOS"]) 
    else:
        return (
            df.groupby("DEPARTAMENTO", observed=False)[["HERIDOS", "MUERTOS"]]
            .sum()
            .sort_values(by="MUERTOS", ascending=False)
        )

# Top N tipos de vehículo involucrados
@timeit
def top_por_tipo_vehiculo(df, n=10):
    if "TIPO_VEHICULO" not in df.columns:
        return pd.Series(dtype="int64", name="TIPO_VEHICULO")
    return df["TIPO_VEHICULO"].value_counts().head(n)

# Tendencias mensuales de accidentes
@timeit
def tendencias_mensuales(df):
    if not set(["ANIO", "MES"]).issubset(df.columns):
        return pd.DataFrame(columns=["ANIO","MES","count"])  # vacío
    g = df.groupby(["ANIO","MES"]).size().reset_index(name="count")
    return g.sort_values(["ANIO","MES"]) 

# Accidentes por hora del día y día de la semana
@timeit
def accidentes_por_dia(df):
    if "DIA_SEMANA" in df.columns:
        day = df.groupby("DIA_SEMANA").size().reset_index(name="count").sort_values("DIA_SEMANA")
    else:
        day = pd.DataFrame(columns=["DIA_SEMANA","count"])  # vacío
    return day

# Función principal para ejecutar todas las métricas
def main():
    df = load()
    df = preprocess(df)
    dep = total_por_departamento(df)
    hm = heridos_muertos_por_departamento(df)
    tv = top_por_tipo_vehiculo(df)
    tmonth = tendencias_mensuales(df)
    day = accidentes_por_dia(df)

    # exportar resúmenes
    dep.to_csv(RESULTS_DIR / "pandas_acc_por_departamento.csv", index=False)
    hm.to_csv(RESULTS_DIR / "pandas_heridos_muertos_por_departamento.csv")
    tmonth.to_csv(RESULTS_DIR / "pandas_tendencia_mensual.csv", index=False)
    # exportar faltantes
    tv.to_csv(RESULTS_DIR / "pandas_top_tipo_vehiculo.csv")
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
