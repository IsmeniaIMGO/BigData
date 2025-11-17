import pandas as pd
import time
from datetime import datetime

DATA_PATH = "data/accidentalidad.csv"

def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} -> {t1-t0:.3f} s")
        return res
    return wrapper

@timeit
def load():
    return pd.read_csv(DATA_PATH, parse_dates=["FECHA"], low_memory=False)

@timeit
def preprocess(df):
    # limpiar columnas, tipos
    df = df.dropna(subset=["FECHA", "DEPARTAMENTO"])
    df["ANIO"] = df["FECHA"].dt.year
    df["MES"] = df["FECHA"].dt.month
    df["HORA"] = df["FECHA"].dt.hour
    df["DIA_SEMANA"] = df["FECHA"].dt.dayofweek + 1
    return df

@timeit
def total_por_departamento(df):
    return df.groupby("DEPARTAMENTO").size().sort_values(ascending=False).reset_index(name="count")

@timeit
def heridos_muertos_por_departamento(df):
    return df.groupby("DEPARTAMENTO")[["HERIDOS","MUERTOS"]].sum().sort_values(by="MUERTOS", ascending=False)

@timeit
def top_por_tipo_vehiculo(df, n=10):
    return df["TIPO_VEHICULO"].value_counts().head(n)

@timeit
def tendencias_mensuales(df):
    g = df.groupby(["ANIO","MES"]).size().reset_index(name="count")
    g["YEAR_MONTH"] = g["ANIO"].astype(str) + "-" + g["MES"].astype(str).str.zfill(2)
    return g.sort_values(["ANIO","MES"])

@timeit
def accidentes_por_hora_dia(df):
    hour = df.groupby("HORA").size().reset_index(name="count").sort_values("HORA")
    day = df.groupby("DIA_SEMANA").size().reset_index(name="count").sort_values("DIA_SEMANA")
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
    dep.to_csv("results/pandas_acc_por_departamento.csv", index=False)
    hm.to_csv("results/pandas_heridos_muertos_por_departamento.csv")
    tmonth.to_csv("results/pandas_tendencia_mensual.csv", index=False)

if __name__ == "__main__":
    main()
