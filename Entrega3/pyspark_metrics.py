from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    year,month,hour,dayofweek,col,count,
    sum as _sum,lit,
    to_timestamp,to_date,
    coalesce as _coalesce,concat,
    lit as _lit,trim,lower,when,
)
from pyspark import StorageLevel
import os
import time
from pathlib import Path

DATA_PATH = "Entrega3/data/accidentalidad.csv"
PARQUET_PATH = "Entrega3/data/accidentalidad.parquet"
RESULTS_DIR = Path("Entrega3/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TIMINGS: list[tuple[str, float]] = []


def _has_winutils() -> bool:
    """Detecta si en Windows hay winutils disponible (HADOOP_HOME configurado)."""
    if os.name != "nt":
        return True
    hhome = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hhome:
        return False
    exe = Path(hhome) / "bin" / "winutils.exe"
    return exe.exists()

def timeit(name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            res = f(*args, **kwargs)
            # Para Spark DataFrames, materializar acción para medir
            if hasattr(res, "count") and callable(getattr(res, "count")):
                # no llamar count en DF muy grandes innecesariamente. Si res es un resultado de acción, omitir.
                try:
                    # si res es muy grande, podríamos usar res.show(5) o res.limit(5).collect() en su lugar
                    res.count()
                except Exception:
                    pass
            t1 = time.perf_counter()
            dt = t1 - t0
            _TIMINGS.append((name, dt))
            print(f"{name} -> {dt:.3f} s")
            return res
        return wrapper
    return decorator

def create_spark(app_name="AccSpark", master="local[*]"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        parallelism = spark.sparkContext.defaultParallelism
        if isinstance(parallelism, int) and parallelism > 0:
            spark.conf.set("spark.sql.shuffle.partitions", str(parallelism))
    except Exception:
        pass
    return spark


def _write_outputs(df, base_name: str):
    """Escribe resultados. En Windows sin winutils, hace fallback a pandas CSV de un solo archivo."""
    csv_dir = RESULTS_DIR / f"{base_name}_csv"
    parquet_dir = RESULTS_DIR / f"{base_name}_parquet"
    flat_csv = RESULTS_DIR / f"{base_name}.csv"

    # Helper: escribir un CSV plano sin depender de pandas/pyarrow
    def _write_single_csv(_df, _path: Path):
        import csv as _csv
        cols = _df.columns
        with open(_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(cols)
            for r in _df.toLocalIterator():
                w.writerow([r[c] for c in cols])

    if os.name == "nt" and not _has_winutils():
        # Fallback: escribir un solo CSV sin pandas (evita error de 'distutils' en Py3.12)
        try:
            _write_single_csv(df.coalesce(1), flat_csv)
            print(f"[INFO] (fallback) CSV escrito: {flat_csv}")
        except Exception as e:
            print(f"[ERROR] Falló el fallback CSV para {base_name}: {e}")
        print(f"[WARN] Parquet omitido para {base_name}: HADOOP_HOME/winutils no detectado en Windows.")
        return

    # Ruta normal con Spark writers
    df.write.mode("overwrite").csv(str(csv_dir), header=True)
    if _has_winutils():
        df.write.mode("overwrite").parquet(str(parquet_dir))
    else:
        print(f"[WARN] Parquet omitido para {base_name}: HADOOP_HOME/winutils no detectado en Windows.")
    # Además, generar un CSV plano único para facilitar la consulta
    try:
        _write_single_csv(df.coalesce(1), flat_csv)
        print(f"[INFO] CSV plano escrito: {flat_csv}")
    except Exception as e:
        print(f"[WARN] No se pudo escribir CSV plano {flat_csv}: {e}")

@timeit("load")
def load(spark):
    # Si existe Parquet previo, úsalo. Si no, convierte CSV → Parquet para reducir I/O en siguientes corridas
    data_file = Path(PARQUET_PATH)
    if data_file.exists() and _has_winutils():
        df = spark.read.parquet(PARQUET_PATH)
    else:
        df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
        # Guardar Parquet para próximas ejecuciones solo si es viable
        if _has_winutils():
            df.write.mode("overwrite").parquet(PARQUET_PATH)
        else:
            print("[WARN] Conversión a Parquet omitida: HADOOP_HOME/winutils no detectado en Windows.")
    return df

@timeit("preprocess")
def preprocess(df):
    # Aseguramos tipos y columnas temporales
    # detectar columna fecha flexible
    date_candidates = ["FECHA", "fecha", "Fecha", "FECHA_HORA", "FECHA_OCURRENCIA", "FECHA_ACCIDENTE"]
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        # Parseo robusto: varios formatos y tolerante a valores como "Aug 2024"
        raw = trim(col(date_col))
        parsed_date = _coalesce(
            to_date(raw),
            to_date(raw, "yyyy-MM-dd"),
            to_date(raw, "dd/MM/yyyy"),
            to_date(raw, "MM/dd/yyyy"),
            to_date(raw, "dd-MM-yyyy"),
            to_date(raw, "yyyy/MM/dd"),
            to_date(raw, "dd-MMM-yyyy"),
            to_date(raw, "dd MMM yyyy"),
            # month-year sin día, asumir día 01
            to_date(concat(_lit("01 "), raw), "dd MMM yyyy"),
            to_date(concat(raw, _lit("-01")), "yyyy-MM-dd"),
            to_date(concat(raw, _lit("/01")), "yyyy/MM/dd"),
        )
        df = df.withColumn("__FECHA_DATE__", parsed_date)
        # Convertir a timestamp de forma segura (Date -> Timestamp no lanza excepción)
        df = df.withColumn("__FECHA_TS__", col("__FECHA_DATE__").cast("timestamp"))
        df = df.withColumn("ANIO", year(col("__FECHA_TS__"))) \
               .withColumn("MES", month(col("__FECHA_TS__"))) \
               .withColumn("HORA", hour(col("__FECHA_TS__"))) \
               .withColumn("DIA_SEMANA", dayofweek(col("__FECHA_TS__")))
    else:
        # columnas nulas para no romper flujos
        df = df.withColumn("ANIO", lit(None).cast("int")) \
               .withColumn("MES", lit(None).cast("int")) \
               .withColumn("HORA", lit(None).cast("int")) \
               .withColumn("DIA_SEMANA", lit(None).cast("int"))
    # Crear columna canónica DEPARTAMENTO si viene como DEPARTAMENTO_ACCIDENTE
    if "DEPARTAMENTO" not in df.columns and "DEPARTAMENTO_ACCIDENTE" in df.columns:
        df = df.withColumn("DEPARTAMENTO", col("DEPARTAMENTO_ACCIDENTE"))
    # Opcional: cache si se reutiliza mucho
    if "DEPARTAMENTO" in df.columns:
        df = df.filter(col("DEPARTAMENTO").isNotNull())
    try:
        num_parts = int(df.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    except Exception:
        num_parts = df.rdd.getNumPartitions()
    part_col = "DEPARTAMENTO" if "DEPARTAMENTO" in df.columns else None
    df = df.repartition(num_parts, part_col) if part_col else df.repartition(num_parts)
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    df.count()  # materializar cache
    return df

@timeit("total_por_departamento")
def total_por_departamento(df):
    res = df.groupBy("DEPARTAMENTO").agg(count("*").alias("count")).orderBy(col("count").desc())
    _write_outputs(res, "spark_acc_por_departamento")
    return res

@timeit("heridos_muertos_por_departamento")
def heridos_muertos_por_departamento(df):
    # Usar columnas numéricas si existen y suman > 0; si no, derivar desde texto "CON HERIDOS"/"CON MUERTOS"
    df2 = df
    use_numeric = False
    if "HERIDOS" in df2.columns and "MUERTOS" in df2.columns:
        num_df = df2.select(
            col("HERIDOS").cast("int").alias("HERIDOS"),
            col("MUERTOS").cast("int").alias("MUERTOS")
        )
        sums = num_df.agg(_sum("HERIDOS").alias("sH"), _sum("MUERTOS").alias("sM")).collect()[0]
        if (sums["sH"] or 0) + (sums["sM"] or 0) > 0:
            use_numeric = True
            df2 = df2.withColumn("HERIDOS", col("HERIDOS").cast("int")).withColumn("MUERTOS", col("MUERTOS").cast("int"))

    if not use_numeric:
        # Buscar patrones en columnas string
        str_cols = [name for (name, dtype) in df2.dtypes if dtype == "string"]
        if str_cols:
            expr_her = None
            expr_mue = None
            for c in str_cols:
                lc = lower(trim(col(c)))
                cond_h = lc.contains("con heridos")
                cond_m = lc.contains("con muertos")
                expr_her = cond_h if expr_her is None else (expr_her | cond_h)
                expr_mue = cond_m if expr_mue is None else (expr_mue | cond_m)
            df2 = df2.withColumn("__HERIDOS_BIN__", when(expr_her, lit(1)).otherwise(lit(0))) \
                     .withColumn("__MUERTOS_BIN__", when(expr_mue, lit(1)).otherwise(lit(0)))
            res = df2.groupBy("DEPARTAMENTO").agg(
                _sum("__HERIDOS_BIN__").alias("HERIDOS"),
                _sum("__MUERTOS_BIN__").alias("MUERTOS")
            ).orderBy(col("MUERTOS").desc())
        else:
            res = df2.groupBy("DEPARTAMENTO").agg(lit(0).alias("HERIDOS"), lit(0).alias("MUERTOS"))
    else:
        res = df2.groupBy("DEPARTAMENTO").agg(_sum("HERIDOS").alias("HERIDOS"), _sum("MUERTOS").alias("MUERTOS")).orderBy(col("MUERTOS").desc())
    _write_outputs(res, "spark_heridos_muertos_por_departamento")
    return res

@timeit("top_por_tipo_vehiculo")
def top_tipo_vehiculo(df, n=10):
    if "TIPO_VEHICULO" not in df.columns:
        print("[WARN] Columna 'TIPO_VEHICULO' no encontrada; se omite métrica.")
        # Devolver DF vacío con esquema esperado
        empty = df.sparkSession.createDataFrame([], schema="TIPO_VEHICULO string, count long")
        _write_outputs(empty, "spark_top_tipo_vehiculo")
        return empty
    res = df.groupBy("TIPO_VEHICULO").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(n)
    _write_outputs(res, "spark_top_tipo_vehiculo")
    return res

@timeit("tendencias_mensuales")
def tendencias_mensuales(df):
    res = df.filter(col("ANIO").isNotNull() & col("MES").isNotNull()) \
            .groupBy("ANIO","MES").agg(count("*").alias("count")).orderBy("ANIO","MES")
    _write_outputs(res, "spark_tendencia_mensual")
    return res

@timeit("accidentes_por_dia")
def accidentes_por_dia(df):
    day = df.filter(col("DIA_SEMANA").isNotNull()).groupBy("DIA_SEMANA").agg(count("*").alias("count")).orderBy("DIA_SEMANA")
    _write_outputs(day, "spark_accidentes_por_dia")
    return day

def main():
    spark = create_spark()
    df = load(spark)
    df = preprocess(df)
    total_por_departamento(df)
    heridos_muertos_por_departamento(df)
    top_tipo_vehiculo(df)
    tendencias_mensuales(df)
    accidentes_por_dia(df)
    # guardar tiempos
    import csv
    with open(RESULTS_DIR / "pyspark_timing.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "metric", "time_s"])
        for metric, secs in _TIMINGS:
            w.writerow(["pyspark", metric, f"{secs:.6f}"])
    spark.stop()

if __name__ == "__main__":
    main()
