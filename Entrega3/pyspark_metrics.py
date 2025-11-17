# pyspark_metrics.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, hour, dayofweek, col, count, sum as _sum, lit
from pyspark import StorageLevel
import time
from pathlib import Path

DATA_PATH = "Entrega3/data/accidentalidad.csv"
PARQUET_PATH = "Entrega3/data/accidentalidad.parquet"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TIMINGS: list[tuple[str, float]] = []

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
    csv_path = f"results/{base_name}_csv"
    parquet_path = f"results/{base_name}_parquet"
    df.write.mode("overwrite").csv(csv_path, header=True)
    df.write.mode("overwrite").parquet(parquet_path)

@timeit("load")
def load(spark):
    # Si existe Parquet previo, úsalo. Si no, convierte CSV → Parquet para reducir I/O en siguientes corridas
    data_file = Path(PARQUET_PATH)
    if data_file.exists():
        df = spark.read.parquet(PARQUET_PATH)
    else:
        df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
        # Guardar Parquet para próximas ejecuciones
        df.write.mode("overwrite").parquet(PARQUET_PATH)
    return df

@timeit("preprocess")
def preprocess(df):
    # Aseguramos tipos y columnas temporales
    # detectar columna fecha flexible
    date_candidates = ["FECHA", "fecha", "Fecha", "FECHA_HORA", "FECHA_OCURRENCIA"]
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df = df.withColumn(date_col, col(date_col).cast("timestamp"))
        df = df.withColumn("ANIO", year(col(date_col))) \
               .withColumn("MES", month(col(date_col))) \
               .withColumn("HORA", hour(col(date_col))) \
               .withColumn("DIA_SEMANA", dayofweek(col(date_col)))
    else:
        # columnas nulas para no romper flujos
        df = df.withColumn("ANIO", lit(None).cast("int")) \
               .withColumn("MES", lit(None).cast("int")) \
               .withColumn("HORA", lit(None).cast("int")) \
               .withColumn("DIA_SEMANA", lit(None).cast("int"))
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

@timeit("acc_por_departamento")
def total_por_departamento(df):
    res = df.groupBy("DEPARTAMENTO").agg(count("*").alias("count")).orderBy(col("count").desc())
    _write_outputs(res, "spark_acc_por_departamento")
    return res

@timeit("heridos_muertos_por_departamento")
def heridos_muertos_por_departamento(df):
    # Asegurarse que HERIDOS y MUERTOS sean ints
    df2 = df.withColumn("HERIDOS", col("HERIDOS").cast("int")).withColumn("MUERTOS", col("MUERTOS").cast("int"))
    res = df2.groupBy("DEPARTAMENTO").agg(_sum("HERIDOS").alias("HERIDOS"), _sum("MUERTOS").alias("MUERTOS")).orderBy(col("MUERTOS").desc())
    _write_outputs(res, "spark_heridos_muertos_por_departamento")
    return res

@timeit("top_tipo_vehiculo")
def top_tipo_vehiculo(df, n=10):
    res = df.groupBy("TIPO_VEHICULO").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(n)
    _write_outputs(res, "spark_top_tipo_vehiculo")
    return res

@timeit("tendencia_mensual")
def tendencias_mensuales(df):
    res = df.groupBy("ANIO","MES").agg(count("*").alias("count")).orderBy("ANIO","MES")
    _write_outputs(res, "spark_tendencia_mensual")
    return res

@timeit("por_hora_y_dia")
def accidentes_por_hora_dia(df):
    hour = df.groupBy("HORA").agg(count("*").alias("count")).orderBy("HORA")
    day = df.groupBy("DIA_SEMANA").agg(count("*").alias("count")).orderBy("DIA_SEMANA")
    _write_outputs(hour, "spark_accidentes_por_hora")
    _write_outputs(day, "spark_accidentes_por_dia")
    return hour, day

def main():
    spark = create_spark()
    df = load(spark)
    df = preprocess(df)
    total_por_departamento(df)
    heridos_muertos_por_departamento(df)
    top_tipo_vehiculo(df)
    tendencias_mensuales(df)
    accidentes_por_hora_dia(df)
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
