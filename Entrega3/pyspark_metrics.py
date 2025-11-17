# pyspark_metrics.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, hour, dayofweek, col, count, sum as _sum
import time

DATA_PATH = "data/accidentalidad.csv"

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
            print(f"{name} -> {t1-t0:.3f} s")
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
    df = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
    return df

@timeit("preprocess")
def preprocess(df):
    # Aseguramos tipos y columnas temporales
    df = df.withColumn("FECHA", col("FECHA").cast("timestamp"))
    df = df.withColumn("ANIO", year(col("FECHA"))) \
           .withColumn("MES", month(col("FECHA"))) \
           .withColumn("HORA", hour(col("FECHA"))) \
           .withColumn("DIA_SEMANA", dayofweek(col("FECHA")))
    # Opcional: cache si se reutiliza mucho
    df = df.filter(col("DEPARTAMENTO").isNotNull())
    try:
        num_parts = int(df.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    except Exception:
        num_parts = df.rdd.getNumPartitions()
    df = df.repartition(num_parts, "DEPARTAMENTO")
    df.cache()
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
    spark.stop()

if __name__ == "__main__":
    main()
