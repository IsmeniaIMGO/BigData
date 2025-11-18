import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Cargar tabla de tiempos anteriormente exportada
df = pd.read_csv("Entrega3/results/timing_comparison.csv")
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")

# Preparar datos pivot para barras lado a lado
pv = df.pivot(index="metric", columns="tool", values="time_s").sort_index()
pv = pv.reindex(columns=["pandas", "pyspark"])  # asegurar orden de columnas

# Asegurar carpeta de salida
out_dir = Path("Entrega3/graficos")
out_dir.mkdir(parents=True, exist_ok=True)

# Configuración del gráfico de barras lado a lado con colores fijos
metrics = list(pv.index)
x = list(range(len(metrics)))
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar([i - width/2 for i in x], pv["pandas"], width=width, color="#1f77b4", label="pandas")
plt.bar([i + width/2 for i in x], pv["pyspark"], width=width, color="#ff0000", label="pyspark")

plt.xlabel("Métrica")
plt.ylabel("Tiempo (segundos)")
plt.title("Comparación de Rendimiento: Pandas vs PySpark")
plt.xticks(ticks=x, labels=metrics, rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# Guardar gráfica y resumen
plt.savefig(out_dir / "timing_comparison.png")
pv.to_csv(out_dir / "timing_comparison_summary.csv")
print("Gráfica y tabla resumen guardadas en 'Entrega3/graficos/'")
plt.close()
