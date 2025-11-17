import matplotlib.pyplot as plt
import pandas as pd

# Cargar tabla de tiempos anteriormente exportada
df = pd.read_csv("results/timing_comparison.csv")

# Crear gráfica
plt.figure(figsize=(10,6))
for tool in ['pandas', 'pyspark']:
    subset = df[df['tool'] == tool]
    plt.bar(subset['metric'], subset['time_s'], alpha=0.7, label=tool)

plt.xlabel("Métrica")
plt.ylabel("Tiempo (segundos)")
plt.title("Comparación de Rendimiento: Pandas vs PySpark")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Guardar gráfica
plt.savefig("results/timing_comparison.png")
# También exportar tabla resumen
df_pivot = df.pivot(index='metric', columns='tool', values='time_s')
df_pivot.to_csv("results/timing_comparison_summary.csv")
print("Gráfica y tabla resumen guardadas en 'results/'")
