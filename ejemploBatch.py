# Ejemplo de procesamiento batch: Calcular ventas totales del día
# Simulación de ventas en una plataforma de comercio electrónico
import csv
import random
from datetime import datetime

# Procesamiento batch: calcular ventas totales del día
def calcular_ventas_totales(nombre_archivo):
	total = 0
	with open(nombre_archivo, 'r', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			total += int(row['cantidad']) * int(row['precio_unitario'])
	return total

if __name__ == "__main__":
	archivo_ventas = 'ventas_dia.csv'  # Usar archivo existente con ventas quemadas
	total = calcular_ventas_totales(archivo_ventas)
	print(f"Ventas totales del día: ${total}")
