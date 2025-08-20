# Ejemplo de procesamiento streaming: Productos más comprados en tiempo real
# Simulación de flujo de ventas en una plataforma de comercio electrónico
import random
import time
from collections import Counter

def simular_streaming_ventas(num_eventos=30, intervalo=0.5):
	productos = ['Laptop', 'Celular', 'Audífonos', 'Monitor', 'Teclado']
	contador = Counter()
	for i in range(num_eventos):
		producto = random.choice(productos)
		contador[producto] += 1
		top_productos = contador.most_common(3)
		print(f"Evento {i+1}: Se vendió {producto}")
		print("Top productos más comprados en vivo:")
		for prod, cant in top_productos:
			print(f"  {prod}: {cant} ventas")
		print("-"*30)
		time.sleep(intervalo)

if __name__ == "__main__":
	simular_streaming_ventas()
