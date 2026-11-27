# =====================================================
# EJEMPLO 1: IMAGEN ORIGINAL Y ESPECTRO DE FRECUENCIA
# =====================================================
print("Iris Chavez NC 0032")
import matplotlib.pyplot as plt
from funciones_frecuencia_0032 import (
    cargar_imagen,
    obtener_espectro,
    guardar_grafica
)

# 1. Cargar la imagen
imagen = cargar_imagen()

# 2. Calcular el espectro de frecuencia
espectro = obtener_espectro(imagen)

# 3. Mostrar dimensiones en consola
print("--- EJEMPLO 1 ---")
print("Imagen del camarógrafo cargada.")
print(f"Dimensiones: Alto={imagen.shape[0]}, Ancho={imagen.shape[1]}")

# 4. Crear la figura con 2 subgráficas (1 fila, 2 columnas)
figura, ejes = plt.subplots(1, 2, figsize=(12, 6))

# 5. Mostrar imagen original
ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen Original (Dominio Espacial) 0032")
ejes[0].axis("off")

# 6. Mostrar espectro de frecuencia
ejes[1].imshow(espectro, cmap="gray")
ejes[1].set_title("Espectro de Frecuencia (Dominio de Frecuencia) 0032")
ejes[1].axis("off")

plt.tight_layout()

# 7. Guardar y mostrar
guardar_grafica(figura, "01_imagen_y_espectro_0032.png")
plt.show()
print("Iris Chavez NC 0032")