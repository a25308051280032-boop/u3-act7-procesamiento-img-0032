# =====================================================
# EJERCICIO 11: Predicción de TEMPERATURA a partir de DÍAS TRANSCURRIDOS
# Aprendizaje Supervisado - Regresión Lineal
# =====================================================

# 1️⃣ IMPORTAR LIBRERÍAS
print("Iris Chavez NC 0032")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ CREAR LOS DATOS (45 días transcurridos)
print("Iris Chavez NC 0032")
np.random.seed(11)  # Semilla #11 correspondiente al número de ejercicio
dias_transcurridos = np.arange(1, 46).reshape(-1, 1)  # Días 1 al 45

# Temperatura simulada: inicia en 16°C y aumenta ~0.25°C por día + variación natural
temperaturas = 16 + 0.25 * dias_transcurridos.flatten() + np.random.normal(0, 1.1, 45)

datos = pd.DataFrame({
    'Dias_Transcurridos': dias_transcurridos.flatten(),
    'Temperatura_C': temperaturas
})

print("📊 Primeros 5 registros de temperatura:")
print(datos.head())

# 3️⃣ SEPARAR EN ENTRENAMIENTO (80%) Y PRUEBA (20%)
print("Iris Chavez NC 0032")
X = datos[['Dias_Transcurridos']]  # Variable predictora (matriz)
y = datos['Temperatura_C']         # Variable objetivo (vector)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

# 4️⃣ ENTRENAR EL MODELO
print("Iris Chavez NC 0032")
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5️⃣ HACER PREDICCIONES SOBRE LOS DATOS DE PRUEBA
print("Iris Chavez NC 0032")
predicciones = modelo.predict(X_test)

# 6️⃣ EVALUAR EL MODELO
print("Iris Chavez NC 0032")
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f"\n🎯 Error cuadrático medio (MSE): {mse:.2f}")
print(f"📈 Coeficiente R²: {r2:.3f}")
print(f"📐 Fórmula aprendida: Temperatura = {modelo.intercept_:.2f} + {modelo.coef_[0]:.3f} * Días")

# 7️⃣ PREDECIR DÍAS FUTUROS (Días 46 al 55)
print("Iris Chavez NC 0032")
dias_futuros = np.arange(46, 56).reshape(-1, 1)
temperaturas_futuras = modelo.predict(dias_futuros)

print("\n🔮 Pronóstico de temperatura para los próximos 10 días:")
for dia, temp in zip(dias_futuros.flatten(), temperaturas_futuras):
    print(f"   Día {dia:2d} → {temp:.2f} °C")

# 8️⃣ VISUALIZAR RESULTADOS
print("Iris Chavez NC 0032")
plt.figure(figsize=(11, 6))
sns.set_theme(style="whitegrid")

# Puntos históricos de entrenamiento y prueba
sns.scatterplot(x=X_train['Dias_Transcurridos'], y=y_train,
                color='dodgerblue', s=80, label='Entrenamiento (80%)')
sns.scatterplot(x=X_test['Dias_Transcurridos'], y=y_test,
                color='crimson', s=100, marker='X', label='Prueba (20%)')

# Línea de tendencia aprendida
x_linea = np.arange(1, 46).reshape(-1, 1)
y_linea = modelo.predict(x_linea)
plt.plot(x_linea, y_linea, color='green', linewidth=2, label='Tendencia del modelo')

# Proyección a futuro
plt.plot(dias_futuros, temperaturas_futuras, color='darkorange',
         linewidth=2, linestyle='--', marker='o', label='Predicción futura (Días 46-55)')

plt.title('🌤️ Predicción de Temperatura según Días Transcurridos', fontsize=14, fontweight='bold')
plt.xlabel('Días transcurridos')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.tight_layout()
plt.show()