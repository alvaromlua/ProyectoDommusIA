import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Cargando datos
# Asume que data es un DataFrame con las columnas adecuadas
data = pd.read_csv("path_to_data.csv")

# Separando las características de entrada y las etiquetas de salida
X = data[['user_energy', 'user_mind', 'user_nature', 'user_tactics', 'user_identity']]
y = data[['ideal_energy', 'ideal_mind', 'ideal_nature', 'ideal_tactics', 'ideal_identity']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio del modelo: {mse}")

# Guardar el modelo para su uso futuro
import joblib
joblib.dump(model, 'roommate_recommender.pkl')
