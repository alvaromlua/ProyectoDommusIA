import numpy as np
import joblib
def predict_ideal_roomate(user_profile):
    """
    Predecir los atributos del roomate ideal para un perfil de usuario dado.

    Args:
    user_profile (list): Lista con los 5 atributos de personalidad del usuario.

    Returns:
    list: Lista con los 5 atributos predichos del roomate ideal.
    """
    model = joblib.load('roommate_recommender.pkl')
    user_profile = np.array(user_profile).reshape(1, -1)  # Transformar a forma adecuada para el modelo
    ideal_profile = model.predict(user_profile)
    return ideal_profile.tolist()[0]

# Ejemplo de uso:
user_profile = [7, 5, 3, 9, 4]  # Energía, Mente, Naturaleza, Tácticas, Identidad del usuario
ideal_roomate = predict_ideal_roomate(user_profile)
print("Perfil ideal del roomate:", ideal_roomate)
