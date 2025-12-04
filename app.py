from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Cargar el modelo entrenado
modelo = joblib.load("modelo_credito_rf.pkl")

# Crear la app de Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "API de predicción de aprobación de crédito - Random Forest"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera un JSON con las 15 variables A1–A15.
    Ejemplo de JSON:
    {
      "A1": "a",
      "A2": 30,
      "A3": 4.0,
      "A4": "u",
      "A5": "g",
      "A6": "c",
      "A7": "v",
      "A8": 2.5,
      "A9": "t",
      "A10": "t",
      "A11": 1.0,
      "A12": "t",
      "A13": "g",
      "A14": 200,
      "A15": 1000
    }
    """
    data = request.get_json()

    # Convertir el diccionario en DataFrame con una fila
    df_entrada = pd.DataFrame([data])

    # Hacer la predicción
    prob_aprobado = modelo.predict_proba(df_entrada)[0, 1]
    pred_clase = modelo.predict(df_entrada)[0]  # 1 o 0

    resultado = {
        "prediccion_clase": int(pred_clase),  # 1 = aprobado, 0 = rechazado
        "probabilidad_aprobado": float(prob_aprobado)
    }

    return jsonify(resultado)

if __name__ == "__main__":
    # Para pruebas locales
    app.run(host="0.0.0.0", port=5000, debug=True)
