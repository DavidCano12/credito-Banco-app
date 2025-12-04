from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado (pipeline completo: preprocess + SMOTE + RandomForest)
MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo_credito_rf.pkl")
modelo = joblib.load(MODELO_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    prob_aprobado = None
    datos_entrada = None

    if request.method == "POST":
        # Recibir datos del formulario
        form = request.form

        # Construir el diccionario con las columnas A1–A15
        # OJO: estos nombres deben coincidir EXACTAMENTE con las columnas del modelo
        entrada = {
            "A1": form.get("A1"),   # variable categórica (ej: 'a', 'b')
            "A2": float(form.get("A2")) if form.get("A2") else None,
            "A3": float(form.get("A3")) if form.get("A3") else None,
            "A4": form.get("A4"),
            "A5": form.get("A5"),
            "A6": form.get("A6"),
            "A7": form.get("A7"),
            "A8": float(form.get("A8")) if form.get("A8") else None,
            "A9": form.get("A9"),
            "A10": form.get("A10"),
            "A11": float(form.get("A11")) if form.get("A11") else None,
            "A12": form.get("A12"),
            "A13": form.get("A13"),
            "A14": float(form.get("A14")) if form.get("A14") else None,
            "A15": float(form.get("A15")) if form.get("A15") else None,
        }

        # Convertir a DataFrame con una sola fila
        df_entrada = pd.DataFrame([entrada])

        # Guardar los datos de entrada para mostrarlos en el HTML
        datos_entrada = entrada

        # Hacer la predicción con el modelo cargado
        prob_aprobado = modelo.predict_proba(df_entrada)[0, 1]
        pred_clase = int(modelo.predict(df_entrada)[0])  # 1 = aprobado, 0 = rechazado

        resultado = "APROBADO" if pred_clase == 1 else "RECHAZADO"
        prob_aprobado = round(prob_aprobado * 100, 2)  # en porcentaje

    return render_template(
        "index.html",
        resultado=resultado,
        prob_aprobado=prob_aprobado,
        datos_entrada=datos_entrada,
    )


if __name__ == "__main__":
    # Modo local
    app.run(host="0.0.0.0", port=5000, debug=True)
