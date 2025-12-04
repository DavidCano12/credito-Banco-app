from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado (pipeline completo: preprocess + SMOTE + RandomForest)
MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo_credito_rf.pkl")
modelo = joblib.load(MODELO_PATH)

# Límites MÁXIMOS solo para variables que manejan dinero
# (valores obtenidos del dataset en Colab, usando percentil 99)
LIMITES_MAX_DINERO = {
    "A11": 7.5,      # Ratio deuda/ingresos
    "A14": 577.5,    # Monto del crédito
    "A15": 988.75,   # Ingreso / ahorros (escala del dataset)
}

def clip_max_dinero(col, valor):
    """
    Si la columna es una de las de dinero (A11, A14, A15),
    recorta el valor al máximo permitido según el dataset.
    Si no es de dinero, lo deja igual.
    """
    if valor is None:
        return None
    max_val = LIMITES_MAX_DINERO.get(col)
    if max_val is None:
        # No es una columna de dinero, se devuelve tal cual
        return valor
    # Si se pasa del máximo, lo recortamos
    return min(valor, max_val)


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

            # Numéricas NO de dinero: solo convertimos a float
            "A2": float(form.get("A2")) if form.get("A2") else None,
            "A3": float(form.get("A3")) if form.get("A3") else None,

            "A4": form.get("A4"),
            "A5": form.get("A5"),
            "A6": form.get("A6"),
            "A7": form.get("A7"),

            # A8 también es numérica, pero no la estamos recortando como dinero
            "A8": float(form.get("A8")) if form.get("A8") else None,

            "A9": form.get("A9"),
            "A10": form.get("A10"),

            # Aquí aplicamos el recorte SOLO a las de dinero
            "A11": clip_max_dinero("A11", float(form.get("A11")) if form.get("A11") else None),

            "A12": form.get("A12"),
            "A13": form.get("A13"),

            "A14": clip_max_dinero("A14", float(form.get("A14")) if form.get("A14") else None),
            "A15": clip_max_dinero("A15", float(form.get("A15")) if form.get("A15") else None),
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
