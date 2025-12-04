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
LIMITES_MAX_DINERO = {
    "A11": 7.5,      # Ratio deuda/ingresos
    "A14": 50000000.1,    # Monto del crédito
    "A15": 500000000.1,   # Ingreso / ahorros (escala del dataset)
}

def clip_max_dinero(col, valor):
    """Recorta el valor al máximo permitido para columnas de dinero."""
    if valor is None:
        return None
    max_val = LIMITES_MAX_DINERO.get(col)
    if max_val is None:
        return valor
    return min(valor, max_val)


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    prob_aprobado = None
    datos_entrada = None

    if request.method == "POST":
        form = request.form

        # 1) Lo que escribió el usuario (para mostrar en el resumen)
        entrada_original = {
            "A1": form.get("A1"),
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

        # 2) Lo que ve el modelo (A12 fijado, dinero recortado)
        entrada_modelo = {
            "A1": entrada_original["A1"],
            "A2": entrada_original["A2"],
            "A3": entrada_original["A3"],
            "A4": entrada_original["A4"],
            "A5": entrada_original["A5"],
            "A6": entrada_original["A6"],
            "A7": entrada_original["A7"],
            "A8": entrada_original["A8"],
            "A9": entrada_original["A9"],
            "A10": entrada_original["A10"],
            "A11": clip_max_dinero("A11", entrada_original["A11"]),
            "A12": "t",  # valor fijo para el modelo
            "A13": entrada_original["A13"],
            "A14": clip_max_dinero("A14", entrada_original["A14"]),
            "A15": clip_max_dinero("A15", entrada_original["A15"]),
        }

        # DataFrame para el modelo
        df_entrada = pd.DataFrame([entrada_modelo])

        # En el HTML mostramos lo que el usuario escribió
        datos_entrada = entrada_original

        # Predicción
        prob_aprobado = modelo.predict_proba(df_entrada)[0, 1]
        pred_clase = int(modelo.predict(df_entrada)[0])  # 1 = aprobado, 0 = rechazado

        resultado = "APROBADO" if pred_clase == 1 else "RECHAZADO"
        prob_aprobado = round(prob_aprobado * 100, 2)

    return render_template(
        "index.html",
        resultado=resultado,
        prob_aprobado=prob_aprobado,
        datos_entrada=datos_entrada,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
