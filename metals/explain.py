"""Nombres, unidades y explicaciones legibles de las variables del MVP."""
import pandas as pd

FEATURE_LABELS = {
    "ret_1m": "Retorno de 1 mes", "ret_3m": "Retorno de 3 meses", "ret_6m": "Retorno de 6 meses", "ret_12m": "Retorno de 12 meses",
    "sma_gap_3": "Distancia a media de 3 meses", "sma_gap_12": "Distancia a media de 12 meses",
    "vol_6m": "Volatilidad de 6 meses", "vol_12m": "Volatilidad de 12 meses", "rsi_6m": "RSI de 6 meses",
    "drawdown_12m": "Caída desde máximo de 12 meses", "gold_silver_ratio": "Razón oro / plata",
    "ratio_change_3m": "Cambio de razón oro / plata", "fedfunds": "Tasa efectiva Fed",
    "treasury10": "Bono del Tesoro a 10 años", "vix": "VIX", "inflation_yoy": "Inflación interanual",
    "fed_change_3m": "Cambio de tasa Fed en 3 meses", "yield_change_3m": "Cambio de rendimiento en 3 meses",
    "news_sentiment": "Sentimiento de noticias", "news_count": "Noticias relevantes", "production_tonnes": "Producción anual en toneladas",
}


def feature_table(row, features):
    output = []
    for key in features:
        value = row.get(key)
        if pd.isna(value):
            rendered = "Sin dato"
        elif key.startswith(("ret_", "sma_gap", "vol_", "drawdown")) or key == "ratio_change_3m":
            rendered = f"{value:.2%}"
        elif key in {"fedfunds", "treasury10", "inflation_yoy"}:
            rendered = f"{value:.2f}%"
        elif key in {"fed_change_3m", "yield_change_3m"}:
            rendered = f"{value:+.2f} puntos porcentuales"
        else:
            rendered = f"{value:,.2f}"
        output.append({"Variable": FEATURE_LABELS.get(key, key), "Valor utilizado": rendered, "Campo del dataset": key})
    return pd.DataFrame(output)


def model_parameters(report, metal_threshold):
    params = report["best_params"]
    weighting = {"balanced": "Balance por clase", "uniform": "Uniforme", "recent_60m": "Mayor peso a datos recientes"}.get(report.get("best_weighting"), report.get("best_weighting", "Uniforme"))
    values = [
        ("Algoritmo", "XGBoost · 3 clases", "Modelo independiente por metal"),
        ("Árboles", str(params["n_estimators"]), "Cantidad de árboles del modelo seleccionado"),
        ("Profundidad máxima", str(params["max_depth"]), "Controla la complejidad de cada árbol"),
        ("Tasa de aprendizaje", "0,04", "Paso de actualización durante entrenamiento"),
        ("Ponderación", weighting, "Se calcula sólo dentro del entrenamiento"),
        ("Banda neutral", f"±{metal_threshold:.0%}", "Variación mensual dentro de esta banda"),
        ("Validación", "3 bloques de 24 observaciones", "Orden cronológico, con purga de 2 filas"),
        ("Prueba histórica", f"{report['holdout']['n']} observaciones", "Ya examinada; se necesita validación futura"),
        ("Selección", "Macro-F1 en desarrollo", "Las métricas de prueba no eligen el candidato"),
        ("Umbral de señal", "55%", "Regla de abstención; no certeza calibrada"),
        ("Ventana de noticias", "31 días · 5 por metal", "Sin duplicados y con fuente verificable"),
    ]
    return pd.DataFrame(values, columns=["Parámetro", "Valor", "Interpretación"])
