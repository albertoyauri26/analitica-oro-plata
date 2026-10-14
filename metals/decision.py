"""Reglas académicas auditables. No ejecuta órdenes ni promete rentabilidad."""
import json
import hashlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .config import ARTIFACTS, LABELS
from .storage import now, save_json

def risk_and_signal(row, probabilities, report, today=None):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
        raise ValueError("Probabilidades inválidas")
    current = pd.Timestamp(today or pd.Timestamp.now()).tz_localize(None).normalize()
    origin = pd.Timestamp(row["date"])
    target = pd.Timestamp(row["target_month"])
    confidence = float(p.max())
    vol = float(row["vol_12m"])
    drawdown = float(row["drawdown_12m"])
    vix = float(row["vix"])
    reasons, score = [], 0
    if not np.isfinite([vol, drawdown, vix]).all():
        return {"risk": "No evaluable", "signal": "Abstenerse", "reasons": ["Faltan variables de riesgo."], "risk_score": None}
    if vol >= 0.30:
        score += 2
        reasons.append("Volatilidad anualizada de 12 retornos mensuales >= 30%.")
    elif vol >= 0.18:
        score += 1
        reasons.append("Volatilidad anualizada >= 18%.")
    if drawdown <= -0.15:
        score += 2
        reasons.append("Caída desde máximo de 12 meses >= 15%.")
    if vix >= 25:
        score += 1
        reasons.append("VIX rezagado >= 25.")
    if confidence < 0.55:
        score += 1
        reasons.append("La mayor probabilidad del modelo es inferior a 55%.")
    risk = "Alto" if score >= 3 else "Medio" if score >= 1 else "Bajo"
    gated = []
    if target < current or (current - origin).days > 62:
        gated.append("Predicción vencida o desactualizada: actualizar datos y reentrenar.")
    if origin > current:
        gated.append("El origen de la predicción está en el futuro.")
    if report["xgboost"]["macro_f1"] <= max(report["baseline_majority"]["macro_f1"], report["baseline_persistence"]["macro_f1"]):
        gated.append("XGBoost no superó ambos modelos de referencia en macro-F1 del holdout.")
    if confidence < 0.55:
        gated.append("Evidencia predictiva insuficiente según regla de 55%.")
    if gated:
        signal = "Abstenerse"
    elif risk == "Alto":
        signal = "Esperar por riesgo alto"
    else:
        signal = {0: "Cautela ante tendencia bajista", 1: "Observar / esperar", 2: "Evaluar oportunidad alcista"}[int(p.argmax())]
    return {"risk": risk, "risk_score": score, "signal": signal,
            "reasons": gated + reasons or ["Sin alertas bajo las reglas configuradas; esto no elimina el riesgo."],
            "rules": "Heurística académica v1; no calibrada con pérdidas financieras."}

def predict(dataset, metal, destination=ARTIFACTS, save=True, news_context=None, hybrid_weight=0.15):
    path = destination / metal
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if hashlib.sha256((path / "model.json").read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("El modelo no coincide con su reporte. Vuelve a entrenar.")
    row = dataset[dataset.metal == metal].sort_values("date").iloc[-1]
    if row[report["features"]].isna().any():
        raise ValueError("Faltan variables del último mes. Actualiza las fuentes antes de predecir.")
    if pd.Timestamp(report["final_train"]["label_available_max"]) > row.date:
        raise ValueError("Modelo entrenado con etiquetas posteriores al origen.")
    if pd.Timestamp(report["final_train"]["origin"]) != row.date:
        raise ValueError("El dataset cambió de mes; reentrena para emitir una nueva predicción.")
    digest = hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest()
    if digest != report["dataset_sha256"]:
        raise ValueError("El dataset cambió desde el entrenamiento. Ejecuta train antes de predict.")
    model = XGBClassifier()
    model.load_model(path / "model.json")
    features = pd.DataFrame([row[report["features"]].astype(float)], columns=report["features"])
    p = model.predict_proba(features)[0]
    result = {"metal": metal, "generated_at": now(), "origin": str(row.date.date()),
              "target_month": str(row.target_month.date()), "source_price_month": str(row.source_price_month.date()),
              "trend": LABELS[int(p.argmax())], "probabilities": {LABELS[i]: float(p[i]) for i in range(3)},
              "max_probability": float(p.max()), "volatility_12m": float(row.vol_12m),
              **risk_and_signal(row, p, report)}
    if news_context is not None:
        from .news_context import contextual_reading
        from .hybrid import combine
        result["news_context"] = news_context
        result["news_reading"] = contextual_reading(news_context, result["trend"])
        result["hybrid"] = combine(result, news_context, weight=hybrid_weight)
    if save:
        save_json(path / "prediction.json", result)
    return result
