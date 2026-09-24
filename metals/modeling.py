"""XGBoost por metal, selección temporal y holdout final separado."""
import hashlib
import platform
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from .config import ARTIFACTS, CORE_FEATURES, EXTRA_FEATURES, LABELS, THRESHOLDS
from .storage import now, save_json

def estimator(params):
    return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                         learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=5, reg_lambda=5, random_state=42,
                         n_jobs=2, tree_method="hist", **params)

def scores(y, pred, probability=None):
    result = {"accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))}
    if probability is not None:
        result["log_loss"] = float(log_loss(y, probability, labels=[0, 1, 2]))
        onehot = np.eye(3)[np.asarray(y, dtype=int)]
        result["multiclass_brier"] = float(np.mean(np.sum((probability - onehot) ** 2, axis=1)))
    return result

def check_train(train, test):
    if train.empty or set(train.target.astype(int)) != {0, 1, 2}:
        raise ValueError("Entrenamiento insuficiente: se requieren ejemplos reales de las tres clases.")
    if train.label_available_at.max() > test.date.min():
        raise ValueError("Fuga temporal: hay etiquetas de entrenamiento todavía no disponibles.")

def temporal_folds(frame):
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=3, test_size=24, gap=2).split(frame):
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        train = train[train.label_available_at <= valid.date.min()]
        check_train(train, valid)
        yield train, valid

def train_metal(dataset, metal, destination=ARTIFACTS):
    frame = dataset[dataset.metal == metal].sort_values("date").copy()
    # Se exige cobertura de las features base; la app nunca oculta una macro ausente.
    usable = frame.dropna(subset=CORE_FEATURES)
    if usable.empty:
        raise ValueError("No hay filas con todas las variables base. Revisa ingesta y cobertura macroeconómica.")
    origin = frame.date.max()
    if usable.date.max() != origin:
        raise ValueError("Faltan datos en el último mes. Actualiza las fuentes antes de entrenar.")
    known = usable[(usable.target.notna()) & (usable.label_available_at <= origin)].copy()
    if len(known) < 180:
        raise ValueError(f"Solo {len(known)} meses útiles para {metal}; se requieren al menos 180.")
    # Holdout reservado. Purga adicional de 2 meses entre desarrollo y prueba.
    holdout = known.iloc[-36:]
    development = known.iloc[:-38]
    development = development[development.label_available_at <= holdout.date.min()]
    features = list(CORE_FEATURES)
    # Las extensiones sólo entran si EXISTEN suficientes datos en el primer train.
    first_train, _ = next(temporal_folds(development))
    excluded = {}
    for col in EXTRA_FEATURES:
        coverage = float(first_train[col].notna().mean())
        if coverage >= 0.8 and first_train[col].notna().sum() >= 60:
            features.append(col)
        else:
            excluded[col] = f"Cobertura inicial {coverage:.1%}; se exige 80% y 60 meses observados."
    candidates = [{"max_depth": d, "n_estimators": n} for d in (2, 3) for n in (80, 160)]
    trials = []
    for params in candidates:
        fold_results = []
        for train, valid in temporal_folds(development):
            model = estimator(params)
            model.fit(train[features], train.target.astype(int))
            p = model.predict_proba(valid[features])
            fold_results.append({"train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
                                 "label_available_max": str(train.label_available_at.max().date()),
                                 "valid_start": str(valid.date.min().date()), "valid_end": str(valid.date.max().date()),
                                 "train_n": len(train), "valid_n": len(valid),
                                 **scores(valid.target, p.argmax(axis=1), p)})
        trials.append({"params": params, "mean_macro_f1": float(np.mean([f["macro_f1"] for f in fold_results])), "folds": fold_results})
    best = max(trials, key=lambda r: r["mean_macro_f1"])
    check_train(development, holdout)
    evaluation = estimator(best["params"])
    evaluation.fit(development[features], development.target.astype(int))
    probabilities = evaluation.predict_proba(holdout[features])
    predicted = probabilities.argmax(axis=1)
    majority = int(development.target.mode().iloc[0])
    persistent = np.where(holdout.ret_1m > THRESHOLDS[metal], 2, np.where(holdout.ret_1m < -THRESHOLDS[metal], 0, 1))
    report = {"metal": metal, "created_at": now(), "unit": "USD/onza troy, promedio mensual",
              "threshold": THRESHOLDS[metal], "labels": LABELS, "features": features, "excluded_context": excluded,
              "best_params": best["params"], "cv_trials": trials,
              "holdout": {"start": str(holdout.date.min().date()), "end": str(holdout.date.max().date()), "n": len(holdout)},
              "xgboost": scores(holdout.target, predicted, probabilities),
              "baseline_majority": scores(holdout.target, np.full(len(holdout), majority)),
              "baseline_persistence": scores(holdout.target, persistent),
              "confusion_matrix": confusion_matrix(holdout.target, predicted, labels=[0, 1, 2]).tolist(),
              "classification_report": classification_report(holdout.target, predicted, labels=[0, 1, 2], target_names=list(LABELS.values()), output_dict=True, zero_division=0),
              "importance": dict(zip(features, map(float, evaluation.feature_importances_))),
              "validation_limitations": ["Fuentes numéricas revisadas; rezagos conservadores, sin archivo de vintages.",
                                         "Probabilidades sin calibración externa; no equivalen a certeza.",
                                         "36 meses de holdout; no demuestra rentabilidad ni causalidad."],
              "versions": {"python": platform.python_version(), "xgboost": xgboost.__version__, "sklearn": sklearn.__version__}}
    path = destination / metal
    path.mkdir(parents=True, exist_ok=True)
    evaluation.save_model(path / "evaluation_model.json")
    test_output = holdout[["date", "target_month", "target_return", "target"]].copy()
    test_output["prediction"] = predicted
    for i in range(3):
        test_output[f"p_{i}"] = probabilities[:, i]
    test_output.to_csv(path / "holdout_predictions.csv", index=False)
    # Modelo final diferente del evaluado: reentrenar sólo con etiquetas conocidas al origen.
    final = estimator(best["params"])
    final.fit(known[features], known.target.astype(int))
    final.save_model(path / "model.json")
    report["model_sha256"] = hashlib.sha256((path / "model.json").read_bytes()).hexdigest()
    report["final_train"] = {"start": str(known.date.min().date()), "end": str(known.date.max().date()), "n": len(known), "origin": str(origin.date()), "label_available_max": str(known.label_available_at.max().date())}
    report["dataset_sha256"] = hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest()
    report["class_counts"] = {LABELS[int(k)]: int(v) for k, v in known.target.value_counts().items()}
    save_json(path / "report.json", report)
    return report
