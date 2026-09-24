"""XGBoost por metal, selección temporal y holdout final separado."""
import hashlib
import platform
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from .config import ARTIFACTS, CORE_FEATURES, EXTRA_FEATURES, LABELS, THRESHOLDS
from .storage import now, save_json

def estimator(params):
    options = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                   learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                   min_child_weight=5, reg_lambda=5, random_state=42,
                   n_jobs=2, tree_method="hist")
    options.update(params)
    return XGBClassifier(**options)


def candidate_specs():
    """Búsqueda acotada, definida sin consultar el holdout ni cambiar etiquetas."""
    structures = [{"max_depth": d, "n_estimators": n} for d in (2, 3) for n in (80, 160)]
    structures.extend([
        {"max_depth": 1, "n_estimators": 80, "min_child_weight": 8, "reg_lambda": 10},
        {"max_depth": 2, "n_estimators": 160, "min_child_weight": 8, "reg_lambda": 10, "reg_alpha": 1},
    ])
    return [{"params": params.copy(), "weighting": weighting}
            for params in structures for weighting in ("uniform", "balanced", "recent_60m")]


def training_weights(train, weighting):
    """Cada fold obtiene pesos exclusivamente de sus propias fechas y etiquetas."""
    if weighting == "uniform":
        return None
    if weighting == "balanced":
        return compute_sample_weight("balanced", train.target.astype(int))
    if weighting == "recent_60m":
        dates = pd.to_datetime(train.date)
        months = dates.dt.year * 12 + dates.dt.month
        weights = np.exp2(-(months.max() - months).to_numpy(dtype=float) / 60.0)
        # Mantener peso medio 1 evita alterar involuntariamente la regularización.
        return weights / weights.mean()
    raise ValueError(f"Ponderación desconocida: {weighting}")

def scores(y, pred, probability=None):
    result = {"accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_precision": float(precision_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
              "macro_recall": float(recall_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
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


def select_model(development, features, candidates=None):
    """No recibe holdout: toda selección usa sólo validaciones de desarrollo."""
    trials = []
    folds = list(temporal_folds(development))
    for spec in candidate_specs() if candidates is None else candidates:
        fold_results = []
        for train, valid in folds:
            model = estimator(spec["params"])
            weights = training_weights(train, spec["weighting"])
            model.fit(train[features], train.target.astype(int), sample_weight=weights)
            p = model.predict_proba(valid[features])
            fold_results.append({"train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
                                 "label_available_max": str(train.label_available_at.max().date()),
                                 "valid_start": str(valid.date.min().date()), "valid_end": str(valid.date.max().date()),
                                 "train_n": len(train), "valid_n": len(valid),
                                 **scores(valid.target, p.argmax(axis=1), p)})
        trials.append({"params": spec["params"], "weighting": spec["weighting"],
                       "mean_macro_f1": float(np.mean([f["macro_f1"] for f in fold_results])),
                       "std_macro_f1": float(np.std([f["macro_f1"] for f in fold_results])),
                       "mean_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
                       "mean_log_loss": float(np.mean([f["log_loss"] for f in fold_results])),
                       "folds": fold_results})
    if not trials:
        raise ValueError("No hay candidatos para seleccionar el modelo.")
    # El desempate también usa únicamente desarrollo. El orden original resuelve
    # empates exactos y deja la variante uniforme primero, sin complejidad extra.
    best = max(trials, key=lambda r: (r["mean_macro_f1"], r["mean_accuracy"], -r["mean_log_loss"]))
    return best, trials

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
    best, trials = select_model(development, features)
    check_train(development, holdout)
    evaluation = estimator(best["params"])
    evaluation.fit(development[features], development.target.astype(int),
                   sample_weight=training_weights(development, best["weighting"]))
    probabilities = evaluation.predict_proba(holdout[features])
    predicted = probabilities.argmax(axis=1)
    majority = int(development.target.mode().iloc[0])
    persistent = np.where(holdout.ret_1m > THRESHOLDS[metal], 2, np.where(holdout.ret_1m < -THRESHOLDS[metal], 0, 1))
    report = {"metal": metal, "created_at": now(), "unit": "USD/onza troy, promedio mensual",
              "threshold": THRESHOLDS[metal], "labels": LABELS, "features": features, "excluded_context": excluded,
              "best_params": best["params"], "best_weighting": best["weighting"], "cv_trials": trials,
              "selection": {"version": "temporal_cv_v2", "candidate_count": len(trials),
                            "criterion": "macro-F1 medio; desempates: accuracy medio, menor log-loss medio",
                            "best_cv_macro_f1": best["mean_macro_f1"],
                            "original_grid_cv_macro_f1": max(t["mean_macro_f1"] for t in trials
                                if t["weighting"] == "uniform" and len(t["params"]) == 2),
                            "holdout_used_for_selection": False},
              "holdout": {"start": str(holdout.date.min().date()), "end": str(holdout.date.max().date()), "n": len(holdout)},
              "xgboost": scores(holdout.target, predicted, probabilities),
              "baseline_majority": scores(holdout.target, np.full(len(holdout), majority)),
              "baseline_persistence": scores(holdout.target, persistent),
              "confusion_matrix": confusion_matrix(holdout.target, predicted, labels=[0, 1, 2]).tolist(),
              "classification_report": classification_report(holdout.target, predicted, labels=[0, 1, 2], target_names=list(LABELS.values()), output_dict=True, zero_division=0),
              "importance": dict(zip(features, map(float, evaluation.feature_importances_))),
              "validation_limitations": ["Fuentes numéricas revisadas; rezagos conservadores, sin archivo de vintages.",
                                         "Probabilidades sin calibración externa; no equivalen a certeza.",
                                         "Ponderar clases o recencia cambia el ajuste; su utilidad depende de la validación temporal.",
                                         "36 observaciones de holdout; no demuestra rentabilidad ni causalidad.",
                                         "Holdout histórico ya examinado en versiones anteriores: comparación retrospectiva, no prueba prospectiva nueva."],
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
    final.fit(known[features], known.target.astype(int),
              sample_weight=training_weights(known, best["weighting"]))
    final.save_model(path / "model.json")
    report["model_sha256"] = hashlib.sha256((path / "model.json").read_bytes()).hexdigest()
    report["final_train"] = {"start": str(known.date.min().date()), "end": str(known.date.max().date()), "n": len(known), "origin": str(origin.date()), "label_available_max": str(known.label_available_at.max().date())}
    report["dataset_sha256"] = hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest()
    report["class_counts"] = {LABELS[int(k)]: int(v) for k, v in known.target.value_counts().items()}
    save_json(path / "report.json", report)
    return report
