"""Fixtures artificiales limitados a pruebas de selección y exportación."""
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier
from metals import modeling


def test_balancing_uses_only_current_training_classes():
    frame = pd.DataFrame({"target": [0, 0, 0, 0, 1, 1, 2]})
    weights = modeling.training_weights(frame, "balanced")
    totals = [weights[frame.target == k].sum() for k in (0, 1, 2)]
    assert totals == pytest.approx([len(frame) / 3] * 3)
    assert weights.mean() == pytest.approx(1)
    assert modeling.training_weights(frame, "uniform") is None


def test_recency_uses_calendar_age_and_mean_one():
    frame = pd.DataFrame({"date": pd.to_datetime(["2010-01-31", "2015-01-31", "2020-01-31"])})
    weights = modeling.training_weights(frame, "recent_60m")
    assert weights[0] / weights[1] == pytest.approx(0.5)
    assert weights[1] / weights[2] == pytest.approx(0.5)
    assert weights.mean() == pytest.approx(1)
    with pytest.raises(ValueError, match="Ponderación desconocida"):
        modeling.training_weights(frame, "unknown")


def test_accuracy_and_macro_precision_are_distinct_metrics():
    result = modeling.scores([0, 0, 1, 2], [0, 0, 0, 2])
    assert result["accuracy"] == 0.75
    assert result["macro_precision"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert result["macro_recall"] == pytest.approx(2 / 3)


def test_all_original_candidates_are_retained():
    specs = modeling.candidate_specs()
    assert len(specs) == 18
    for depth in (2, 3):
        for count in (80, 160):
            assert {"params": {"max_depth": depth, "n_estimators": count}, "weighting": "uniform"} in specs


def test_selection_fits_only_purged_training_rows(monkeypatch):
    dates = pd.date_range("2000-01-31", periods=200, freq="ME")
    frame = pd.DataFrame({"date": dates, "label_available_at": dates + pd.offsets.MonthEnd(2),
                          "target": np.arange(200) % 3, "feature": np.arange(200)})
    fitted, valid_rows = [], []

    class Classifier:
        def __init__(self, params):
            self.good = params["good"]

        def fit(self, x, y, sample_weight=None):
            fitted.append(x.index.to_numpy())
            if sample_weight is not None:
                assert len(sample_weight) == len(x)

        def predict_proba(self, x):
            valid_rows.append(x.index.to_numpy())
            predicted = x.feature.to_numpy() % 3 if self.good else np.zeros(len(x), dtype=int)
            p = np.full((len(x), 3), 0.1)
            p[np.arange(len(x)), predicted] = 0.8
            return p

    monkeypatch.setattr(modeling, "estimator", Classifier)
    candidates = [{"params": {"good": False}, "weighting": "uniform"},
                  {"params": {"good": True}, "weighting": "balanced"}]
    best, trials = modeling.select_model(frame, ["feature"], candidates)
    assert best["params"]["good"] is True
    assert best["mean_macro_f1"] == 1
    assert len(trials) == 2
    for train_idx, valid_idx in zip(fitted, valid_rows):
        assert len(set(train_idx) & set(valid_idx)) == 0
        assert frame.loc[train_idx, "label_available_at"].max() <= frame.loc[valid_idx, "date"].min()
        assert max(train_idx) + 2 < min(valid_idx)


def test_weighted_model_exports_without_inference_transform(tmp_path):
    # Se exportan árboles: los pesos sólo intervienen al ajustar, no al predecir.
    frame = pd.DataFrame({"feature": np.arange(30, dtype=float), "target": np.arange(30) % 3})
    model = modeling.estimator({"max_depth": 1, "n_estimators": 4, "reg_lambda": 10})
    model.fit(frame[["feature"]], frame.target,
              sample_weight=modeling.training_weights(frame, "balanced"))
    path = tmp_path / "weighted.json"
    model.save_model(path)
    loaded = XGBClassifier()
    loaded.load_model(path)
    np.testing.assert_allclose(model.predict_proba(frame[["feature"]]), loaded.predict_proba(frame[["feature"]]))
