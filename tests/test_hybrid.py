"""Datos inventados exclusivamente para verificar límites del escenario híbrido."""
from copy import deepcopy
import math
import pytest
from metals.hybrid import combine

TODAY = "2026-09-28T12:00:00Z"


def example(score=0.8, relevant=5):
    base = {"metal": "gold", "origin": "2026-08-31", "target_month": "2026-09-30",
            "source_price_month": "2026-07-31", "generated_at": "2026-09-28T11:55:00Z",
            "probabilities": {"Bajista": 0.4, "Neutral": 0.25, "Alcista": 0.35},
            "trend": "Bajista", "max_probability": 0.4, "signal": "Abstenerse", "risk": "Alto",
            "reasons": ["Confianza base insuficiente."]}
    audit = [{"id": str(i), "published_at": "2026-09-26T00:00:00Z",
              "fetched_at": "2026-09-27T00:00:00Z", "analyzed_at": "2026-09-27T00:05:00Z",
              "relevant": i < relevant, "score": score if i < relevant else None, "provider": "finbert_local"}
             for i in range(5)]
    context = {"metal": "gold", "as_of": TODAY, "requested": 5, "selected": 5, "analyzed": 5,
               "pending": 0, "relevant": relevant, "mean_sentiment": score if relevant else None,
               "news_ids": [str(i) for i in range(5)], "article_audit": audit, "score_semantics": "financial_tone"}
    return base, context


@pytest.mark.parametrize("score", [-1, -0.4, 0, 0.7, 1])
def test_convex_scenario_is_bounded_and_preserves_neutral(score):
    base, context = example(score)
    result = combine(base, context, weight=0.2, today=TODAY)
    assert result["status"] == "applied_experimental"
    p = result["hybrid_probabilities"]
    assert sum(p.values()) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in p.values())
    assert p["Neutral"] == base["probabilities"]["Neutral"]
    assert max(abs(d) for d in result["probability_changes"].values()) <= 0.2
    assert result["applied_weight"] == pytest.approx(0.2 * abs(score))
    assert math.copysign(1, p["Alcista"] - base["probabilities"]["Alcista"]) == math.copysign(1, score)
    if score == 0:
        assert p == base["probabilities"]


def test_relevant_coverage_reduces_the_adjustment():
    base, context = example(score=1, relevant=3)
    result = combine(base, context, weight=0.2, today=TODAY)
    assert result["applied_weight"] == pytest.approx(0.12)
    assert result["hybrid_probabilities"]["Alcista"] == pytest.approx(0.35 + 0.12 * 0.4)


def test_changed_trend_never_upgrades_base_signal_or_confidence():
    base, context = example(score=1)
    result = combine(base, context, weight=0.2, today=TODAY)
    assert result["trend"] == "Alcista"
    assert result["baseline_trend"] == "Bajista"
    assert result["baseline_confidence"] == 0.4
    assert result["signal"] == "Abstenerse"
    assert result["risk"] == "Alto"
    assert result["decision_uses_hybrid"] is False
    assert result["validated"] is False
    assert any("FinBERT" in text and "no retorno" in text for text in result["limitations"])


def test_zero_weight_disables_without_news():
    base, _ = example()
    result = combine(base, {}, weight=0, today=TODAY)
    assert result["status"] == "disabled"
    assert result["hybrid_probabilities"] == base["probabilities"]


@pytest.mark.parametrize("weight", [-0.1, 0.21, float("nan"), float("inf"), None, True])
def test_invalid_weight_is_not_silently_clamped(weight):
    base, context = example()
    with pytest.raises(ValueError):
        combine(base, context, weight=weight, today=TODAY)


def test_invalid_base_probabilities_fail():
    base, context = example()
    base["probabilities"]["Alcista"] = 1
    with pytest.raises(ValueError, match="Probabilidades base"):
        combine(base, context, today=TODAY)


@pytest.mark.parametrize("field", ["published_at", "fetched_at", "analyzed_at"])
def test_future_news_capture_or_analysis_cannot_enter(field):
    base, context = example()
    context["article_audit"][0][field] = "2026-09-29T00:00:00Z"
    result = combine(base, context, today=TODAY)
    assert result["status"] == "blocked"
    assert result["hybrid_probabilities"] == base["probabilities"]
    assert any("no estaba disponible" in text for text in result["reasons"])


@pytest.mark.parametrize("change", ["stale", "future", "wrong_metal", "pending", "duplicate", "missing_audit", "tampered_mean", "old_article"])
def test_invalid_snapshot_cannot_adjust_the_prediction(change):
    base, context = example()
    if change == "stale":
        context["as_of"] = "2026-09-26T00:00:00Z"
    elif change == "future":
        context["as_of"] = "2026-09-29T00:00:00Z"
    elif change == "wrong_metal":
        context["metal"] = "silver"
    elif change == "pending":
        context["article_audit"][0]["analyzed_at"] = None
    elif change == "duplicate":
        context["article_audit"][0]["id"] = "1"
    elif change == "missing_audit":
        del context["article_audit"]
    elif change == "tampered_mean":
        context["mean_sentiment"] = -1
    else:
        context["article_audit"][0]["published_at"] = "2026-08-01T00:00:00Z"
    result = combine(base, context, today=TODAY)
    assert result["status"] == "blocked"
    assert result["applied_weight"] == 0
    assert result["hybrid_probabilities"] == base["probabilities"]
    assert result["reasons"]


def test_three_relevant_articles_are_required():
    base, context = example(relevant=2)
    result = combine(base, context, today=TODAY)
    assert result["status"] == "blocked"
    assert any("al menos tres" in text for text in result["reasons"])


@pytest.mark.parametrize("change", ["expired", "future_target", "stale_price", "future_model"])
def test_only_a_current_month_prediction_can_be_combined(change):
    base, context = example()
    current = TODAY
    if change == "expired":
        current = "2026-10-01T00:00:00Z"
        context["as_of"] = current
    elif change == "future_target":
        base.update(origin="2026-09-30", target_month="2026-10-31", source_price_month="2026-08-31")
    elif change == "stale_price":
        base["source_price_month"] = "2026-06-30"
    else:
        base["generated_at"] = "2026-09-29T00:00:00Z"
    result = combine(base, context, today=current)
    assert result["status"] == "blocked"
    assert result["hybrid_probabilities"] == base["probabilities"]


def test_inputs_are_unchanged_and_baseline_reasons_are_copied():
    base, context = example()
    originals = deepcopy((base, context))
    result = combine(base, context, today=TODAY)
    assert (base, context) == originals
    result["baseline_reasons"].append("Cambio externo")
    assert (base, context) == originals


def test_equivalent_timezone_snapshot_is_accepted():
    base, context = example()
    context["as_of"] = "2026-09-28T07:00:00-05:00"
    assert combine(base, context, today=TODAY)["status"] == "applied_experimental"
