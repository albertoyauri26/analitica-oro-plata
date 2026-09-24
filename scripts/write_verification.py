"""Anexo generado desde datos/modelos reales; no lee credenciales."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metals.storage import connect, read, now
from metals.news_filter import filter_news
from metals.news_context import snapshot


def write(tests_passed=None):
    conn = connect()
    tables = {name: read(conn, name) for name in ("prices", "macro", "news", "sentiment", "production", "ingestion_log")}
    conn.close()
    news = filter_news(tables["news"])
    reports = {m: json.loads((ROOT / "artifacts" / m / "report.json").read_text(encoding="utf-8")) for m in ("gold", "silver")}
    text = ["# Resultados verificados de esta instalación", "", f"Generado desde archivos reales: {now()}.", "",
            "## Datos", "", f"Precios: {len(tables['prices'])} observaciones. Macro: {len(tables['macro'])}.",
            f"Noticias del tema: {len(news)}; registros antiguos excluidos: {len(tables['news']) - len(news)}. Producción anual importada: {len(tables['production'])}.", "",
            "## Evaluación temporal", "", "| Metal | Accuracy | Precisión macro | Macro-F1 | F1 mayoritaria | F1 persistencia |", "|---|---:|---:|---:|---:|---:|"]
    for metal, report in reports.items():
        score = report["xgboost"]
        text.append(f"| {metal} | {score['accuracy']:.2%} | {score['macro_precision']:.2%} | {score['macro_f1']:.4f} | {report['baseline_majority']['macro_f1']:.4f} | {report['baseline_persistence']['macro_f1']:.4f} |")
    text += ["", "Selección entre 18 candidatos únicamente con validación temporal de desarrollo. Holdout de 36 observaciones, ya examinado en versiones anteriores: comparación retrospectiva, no prueba prospectiva inédita.",
             "", "La plata mejora accuracy y macro-F1 respecto de la versión anterior, pero disminuye su precisión macro. El oro no mejora. Consulta [comparación completa](12_MODELO_Y_METRICAS.md).", "", "## Noticias y Gemini", "",
             "| Metal | Seleccionadas | Analizadas | Relevantes | Pendientes | Score medio |", "|---|---:|---:|---:|---:|---:|"]
    for metal in reports:
        context = snapshot(tables["news"], tables["sentiment"], metal)
        score_text = "Sin dato" if context["mean_sentiment"] is None else f"{context['mean_sentiment']:+.2f}"
        text.append(f"| {metal} | {context['selected']}/5 | {context['analyzed']} | {context['relevant']} | {context['pending']} | {score_text} |")
    gemini_logs = tables["ingestion_log"][tables["ingestion_log"].source.str.startswith("Gemini", na=False)].sort_values("id")
    if len(gemini_logs):
        last = gemini_logs.iloc[-1]
        text += ["", f"Último registro Gemini ({last.fetched_at}): {last.status}. {last.detail}"]
    text += ["", "Los lotes se basan en los últimos 31 días; una noticia sobre ambos metales puede formar parte de ambos lotes y se consulta una vez. Pendiente no significa neutral. El texto disponible son titulares y fragmentos públicos breves, no artículos completos.",
             "", "El contexto aparece junto a la predicción y se exporta en su JSON. El XGBoost todavía excluye sentimiento/producción por falta de historial suficiente; no se atribuye la mejora de plata a Gemini.",
             "", "## Últimas salidas", ""]
    for metal in reports:
        prediction = json.loads((ROOT / "artifacts" / metal / "prediction.json").read_text(encoding="utf-8"))
        text.append(f"- {metal}: objetivo {prediction['target_month'][:7]}, {prediction['trend']}, probabilidad máxima {prediction['max_probability']:.1%}, riesgo {prediction['risk']}, señal {prediction['signal']}.")
    text += ["", "## Software y límites", "",
             f"- {tests_passed} pruebas automatizadas aprobadas en esta comprobación." if tests_passed is not None else "- Ejecuta `python -m pytest -q` para verificar esta copia.",
             "- Exportación y carga de modelos JSON comprobadas; revisión de la app con Oro y Plata.",
             "- Fuentes revisadas y probabilidades no calibradas externamente: el experimento no demuestra rentabilidad.",
             "- La combinación de pandas/NumPy genera avisos de deprecación; consulta requirements-lock.txt para las versiones verificadas.",
             "- La instalación en el equipo físico del docente se comprueba allí con su guía. Los ZIP excluyen .env y credenciales.",
             "- Hugging Face/Docker preparados, aún no publicados ni construidos en una cuenta externa."]
    path = ROOT / "docs" / "08_VERIFICACION.md"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Informe actualizado: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int)
    write(parser.parse_args().tests_passed)
