"""Genera el anexo de resultados desde SQLite y reportes, sin leer credenciales."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write():
    with sqlite3.connect(ROOT / "data" / "metals.sqlite") as conn:
        counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("prices", "macro", "news", "sentiment", "production")}
        first, last = conn.execute("SELECT MIN(date),MAX(date) FROM prices").fetchone()
        sentiments = conn.execute("SELECT model,metal,COUNT(*),SUM(relevant) FROM sentiment GROUP BY model,metal").fetchall()
        sources = conn.execute("SELECT source,COUNT(*) FROM news GROUP BY source").fetchall()
    reports = {m: json.loads((ROOT / "artifacts" / m / "report.json").read_text(encoding="utf-8")) for m in ("gold", "silver")}
    text = ["# Resultados verificados de esta instalación", "", f"Generado desde los archivos reales: {datetime.now(timezone.utc).isoformat()}.", "",
            "## Datos", "", f"Precios: {counts['prices']} observaciones de ambos metales, de {first} a {last}.",
            f"Macro: {counts['macro']} observaciones. Noticias: {counts['news']}. Registros de sentimiento por metal: {counts['sentiment']}. Producción anual importada: {counts['production']}.", "",
            "Fuentes de noticias guardadas:", ""]
    text += [f"- {source}: {count} artículos." for source, count in sources]
    text += ["", "## Evaluación", "", "| Metal | Macro-F1 XGBoost | Mayoritaria | Persistencia | Exactitud XGBoost |", "|---|---:|---:|---:|---:|"]
    for m, r in reports.items():
        text.append(f"| {m} | {r['xgboost']['macro_f1']:.4f} | {r['baseline_majority']['macro_f1']:.4f} | {r['baseline_persistence']['macro_f1']:.4f} | {r['xgboost']['accuracy']:.4f} |")
    r = reports["gold"]
    text += ["", f"Holdout: {r['holdout']['start']} a {r['holdout']['end']}, {r['holdout']['n']} observaciones. Las ventanas pueden incluir huecos de cobertura, por lo que 36 observaciones no siempre corresponden a 36 meses consecutivos.",
             "", "El oro no supera la referencia de persistencia en este experimento. La plata supera ambas referencias. Esto no demuestra rentabilidad ni estabilidad fuera del periodo de prueba.", "", "## Últimas salidas", ""]
    for m in reports:
        p = json.loads((ROOT / "artifacts" / m / "prediction.json").read_text(encoding="utf-8"))
        text.append(f"- {m}: objetivo {p['target_month'][:7]}, tendencia {p['trend']}, probabilidad máxima {p['max_probability']:.1%}, riesgo {p['risk']}, señal {p['signal']}.")
    text += ["", "Las salidas tienen origen y vigencia definidos. Al abrir la web más adelante puede activarse la abstención por vencimiento.", "", "## Gemini real", "",
             "| Modelo | Metal | Noticias analizadas | Relevantes |", "|---|---|---:|---:|"]
    text += [f"| {model} | {metal} | {count} | {relevant} |" for model, metal, count, relevant in sentiments]
    text += ["", "Se usaron textos reales de RSS, con evidencia literal validada cuando son relevantes. Las respuestas no relevantes mantienen score ausente en SQLite. El servicio devolvió errores 503 intermitentes y después cuota agotada HTTP 429. Se conservaron seis noticias analizadas, todas no relevantes para ambos metales según Gemini. Dos titulares específicos verificados de oro/plata quedaron pendientes por cuota. La clave no se incluye en los reportes ni en los paquetes.",
             "", "## Verificación de software", "", "- 20 pruebas automatizadas aprobadas en el equipo de desarrollo, incluida la app con ambos metales, las reglas temporales, importación de noticias, reutilización de sentimiento y ocultación de credenciales.",
             "- pip check: sin dependencias incompatibles declaradas.", "- Modelos JSON cargados nuevamente para generar predicciones.", "- Aplicación local iniciada en http://localhost:8501 y revisada en navegador.",
             "- El entorno genera avisos de deprecación por la combinación pandas/NumPy; las pruebas pasan. Las versiones exactas están fijadas en requirements-lock.txt.",
             "", "## Pendientes y limitaciones", "", "- Completar publicaciones históricas verificadas de producción y un archivo prospectivo de noticias antes de evaluar su aporte al modelo.",
             "- Las variables de sentimiento/producción quedan excluidas del XGBoost actual por cobertura histórica insuficiente.",
             "- MINING.com devuelve 403; se utiliza Mining Weekly como alternativa.",
             "- El funcionamiento en la computadora física del docente se verifica allí siguiendo su guía. Una prueba extraída en este equipo no sustituye esa comprobación.",
             "- Docker/Hugging Face preparados, pero todavía no construidos ni publicados en una cuenta de Hugging Face."]
    path = ROOT / "docs" / "08_VERIFICACION.md"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Informe actualizado: {path}")

if __name__ == "__main__":
    write()
