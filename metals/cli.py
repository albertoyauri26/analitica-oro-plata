"""Comandos reproducibles: python -m metals.cli --help."""
import argparse
import json
import sqlite3
import sys
import requests
from .config import ARTIFACTS, DATA, SERIES
from .storage import connect, read

def main():
    parser = argparse.ArgumentParser(description="Sistema académico de analítica predictiva para oro y plata")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Descargar precios, macro y noticias reales")
    sub.add_parser("build", help="Construir dataset mensual")
    sub.add_parser("train", help="Validación temporal, modelos y reportes")
    sub.add_parser("predict", help="Tendencia, riesgo y señal")
    sub.add_parser("run", help="Ejecutar ingesta, dataset, entrenamiento y predicción")
    sub.add_parser("status", help="Consultar cobertura almacenada")
    sub.add_parser("gemini-check", help="Verificar credencial y modelos disponibles sin mostrar la clave")
    sentiment = sub.add_parser("sentiment", help="Analizar noticias con Gemini (usa tu cuota)")
    sentiment.add_argument("--limit", type=int, default=5, choices=range(1, 101), metavar="1-100")
    sentiment.add_argument("--source", help="Opcional: Federal Reserve, Mining Weekly u otra fuente guardada")
    imp = sub.add_parser("import-production", help="Importar producción anual verificada")
    imp.add_argument("path")
    news_import = sub.add_parser("import-news", help="Importar títulos/extractos reales con URL y fecha verificadas")
    news_import.add_argument("path")
    wb = sub.add_parser("import-market", help="Importar XLSX Pink Sheet descargado manualmente")
    wb.add_argument("path")
    macro = sub.add_parser("import-macro", help="Importar CSV original de FRED")
    macro.add_argument("series", choices=list(SERIES))
    macro.add_argument("path")
    args = parser.parse_args()
    conn = connect()
    try:
        if args.command == "gemini-check":
            from .sentiment import check_connection
            print(json.dumps(check_connection(), ensure_ascii=False, indent=2))
        if args.command in {"ingest", "run"}:
            from .ingest import ingest_all
            ingest_all(conn)
        if args.command == "import-market":
            from .ingest import market
            print(f"Precios importados: {market(conn, args.path)}")
        if args.command == "import-macro":
            from .ingest import macro
            print(f"Observaciones importadas: {macro(conn, args.series, args.path)}")
        if args.command == "import-production":
            from .ingest import import_production
            print(f"Filas importadas: {import_production(conn, args.path)}")
        if args.command == "import-news":
            from .ingest import import_news
            print(f"Noticias importadas: {import_news(conn, args.path)}")
        if args.command == "sentiment":
            from .sentiment import analyze
            print(f"Noticias analizadas: {analyze(conn, args.limit, args.source)}")
        if args.command in {"build", "train", "predict", "run"}:
            from .features import build_dataset
            dataset = build_dataset(conn)
            print(f"Dataset: {len(dataset)} filas. Guardado en {DATA / 'analytical_dataset.csv'}")
        if args.command in {"train", "run"}:
            from .modeling import train_metal
            for metal in ("gold", "silver"):
                report = train_metal(dataset, metal)
                print(f"{metal}: macro-F1 holdout = {report['xgboost']['macro_f1']:.3f}")
        if args.command in {"predict", "run"}:
            from .decision import predict
            for metal in ("gold", "silver"):
                print(json.dumps(predict(dataset, metal), indent=2, ensure_ascii=False))
        if args.command == "status":
            for table in ("prices", "macro", "news", "sentiment", "production"):
                frame = read(conn, table)
                print(f"{table}: {len(frame)} registros")
                if not frame.empty and "date" in frame:
                    print(f"  {frame.date.min()} a {frame.date.max()}")
            print(read(conn, "ingestion_log").tail(10).drop(columns=["sha256"]).to_string(index=False))
    except (ValueError, FileNotFoundError, sqlite3.Error, requests.RequestException) as exc:
        print(f"ERROR: {exc}\nConsulta docs/03_SOLUCION_ERRORES.md", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
