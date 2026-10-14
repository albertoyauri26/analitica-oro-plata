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
    prediction_cmd = sub.add_parser("predict", help="Tendencia, riesgo, señal y escenario con sentimiento")
    prediction_cmd.add_argument("--news-weight", type=float, default=0.15, help="Peso experimental de noticias de 0 a 0.20; 0 desactiva la mezcla")
    sub.add_parser("run", help="Ejecutar ingesta, dataset, entrenamiento y predicción")
    sub.add_parser("status", help="Consultar cobertura almacenada")
    sub.add_parser("gemini-check", help="Verificar credencial y modelos disponibles sin mostrar la clave")
    sub.add_parser("mistral-check", help="Verificar credencial y modelos de Mistral sin mostrar la clave")
    news_refresh = sub.add_parser("news-refresh", help="Recopilar noticias específicas por metal; opcionalmente analizarlas")
    news_refresh.add_argument("--per-metal", type=int, default=5, choices=range(1, 21), metavar="1-20")
    news_refresh.add_argument("--analyze", action="store_true", help="Proveedor de SENTIMENT_PROVIDER en .env; los LLM consumen cuota")
    sentiment = sub.add_parser("sentiment", help="Analizar noticias: LLM (Gemini o Mistral) y API local alternativa")
    sentiment.add_argument("--provider", choices=("auto", "gemini", "mistral", "finbert"), help="Proveedor explícito para esta ejecución; gemini y mistral no usan respaldo local")
    news_quantity = sentiment.add_mutually_exclusive_group()
    news_quantity.add_argument("--limit", type=int, choices=range(1, 101), metavar="1-100", help="Límite total de artículos pendientes, modo manual")
    news_quantity.add_argument("--per-metal", type=int, choices=range(1, 21), metavar="1-20", help="Por defecto: 5 por cada metal")
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
        if args.command in {"gemini-check", "mistral-check"}:
            from .sentiment import check_connection
            print(json.dumps(check_connection(args.command.removesuffix("-check")), ensure_ascii=False, indent=2))
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
            print(f"Noticias analizadas: {analyze(conn, args.limit or 5, args.source, per_metal=None if args.limit else args.per_metal or 5, provider=args.provider)}")
        if args.command == "news-refresh":
            from .news_sources import fetch_targeted_news
            from .storage import upsert, log
            rows, diagnostics = fetch_targeted_news(per_metal=args.per_metal)
            upsert(conn, "news", rows)
            print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
            log(conn, "Noticias por metal", "ok" if rows else "empty", len(rows), json.dumps(diagnostics, ensure_ascii=False)[:1500])
            for result in diagnostics["sources"]:
                log(conn, result["source"], result["status"], result["rows"], result["detail"], result["sha256"])
            if args.analyze:
                from .sentiment import analyze
                print(f"Noticias analizadas: {analyze(conn, per_metal=args.per_metal)}")
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
            from .news_context import snapshot
            for metal in ("gold", "silver"):
                context = snapshot(read(conn, "news"), read(conn, "sentiment"), metal)
                print(json.dumps(predict(dataset, metal, news_context=context, hybrid_weight=getattr(args, "news_weight", 0.15)), indent=2, ensure_ascii=False))
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
