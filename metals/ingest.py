"""Descargas reales. Un fallo nunca se reemplaza por datos simulados."""
import calendar
import hashlib
import io
import html
import re
import time
from pathlib import Path
from urllib.parse import urlparse
import feedparser
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import DATA, START, WB_URL, WB_PAGE, SERIES, FEEDS
from .storage import now, upsert, log
from .news_filter import matched_metals, FILTER_VERSION

def session():
    s = requests.Session()
    s.headers["User-Agent"] = "MetalsAcademicMVP/0.1 (research; public data)"
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return s

def download(url, filename):
    response = session().get(url, timeout=(10, 60))
    response.raise_for_status()
    body = response.content
    digest = hashlib.sha256(body).hexdigest()
    dest = DATA / "raw" / f"{digest[:12]}_{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(body)
    return body, digest

def parse_world_bank(body):
    raw = pd.read_excel(io.BytesIO(body), sheet_name="Monthly Prices", header=None)
    # Detectar cabecera por nombre, sin depender de una posición fija de columna.
    headers = [i for i, row in raw.iterrows() if "Gold" in row.values and "Silver" in row.values]
    if len(headers) != 1:
        raise ValueError("Cambió la estructura de Pink Sheet: falta cabecera Gold/Silver.")
    h = headers[0]
    columns = {str(v).strip(): i for i, v in raw.iloc[h].items()}
    dates = raw.iloc[:, 0].astype(str)
    mask = dates.str.fullmatch(r"\d{4}M\d{2}")
    index = pd.to_datetime(dates[mask].str.replace("M", "-", regex=False)) + pd.offsets.MonthEnd(0)
    frame = pd.DataFrame({"date": index.to_numpy()})
    for source, metal in [("Gold", "gold"), ("Silver", "silver")]:
        frame[metal] = pd.to_numeric(raw.loc[mask, columns[source]], errors="coerce").to_numpy()
    if frame.date.duplicated().any():
        raise ValueError("Pink Sheet contiene meses duplicados.")
    if (frame[["gold", "silver"]].dropna() <= 0).any().any():
        raise ValueError("Precio no positivo en la fuente.")
    return frame.sort_values("date")

def market(conn, local_file=None):
    url = WB_URL
    if not local_file:
        # El Banco Mundial cambia la carpeta del histórico entre publicaciones.
        page = session().get(WB_PAGE, timeout=(10, 45))
        page.raise_for_status()
        links = re.findall(r'https://thedocs\.worldbank\.org/[^\s"<>]+CMO-Historical-Data-Monthly\.xlsx', html.unescape(page.text))
        if links:
            url = links[0]
        else:
            log(conn, "World Bank enlace", "warning", 0, "No se encontró enlace en portada; se intenta URL oficial configurada.")
    body, sha = (Path(local_file).read_bytes(), "") if local_file else download(url, "world_bank.xlsx")
    if not sha:
        sha = hashlib.sha256(body).hexdigest()
    frame = parse_world_bank(body)
    cutoff = pd.Timestamp.now().normalize().replace(day=1)
    frame = frame[(frame.date >= START) & (frame.date < cutoff)]
    rows = []
    timestamp = now()
    for record in frame.to_dict("records"):
        for metal in ("gold", "silver"):
            if pd.notna(record[metal]):
                rows.append(dict(date=record["date"].strftime("%Y-%m-%d"), metal=metal,
                                 price=float(record[metal]), source=url, fetched_at=timestamp))
    if not rows:
        raise ValueError("No se recibieron precios válidos.")
    if (pd.Timestamp.now().normalize() - frame.date.max()).days > 100:
        log(conn, "World Bank vigencia", "warning", len(rows), f"Archivo antiguo: último mes {frame.date.max():%Y-%m}.")
    upsert(conn, "prices", rows)
    log(conn, "World Bank", "ok", len(rows), f"USD/oz troy, promedios mensuales; hasta {frame.date.max():%Y-%m}", sha)
    return len(rows)

def macro(conn, series_id, local_file=None):
    if series_id not in SERIES:
        raise ValueError("Serie no permitida")
    download_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    body, sha = (Path(local_file).read_bytes(), "") if local_file else download(download_url, f"{series_id}.csv")
    frame = pd.read_csv(io.BytesIO(body))
    if series_id not in frame or len(frame.columns) != 2:
        raise ValueError(f"CSV FRED inválido para {series_id}.")
    frame.columns = ["date", "value"]
    frame.date = pd.to_datetime(frame.date, errors="coerce")
    frame.value = pd.to_numeric(frame.value, errors="coerce")
    frame = frame.dropna()
    frame = frame[(frame.date >= START) & (frame.date <= pd.Timestamp.now().normalize())]
    if frame.empty or frame.date.duplicated().any():
        raise ValueError(f"Serie {series_id} vacía o duplicada.")
    timestamp = now()
    rows = [dict(date=r.date.strftime("%Y-%m-%d"), series=series_id, value=float(r.value),
                 source=f"https://fred.stlouisfed.org/series/{series_id}", fetched_at=timestamp)
            for r in frame.itertuples()]
    upsert(conn, "macro", rows)
    log(conn, series_id, "ok", len(rows), "Valores revisados actuales; rezagos en features, sin garantía point-in-time", sha or hashlib.sha256(body).hexdigest())
    return len(rows)

def clean_text(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()

def news(conn, source, url, limit=40):
    body, sha = download(url, re.sub(r"\W", "_", source) + ".xml")
    feed = feedparser.parse(body)
    if not feed.entries:
        raise ValueError("RSS vacío o no válido. Consulta la web original o importa un CSV con fuente y fecha.")
    rows = []
    timestamp = now()
    excluded = 0
    for entry in feed.entries[:limit]:
        title = clean_text(entry.get("title", ""))[:500]
        summary = clean_text(entry.get("summary", ""))[:2000]
        if not matched_metals(title, summary):
            excluded += 1
            continue
        date = entry.get("published_parsed") or entry.get("updated_parsed")
        link = entry.get("link", "")
        if urlparse(link).scheme not in {"http", "https"}:
            continue
        if str(entry.get("published", "")).isdigit():
            # Algunos RSS añaden <published> Unix y sobrescriben pubDate al parsear.
            published = pd.Timestamp(int(entry["published"]), unit="s", tz="UTC")
        elif date:
            published = pd.Timestamp(calendar.timegm(date), unit="s", tz="UTC")
        else:
            continue
        if published > pd.Timestamp(timestamp):
            continue
        rows.append(dict(id=hashlib.sha256(link.encode()).hexdigest(),
                         title=title, summary=summary, url=link, source=source,
                         published_at=published.isoformat(), fetched_at=timestamp))
    upsert(conn, "news", rows)
    log(conn, source, "ok" if rows else "empty", len(rows), f"RSS: {FILTER_VERSION}; {excluded} excluidas por tema; primera captura conservada", sha)
    return len(rows)

def import_production(conn, path):
    frame = pd.read_csv(path)
    required = {"metal", "year", "tonnes", "available_at", "source"}
    if not required.issubset(frame):
        raise ValueError(f"Columnas requeridas: {sorted(required)}")
    timestamp = now()
    rows = []
    for r in frame.to_dict("records"):
        date = pd.Timestamp(r["available_at"])
        if r["metal"] not in {"gold", "silver"} or not float(r["tonnes"]) > 0:
            raise ValueError("Metal o producción no válido")
        if int(r["year"]) != float(r["year"]) or date.year <= int(r["year"]):
            raise ValueError("Usa producción anual observada, publicada después del año informado.")
        if date > pd.Timestamp.now() or not str(r["source"]).startswith("https://"):
            raise ValueError("Fecha futura o fuente ausente")
        rows.append(dict(metal=r["metal"], year=int(r["year"]), tonnes=float(r["tonnes"]),
                         available_at=date.isoformat(), source=r["source"], fetched_at=timestamp))
    upsert(conn, "production", rows)
    log(conn, "Producción importada", "ok", len(rows), str(Path(path).name))
    return len(rows)

def import_news(conn, path):
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"title", "summary", "url", "source", "published_at"}
    if not required.issubset(frame):
        raise ValueError(f"Columnas requeridas: {sorted(required)}")
    rows = []
    excluded = 0
    for item in frame.to_dict("records"):
        url = str(item["url"])
        if urlparse(url).scheme != "https" or not item["title"] or not item["source"]:
            raise ValueError("Cada noticia requiere título, fuente y URL HTTPS.")
        published = pd.to_datetime(item["published_at"], utc=True)
        if len(str(item["published_at"])) == 10:
            # Fuente con precisión de día: fin del día como disponibilidad conservadora.
            published = published + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if pd.isna(published) or published > pd.Timestamp.now(tz="UTC"):
            raise ValueError("Fecha de noticia inválida o futura.")
        if not matched_metals(str(item["title"])[:500], str(item["summary"])[:2000]):
            excluded += 1
            continue
        rows.append(dict(id=hashlib.sha256(url.encode()).hexdigest(), title=str(item["title"])[:500],
                         summary=str(item["summary"])[:2000], url=url, source=str(item["source"]),
                         published_at=published.isoformat(), fetched_at=now()))
    upsert(conn, "news", rows)
    log(conn, "Noticias CSV", "ok" if rows else "empty", len(rows), f"{Path(path).name}; {FILTER_VERSION}; {excluded} excluidas por tema", hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return len(rows)

def ingest_all(conn):
    results = {}
    jobs = [("World Bank", lambda: market(conn))]
    jobs += [(key, lambda key=key: macro(conn, key)) for key in SERIES]
    jobs += [(name, lambda name=name, url=url: news(conn, name, url)) for name, url in FEEDS.items()]
    def targeted():
        from .news_sources import fetch_targeted_news
        rows, diagnostics = fetch_targeted_news(per_metal=5)
        upsert(conn, "news", rows)
        for item in diagnostics["sources"]:
            log(conn, item["source"], item["status"], item["rows"], item["detail"], item["sha256"])
        log(conn, "Lote por metal", "ok" if rows else "empty", len(rows), str(diagnostics["coverage"]))
        return len(rows)
    jobs.append(("Noticias específicas oro/plata", targeted))
    for name, job in jobs:
        try:
            count = job()
            results[name] = f"OK: {count} registros" if count else "SIN REGISTROS VÁLIDOS: revisar fechas y formato de la fuente."
        except (requests.RequestException, ValueError, OSError, KeyError) as exc:
            # Estas fuentes públicas no incluyen claves en sus URLs.
            detail = f"{type(exc).__name__}: {str(exc)[:350]}"
            log(conn, name, "error", 0, detail)
            results[name] = detail + " | Se conservan los registros previos, si existen."
        print(f"{name}: {results[name]}", flush=True)
    return results
