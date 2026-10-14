"""SQLite almacena datos normalizados y conserva la trazabilidad de ingesta."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from .config import DB, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
 date TEXT NOT NULL, metal TEXT NOT NULL, price REAL NOT NULL CHECK(price>0),
 source TEXT NOT NULL, fetched_at TEXT NOT NULL, PRIMARY KEY(date,metal));
CREATE TABLE IF NOT EXISTS macro (
 date TEXT NOT NULL, series TEXT NOT NULL, value REAL NOT NULL,
 source TEXT NOT NULL, fetched_at TEXT NOT NULL, PRIMARY KEY(date,series));
CREATE TABLE IF NOT EXISTS news (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, summary TEXT, url TEXT NOT NULL,
 source TEXT NOT NULL, published_at TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sentiment (
 news_id TEXT NOT NULL, metal TEXT NOT NULL, score REAL, relevant INTEGER NOT NULL,
 explanation TEXT, evidence TEXT, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
 analyzed_at TEXT NOT NULL, PRIMARY KEY(news_id,metal,model,prompt_version));
CREATE TABLE IF NOT EXISTS production (
 metal TEXT NOT NULL, year INTEGER NOT NULL, tonnes REAL NOT NULL CHECK(tonnes>0),
 available_at TEXT NOT NULL, source TEXT NOT NULL, fetched_at TEXT NOT NULL,
 PRIMARY KEY(metal,year,available_at));
CREATE TABLE IF NOT EXISTS ingestion_log (
 id INTEGER PRIMARY KEY, source TEXT, fetched_at TEXT, status TEXT,
 rows_count INTEGER, detail TEXT, sha256 TEXT);
"""

def now():
    return datetime.now(timezone.utc).isoformat()

def connect(path=None):
    ensure_dirs()
    conn = sqlite3.connect(path or DB, timeout=30)
    conn.executescript(SCHEMA)
    return conn

def upsert(conn, table, rows):
    if not rows:
        return
    allowed = {"prices", "macro", "news", "sentiment", "production"}
    if table not in allowed:
        raise ValueError("Tabla no permitida")
    cols = list(rows[0])
    # Noticias y análisis son observaciones de primera captura: nunca retrofecharlos.
    mode = "IGNORE" if table in {"news", "sentiment", "production"} else "REPLACE"
    if table == "news":
        with conn:
            conn.executemany(f"INSERT INTO news ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) "
                             "ON CONFLICT(id) DO UPDATE SET published_at=excluded.published_at",
                             [tuple(r[c] for c in cols) for r in rows])
        return
    with conn:
        conn.executemany(f"INSERT OR {mode} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                         [tuple(r[c] for c in cols) for r in rows])

def log(conn, source, status, count, detail="", sha=""):
    with conn:
        conn.execute("INSERT INTO ingestion_log(source,fetched_at,status,rows_count,detail,sha256) VALUES (?,?,?,?,?,?)",
                     (source, now(), status, count, detail, sha))

def read(conn, table):
    if table not in {"prices", "macro", "news", "sentiment", "production", "ingestion_log"}:
        raise ValueError("Tabla no permitida")
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)

def dataset_sha256(dataset):
    # Finales de línea fijos: la huella debe coincidir entre Windows (entrenamiento) y Linux (Render).
    return hashlib.sha256(dataset.to_csv(index=False, lineterminator="\n").encode()).hexdigest()

def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
