"""Crear un ZIP revisable para Spaces. No publica ni necesita credenciales."""
import argparse
from contextlib import closing
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def package(include_data=False, target_kind="space"):
    required = ["app.py", "Dockerfile", "requirements.txt", "requirements-lock.txt", "README.md", ".dockerignore", ".streamlit/config.toml"]
    files = [ROOT / p for p in required]
    if target_kind == "docente":
        files.extend(ROOT / name for name in ("INSTALAR.cmd", "INICIAR.cmd", "ACTUALIZAR.cmd", "LEEME_DOCENTE.md", "pytest.ini", ".env.example", ".gitignore", ".vscode/settings.json", ".vscode/launch.json"))
    for folder in ("metals", "docs", "templates"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".md", ".csv"})
    if target_kind == "docente":
        for folder in ("scripts", "tests"):
            files.extend(p for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts)
    if include_data:
        for metal in ("gold", "silver"):
            for name in ("model.json", "evaluation_model.json", "report.json", "prediction.json", "holdout_predictions.csv"):
                files.append(ROOT / "artifacts" / metal / name)
    missing = [str(p.relative_to(ROOT)) for p in files if not p.exists()]
    if missing:
        raise ValueError(f"Faltan archivos: {missing}. Instala, descarga, entrena y predice primero.")
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"metals-{target_kind}-{stamp}.zip"
    temporary_zip = target.with_suffix(".zip.partial")
    manifest = []
    with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            name = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            z.writestr(name, payload)
            manifest.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest()})
        if include_data:
            source_db = ROOT / "data" / "metals.sqlite"
            if not source_db.exists():
                raise ValueError("Falta SQLite")
            # SQLite backup obtiene una copia coherente aunque la app esté abierta.
            with tempfile.TemporaryDirectory(dir=out) as scratch:
                backup = Path(scratch) / "metals.sqlite"
                with closing(sqlite3.connect(source_db)) as src, closing(sqlite3.connect(backup)) as dst:
                    src.backup(dst)
                payload = backup.read_bytes()
                z.writestr("data/metals.sqlite", payload)
                manifest.append({"path": "data/metals.sqlite", "sha256": hashlib.sha256(payload).hexdigest()})
        z.writestr("PACKAGE_MANIFEST.json", json.dumps({"created_at": stamp, "files": manifest}, indent=2))
    temporary_zip.replace(target)
    print(f"Paquete creado: {target}")
    print(f"Archivos: {len(manifest)}. .env y credenciales no están en la lista permitida.")
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-data", action="store_true", help="Incluye SQLite y modelos locales para una demostración sin descarga")
    parser.add_argument("--target", choices=["space", "docente"], default="space")
    args = parser.parse_args()
    package(args.include_data, args.target)
