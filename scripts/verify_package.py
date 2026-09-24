"""Inspecciona y extrae el paquete del docente sin mostrar credenciales."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

def verify(path):
    key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY", "")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        required = {"app.py", "INSTALAR.cmd", "INICIAR.cmd", "data/metals.sqlite", "artifacts/gold/model.json", "artifacts/silver/model.json", "docs/09_INSTALACION_DOCENTE.md", "PACKAGE_MANIFEST.json"}
        if not required.issubset(names):
            raise ValueError(f"Paquete incompleto: {required - set(names)}")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError("Ruta no segura dentro del ZIP")
            if p.name == ".env" or ".venv" in p.parts or ".git" in p.parts:
                raise ValueError("Archivo privado no permitido")
            if key and key.encode() in archive.read(name):
                raise ValueError("Credencial detectada en el paquete. No distribuir.")
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        for entry in manifest["files"]:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise ValueError("Huella de archivo incompatible con manifiesto")
        out = ROOT / "dist" / (Path(path).stem + "-verificado")
        out.mkdir(parents=True, exist_ok=False)
        archive.extractall(out)
    print(f"Verificados {len(names)} archivos y sus huellas. Sin credencial ni .env.")
    print(f"Extraido para prueba: {out}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    verify(args.path)
