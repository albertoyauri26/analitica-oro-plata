"""Entrada ASGI para Vercel: Streamlit sirve app.py con st.App, sin el servidor de `streamlit run`."""
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if os.environ.get("VERCEL"):
    # En Vercel sólo /tmp admite escritura: cada instancia copia ahí la base y los modelos al arrancar.
    copy = Path(tempfile.gettempdir()) / "aurum"
    for name in ("data", "artifacts"):
        if not (copy / name).exists():
            shutil.copytree(ROOT / name, copy / name)
    os.environ.setdefault("METALS_DATA_DIR", str(copy / "data"))
    os.environ.setdefault("METALS_ARTIFACT_DIR", str(copy / "artifacts"))

import streamlit as st  # noqa: E402  (después de fijar las rutas que lee metals.config)

app = st.App(ROOT / "app.py")
