"""API HTTP local de FinBERT; sólo escucha en 127.0.0.1.

Puede iniciarse manualmente con python -m metals.local_sentiment_api o utilizarse
automáticamente durante analyze. No requiere una cuenta o clave de un tercero.
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests

from .sentiment_providers import LOCAL_MODEL, LOCAL_PROMPT_VERSION, get_engine, prepare_model

DEFAULT_URL = "http://127.0.0.1:8766"
_server = None
_start_lock = threading.Lock()


def local_endpoint():
    url = os.environ.get("SENTIMENT_LOCAL_URL", DEFAULT_URL).rstrip("/")
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment or not parsed.port):
        raise ValueError("SENTIMENT_LOCAL_URL debe ser http://127.0.0.1:PUERTO; no se envían noticias a destinos externos.")
    return url, parsed.port


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # No registrar textos completos de noticias en la terminal.
        pass

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "Ruta no disponible"})
        self.reply(200, {"status": "ok", "model": LOCAL_MODEL, "prompt_version": LOCAL_PROMPT_VERSION})

    def do_POST(self):
        if self.path != "/analyze":
            return self.reply(404, {"error": "Ruta no disponible"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 30000 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(400, {"error": "Se requiere JSON breve"})
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text")
            self.reply(200, get_engine().analyze(text))
        except (ValueError, TypeError, AttributeError) as exc:
            self.reply(400, {"error": str(exc)[:350]})
        except Exception:
            self.reply(503, {"error": "No se pudo cargar o ejecutar FinBERT. Revisa dependencias y modelo local."})


def local_client():
    client = requests.Session()
    client.trust_env = False  # Los proxies externos no reciben el texto local.
    return client


def ensure_local_api():
    global _server
    url, port = local_endpoint()
    with _start_lock:
        with local_client() as client:
            try:
                response = client.get(url + "/health", timeout=2)
                body = response.json()
                if response.ok and body.get("model") == LOCAL_MODEL and body.get("prompt_version") == LOCAL_PROMPT_VERSION:
                    return url
                raise ValueError("El puerto local pertenece a otro servicio; cambia SENTIMENT_LOCAL_URL.")
            except requests.ConnectionError:
                pass
        # Precargar antes de ofrecer health=ok; la primera descarga puede tardar.
        get_engine()
        try:
            _server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            raise ValueError("No se pudo abrir el puerto local para FinBERT.") from exc
        _server.daemon_threads = True
        threading.Thread(target=_server.serve_forever, daemon=True, name="finbert-local-api").start()
    return url


def analyze_local(text):
    url = ensure_local_api()
    with local_client() as client:
        response = client.post(url + "/analyze", json={"text": text}, timeout=(5, 90))
        if not response.ok:
            raise ValueError(f"FinBERT local HTTP {response.status_code}: {response.json().get('error', 'error de inferencia')}")
        body = response.json()
    if body.get("model") != LOCAL_MODEL or body.get("prompt_version") != LOCAL_PROMPT_VERSION:
        raise ValueError("La API local respondió con un modelo distinto al fijado.")
    return body


def main():
    parser = argparse.ArgumentParser(description="API local de sentimiento financiero FinBERT")
    parser.add_argument("--prepare", action="store_true", help="Descargar y verificar el modelo sin iniciar servidor")
    args = parser.parse_args()
    if args.prepare:
        print(f"Modelo preparado: {prepare_model()}")
        return
    get_engine()
    url, port = local_endpoint()
    print(f"FinBERT local disponible en {url}; detener con Ctrl+C.", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
