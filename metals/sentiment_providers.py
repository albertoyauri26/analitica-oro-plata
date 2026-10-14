"""FinBERT local: pesos públicos fijados y tono financiero por fragmento.

No interpreta causalidad sobre el precio del metal. No utiliza reglas léxicas
para fabricar el sentimiento: las probabilidades salen de la red FinBERT.
"""
import hashlib
import json
import re
import threading

import requests

from .config import DATA
from .news_filter import matched_metals

MODEL_REPO = "ProsusAI/finbert"
MODEL_REVISION = "7db323f79b751944bcfa66298ec06977e4518306"
MODEL_SHA256 = "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00"
LOCAL_MODEL = f"finbert-local/{MODEL_REPO}@{MODEL_REVISION}"
LOCAL_PROMPT_VERSION = "finbert-financial-tone-v1"
MODEL_FILES = ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors")
_engine = None
_engine_lock = threading.Lock()


def model_directory():
    return DATA / "models" / "finbert" / MODEL_REVISION


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model():
    """Descarga sólo JSON, vocabulario y safetensors; nunca pickle/código remoto."""
    dest = model_directory()
    dest.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        path = dest / filename
        if path.exists():
            if filename == "model.safetensors" and sha256(path) != MODEL_SHA256:
                raise ValueError("El archivo FinBERT no coincide con su SHA-256. Revisa data/models antes de usarlo.")
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}"
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            partial = path.with_suffix(path.suffix + ".partial")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if filename == "model.safetensors" and sha256(partial) != MODEL_SHA256:
            raise ValueError("Descarga FinBERT incompleta o hash distinto; no se carga el archivo.")
        partial.replace(path)
    manifest = {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                "files": {name: sha256(dest / name) for name in MODEL_FILES}}
    (dest / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def metal_fragments(text, metal):
    """Fragmentos literales: separar contrastes antes de clasificar cada metal."""
    parts = re.split(r"\n+|(?<=[.!?;])\s+|\s+\b(?:while|whereas|but)\b\s+", text, flags=re.I)
    fragments = [part.strip() for part in parts if metal in matched_metals(part)]
    # El prefiltro puede necesitar el contexto combinado del título y resumen.
    if not fragments and metal in matched_metals(text):
        fragments = [text]
    return list(dict.fromkeys(fragments))


class FinBertEngine:
    def __init__(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise ValueError("Faltan dependencias FinBERT. Ejecuta pip install -r requirements-sentiment.txt; consulta docs/14_PROVEEDORES_SENTIMIENTO.md.") from exc
        directory = prepare_model()
        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels.values()) != {"positive", "negative", "neutral"}:
            raise ValueError("Etiquetas FinBERT inesperadas; no se interpretan índices por suposición.")
        self.lock = threading.Lock()

    def probabilities(self, texts):
        with self.lock, self.torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            values = self.model(**inputs).logits.softmax(dim=-1).cpu().tolist()
        return [{self.labels[i]: float(value) for i, value in enumerate(row)} for row in values]

    def analyze(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 6000:
            raise ValueError("El texto debe tener entre 1 y 6000 caracteres.")
        if re.search(r"\b(?:oro|plata)\b", text, re.I) and not re.search(r"\b(?:gold|silver)\b", text, re.I):
            raise ValueError("FinBERT está entrenado para inglés. Usa Gemini para este texto en español; no se inventa una traducción.")
        results = []
        for metal in ("gold", "silver"):
            fragments = metal_fragments(text, metal)
            if not fragments:
                results.append({"metal": metal, "relevant": False, "score": 0.0,
                                "explanation": "Sin mención temática del metal en el texto proporcionado.", "evidence": ""})
                continue
            predictions = self.probabilities(fragments)
            probabilities = {label: sum(row[label] for row in predictions) / len(predictions)
                             for label in ("positive", "negative", "neutral")}
            score = probabilities["positive"] - probabilities["negative"]
            label = max(probabilities, key=probabilities.get)
            language_label = {"positive": "positivo", "negative": "negativo", "neutral": "neutral"}[label]
            strongest = max(range(len(predictions)), key=lambda i: max(predictions[i].values()))
            results.append({"metal": metal, "relevant": True, "score": score,
                "explanation": (f"FinBERT: tono financiero {language_label}. "
                    f"P(positivo)={probabilities['positive']:.1%}, P(neutral)={probabilities['neutral']:.1%}, "
                    f"P(negativo)={probabilities['negative']:.1%}. "
                    "Índice=P(positivo)-P(negativo). Relevancia temática por mención del metal; "
                    "no determina el efecto causal sobre su precio."),
                "evidence": fragments[strongest][:300]})
        return {"results": results, "model": LOCAL_MODEL, "prompt_version": LOCAL_PROMPT_VERSION,
                "semantics": "financial_text_tone"}


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FinBertEngine()
    return _engine
