# Sentimiento por metal con continuidad local

La aplicación dispone de dos LLM, **Mistral** y **Gemini**, y de una **API HTTP local con FinBERT**. `SENTIMENT_PROVIDER` en `.env` elige el proveedor:

| Valor | Comportamiento |
|---|---|
| `mistral` | Sólo Mistral (LLM). Sin FinBERT: un fallo deja la noticia pendiente para la próxima ejecución. |
| `gemini` | Sólo Gemini (LLM). Sin FinBERT, igual que `mistral`. |
| `auto` | Gemini y, si falla tras los reintentos, FinBERT para las noticias pendientes del lote. |
| `finbert` | Sólo el modelo local. |

Ambos LLM reciben el mismo prompt, devuelven JSON restringido al mismo esquema y pasan la misma validación de evidencia literal. Los resultados indican el proveedor que realmente respondió.

### Mistral

1. Crear una clave en [console.mistral.ai](https://console.mistral.ai). El plan gratuito *Experiment* no pide tarjeta; requiere verificar un teléfono y Mistral puede usar los datos para mejorar sus modelos. El plan de pago no los usa.
2. En `.env`: `SENTIMENT_PROVIDER=mistral`, `MISTRAL_API_KEY=<clave>` y opcionalmente `MISTRAL_MODEL` (por defecto `ministral-14b-2512`, versión fija incluida en el plan gratuito: 30 solicitudes por minuto; Mistral Small y Medium muestran límite 0 en ese plan).
3. Verificar sin mostrar la clave: `.\.venv\Scripts\python.exe -m metals.cli mistral-check`.

Las esperas entre noticias (`MISTRAL_MIN_INTERVAL`, 3 s) y los reintentos ante HTTP 503 o 429 por minuto (`MISTRAL_ATTEMPTS`, 3) son configurables. Para Gemini existen `GEMINI_MIN_INTERVAL` y `GEMINI_ATTEMPTS`; su nivel gratuito admite unas 20 solicitudes diarias por modelo.

## Qué hace cada modelo

| Proveedor | Entrada y salida | Interpretación |
|---|---|---|
| Gemini | Título y extracto breve; relevancia, evidencia literal y puntuación entre −1 y +1 por metal | Interpretación del contexto solicitado para el metal; no causalidad demostrada |
| FinBERT local | Fragmentos en inglés que mencionan oro o plata en contexto financiero/minero; probabilidades positivo, neutral y negativo | Tono financiero del texto, no predicción del precio ni efecto causal |

FinBERT es un clasificador neuronal de texto financiero publicado por [ProsusAI](https://huggingface.co/ProsusAI/finbert). Su tarjeta describe tres salidas de sentimiento; no lo presenta como un predictor del oro o de la plata. El proyecto calcula el índice `P(positivo) − P(negativo)`, conservando las probabilidades en la explicación del resultado.

El prefiltro identifica el metal mencionado. El adaptador separa oraciones y contrastes como “while”, “whereas” y “but” antes de aplicar FinBERT a cada metal; una noticia sobre ambos metales puede producir tonos diferentes. Si una misma frase trata ambos metales, puede compartir tono. La relevancia de FinBERT es temática mediante el prefiltro, no una evaluación causal equivalente a la de Gemini.

**No se reemplaza un error por sentimiento neutral.** Los fallos quedan pendientes. Un resultado neutral es una clasificación real del proveedor. El modelo FinBERT utilizado está orientado al inglés; no se inventan traducciones para noticias en español. Para textos españoles corresponde Gemini o incorporar posteriormente un modelo multilingüe validado.

## Instalación en Visual Studio Code

En la terminal PowerShell, con la carpeta del proyecto abierta:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-sentiment.txt
.\.venv\Scripts\python.exe -m metals.local_sentiment_api --prepare
```

Se usa CPU; no requiere tarjeta gráfica. La primera preparación necesita internet. En Windows/Python 3.13 se observó una descarga de aproximadamente 619 MB para PyTorch y 438 MB para los pesos FinBERT, además de sus dependencias. Debe reservarse espacio adicional para los archivos instalados y la memoria necesaria para ejecutar el modelo. Las dependencias se instalan en `.venv`; los pesos se guardan dentro de `data/models/finbert/`, sin modificar los modelos XGBoost.

Una vez preparados los pesos y las dependencias, la inferencia local funciona sin internet. La recopilación de noticias nuevas sí requiere conexión. El ZIP de demostración puede conservar resultados SQLite para consultarlos en otra computadora sin reinstalar FinBERT; para **generar nuevos análisis locales**, instalar este complemento y preparar el modelo en esa computadora.

## Uso habitual

```powershell
.\.venv\Scripts\python.exe -m metals.cli news-refresh --per-metal 5 --analyze
```

El comando recopila hasta cinco noticias específicas por metal y analiza las pendientes. Para analizar únicamente el lote ya guardado:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --per-metal 5
```

En `.env`, los valores opcionales son:

```dotenv
SENTIMENT_PROVIDER=auto
SENTIMENT_LOCAL_URL=http://127.0.0.1:8766
```

`auto` usa Gemini cuando responde y FinBERT ante el primer fallo del lote. `gemini` exige ese proveedor y conserva pendientes si falla. `finbert` usa directamente la API local, sin llamar a Google. No hace falta copiar o compartir la clave Gemini con el docente si se utiliza FinBERT local.

El comando inicia automáticamente la API local cuando la necesita. Durante una ejecución CLI el servidor vive dentro de ese proceso y termina con él; no hay que abrir otra ventana ni dejar un servicio permanente. Para probarlo manualmente en una terminal independiente:

```powershell
.\.venv\Scripts\python.exe -m metals.local_sentiment_api
```

La API expone `GET /health` y `POST /analyze` con JSON `{"text":"texto financiero en inglés"}`. Escucha exclusivamente en `127.0.0.1`. El cliente admite únicamente una dirección HTTP de ese mismo host, desactiva proxies para las peticiones locales y comprueba el identificador del modelo devuelto. No publica un servicio en internet. Detener el servidor manual con Ctrl+C.

## Fallos, caché y trazabilidad

- Un fallo de Gemini (incluidos 429 y 503), respuesta inválida o error de red abre el circuito para el resto del lote. No se reintenta la cuota agotada ni se prueban otros modelos remotos para eludirla.
- Cada artículo se persiste cuando termina: un fallo posterior conserva lo ya analizado.
- El modo `auto` reutiliza resultados existentes de Gemini o FinBERT para esa noticia y metal. Volver a ejecutar no genera nuevas llamadas para resultados ya guardados.
- Las filas guardan noticia, metal, evidencia literal, explicación, fecha de análisis, modelo y versión del procedimiento. FinBERT se identifica mediante el prefijo `finbert-local/` y la versión `finbert-financial-tone-v1`.
- Si FinBERT no puede cargar, comprobar los comandos de instalación, espacio disponible y el puerto local. Si otra aplicación ocupa 8766, configurar otro puerto en `SENTIMENT_LOCAL_URL`, manteniendo `127.0.0.1`.
- No se retrofechan los resultados: las noticias analizadas hoy sólo son información disponible desde hoy. La captura actual no sustituye un archivo histórico de noticias para validar incrementos de accuracy.

## Versión reproducible y procedencia

Se fija la revisión `7db323f79b751944bcfa66298ec06977e4518306` del repositorio `ProsusAI/finbert`: [conversión publicada por SFconvertbot](https://huggingface.co/ProsusAI/finbert/commit/7db323f79b751944bcfa66298ec06977e4518306). Esa revisión añade el formato safetensors mediante una propuesta automática dentro del repositorio; no se afirma que esté fusionada en su rama principal.

Sólo se descargan configuración, vocabulario y pesos safetensors. No se carga `pytorch_model.bin` ni se ejecuta código remoto: `trust_remote_code=False`, `local_files_only=True` y `use_safetensors=True`. El SHA-256 publicado de los pesos es `e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00`; el descargador exige su coincidencia. Un archivo `provenance.json` conserva los hashes de los archivos utilizados. Las versiones opcionales están fijadas en `requirements-sentiment.txt` y son independientes de la instalación base del dashboard.

La [documentación oficial de Transformers](https://huggingface.co/docs/transformers/installation) explica el uso con PyTorch y la carga de archivos locales. La ejecución local elimina la dependencia de la cuota de una API externa; no demuestra por sí misma mejor accuracy del sistema predictivo. El aporte al precio o tendencia debe evaluarse con datos históricos y validación temporal separada.
