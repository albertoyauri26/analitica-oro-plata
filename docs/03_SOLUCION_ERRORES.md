# Solución de errores

Trabaja desde la carpeta que contiene `app.py`. Lee la última línea del error antes de cambiar nada. Conserva el mensaje para revisarlo juntos y elimina cualquier clave de API antes de compartirlo.

| Síntoma | Causa probable | Acción |
|---|---|---|
| `python` no se reconoce | Python no está en PATH | Reabre VS Code después de instalar Python; comprueba `python --version` |
| `py -3.11` no puede crear proceso | Registro de una versión que ya no existe | Usa el `python` 3.13 instalado en este equipo |
| `No module named ...` | Intérprete incorrecto o dependencia ausente | Usa `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Bloqueo de `Activate.ps1` | Política de PowerShell | Usa directamente `.\.venv\Scripts\python.exe`; no necesitas activar |
| `ensurepip` / acceso denegado a Temp | Permisos del proceso o del entorno de ejecución | Ejecuta la creación de `.venv` en tu terminal local; en Codex autoriza sólo la instalación solicitada |
| FRED 429 | Demasiadas consultas | Conserva la base y reintenta más tarde; usa el CSV oficial descargado manualmente |
| 403, bloqueo regional o red | Proveedor, firewall o proxy | Abre la fuente oficial en navegador; descarga original e impórtalo. No uses cifras sustitutas |
| Error de certificados | Red corporativa o certificados | Comprueba fecha del equipo y configuración de red. No desactives TLS |
| Pink Sheet sin Gold/Silver | Cambio en cabecera o archivo incorrecto | Descarga el histórico mensual, pestaña Monthly Prices. Revisa `parse_world_bank` |
| Falta precio o macro del último mes | Descarga incompleta o publicación aún no disponible | Consulta `status` y fuentes; el sistema bloquea un nuevo entrenamiento incompleto |
| Menos de 180 meses útiles | Poca historia o nulos extensos | Descarga las series desde 2000; revisa qué fuente falta |
| Falta una clase en entrenamiento | Historia insuficiente o umbral inadecuado | Examina conteos, sin rellenarlos. Revisa el diseño antes de cambiar umbrales |
| Gemini: clave ausente | `.env` no configurado | Copia `.env.example` y agrega tu clave local |
| Gemini HTTP 400/404 | Modelo o esquema no admitido | Revisa nombre disponible en AI Studio y documentación de salidas estructuradas |
| Gemini HTTP 401/403 | Clave/permisos | Revisa tu clave y disponibilidad de servicio |
| Gemini HTTP 429 | Cuota o límite | Espera y revisa cuota; baja `--limit`. El error no se transforma en Neutral |
| Gemini HTTP 503 | Servicio temporalmente saturado | Los reintentos son limitados; conserva los resultados y vuelve a intentar después. No implica que la clave sea inválida |
| Evidencia no coincide | Gemini generó una cita inexistente | El resultado se rechaza. Revisa texto y prompt antes de reintentar |
| Dataset cambió desde entrenamiento | Nueva ingesta o sentimiento | Ejecuta `train` y después `predict` |
| Predicción vencida | El mes objetivo ya pasó | Actualiza con `run`; usa la anterior sólo como resultado histórico |
| Puerto 8501 ocupado | Otra app usa el puerto | Detén la otra sesión o usa `--server.port 8502` |
| App no se abre | Terminal cerrada o error de ejecución | Reejecuta `streamlit run app.py` y abre `http://localhost:8501` |
| Error al cargar XGBoost | Archivo dañado o versiones distintas | Reinstala versiones del lock o reentrena; no cargues archivos de origen desconocido |
| Space vacío tras reinicio | Disco efímero sin snapshot | Publica el paquete con SQLite y modelos, según la guía de despliegue |

Comandos para diagnosticar:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m metals.cli status
.\.venv\Scripts\python.exe -m pytest -q
```

Para otra puerta de la aplicación:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

No borres `data` ni `artifacts` como primera solución. Contienen fuentes, fechas y experimentos que sirven para diagnosticar y reproducir el trabajo.
