# Actualización de septiembre de 2026

Ejecución verificada el 12 de octubre de 2026. Se conservó una copia previa de la base SQLite y de los modelos en `dist/before-september-update-20261012T211927Z`.

## Datos y predicciones

La descarga del Banco Mundial incorporó precios mensuales hasta el 30 de septiembre de 2026 para ambos metales. FRED proporcionó FEDFUNDS hasta septiembre, DGS10 hasta el 8 de octubre y VIXCLS hasta el 9 de octubre. CPIAUCSL continúa en agosto; no se imputó una observación inventada de septiembre.

Se reconstruyó el dataset (642 filas), se reentrenaron ambos modelos y se guardaron predicciones con origen 30 de septiembre y objetivo 31 de octubre. La metodología conserva el rezago de los indicadores técnicos: las variables de precio de esa predicción proceden de agosto. Incluir septiembre en el dashboard y en el dataset no significa eliminar ese rezago. Las métricas son retrospectivas y no prueban una mejora futura.

| Modelo | Accuracy | Precisión macro | F1 macro |
|---|---:|---:|---:|
| Oro | 44,44% | 32,11% | 0,3202 |
| Plata | 50,00% | 68,28% | 0,4301 |

## Prueba real de Gemini

La consulta de modelos autenticó correctamente la credencial y confirmó que el modelo configurado estaba disponible. La llamada real de análisis devolvió HTTP 503. No se generaron resultados nuevos de sentimiento con Gemini en esta ejecución.

Se utilizó después el respaldo local FinBERT: cinco noticias recientes de oro y cinco de plata analizadas y guardadas, con evidencia y proveedor. La selección corresponde a los últimos 31 días al ejecutar la actualización, no exclusivamente a noticias de septiembre. No se retrofecharon capturas ni análisis para introducir información futura en el entrenamiento histórico.

Los dos escenarios experimentales de sentimiento quedaron activos. Sus probabilidades ajustadas no tienen las métricas de validación del modelo XGBoost ni modifican por sí solas la señal de riesgo.

## Repetir la prueba en Visual Studio Code

Desde la carpeta raíz, usando la terminal PowerShell:

```powershell
.\.venv\Scripts\python.exe -m metals.cli gemini-check
.\.venv\Scripts\python.exe -m metals.cli sentiment --per-metal 5 --provider gemini
```

El parámetro `--provider gemini` fuerza la prueba con Gemini sin respaldo y sin modificar `.env`. Si no hay noticias pendientes para ese proveedor, reutiliza los resultados guardados; ese caso no representa una nueva llamada de generación.

Para ejecutar el respaldo explícitamente:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --per-metal 5 --provider finbert
.\.venv\Scripts\python.exe -m metals.cli predict
```

Abre http://localhost:8501 y pulsa **Actualizar vista**. En Noticias y minería puedes revisar cada puntuación, evidencia, fecha y proveedor.

## Verificación

Pasaron 28 pruebas de integración de sentimiento, proveedores, pipeline y dashboard, incluyendo oro y plata. Tras incorporar la selección explícita de proveedor, se verificaron nuevamente las nueve pruebas de proveedores, incluyendo que Gemini explícito no active ni reutilice FinBERT. El servidor respondió correctamente a su comprobación de salud.

La fuente genérica MINING.com respondió 403; las fuentes específicas de oro y plata sí permitieron completar el lote. Los registros de ingesta conservan también los intentos fallidos por restricciones de red del entorno de ejecución.

Resumen verificable: `artifacts/update_2026-10-12.json`. Los modelos, reportes y predicciones vigentes están en `artifacts/gold` y `artifacts/silver`.
