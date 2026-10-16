# Noticias específicas para oro y plata

El recolector consulta RSS públicos especializados, comprueba el tema de cada título y extracto, elimina duplicados por URL y selecciona hasta **cinco noticias por metal publicadas en los últimos 31 días**. La cifra cinco es un objetivo de cobertura: una fuente caída o un número menor de noticias válidas se informa como déficit, sin inventar artículos ni fechas.

## Fuentes verificadas el 24 de septiembre de 2026

| Publicación | Categoría y RSS público | Respuesta observada |
|---|---|---|
| The Northern Miner | [Categoría oro](https://www.northernminer.com/commodity/gold/) · [RSS oro](https://www.northernminer.com/commodity/gold/feed/) | HTTP 200; 20 entradas |
| The Northern Miner | [Categoría plata](https://www.northernminer.com/commodity/silver/) · [RSS plata](https://www.northernminer.com/commodity/silver/feed/) | HTTP 200; 20 entradas |
| Mining Australia | [Categoría plata](https://miningaustralia.com.au/category/commodities/silver-mining-news/) · [RSS plata](https://miningaustralia.com.au/category/commodities/silver-mining-news/feed/) | HTTP 200; 10 entradas |

### Fuentes de mercado añadidas el 15 de octubre de 2026

Los RSS mineros publican sobre todo noticias de empresas (exploración, minas, resultados), que el LLM marca como no relevantes para el precio del metal: en oro quedaba una sola noticia relevante. Se añadieron dos búsquedas públicas de Google News que agregan noticias del precio de medios financieros (Kitco, Bloomberg, Fortune, TheStreet, Barchart):

| Fuente | RSS | Prueba del 15/10/2026 |
|---|---|---|
| Google News — Gold price | [búsqueda "gold price", últimos 30 días](https://news.google.com/rss/search?q=%22gold+price%22+when:30d&hl=en-US&gl=US&ceid=US:en) | 97 noticias útiles |
| Google News — Silver price | [búsqueda "silver price", últimos 30 días](https://news.google.com/rss/search?q=%22silver+price%22+when:30d&hl=en-US&gl=US&ceid=US:en) | 98 noticias útiles |

El enlace de cada noticia pasa por Google News y el medio original figura al final del título. Se guardan sólo metadatos (título, extracto, enlace y fecha), igual que con las demás fuentes.

Las tres URLs RSS mineras se localizaron en los enlaces `rel="alternate"` de sus respectivas páginas. La cantidad y el contenido del feed pueden cambiar. Se obtiene lo que cada medio publica en su RSS, sin descargar el cuerpo del artículo ni contenido de suscripción. La categoría de una página por sí sola no demuestra relevancia: también se exige una mención temática en los metadatos del artículo.

La prueba de conexión a las 16:52 UTC del 24/09/2026 devolvió diez URLs únicas y cobertura de cinco noticias para oro y cinco para plata, sin déficit. Esta es una observación de la prueba, no una garantía de cobertura futura ni una validación de todas las afirmaciones periodísticas. El CSV `templates/news_verified_2026-09-24_batch.csv` conserva esa selección verificable para importación manual; no se usa para rellenar artificialmente una actualización fallida.

## Ejecutar desde la terminal de Visual Studio Code

Con la carpeta del proyecto abierta, recopilar noticias:

```powershell
.\.venv\Scripts\python.exe -m metals.cli news-refresh --per-metal 5
```

El comando muestra `coverage` (cantidad encontrada) y `deficits` (cuántas faltan) por metal, además del estado de cada fuente. Después, analizar las noticias seleccionadas con Gemini:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --per-metal 5
```

También se puede recopilar y solicitar el análisis en un solo paso:

```powershell
.\.venv\Scripts\python.exe -m metals.cli news-refresh --per-metal 5 --analyze
```

El análisis usa la credencial local de `.env` y consume la cuota disponible de Gemini. Un error 429 significa que falta cuota; se conservan las noticias y los análisis ya guardados. No se asigna sentimiento neutral a las noticias pendientes. La extracción RSS puede funcionar aunque Gemini no esté disponible.

Para importar explícitamente la selección de ejemplo real:

```powershell
.\.venv\Scripts\python.exe -m metals.cli import-news templates/news_verified_2026-09-24_batch.csv
```

Este CSV es una captura fechada. Después de 31 días sus artículos dejan de cumplir la ventana de actualidad y corresponde recopilar noticias nuevas.

## Qué se conserva y cómo se selecciona

1. URL original, editor, fecha de publicación y primera captura local.
2. Título y un fragmento literal del resumen RSS: **máximo 25 palabras entre ambos**. Los puntos suspensivos indican cortes. El fragmento se elige para conservar una mención del metal; no se inventa un resumen.
3. Archivo RSS original en `data/raw`, identificado por SHA-256; el diagnóstico de cada fuente incluye el hash completo.
4. Sólo noticias con URL HTTPS, fecha interpretable, no futura y dentro de la ventana solicitada.
5. Orden por publicación reciente y eliminación de duplicados por URL. Una noticia sobre ambos metales se guarda una vez y cuenta para ambos cupos.

No se deduce que una noticia sea sobre oro o plata sólo por provenir de un medio minero. Las menciones a medallas o usos figurados se excluyen mediante `metals/news_filter.py`. Gemini evalúa después la relevancia efectiva y el sentimiento para cada metal; el prefiltro no reemplaza esa evaluación.

## Cómo interpretar su aporte

El sentimiento corresponde únicamente al texto breve proporcionado, no a una lectura completa del artículo ni a una garantía de movimiento del precio. Una mejora operativa de una empresa puede beneficiar sus acciones y tener un efecto indeterminado sobre el metal mundial. Por ello, Gemini puede marcar una noticia temática como insuficiente para inferir un efecto y el tablero debe mostrarlo.

Cinco noticias actuales sirven para resumir el contexto actual y complementar la decisión. Por sí solas no aportan una serie histórica suficiente para demostrar mayor accuracy ni para entrenar de manera válida el efecto mensual de las noticias. La captura y el análisis de hoy no se retrofechan para mejorar artificialmente una evaluación de años anteriores.

El módulo `metals/news_sources.py` implementa `fetch_targeted_news(per_metal=5, window_days=31)`. Devuelve `(rows, diagnostics)`, sin modificar SQLite ni llamar a Gemini; el comando CLI se encarga de guardar los resultados. Las pruebas del parser comprueban fechas futuras/antiguas, noticias de otros minerales, URLs inválidas, deduplicación, cupos y conservación del metal en extractos breves.
