# Fuentes, trazabilidad y diccionario

Fuentes verificadas para el diseño: septiembre de 2026. La fecha de descarga efectiva y la cobertura están en `ingestion_log` y en el dashboard.

| Necesidad | Fuente y serie | Frecuencia y unidad | Integración y alternativa |
|---|---|---|---|
| Oro | [Banco Mundial, Pink Sheet](https://www.worldbank.org/en/research/commodity-markets), Gold | Promedio mensual, USD/oz troy | XLSX automático; importación del mismo original si falla |
| Plata | Misma fuente, Silver | Promedio mensual, USD/oz troy | Mismo flujo |
| Tasa de fondos federales | [FRED FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS), Federal Reserve Board | Mensual, porcentaje anual, tasa efectiva | CSV público; descarga manual FRED |
| Treasury 10 años | [FRED DGS10](https://fred.stlouisfed.org/series/DGS10), Federal Reserve Board | Diaria, porcentaje anual | Promedio mensual de observaciones disponibles |
| VIX | [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS), Cboe | Diario, puntos de índice | Promedio mensual; atribuir a Cboe/FRED |
| Inflación | [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL), BLS | Índice mensual ajustado estacionalmente | Inflación interanual = (CPI/CPI de 12 meses atrás − 1) × 100 |
| Noticias económicas | [RSS oficial de la Reserva Federal](https://www.federalreserve.gov/feeds/feeds.htm) | Comunicados fechados | `press_monetary.xml`; si falla, mantener estado pendiente |
| Noticias mineras | [MINING.com](https://www.mining.com/) y [RSS](https://www.mining.com/feed/) | Titulares/extractos de minería y materias primas | Se conservan fuente y enlace; relevancia por metal la analiza Gemini |
| Alternativa minera | [Mining Weekly, RSS internacional](https://www.miningweekly.com/page/international-home/feed) | Titulares/extractos y fechas de publicación | Se consulta también; conserva la misma estructura de trazabilidad |
| Noticias por metal | [Mining Weekly oro](https://www.miningweekly.com/page/gold) y [plata](https://www.miningweekly.com/page/silver) | Páginas públicas temáticas | Los endpoints RSS temáticos probados devolvieron 404. Se añade una selección manual de títulos/extractos verificables mediante CSV |
| Producción | [USGS Mineral Commodity Summaries](https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries) | Anual, toneladas métricas de metal contenido | Importación CSV verificada; no interpolar a producción mensual |
| Archivo de publicaciones macro | [ALFRED/FRED API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) | Valores por fecha de publicación o vintage | Ampliación pendiente para una evaluación estrictamente point-in-time |

Pink Sheet permite un MVP reproducible sin una clave de mercado. Los promedios no equivalen al cierre de una sesión ni a precios negociables. Si más adelante se eligen futuros COMEX o ETFs, se debe construir otro dataset y definir vencimientos, ajustes, instrumentos y costes. No se mezclarán como si fueran la misma serie.

## Descarga manual si falla una fuente

Descarga **Monthly prices** desde la página oficial del Banco Mundial. Guarda el archivo original en `data/manual`. Después:

```powershell
.\.venv\Scripts\python.exe -m metals.cli import-market "data/manual/CMO-Historical-Data-Monthly.xlsx"
```

En cada página FRED selecciona Download → CSV. Mantén las dos columnas originales (`observation_date` o `DATE`, identificador de serie):

```powershell
.\.venv\Scripts\python.exe -m metals.cli import-macro FEDFUNDS "data/manual/FEDFUNDS.csv"
.\.venv\Scripts\python.exe -m metals.cli import-macro DGS10 "data/manual/DGS10.csv"
.\.venv\Scripts\python.exe -m metals.cli import-macro VIXCLS "data/manual/VIXCLS.csv"
.\.venv\Scripts\python.exe -m metals.cli import-macro CPIAUCSL "data/manual/CPIAUCSL.csv"
```

Después ejecuta `train` y `predict`. Las importaciones validan estructura y fechas, pero no prueban por sí solas autenticidad: debes conservar el archivo oficial y su procedencia.

## Noticias verificadas fuera del RSS

`templates/news_verified_2026-09-24.csv` contiene dos títulos y extractos breves reales, comprobados en las páginas originales el 24 de septiembre de 2026. No es un histórico exhaustivo ni una selección aleatoria. Los enlaces están junto a cada texto. Las fechas originales tienen precisión de día; la importación usa el final de ese día como disponibilidad conservadora, no como hora exacta de publicación.

```powershell
.\.venv\Scripts\python.exe -m metals.cli import-news templates/news_verified_2026-09-24.csv
.\.venv\Scripts\python.exe -m metals.cli sentiment --limit 2 --source "Mining Weekly seleccion verificada"
```

Puedes importar otros títulos/extractos con las columnas `title,summary,url,source,published_at`. Debes comprobar su contenido y fecha contra la URL original. La primera captura es la fecha de importación; nunca se traslada al pasado.

## Producción minera sin inventar cifras

Usa `templates/production_schema.csv` como cabecera. No contiene ejemplos numéricos. Copia a `data/manual/production.csv` y completa sólo datos comprobados del USGS para **producción minera mundial anual, toneladas de metal contenido**. No mezcles producción de Estados Unidos con el total mundial ni producción refinada con minera.

| Columna | Contenido |
|---|---|
| `metal` | `gold` o `silver` |
| `year` | Año de la producción anual observada |
| `tonnes` | Toneladas métricas, número positivo; conservar precisión del informe |
| `available_at` | Fecha de publicación de esa cifra/versión del informe |
| `source` | URL HTTPS del informe o tabla original |

```powershell
.\.venv\Scripts\python.exe -m metals.cli import-production "data/manual/production.csv"
```

Se selecciona el último año publicado disponible en cada fecha, hasta un máximo de 730 días desde la publicación. Es una **observación anual disponible**, no una estimación mensual. Para usarla en entrenamiento necesitas suficientes publicaciones históricas y fechas verificadas. Una tabla de este año con revisiones de años pasados no puede atribuirse automáticamente a fechas antiguas.

La minería se integra desde el inicio mediante noticias y mediante este conector de producción. El modelo base deja la variable numérica ausente hasta contar con datos verificados suficientes.

## Tablas SQLite

| Tabla | Clave | Contenido |
|---|---|---|
| `prices` | fecha, metal | Precio observado, fuente, captura |
| `macro` | fecha, serie | Valor macro observado, fuente, captura |
| `news` | SHA-256 de URL | Título, extracto, URL, fuente, publicación, primera captura |
| `sentiment` | noticia, metal, modelo, versión de prompt | Relevancia, puntuación, evidencia, explicación, fecha |
| `production` | metal, año, publicación | Producción anual y fuente |
| `ingestion_log` | id | Estado, conteo, detalle y SHA-256 de descarga |

## Diccionario del dataset

| Campos | Definición |
|---|---|
| `date`, `metal` | Origen temporal de la predicción, fin de mes, y activo |
| `price`, `source_price_month` | Último promedio usado como predictor y su mes, t−1 |
| `ret_1m/3m/6m/12m` | Cambio relativo de precio en esa ventana, terminado en t−1 |
| `sma_gap_3/12` | Precio/media móvil simple − 1, ventana de meses |
| `vol_6m/12m` | Desviación muestral de retornos mensuales × raíz de 12 |
| `rsi_6m` | RSI con medias simples de ganancias/pérdidas de 6 meses, 0–100 |
| `drawdown_12m` | Precio/máximo móvil de 12 meses − 1 |
| `gold_silver_ratio` | Precio oro/precio plata, mismas fechas y unidades |
| `ratio_change_3m` | Cambio relativo trimestral de la razón |
| `fedfunds`, `treasury10`, `vix` | Valores macro, efectivos t−2 tras los dos desplazamientos |
| `cpi`, `inflation_yoy` | Índice y su variación interanual porcentual, rezagados |
| `fed_change_3m`, `yield_change_3m` | Cambios trimestrales en puntos porcentuales |
| `news_sentiment`, `news_count` | Media y cantidad de noticias relevantes disponibles, ventana 31 días |
| `production_tonnes` | Última producción mundial anual publicada disponible |
| `target_return` | P(t+1)/P(t) − 1, sólo para etiquetado/evaluación |
| `target` | 0 Bajista, 1 Neutral, 2 Alcista |
| `target_month` | Mes t+1 cuya variación se clasifica |
| `label_available_at` | Fin de t+2, disponibilidad conservadora de la etiqueta |

No se permite `bfill`, interpolación de precios ni clasificación del futuro ausente como Neutral. Los nulos significan dato no disponible. El dataset conserva esos nulos y el entrenamiento informa qué filas/variables usa.

Los precios y macro pueden revisarse al volver a descargar. Se conservan los originales por huella, aunque las tablas normalizadas toman la última descarga. La reproducción de un experimento debe conservar juntos base de datos, dataset, originales, modelo y reporte.

## Uso y atribución

Consulta los [términos del Banco Mundial](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) y las notas de cada serie FRED antes de redistribuir datos. El hecho de que una descarga sea pública no elimina derechos de terceros. En una presentación atribuye proveedor original y distribuidor. Las noticias conservan extractos RSS y enlaces; no descargamos artículos completos. El paquete web puede incorporar tu copia de presentación, con las mismas fuentes y condiciones de uso.
