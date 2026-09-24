# Cinco noticias por metal y su relación con la predicción

## Ejecutarlo en tu Visual Studio Code

Abre la carpeta del proyecto y una segunda terminal PowerShell (la primera puede mantener Streamlit abierto). Ejecuta:

```powershell
.\.venv\Scripts\python.exe -m metals.cli news-refresh --per-metal 5 --analyze
```

Este comando descarga los RSS públicos especializados, conserva sus originales, selecciona hasta cinco noticias recientes de oro y cinco de plata y pide a Gemini el sentimiento. Una noticia que trata ambos metales se consulta una sola vez. La clave ya configurada vive en `.env`; no se necesita volver a pegarla.

Para descargar sin consumir Gemini, omite `--analyze`. Para continuar sólo análisis pendientes de noticias guardadas:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --per-metal 5
```

Al terminar pulsa **Actualizar vista** en la web. Selecciona Oro o Plata y abre **Noticias y minería**. Debes ver `5 / 5` seleccionadas cuando hay suficientes artículos, el número analizado, las pendientes y el sentimiento relevante medio. Expande cada artículo para consultar la fuente original, fecha, evidencia y explicación.

Una respuesta HTTP 503 significa saturación del servicio. Una respuesta 429 significa límite de cuota: no cambia de modelo para evadirlo ni repite indefinidamente. En ambos casos los resultados completados quedan guardados. Puedes reanudar con el segundo comando cuando el servicio/cuota esté disponible. No se asigna sentimiento a una noticia no analizada.

## Cómo se filtran las noticias

- Se exige una mención explícita de oro/gold/XAU o plata/silver/XAG junto con vocabulario de mercado o minería en título/extracto.
- Se descartan noticias generales de tasas, cobre o litio que no mencionan esos metales; también se excluyen varios usos no minerales (medallas, cine, etc.). Es un prefiltro por reglas, no un clasificador semántico infalible.
- La ventana es de 31 días. No se incluyen fechas futuras ni artículos capturados después del momento consultado. Se seleccionan los más recientes y se deduplican URL y título.
- No se completa un lote faltante con otro metal ni con datos inventados. La app informa el déficit.
- Los registros generales capturados en la versión anterior permanecen sólo para auditoría; no se envían a Gemini ni se usan como contexto predictivo.
- Gemini confirma relevancia respecto de cada metal. Un artículo temáticamente candidato puede resultar no relevante para interpretar el precio. Ese caso tiene score ausente, no cero.

Las fuentes y sus alternativas se explican en [Fuentes de noticias](13_FUENTES_NOTICIAS.md). Se analizan titulares y fragmentos públicos breves, no artículos completos; esto limita la interpretación.

## Qué aporta a la predicción ahora

La pestaña **Predictivo** muestra conjuntamente la tendencia XGBoost y el contexto reciente de las cinco noticias. Una regla transparente compara la dirección de la tendencia con el tono medio: coincidencia, discrepancia o lectura no concluyente. El resultado queda también en `news_context` y `news_reading` del JSON de predicción.

El score medio oscila entre −1 y +1. No es una probabilidad. Para describir el conjunto, valores superiores a +0,1 se consideran positivos, inferiores a −0,1 negativos y el intervalo intermedio mixto/neutral. Estos cortes son descriptivos, no una calibración aprendida.

**No se suman puntos arbitrarios a las probabilidades del modelo.** La coincidencia de las noticias con una tendencia no permite decir que su probabilidad subió ni que mejoró el accuracy. Las noticias de septiembre capturadas hoy se presentan como contexto de hoy; no se introducen en una predicción cuyo origen fue agosto.

## Cómo aportará al entrenamiento

El dataset dispone de `news_sentiment` y `news_count`. Una observación puede utilizarse sólo después de la mayor fecha entre publicación, captura y análisis. Se conservan ausencias cuando no hay noticias relevantes, sin convertirlas artificialmente en neutrales.

El entrenamiento exige al menos 60 meses observados y 80% de cobertura en su primer bloque para admitir una variable de contexto. Los cinco artículos recientes no cumplen esa condición. Por eso, **el XGBoost actual todavía no aprende del sentimiento**; el reporte lo declara. Para medir su contribución hará falta un archivo histórico compatible con las fechas y comparar modelos con/sin sentimiento en nuevos periodos temporales.

Para guardar una nueva salida con el contexto disponible:

```powershell
.\.venv\Scripts\python.exe -m metals.cli predict
```

Si se informa cambio de dataset, ejecuta primero `train` y luego `predict`. No reentrenes para cada artículo sólo para cambiar un número: el origen mensual y la cobertura histórica deben justificarlo.

Consulta [Mejora del modelo y métricas](12_MODELO_Y_METRICAS.md) para distinguir accuracy, precisión macro, recall, macro-F1 y probabilidad de una predicción.
