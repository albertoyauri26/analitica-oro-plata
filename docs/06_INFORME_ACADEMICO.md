# SISTEMA DE ANALÍTICA PREDICTIVA PARA LA INVERSIÓN EN METALES PRECIOSOS

**Tipo:** proyecto aplicado de Machine Learning y analítica predictiva. **Entregable:** MVP local reproducible y aplicación web preparada para despliegue. Completa autor, institución, asignatura, docente y fecha de entrega según tu curso.

## Resumen

El proyecto integra series históricas de oro y plata, indicadores macroeconómicos y contexto económico/minero en un sistema modular. Se construye un dataset mensual con variables técnicas y rezagos temporales, se entrenan clasificadores XGBoost para tres tendencias y se evalúan mediante particiones expansivas y una prueba final separada. Un motor de reglas transforma predicción, incertidumbre y volatilidad en una señal orientativa y un nivel de riesgo. La aplicación Streamlit permite inspeccionar datos, métricas, fuentes y limitaciones. Gemini está conectado para producir sentimiento estructurado sobre textos reales; su uso como predictor depende de disponer de historia suficiente y correctamente fechada. Las conclusiones sobre utilidad predictiva deben basarse en los reportes generados, no en una precisión supuesta.

## 1. Planteamiento del problema

Las series de metales, indicadores macroeconómicos y noticias suelen consultarse de manera separada. Esto dificulta formular y evaluar de forma reproducible una hipótesis sobre la tendencia mensual. Además, un análisis que mezcla datos publicados después de una decisión puede presentar resultados artificialmente favorables. Se necesita un prototipo que integre estas fuentes y exponga tanto los resultados como su procedencia y restricciones temporales.

La pregunta de investigación es: **¿En qué medida un modelo XGBoost con información histórica rezagada mejora la clasificación mensual de oro y plata frente a reglas de clase mayoritaria y persistencia?** Una segunda pregunta, para la ampliación con datos suficientes, es si sentimiento y producción aportan mejora fuera de muestra.

## 2. Objetivos

Objetivo general: desarrollar y evaluar un MVP que clasifique tendencias mensuales y genere señales orientativas explicables para oro y plata.

Objetivos específicos: integrar fuentes verificables; almacenar datos y trazabilidad; calcular variables sin incorporar resultados futuros como predictores; conectar el análisis de noticias; entrenar y comparar modelos; medir desempeño por clase; construir reglas de riesgo auditables; presentar dashboards; exportar modelos y preparar un despliegue reproducible después de verificar la ejecución local.

## 3. Justificación y alcance

El valor académico reside en enlazar ingeniería de datos, modelado, evaluación y comunicación. La aplicación permite contrastar una hipótesis con observaciones y examinar errores. La reproducibilidad se apoya en código modular, versiones de paquetes, archivos originales y hashes.

Se trabaja con promedios mensuales en dólares, no con cotizaciones intradía. Se cubren dos metales y cuatro indicadores macro de Estados Unidos. El contexto de producción se interpreta con frecuencia anual. No se incluye ejecución de operaciones, cartera óptima, rentabilidad garantizada ni recomendaciones personalizadas. La implementación no exige GPU.

## 4. Fundamentos

El aprendizaje supervisado estima una relación entre variables históricas y etiquetas observadas. XGBoost combina árboles de decisión mediante boosting y regularización. La clasificación multiclase diferencia ascenso, estabilidad dentro de una banda y descenso. Los indicadores técnicos resumen trayectoria y dispersión; no constituyen por sí mismos pruebas de causalidad económica.

La analítica descriptiva muestra qué ocurrió. La predictiva estima una clase futura y evalúa sus errores. La prescriptiva de este MVP aplica reglas académicas a las salidas, incluyendo abstención. El análisis de sentimiento convierte texto en variables estructuradas, cuya interpretación y disponibilidad temporal requieren validación.

## 5. Diseño y arquitectura

Se respeta la secuencia fuentes → ingesta/procesamiento → dataset analítico → modelado → decisión/riesgo → dashboards → despliegue. SQLite separa precios, macro, noticias, sentimiento, producción y bitácora. Los archivos originales se conservan por SHA-256. El dataset y los modelos se generan mediante CLI; la aplicación presenta los resultados del mismo código de inferencia.

Python facilita limpieza, experimentación y despliegue. pandas procesa series, XGBoost realiza clasificación, scikit-learn calcula particiones y métricas, Plotly representa gráficos y Streamlit ofrece la interfaz. La exportación JSON evita depender de objetos serializados con pickle.

## 6. Método experimental

Consulta el protocolo completo en `04_METODOLOGIA.md`. La unidad es mes/metal. Las bandas de neutralidad son ±2% y ±3%. Los predictores de mercado terminan un mes antes del origen y la macro dos meses antes. Se reservan 36 observaciones finales como prueba y se seleccionan hiperparámetros en tres validaciones anteriores de 24 observaciones con purga.

La hipótesis nula práctica es que XGBoost no mejora las referencias en macro-F1 de la prueba reservada. Un resultado superior constituye evidencia descriptiva en ese periodo, no una demostración estadística universal. No se realizan tests de significancia ni intervalos de confianza en esta versión. Si se requieren, debe añadirse un diseño adecuado a la dependencia temporal, por ejemplo bootstrap por bloques preespecificado.

## 7. Análisis de noticias

La ingestión retiene texto público de RSS y enlaces. Gemini clasifica relevancia y tono respecto a cada metal. Se exige una cita literal para sustentar la interpretación y se guarda versión del modelo/prompt. La primera captura y la fecha de análisis se conservan. El protocolo evita reconstruir señales históricas con respuestas producidas después de una fecha de decisión.

Se propone una evaluación humana posterior con una muestra doblemente anotada: relevancia, signo, evidencia correcta y desacuerdos. La extracción estructurada no demuestra que el juicio del LLM sea económicamente acertado. El conector está implementado; la validación real depende de ejecutar la API con credenciales y revisar sus resultados.

## 8. Resultados

Los resultados numéricos reproducibles están en `artifacts/gold/report.json`, `artifacts/silver/report.json` y `08_VERIFICACION.md`. Cada reporte incluye rangos temporales, tamaños, distribución de clases, parámetros elegidos, métricas del modelo y baselines. `holdout_predictions.csv` permite localizar qué meses se clasificaron mal.

Al redactar la versión final para la universidad, incorpora esas cifras y la fecha exacta del experimento. No sustituyas ausencias por números ilustrativos. Reporta también las variables excluidas por cobertura y cualquier fuente inaccesible. La señal de abstención, cuando se activa, demuestra la aplicación de controles definidos, no que el programa haya fallado.

## 9. Discusión

El tamaño mensual del conjunto es moderado y la distribución de clases depende del umbral. La dinámica económica cambia y un único holdout no representa todos los regímenes. Los precios promedio suavizan información diaria. La macro revisada puede introducir sesgos aun usando rezagos. Las noticias RSS son una muestra parcial y no un archivo exhaustivo. La producción mundial anual aporta pocas observaciones independientes.

La mejora respecto a una referencia debe interpretarse junto con errores por clase y probabilidades. La importancia de una variable indica utilización interna del modelo, no causalidad ni una explicación suficiente de una predicción particular.

## 10. Conclusiones y trabajo futuro

La implementación materializa la integración de datos, modelos, decisiones y visualización en un flujo reproducible. La conclusión empírica sobre capacidad predictiva queda condicionada a los reportes reales de cada ejecución. La evidencia obtenida no autoriza inferir rentabilidad de una estrategia operativa.

Prioridades de ampliación: archivo de vintages y fechas de publicación, colección prospectiva de sentimiento, historia anual de producción verificada, auditoría humana de Gemini, comparación con regresión logística, calibración temporal de probabilidades, análisis de estabilidad por periodos y definición de instrumento/costes antes de un eventual backtest.

## Referencias

- Banco Mundial. Commodity Markets / Pink Sheet. https://www.worldbank.org/en/research/commodity-markets
- Federal Reserve Bank of St. Louis. FRED: FEDFUNDS, DGS10, VIXCLS, CPIAUCSL. Enlaces individuales en `02_FUENTES_Y_DATOS.md`.
- Federal Reserve Board. RSS feeds. https://www.federalreserve.gov/feeds/feeds.htm
- USGS. Mineral Commodity Summaries. https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries
- Google. Gemini structured outputs. https://ai.google.dev/gemini-api/docs/structured-output
- XGBoost. Model IO. https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
- scikit-learn. TimeSeriesSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Hugging Face. Docker Spaces. https://huggingface.co/docs/hub/spaces-sdks-docker
