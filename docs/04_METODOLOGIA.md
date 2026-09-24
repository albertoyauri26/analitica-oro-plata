# Metodología del MVP

## Pregunta y unidad de análisis

Clasificación supervisada mensual por metal. Una fila corresponde al fin del mes t. Se predice la clase del cambio del promedio de precio del mes t+1 respecto al de t:

`r(t+1) = P(t+1) / P(t) - 1`.

Oro: Bajista si r < −2%, Neutral entre −2% y +2% inclusive, Alcista si r > +2%. Plata: las mismas reglas con ±3%. Son umbrales de diseño académico definidos antes de evaluar, no estimados para maximizar resultados. Se puede estudiar sensibilidad en un experimento nuevo, manteniendo su propia prueba temporal.

## Fechas y disponibilidad

El promedio de t no está necesariamente publicado al finalizar t. Para evitar usarlo anticipadamente, las variables técnicas y de precio terminan en **t−1**. Las macro se convierten a promedio mensual y se desplazan un mes, y después participan del desplazamiento general de otro mes: el valor final representa **t−2**. Es un supuesto conservador, no un calendario de publicaciones verificado.

La etiqueta de t usa P(t+1), por lo que se considera disponible al final de **t+2**. El entrenamiento final sólo acepta etiquetas cuya fecha de disponibilidad no supera el origen de predicción. En cada partición se verifica la misma condición respecto al inicio de validación.

Ejemplo de calendario, sin inventar precios: en una fila con origen 31 de agosto, se usan precios hasta julio y macro hasta junio. Se clasifica el cambio del promedio de septiembre frente al de agosto. La etiqueta se considera disponible el 31 de octubre. Esos datos de septiembre/octubre nunca son predictores de la fila de agosto.

La inferencia corresponde al origen mensual indicado, no a todos los eventos conocidos el día en que se abre el dashboard. Las noticias recibidas después de ese origen se muestran como contexto reciente, pero no se introducen en esa predicción anterior.

## Limpieza y variables

Se verifican columnas, fechas válidas, precios positivos y duplicados. Se reindexa al calendario mensual completo. Los meses ausentes permanecen ausentes: no hay interpolación ni relleno hacia atrás. `pct_change` se calcula sin rellenar datos faltantes.

Las variables base son retornos de 1, 3, 6 y 12 meses; distancia a medias móviles de 3 y 12 meses; volatilidad de 6 y 12 meses anualizada por raíz de 12; RSI de 6 meses con medias simples; drawdown móvil de 12 meses; razón oro/plata y cambio trimestral; tasa efectiva federal, Treasury 10 años, VIX, inflación interanual y cambios trimestrales de tasas. La volatilidad deriva de promedios mensuales y puede diferir de la volatilidad calculada con cierres diarios.

El modelo de árboles no necesita estandarización de variables. Las filas sin cualquiera de las variables base se excluyen y su cobertura se puede inspeccionar. Esto hace visible la ausencia de una fuente requerida.

## Noticias, sentimiento y minería

Gemini recibe únicamente título y extracto RSS. Devuelve relevancia, score entre −1 y +1, explicación y evidencia literal para cada metal. Se valida esquema, unicidad de metales, límites y presencia de evidencia dentro del texto. Una ausencia o error no significa sentimiento neutral.

Para integrar una noticia al dataset, la disponibilidad efectiva es el máximo entre publicación, primera captura y análisis. Además, su publicación debe caer en la ventana reciente de 31 días. No se rellena el pasado con análisis generados hoy. El conocimiento preentrenado del LLM también puede contener hechos posteriores a un artículo antiguo; la evaluación estricta del sentimiento exigiría un diseño prospectivo y auditoría humana.

La producción anual requiere publicación fechada. La última observación disponible puede repetirse como contexto hasta 730 días; no representa una producción mensual interpolada. Su exactitud y fecha se deben comprobar en el informe original.

Las tres variables complementarias sólo se seleccionan si el primer bloque de entrenamiento tiene al menos 60 valores mensuales observados y 80% de cobertura. La selección se hace antes de ver validación y holdout. Si no cumplen, el reporte explica la exclusión. Por tanto, **el primer modelo puede usar únicamente precios y macro, mientras noticias/minería se muestran en el dashboard**. No se afirmará que el sentimiento mejoró el modelo sin un experimento que realmente lo utilice.

## Entrenamiento y evaluación

Se entrenan dos clasificadores XGBoost independientes. Semilla 42, objetivo multiclase, 3 clases, CPU y dos hilos. Se comparan cuatro combinaciones de profundidad 2/3 y 80/160 árboles. La tasa de aprendizaje y regularización permanecen fijas.

Los últimos 36 meses con etiqueta utilizable se reservan como holdout. El conjunto de desarrollo anterior se separa del holdout mediante una purga de dos filas. Dentro de desarrollo se hacen tres particiones expansivas, cada validación con 24 meses y purga de dos filas. Si hubiera huecos por datos faltantes, las ventanas contienen observaciones disponibles, no necesariamente 24 meses consecutivos; el control adicional por fechas impide etiquetas futuras.

La selección usa la media de macro-F1 de validación. El holdout se evalúa una vez con esos parámetros. No se ajustan parámetros ni umbrales usando sus resultados. Las referencias son: clase más frecuente del entrenamiento y persistencia de la clase del último retorno mensual conocido.

Se reportan exactitud, balanced accuracy, macro-F1, log-loss, Brier multiclase, matriz de confusión y métricas por clase. El Brier es la media de la suma de errores cuadrados de las tres probabilidades, con rango 0–2 bajo esta convención. No se presenta accuracy como rentabilidad.

Se exporta el modelo evaluado por separado. Después se reentrena un modelo final con todas las etiquetas conocidas al origen. Las métricas históricas corresponden al primero, no a una prueba nueva del segundo. Los modelos se exportan en JSON nativo de XGBoost junto con lista/orden de variables, versiones y huellas SHA-256.

## Motor de riesgo y decisión

Puntuación heurística:

| Condición | Puntos |
|---|---:|
| Volatilidad anualizada ≥30% | 2 |
| Volatilidad anualizada ≥18% y <30% | 1 |
| Drawdown de 12 meses ≤−15% | 2 |
| VIX rezagado ≥25 | 1 |
| Mayor probabilidad del modelo <55% | 1 |

Riesgo bajo = 0; medio = 1–2; alto = 3 o más. Si faltan variables de riesgo: No evaluable y Abstenerse. Estos cortes no son probabilidades de pérdida ni una escala validada de riesgo personal.

Se abstiene si la predicción está vencida, tiene antigüedad excesiva, el origen es futuro, su mayor probabilidad es menor a 55% o XGBoost no supera ambas referencias en macro-F1. Si pasa esos controles pero el riesgo es alto, espera. En los demás casos ofrece una interpretación de la tendencia. Las reglas se escribieron antes de observar el holdout y no se optimizaron sobre éste.

## Límites y validez

Los archivos del Banco Mundial y FRED son versiones actuales revisadas. Los rezagos **no eliminan el sesgo por revisiones históricas**. Este MVP ofrece una validación retrospectiva temporal aproximada; para una reconstrucción estricta deben usarse archivos de publicaciones y ALFRED con joins por fecha de disponibilidad, además de versiones históricas de precios. Esa extensión no está implementada en esta versión.

Un holdout de 36 meses tiene incertidumbre y puede representar un régimen económico particular. La búsqueda de cuatro modelos no agota alternativas. No se hacen afirmaciones causales. Las probabilidades no están calibradas externamente. No se incluye un backtest de ganancias porque promedios mensuales no son precios ejecutables y no hay un instrumento, comisiones, spread ni reglas de negociación definidos.

El éxito académico se mide por reproducibilidad, integridad temporal, comparación honesta, interpretación y funcionamiento del sistema. La utilidad predictiva se determina con las métricas reales, incluso si el resultado es negativo.

Referencias técnicas: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [exportación XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), [Gemini, salidas estructuradas](https://ai.google.dev/gemini-api/docs/structured-output), [FRED y vintages](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
