# Mejora del modelo y lectura de las métricas

## Qué cambió en esta versión

El sistema mantiene dos clasificadores XGBoost, uno para oro y otro para plata. Se predice una **clase mensual** (Bajista, Neutral o Alcista), no un precio objetivo. Los umbrales, fechas, rezagos y particiones temporales permanecen iguales.

Se amplió la búsqueda de cuatro a **18 candidatos previamente definidos**. Se conservan las cuatro configuraciones originales y se añaden dos configuraciones con árboles menos profundos o mayor regularización. Cada configuración se prueba con tres alternativas:

1. Pesos uniformes: cada observación tiene la misma influencia.
2. Pesos balanceados: las clases menos frecuentes reciben mayor peso. La frecuencia se calcula únicamente con las etiquetas del entrenamiento de cada partición.
3. Pesos por recencia: una observación de hace 60 meses tiene la mitad del peso de la más reciente, antes de normalizar el peso medio a uno. La antigüedad se calcula en meses calendario dentro del entrenamiento.

Las opciones intentan controlar complejidad, desbalance y cambios de régimen. Ninguna garantiza una mejora. El modelo puede conservar la configuración original si obtiene la mejor validación.

## Cómo se elige sin usar el futuro

Dentro del conjunto de desarrollo se mantienen tres validaciones expansivas con 24 observaciones cada una, purga de dos filas y control adicional de disponibilidad de etiquetas. La media de **macro-F1** determina la elección; sólo en caso de empate se compara accuracy y después log-loss. No se cambian umbrales de tendencia para obtener mejores resultados.

La función de selección sólo recibe desarrollo y variables predictoras: no recibe el holdout. Los pesos de un fold se calculan únicamente con su entrenamiento. Se evalúa una sola configuración elegida en las 36 observaciones finales. Después se exporta por separado el modelo final, ajustado con todas las etiquetas disponibles al origen.

Los pesos sólo afectan al entrenamiento. La predicción sigue utilizando el JSON nativo de XGBoost y el orden de variables registrado. No hay un cambio oculto de probabilidades durante inferencia.

El reporte incorpora `best_weighting`, `selection`, resultados completos de los candidatos y métricas de precisión y recall. La desviación de macro-F1 entre los tres folds permite ver si el resultado cambia mucho entre periodos.

## Resultado observado con el snapshot de septiembre de 2026

Evaluación realizada sobre el dataset con origen **31 de agosto de 2026**, 303 observaciones utilizables para el entrenamiento final y 36 para la comparación histórica. El periodo de holdout abarca junio de 2023 a junio de 2026; hay un hueco por cobertura de variables, por lo que se trata de 36 observaciones, no 36 meses consecutivos.

| Métrica | Oro anterior | Oro actual | Plata anterior | Plata actual |
|---|---:|---:|---:|---:|
| Macro-F1 medio en validación de desarrollo | 0,4569 | 0,4569 | 0,3606 | **0,4245** |
| Accuracy en holdout | 27,78% | 27,78% | 44,44% | **50,00%** |
| Precisión macro en holdout | 20,35% | 20,35% | 69,10% | **62,16%** |
| Recall macro en holdout | 21,29% | 21,29% | 40,40% | **46,51%** |
| Macro-F1 en holdout | 0,2058 | 0,2058 | 0,3423 | **0,4333** |
| Log-loss en holdout (menor es mejor) | 1,0036 | 1,0036 | 1,1062 | **1,0941** |

La plata seleccionó profundidad 3, 160 árboles y ponderación balanceada. Mejoraron su accuracy, recall y macro-F1, pero **su precisión macro disminuyó**: no se puede afirmar que todas las métricas mejoraron. El oro conservó profundidad 3, 80 árboles y pesos uniformes, con resultados idénticos.

En oro, el macro-F1 de persistencia fue 0,4508 y el de XGBoost 0,2058: el modelo sigue sin superar esa referencia. En plata, el modelo obtuvo 0,4333 frente a 0,1867 de clase mayoritaria y 0,1669 de persistencia. Estos resultados describen la muestra histórica, no aseguran el siguiente mes.

La mayor probabilidad de la predicción final fue aproximadamente 43,02% en oro y 49,76% en plata, ambas por debajo del umbral de confianza del 55%. La mejora de las métricas históricas no obliga a emitir una señal favorable.

## Qué significa cada medida

- **Accuracy o exactitud:** proporción total de clases acertadas.
- **Precisión macro:** para cada clase, qué proporción de las predicciones de esa clase fue correcta; después se promedian las tres clases con el mismo peso.
- **Recall macro:** para cada clase real, qué proporción se detectó; después se promedian las tres clases.
- **Macro-F1:** promedio de las tres medidas F1 por clase. Considera precisión y recall y evita elegir sólo por la clase más frecuente.
- **Probabilidad máxima:** salida del modelo para una fila nueva. No es el accuracy del modelo ni una certeza de rentabilidad.

Si una clase no recibe predicciones, su precisión se define como cero en estos reportes. Esto evita ocultar clases que el modelo deja de reconocer.

## Límites y siguiente experimento

**El holdout ya fue examinado en versiones anteriores.** Aunque el código no lo usa para seleccionar candidatos, esta nueva comparación no constituye una prueba prospectiva inédita. Se necesita registrar las predicciones futuras sin modificarlas después de conocer su resultado, y evaluarlas cuando el precio y la etiqueta estén disponibles. No se seguirá cambiando candidatos hasta encontrar uno que suba esa misma prueba.

La búsqueda de 18 candidatos también puede sobreajustar las validaciones de desarrollo. Se conserva el informe completo y se evita una búsqueda ilimitada. Los precios y series macro revisados mantienen la limitación de disponibilidad histórica descrita en la metodología.

Cinco noticias recientes por metal aportan contexto del momento. No crean un historial mensual de sentimiento ni demuestran una mejora del clasificador. Para evaluar su contribución al XGBoost se necesitan noticias y análisis almacenados antes de cada origen, suficientes meses y una comparación temporal entre modelos con y sin sentimiento. No se introducen análisis de hoy en filas históricas.

## Cómo reproducir en Visual Studio Code

Desde una terminal PowerShell abierta en la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m metals.cli train
.\.venv\Scripts\python.exe -m metals.cli predict
.\.venv\Scripts\python.exe -m pytest -q tests/test_model_selection.py
```

El primer comando reconstruye el dataset desde SQLite, ejecuta la selección y escribe los modelos e informes en `artifacts/gold` y `artifacts/silver`. El segundo comprueba que el dataset coincide con el entrenamiento y genera la predicción. Si se cambian datos, las cifras pueden cambiar; los valores de la tabla corresponden al snapshot indicado, no son objetivos que el código deba alcanzar.

Referencias técnicas: [pesos por muestra en XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier.fit), [ponderación balanceada en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html), [validación temporal](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## Backtesting con comisiones en el holdout

`metals/backtest.py` simula sobre los 36 meses del holdout (2023-07 a 2026-07) una estrategia que invierte el mes siguiente sólo si el modelo predice Alcista y se queda fuera en otro caso, frente a comprar y mantener. Cobra 0,1 % por cada entrada o salida. Usa los retornos entre promedios mensuales de la etiqueta, no precios ejecutables, y no incluye impuestos ni deslizamiento. La pestaña Predictivo muestra el resultado.

| Metal | Estrategia | Retorno acumulado | Caída máxima | Operaciones |
|---|---|---:|---:|---:|
| Oro | Modelo (13 meses invertido) | +45,8 % | −1,9 % | 11 |
| Oro | Comprar y mantener | +104,8 % | −18,9 % | 1 |
| Plata | Modelo (29 meses invertido) | +81,4 % | −24,7 % | 9 |
| Plata | Comprar y mantener | +81,9 % | −36,2 % | 1 |

En oro la estrategia gana menos que comprar y mantener en un periodo muy alcista, con una caída máxima mucho menor. En plata obtiene un retorno similar con menor caída. Es un resultado retrospectivo sobre un holdout ya examinado: no demuestra rentabilidad futura.
