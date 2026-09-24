# Resultados verificados de esta instalación

Generado desde archivos reales: 2026-09-24T17:02:13.538941+00:00.

## Datos

Precios: 640 observaciones. Macro: 14077.
Noticias del tema: 12; registros antiguos excluidos: 23. Producción anual importada: 0.

## Evaluación temporal

| Metal | Accuracy | Precisión macro | Macro-F1 | F1 mayoritaria | F1 persistencia |
|---|---:|---:|---:|---:|---:|
| gold | 27.78% | 20.35% | 0.2058 | 0.1867 | 0.4508 |
| silver | 50.00% | 62.16% | 0.4333 | 0.1867 | 0.1669 |

Selección entre 18 candidatos únicamente con validación temporal de desarrollo. Holdout de 36 observaciones, ya examinado en versiones anteriores: comparación retrospectiva, no prueba prospectiva inédita.

La plata mejora accuracy y macro-F1 respecto de la versión anterior, pero disminuye su precisión macro. El oro no mejora. Consulta [comparación completa](12_MODELO_Y_METRICAS.md).

## Noticias y Gemini

| Metal | Seleccionadas | Analizadas | Relevantes | Pendientes | Score medio |
|---|---:|---:|---:|---:|---:|
| gold | 5/5 | 1 | 1 | 4 | +0.00 |
| silver | 5/5 | 0 | 0 | 5 | Sin dato |

Último registro Gemini (2026-09-24T16:57:34.098592+00:00): error. HTTP 503; 0 noticias completadas antes del fallo. Modelo gemini-3.6-flash.

Los lotes se basan en los últimos 31 días; una noticia sobre ambos metales puede formar parte de ambos lotes y se consulta una vez. Pendiente no significa neutral. El texto disponible son titulares y fragmentos públicos breves, no artículos completos.

El contexto aparece junto a la predicción y se exporta en su JSON. El XGBoost todavía excluye sentimiento/producción por falta de historial suficiente; no se atribuye la mejora de plata a Gemini.

## Últimas salidas

- gold: objetivo 2026-09, Alcista, probabilidad máxima 43.0%, riesgo Alto, señal Abstenerse.
- silver: objetivo 2026-09, Bajista, probabilidad máxima 49.8%, riesgo Alto, señal Abstenerse.

## Software y límites

- 43 pruebas automatizadas aprobadas en esta comprobación.
- Exportación y carga de modelos JSON comprobadas; revisión de la app con Oro y Plata.
- Fuentes revisadas y probabilidades no calibradas externamente: el experimento no demuestra rentabilidad.
- La combinación de pandas/NumPy genera avisos de deprecación; consulta requirements-lock.txt para las versiones verificadas.
- La instalación en el equipo físico del docente se comprueba allí con su guía. Los ZIP excluyen .env y credenciales.
- Hugging Face/Docker preparados, aún no publicados ni construidos en una cuenta externa.
