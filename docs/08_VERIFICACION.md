# Resultados verificados de esta instalación

Generado desde los archivos reales: 2026-09-24T06:50:58.719678+00:00.

## Datos

Precios: 640 observaciones de ambos metales, de 2000-01-31 a 2026-08-31.
Macro: 14077 observaciones. Noticias: 25. Registros de sentimiento por metal: 12. Producción anual importada: 0.

Fuentes de noticias guardadas:

- Federal Reserve: 15 artículos.
- Mining Weekly: 8 artículos.
- Mining Weekly seleccion verificada: 2 artículos.

## Evaluación

| Metal | Macro-F1 XGBoost | Mayoritaria | Persistencia | Exactitud XGBoost |
|---|---:|---:|---:|---:|
| gold | 0.2058 | 0.1867 | 0.4508 | 0.2778 |
| silver | 0.3423 | 0.1867 | 0.1669 | 0.4444 |

Holdout: 2023-06-30 a 2026-06-30, 36 observaciones. Las ventanas pueden incluir huecos de cobertura, por lo que 36 observaciones no siempre corresponden a 36 meses consecutivos.

El oro no supera la referencia de persistencia en este experimento. La plata supera ambas referencias. Esto no demuestra rentabilidad ni estabilidad fuera del periodo de prueba.

## Últimas salidas

- gold: objetivo 2026-09, tendencia Alcista, probabilidad máxima 43.0%, riesgo Alto, señal Abstenerse.
- silver: objetivo 2026-09, tendencia Bajista, probabilidad máxima 43.6%, riesgo Alto, señal Abstenerse.

Las salidas tienen origen y vigencia definidos. Al abrir la web más adelante puede activarse la abstención por vencimiento.

## Gemini real

| Modelo | Metal | Noticias analizadas | Relevantes |
|---|---|---:|---:|
| gemini-3.6-flash | gold | 6 | 0 |
| gemini-3.6-flash | silver | 6 | 0 |

Se usaron textos reales de RSS, con evidencia literal validada cuando son relevantes. Las respuestas no relevantes mantienen score ausente en SQLite. El servicio devolvió errores 503 intermitentes y después cuota agotada HTTP 429. Se conservaron seis noticias analizadas, todas no relevantes para ambos metales según Gemini. Dos titulares específicos verificados de oro/plata quedaron pendientes por cuota. La clave no se incluye en los reportes ni en los paquetes.

## Verificación de software

- 20 pruebas automatizadas aprobadas en el equipo de desarrollo, incluida la app con ambos metales, las reglas temporales, importación de noticias, reutilización de sentimiento y ocultación de credenciales.
- pip check: sin dependencias incompatibles declaradas.
- Modelos JSON cargados nuevamente para generar predicciones.
- Aplicación local iniciada en http://localhost:8501 y revisada en navegador.
- El entorno genera avisos de deprecación por la combinación pandas/NumPy; las pruebas pasan. Las versiones exactas están fijadas en requirements-lock.txt.

## Pendientes y limitaciones

- Completar publicaciones históricas verificadas de producción y un archivo prospectivo de noticias antes de evaluar su aporte al modelo.
- Las variables de sentimiento/producción quedan excluidas del XGBoost actual por cobertura histórica insuficiente.
- MINING.com devuelve 403; se utiliza Mining Weekly como alternativa.
- El funcionamiento en la computadora física del docente se verifica allí siguiendo su guía. Una prueba extraída en este equipo no sustituye esa comprobación.
- Docker/Hugging Face preparados, pero todavía no construidos ni publicados en una cuenta de Hugging Face.
