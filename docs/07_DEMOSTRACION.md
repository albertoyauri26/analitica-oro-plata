# Guion de demostración y criterios de aceptación

## Presentación de 8–10 minutos

1. **Problema y objetivo, 1 minuto.** Explica que se clasifica un cambio mensual de promedios de oro/plata y se genera orientación con riesgo.
2. **Arquitectura y fuentes, 1 minuto.** Muestra el flujo del README y las fuentes originales. Diferencia precio promedio de cierre negociable.
3. **Datos, 1 minuto.** Abre Descriptivo. Cambia de metal, observa fechas y unidades. Muestra una descarga y su fuente.
4. **Experimento, 2 minutos.** Abre Predictivo. Explica las clases, probabilidades, prueba temporal, baselines y matriz de confusión. Lee las métricas reales.
5. **Decisión, 1 minuto.** Muestra el nivel de riesgo y sus razones. Si aparece Abstenerse, explica qué regla se activó.
6. **Noticias/minería, 1 minuto.** Abre una fuente y, si ejecutaste Gemini, su evidencia. Explica qué variables contextuales pudieron entrar en entrenamiento.
7. **Reproducibilidad y límites, 1 minuto.** Muestra versiones, hash, nulos y el protocolo de fechas. Explica la limitación por revisiones y el despliegue preparado.

## Criterios de aceptación

| Requisito | Evidencia verificable |
|---|---|
| Fuentes reales | Originales en data/raw, enlaces y bitácora |
| Ingesta, limpieza, almacenamiento | CLI ingest/status y tablas SQLite |
| Variables y dataset | CSV analítico con fechas y nulos legítimos |
| Noticias y Gemini | RSS real; API configurable; resultados reales sólo tras credenciales |
| XGBoost | Modelos JSON independientes para ambos metales |
| Validación temporal | Fechas de cada fold, purga, holdout, baselines |
| Riesgo y decisión | Reglas, razones y abstención en prediction.json |
| Tres dashboards | Pestañas Descriptivo, Predictivo, Prescriptivo |
| Exportación | Modelo final/evaluado, reportes y predicciones |
| Aplicación local | Streamlit visible y tests aprobados |
| Despliegue | ZIP, Dockerfile y guía; URL sólo cuando se publique y verifique |

## Preguntas frecuentes del docente

**¿Por qué no dividir aleatoriamente?** Una mezcla aleatoria permitiría aprender de periodos posteriores a la fecha evaluada.

**¿Por qué XGBoost?** Permite modelar interacciones no lineales en variables tabulares con regularización y ejecutarse en CPU. Su selección debe justificarse comparando métricas, no por reputación.

**¿Por qué no rellenar noticias ausentes con cero?** Cero significa sentimiento neutral observado; ausencia significa que no tenemos análisis.

**¿Se usó Gemini para predecir el precio?** Se usa para estructurar sentimiento. XGBoost estima la clase. Sólo se incorporan variables de sentimiento cuando cumplen cobertura y disponibilidad.

**¿Se puede invertir con esta salida?** El MVP evalúa clasificación y aplica reglas académicas. No demuestra ejecución rentable con instrumentos y costes reales.

**¿Hay fuga temporal?** Se impiden mezclas de fechas y etiquetas futuras bajo el calendario documentado. Las series revisadas carecen de vintages, por lo que no se afirma una reconstrucción histórica perfecta.

**¿Por qué una señal de abstención?** Porque una clase estimada no basta si hay poca evidencia, datos vencidos o desempeño inferior a referencias.
