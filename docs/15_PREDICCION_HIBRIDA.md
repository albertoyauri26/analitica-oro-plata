# Escenario experimental con sentimiento de noticias

Esta función hace que el sentimiento intervenga de forma visible y reproducible en un **escenario del mes en curso**. Conserva al lado la predicción de XGBoost para que se pueda comparar qué cambia y por qué. La mezcla es una regla de investigación: no es un nuevo modelo entrenado, no tiene accuracy propio y no reemplaza las métricas históricas.

## Qué entra al cálculo

- La distribución de XGBoost para Bajista, Neutral y Alcista, junto con metal, origen, mes objetivo y mes del precio utilizado.
- Un snapshot fechado con exactamente **cinco noticias distintas y analizadas para el metal seleccionado**.
- Al menos **tres noticias relevantes**, con un sentimiento válido entre −1 y +1 y proveedor identificable.
- Un peso elegido entre **0 y 0,20**. El valor inicial de la función es 0,15; cero la desactiva.

El snapshot debe tener como máximo 24 horas de antigüedad. Todas las fechas de publicación, primera captura y análisis deben ser anteriores o iguales a su emisión; las publicaciones deben pertenecer a los últimos 31 días. Los contadores y el sentimiento medio se contrastan con los registros individuales. No basta con escribir "cinco noticias" en un contador.

El metal del contexto debe coincidir con el del modelo. El objetivo debe ser el mes calendario actual, el origen el cierre del mes anterior y el precio utilizado el mes anterior al origen, respetando el contrato del modelo mensual. Si el objetivo ya venció, se solicita actualizar el modelo antes de aplicar el escenario.

## Regla transparente

Sean:

- `p`: distribución original de XGBoost.
- `s`: sentimiento medio de las noticias relevantes, entre −1 y +1.
- `c`: cantidad de noticias relevantes dividida entre cinco.
- `w`: peso elegido por el usuario, de 0 a 0,20.
- `a = w × c × |s|`: coeficiente efectivo aplicado.

La probabilidad Neutral se conserva. Si `s > 0`, se transfiere `a × p(Bajista)` desde Bajista hacia Alcista. Si `s < 0`, se transfiere `a × p(Alcista)` desde Alcista hacia Bajista. Si `s = 0`, no se mueve ninguna probabilidad.

Esto equivale a una mezcla convexa `h = (1−a) × p + a × q`, donde `q` conserva la masa Neutral y coloca la masa direccional en el signo del sentimiento. El resultado suma uno, no contiene valores negativos y ninguna clase puede cambiar más de 20 puntos porcentuales. En condiciones habituales el cambio es bastante menor porque se reduce por cobertura y por magnitud del sentimiento.

**Ejemplo matemático, no datos del mercado:** una distribución Bajista 40%, Neutral 25% y Alcista 35%, con tres noticias relevantes, sentimiento medio +0,8 y peso 0,15, produce `a=0,072`. Se transfieren 2,88 puntos porcentuales desde Bajista a Alcista: 37,12%, 25% y 37,88%. La etiqueta más alta cambia, aunque la evidencia siga siendo débil. Esto muestra por qué se presenta como un escenario condicionado y no como una orden de compra.

Tono neutral no equivale a retorno neutral. Por eso un sentimiento cero deja intacta la distribución original, en lugar de aumentar artificialmente la clase Neutral.

## Qué se mantiene protegido

La señal orientativa, el riesgo, las razones de abstención y la confianza utilizada para decidir siguen siendo los del modelo base. **Una abstención no se convierte en una señal favorable por esta regla**, aunque la clase más probable del escenario cambie. Tampoco se vuelve a calcular el umbral de confianza del 55% con la mezcla.

Los JSON del modelo, la selección temporal, el holdout y sus métricas no se modifican. La función no escribe archivos ni altera las entradas. Devuelve las dos distribuciones, diferencia por clase, peso solicitado, peso aplicado, proveedor, fechas, parámetros y razones.

Los estados son:

| Estado técnico | Qué significa |
|---|---|
| `disabled` | Peso cero; se conserva XGBoost. |
| `blocked` | Fechas, cobertura, metal o trazabilidad insuficientes; se conserva XGBoost y se explican las causas. |
| `applied_experimental` | Cumple los controles y muestra el escenario, aún sin validación predictiva propia. |

## Gemini y FinBERT no prueban una relación con el precio

FinBERT clasifica el tono financiero del texto en positivo, negativo o neutral. No fue diseñado en este proyecto para identificar causalmente cómo una noticia afecta al precio futuro del oro o de la plata. Una noticia favorable sobre más producción puede incluso aumentar la oferta y presionar el precio en otra dirección. El tono y el impacto sobre precio son conceptos diferentes. [Ficha oficial de ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert).

Las valoraciones contextuales de Gemini tampoco convierten automáticamente esa relación en una predicción demostrada. Si se mezclan proveedores, sus escalas pueden no ser comparables. El escenario usa sus puntuaciones como **hipótesis explícita de investigación**, con el proveedor visible y un peso acotado; no presenta la puntuación como una probabilidad de subida del metal.

Sumar uno y estar entre cero y uno no demuestra calibración. Para interpretar una salida del 80% como frecuencia real cercana al 80% haría falta comprobarlo con resultados observados y datos separados. Este ajuste no realiza esa calibración. [Documentación de calibración de scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

## Fechas y evaluación posterior

La predicción mensual puede tener origen 31 de agosto y objetivo septiembre. Noticias analizadas el 28 de septiembre pueden alimentar un escenario **emitido el 28 de septiembre** para ese mismo mes en curso. Nunca se registra ese escenario como si existiera el 31 de agosto. Tampoco convierte el objetivo en octubre ni es una predicción para los siguientes 30 días.

El siguiente experimento debe guardar prospectivamente, para cada origen y sin sobrescribir resultados después de conocer el precio: la predicción base, el escenario, el lote de noticias, sus proveedores y el peso elegido. Cuando haya suficientes periodos con etiqueta conocida, se podrá comparar modelos con y sin sentimiento, calibración y estabilidad. El peso no debe elegirse repetidamente sobre el mismo holdout para aparentar mejoras.

## Ubicación y verificación

El cálculo está en `metals/hybrid.py`, función `combine(base_prediction, news_context, weight=0.15, today=None)`. El snapshot necesita `metal`, `as_of`, contadores, `news_ids`, `mean_sentiment` y `article_audit`. Cada artículo auditado incluye `id`, `published_at`, `fetched_at`, `analyzed_at`, `relevant`, `score` y `provider`.

En la terminal de Visual Studio Code, desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_hybrid.py
```

Las pruebas verifican límites, suma de probabilidades, efecto direccional, conservación de Neutral, lote incompleto, duplicados, metal incorrecto, fechas futuras o vencidas, zonas horarias y conservación de la abstención. Sus ejemplos artificiales se usan sólo para probar el programa; no se importan como noticias ni resultados del mercado.
