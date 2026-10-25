# Recomendación prescriptiva con LLM (Mistral)

El reporte prescriptivo combina **reglas fijas** y un **modelo de lenguaje**. Las reglas de `metals/decision.py` calculan el riesgo y deciden qué acciones están permitidas; el LLM elige una de esas acciones y la justifica sólo con los datos del sistema; el código valida la respuesta antes de mostrarla. Código: `metals/advisor.py`. Pruebas: `tests/test_advisor.py`.

## Proveedor, modelo y credencial

| Dato | Valor |
|---|---|
| Proveedor y API | Mistral AI, endpoint `chat/completions` con salida JSON restringida a un esquema |
| Modelo | `ministral-14b-2512` (incluido en el plan gratuito *Experiment*; Mistral Small y Medium tenían límite 0 en ese plan) |
| Selección del proveedor | `SENTIMENT_PROVIDER` (`mistral`; con `gemini` usa Google Gemini con el mismo prompt y validación) |
| Variable de la clave | `MISTRAL_API_KEY` (en `.env` local y en Environment de Render; nunca en el repositorio) |

Se eligió Mistral porque su plan gratuito admite el volumen del proyecto (unas 1 000 llamadas de noticias y recomendaciones por mes) y el plan gratuito de Gemini limitaba a unas 20 solicitudes diarias por modelo.

## Qué deciden las reglas y qué decide el LLM

| Señal de las reglas | Acciones que puede elegir el LLM |
|---|---|
| Abstenerse | Abstenerse, Observar |
| Esperar por riesgo alto | Esperar, Abstenerse, Observar |
| Evaluar oportunidad alcista | Evaluar compra gradual (simulada), Observar, Esperar |
| Cautela ante tendencia bajista | Reducir exposición (simulada), Esperar, Observar |
| Observar / esperar | Observar, Esperar |

Además: la compra simulada sólo existe con tendencia Alcista; el perfil **conservador** no la recibe si el riesgo no es Bajo; la reducción de exposición sólo existe con tendencia Bajista. Las reglas conservan los bloqueos (abstención por baja calidad del modelo, probabilidad menor a 55 %, predicción vencida, riesgo alto). El LLM aporta la elección dentro de esas opciones y la explicación adaptada al perfil.

## Entrada del LLM

El sistema arma un JSON sólo con valores calculados: predicción XGBoost y probabilidades, macro-F1 del holdout y de las dos referencias, nivel de riesgo, señal y motivos de las reglas, volatilidad de 12 meses, caída desde el máximo de 12 meses, VIX, sentimiento medio y hasta tres titulares relevantes, perfil del inversionista y acciones permitidas. La pestaña Prescriptivo muestra la entrada exacta en "Ver la entrada enviada al LLM". Ejemplo real en el caso 2.

## Prompt de sistema completo (versión `advisor-v2`)

> Eres el asistente prescriptivo de AURUM, un proyecto académico de analítica de oro y plata. Recibes un JSON con datos reales del sistema: predicción XGBoost, calidad del modelo, riesgo calculado por reglas, indicadores, sentimiento de noticias, perfil del inversionista y acciones permitidas. Reglas obligatorias: 1) Usa sólo los datos del JSON. No inventes cifras, precios, fechas ni noticias: cada número que escribas debe aparecer en el JSON (puedes expresar una probabilidad 0.65 como 65%). El sentimiento_medio está en escala de -1 a +1 y el macro-F1 va de 0 a 1: no los expreses como porcentaje ni los compares con el umbral de 55%, que sólo aplica a la probabilidad máxima. 2) En 'accion' elige exactamente una opción de acciones_permitidas; en 'alternativa' elige otra de esa lista (o la misma si es la única). Nunca propongas una acción fuera de la lista. 3) No contradigas la señal ni el nivel de riesgo de las reglas. 4) No es asesoría financiera: no indiques comprar o vender en un bróker, montos, apalancamiento ni plazos distintos al mes objetivo. 5) Adapta el tono al perfil del inversionista. 6) Responde sólo el JSON pedido, en español: 'justificacion' de máximo cuatro frases, 'riesgos' de uno a tres elementos breves y 'confianza' entre 0 y 1 sobre tu propia recomendación.

**Cómo se diseñó y probó:** la versión 1 se probó con tres casos reales; el LLM expresaba el sentimiento (0,767) y el macro-F1 como porcentajes y comparaba el macro-F1 con el umbral de 55 %. La versión 2 añadió la aclaración de escalas y se repitieron los casos (abajo): esos errores desaparecieron. Las pruebas automáticas cubren acciones no permitidas, cifras inventadas, perfil conservador, caché, fallos de la API y límite diario.

## Salida estructurada y validación

El LLM devuelve un JSON con `accion`, `justificacion`, `riesgos` (1 a 3), `confianza` (0 a 1) y `alternativa`. Antes de mostrarlo, el código comprueba:

1. **Formato:** todos los campos, tipos y rangos (Pydantic, sin campos extra).
2. **Acción permitida:** `accion` y `alternativa` deben estar en `acciones_permitidas`; así es coherente con la predicción y las reglas.
3. **Cifras reales:** cada número de la justificación y los riesgos debe existir en la entrada (admite 0,65 escrito como 65 %). Un número que no esté en los datos invalida la respuesta.

Si cualquier comprobación falla, o la API no responde, la página muestra la **recomendación por reglas** con un aviso; nunca se muestra una respuesta no validada. Límite conocido: la validación detecta cifras inventadas y acciones fuera de las reglas, pero no errores de interpretación sin números (ver caso 3).

## Protecciones

| Protección | Implementación |
|---|---|
| Aviso | "Proyecto académico: no es asesoría financiera ni ejecuta operaciones" junto a cada recomendación |
| Límite de tokens | 600 tokens de respuesta (`MAX_TOKENS`) |
| Límite de llamadas | 50 consultas por día (`DAILY_LIMIT`); al alcanzarlo se usan las reglas |
| Sin gasto al abrir la página | La recomendación se genera con un botón; si los datos no cambiaron, se reutiliza la guardada |
| Errores de la API | Reintentos ante saturación (503/429 sin capacidad) con esperas de 10 y 30 s; aviso visible si falla |
| Registro | Tabla `recommendations` de SQLite: entrada, salida, modelo, versión del prompt, estado (ok, invalid, error), detalle, tiempo y tokens |

## Costo estimado por 100 recomendaciones

Tokens medidos en los tres casos reales: 1 038, 1 079 y 1 056 por recomendación (entrada más salida), unos 106 000 tokens por cada 100 recomendaciones. En el plan gratuito *Experiment* de Mistral el costo es US$ 0 dentro de sus límites. Con el precio publicado por agregadores para `ministral-14b-2512` (US$ 0,20 por millón de tokens de entrada y de salida; verificar en mistral.ai/pricing), 100 recomendaciones costarían unos **US$ 0,02**.

## Tres casos reales (15/10/2026, holdout y datos del sistema)

### Caso 1 · Oro, perfil moderado

- **Entrada (resumen):** tendencia Alcista con 38,2 % (Neutral 36,5 %, Bajista 25,3 %); macro-F1 0,32 frente a persistencia 0,407; riesgo Medio; señal de reglas *Abstenerse*; volatilidad 0,223; caída −0,121; VIX 17,091; sentimiento +0,767 con 3 noticias relevantes. Acciones permitidas: Abstenerse, Observar.
- **Respuesta:** Abstenerse (alternativa Observar), confianza 0,85. "El modelo XGBoost no supera los umbrales de calidad (macro-F1 de holdout 0.32) ni cumple la regla de probabilidad máxima del 55%… Aunque el sentimiento en noticias es positivo (0.767), la señal de las reglas prioriza precaución. La caída del 12.1% desde máximos en 12 meses refuerza la cautela." Tiempo 7,3 s; 1 038 tokens.
- **Análisis crítico:** razonable. Respeta la abstención de las reglas, cita cifras reales y explica correctamente que el sentimiento positivo no compensa un modelo que no supera la persistencia. El tercer riesgo ("reversión a la media") es genérico.

### Caso 2 · Plata, perfil moderado

- **Entrada real:**
```json
{"metal": "Plata", "mes_objetivo": "2026-10", "perfil_inversionista": "moderado",
 "prediccion_xgboost": {"tendencia": "Bajista", "probabilidades": {"Bajista": 0.65, "Neutral": 0.092, "Alcista": 0.259}},
 "calidad_modelo": {"macro_f1_holdout": 0.43, "macro_f1_clase_mayoritaria": 0.187, "macro_f1_persistencia": 0.165},
 "riesgo_reglas": {"nivel": "Alto", "senal": "Esperar por riesgo alto",
                   "motivos": ["Volatilidad anualizada de 12 retornos mensuales >= 30%.", "Caída desde máximo de 12 meses >= 15%."]},
 "indicadores": {"volatilidad_12m": 0.614, "caida_desde_maximo_12m": -0.29, "vix": 17.091},
 "noticias": {"sentimiento_medio": -0.925, "relevantes": 4, "analizadas": 5,
              "titulares": ["Silver Price Today: Silver Falls 3.61% on October 15, 2026 - USA Today (sentimiento -1.00)",
                            "Silver Price Forecast: XAG/USD rebounds, but bearish technical outlook persists - FXStreet (sentimiento -0.70)",
                            "Silver Price Trades Near Its 2026 Low: Bargain or Value Trap? - BeInCrypto (sentimiento -1.00)"]},
 "acciones_permitidas": ["Esperar", "Abstenerse", "Observar"]}
```
- **Respuesta:** Abstenerse (alternativa Observar), confianza 0,85. Cita el 65 % bajista, el sentimiento −0,925, la volatilidad de 61,4 % y la caída de 29 %. Tiempo 10,5 s; 1 079 tokens.
- **Análisis crítico:** coherente con el riesgo Alto y más prudente que la señal de reglas (eligió Abstenerse en lugar de Esperar, ambas permitidas). Dos imprecisiones: llama "baja" a la calidad del modelo cuando el macro-F1 0,43 supera a ambas referencias (0,187 y 0,165), y habla de "máximos históricos" cuando el dato es el máximo de 12 meses. También menciona "bajo soporte en indicadores macro", que no está en la entrada.

### Caso 3 · Plata, perfil agresivo

- **Entrada:** la del caso 2 con perfil agresivo.
- **Respuesta:** Esperar (alternativa Observar), confianza 0,75. Tiempo 7,8 s; 1 056 tokens.
- **Análisis crítico:** la acción coincide exactamente con la señal de reglas y las cifras son reales. Error de interpretación: describe el perfil agresivo como uno "que prioriza preservación de capital", lo contrario de su definición. La validación no detecta este tipo de error porque no involucra cifras ni acciones; por eso la recomendación final siempre queda acotada por las reglas.

## Limitaciones

- La recomendación es académica: no considera capital, impuestos, costos reales ni objetivos personales.
- El LLM puede cometer errores de interpretación sin números (casos 2 y 3); las reglas limitan su impacto a acciones permitidas.
- En Render (25/10/2026, oro, perfil agresivo) el LLM volvió a comparar el macro-F1 con 0,55 pese a la instrucción del prompt; la acción (Abstenerse) fue correcta. En una prueba anterior escribió el umbral de 55 % como 0,55 y el validador lo rechazó por error: se corrigió para aceptar ambas formas (commit dd561da).
- En Render gratuito el registro de recomendaciones se reinicia con la instancia; la copia del repositorio conserva las del último despliegue.
