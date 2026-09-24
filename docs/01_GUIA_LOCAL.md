# Guía de ejecución y aprendizaje en Visual Studio Code

## 1. Entender qué estamos construyendo

Este proyecto responde: «Con la información histórica utilizable al cierre de un mes, ¿qué clase de cambio tendrá el promedio de precio del mes siguiente?». El modelo aprende ejemplos pasados etiquetados. El motor de riesgo aplica reglas explícitas a sus resultados. La aplicación permite inspeccionar ambas partes.

Trabajaremos en esta secuencia: **instalar → descargar → inspeccionar → construir variables → entrenar → interpretar → probar la web → conectar Gemini → preparar la presentación → publicar**. Cada paso tiene una comprobación. No avances a publicación si la prueba local falla.

## 2. Abrir la carpeta correcta

Abre Visual Studio Code. Elige **Archivo → Abrir carpeta** y selecciona:

```text
C:\Users\MANUEL ALONSO\Documents\ChatGPT\Analitica predictiva para Oro y Plata,Fun business
```

En el explorador de VS Code debes ver `app.py`, `requirements.txt`, `metals` y `docs`. No crees otra carpeta dentro para copiar los mismos archivos. Abre **Terminal → Nueva terminal**. Usa PowerShell.

```powershell
Get-Location
python --version
```

`Get-Location` debe indicar la carpeta del proyecto. Esta instalación dispone de Python 3.13. El registro del lanzador también menciona Python 3.11, pero su ejecutable no existe; por eso usamos `python`, que apunta al Python funcional. Las dependencias se probarán en ese entorno.

## 3. Crear un entorno aislado

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` guarda las bibliotecas de este proyecto. No modifica las de otros proyectos. `requirements.txt` enumera las dependencias y rangos compatibles. `requirements-lock.txt`, cuando exista, registra las versiones exactas comprobadas. Para reproducir esa instalación, usa `-r requirements-lock.txt`.

Si estos pasos ya se ejecutaron durante nuestra sesión, no necesitas repetirlos. Comprueba:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, xgboost, streamlit; print('Entorno listo')"
```

Instala la extensión oficial **Python de Microsoft** en VS Code. Pulsa `Ctrl+Shift+P`, busca **Python: Select Interpreter** y selecciona `.venv\Scripts\python.exe`. La ruta ya está sugerida en `.vscode/settings.json`.

Los comandos usan la ruta completa de Python para funcionar aunque PowerShell bloquee `Activate.ps1`. No necesitas cambiar la política de seguridad ni abrir una terminal de administrador.

## 4. Descargar y revisar datos

```powershell
.\.venv\Scripts\python.exe -m metals.cli ingest
.\.venv\Scripts\python.exe -m metals.cli status
```

La ingesta descarga Pink Sheet, cuatro CSV de FRED y los canales RSS configurados. Las noticias pasan un filtro de oro/plata y se añaden fuentes RSS especializadas para seleccionar hasta cinco por metal. Cada resultado informa su cantidad de registros o su error. `data/raw` conserva los originales. `data/metals.sqlite` almacena las tablas normalizadas. Consulta `11_NOTICIAS_Y_SENTIMIENTO.md` para completar el análisis Gemini.

Comprueba que `prices` y `macro` tengan registros y fechas razonables. Si las noticias fallan, puedes continuar con el modelo base. Si falta una serie macro requerida, revisa el registro: el entrenamiento no debe ocultar ese problema. En el documento de fuentes encontrarás la importación manual.

**Aprendizaje:** ingesta significa traer los datos; limpieza significa interpretar fechas, convertir valores a números, descartar ausencias explícitas de la fuente y verificar duplicados. No significa fabricar una serie completa.

## 5. Construir el dataset

```powershell
.\.venv\Scripts\python.exe -m metals.cli build
```

Abre `data/analytical_dataset.csv` con VS Code. Cada fila corresponde a un metal y un mes de origen. Encontrarás retornos históricos, distancias a medias móviles, volatilidad, RSI mensual, macro rezagada y el objetivo.

Las primeras filas tienen celdas vacías porque aún no existen suficientes meses para calcular ventanas de 12 meses. Las últimas etiquetas pueden estar vacías porque su resultado es futuro. Eso es correcto. `target` codifica Bajista=0, Neutral=1, Alcista=2.

**Ejercicio:** revisa `date`, `source_price_month`, `target_month` y `label_available_at` en la misma fila. Explica las cuatro fechas antes de interpretar una métrica.

## 6. Entrenar y evaluar

```powershell
.\.venv\Scripts\python.exe -m metals.cli train
.\.venv\Scripts\python.exe -m metals.cli predict
```

En `artifacts/gold` y `artifacts/silver` encontrarás:

- `evaluation_model.json`: modelo entrenado antes del holdout, usado para las métricas.
- `model.json`: modelo final reentrenado para inferencia.
- `report.json`: variables, parámetros, fechas, métricas, exclusiones y huellas de datos/modelo.
- `holdout_predictions.csv`: una fila por mes de prueba con predicción y resultado real.
- `prediction.json`: última tendencia, probabilidades, señal y razones del riesgo.

Macro-F1 resume el desempeño en las tres clases. Compáralo con clase mayoritaria y persistencia. Un resultado inferior a una referencia también es un hallazgo académico válido. El sistema se abstiene si no supera ambas referencias; no cambies reglas sólo para obtener una señal atractiva.

## 7. Abrir la aplicación

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre `http://localhost:8501`. Mantén esa terminal abierta. Para detener la aplicación pulsa `Ctrl+C` en ella.

Recorre **Descriptivo**, **Predictivo**, **Prescriptivo**, **Noticias y minería** y **Calidad y metodología**. Cambia de Oro a Plata en el lateral. Descarga un reporte y verifica que coincide con la carpeta `artifacts`.

VS Code también incluye una configuración de depuración: abre Ejecutar y depurar, elige **Dashboard de metales** y pulsa F5. Si falta `debugpy`, usa el comando de terminal anterior.

## 8. Activar Gemini

1. Abre [Google AI Studio](https://aistudio.google.com/apikey) en tu cuenta y crea una clave para Gemini.
2. Copia `.env.example` a `.env` desde VS Code. No reemplaces un `.env` existente.
3. Escribe tu clave únicamente en `GEMINI_API_KEY=`. Mantén la clave en tu equipo.
4. Verifica que `GEMINI_MODEL` sea un modelo disponible para tu cuenta y compatible con salida JSON estructurada. El nombre es configurable porque el catálogo cambia.
5. Ejecuta una prueba pequeña:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --limit 2
.\.venv\Scripts\python.exe -m metals.cli build
.\.venv\Scripts\python.exe -m metals.cli train
.\.venv\Scripts\python.exe -m metals.cli predict
```

Reentrenamos porque cualquier cambio del dataset invalida su huella anterior. Actualiza la vista del dashboard. El análisis debe indicar modelo, fecha, explicación y evidencia literal del texto. Si no hay clave o falla la API, debe aparecer pendiente, nunca como neutral.

Las noticias recién analizadas aparecen en el panel inmediatamente, pero las variables mensuales sólo las incorporan cuando su fecha de disponibilidad entra en un mes cerrado. Sin historia suficiente, permanecen excluidas del modelo. Esto evita evaluar con información creada después del hecho.

## 9. Comprobar y actualizar

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Los tests comprueban límites de clase, orden temporal, huecos mensuales, sentimiento y reglas. Usan pequeñas entradas artificiales **sólo como fixtures de prueba**, nunca como precios del proyecto ni como datos de entrenamiento reales.

Para actualizar mensualmente repite `python -m metals.cli run` con el Python del entorno. Conserva los reportes que uses en tu presentación antes de actualizar. En tu siguiente sesión podemos revisar juntos las métricas reales y la interpretación de cada pestaña.
