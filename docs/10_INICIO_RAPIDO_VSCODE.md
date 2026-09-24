# Ejecutarlo ahora desde tu Visual Studio Code

## Abrir tu proyecto

1. Abre **Visual Studio Code**.
2. Selecciona **Archivo → Abrir carpeta**.
3. Abre `C:\Users\MANUEL ALONSO\Documents\ChatGPT\Analitica predictiva para Oro y Plata,Fun business`.
4. Verifica que aparecen `app.py`, `metals`, `docs` y `.venv`.
5. Abre **Terminal → Nueva terminal**. La terminal debe estar situada en esta misma carpeta.

## Iniciar el dashboard

El entorno y los datos ya están preparados en esta computadora. Ejecuta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre **http://localhost:8501**. Deja la terminal abierta. Si ya está funcionando otra instancia, usa esa página o detén la anterior con `Ctrl+C` antes de volver a iniciar.

## Elegir el intérprete para editar código

Pulsa `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`. Si el menú no aparece, instala la extensión oficial **Python de Microsoft** desde Extensiones. Esto ayuda al editor a reconocer las bibliotecas instaladas.

## Actualizar datos y modelos

Abre una **segunda terminal** para mantener la aplicación abierta y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m metals.cli run
```

Este comando descarga, limpia, guarda, construye variables, evalúa XGBoost y genera la señal. Al terminar pulsa **Actualizar vista** en el dashboard. Un RSS bloqueado queda registrado y se utiliza la fuente alternativa disponible.

## Ejecutar sentimiento Gemini

La clave está configurada localmente. No la pegues en el código ni en reportes. Para diagnosticar la conexión y procesar un lote pequeño:

```powershell
.\.venv\Scripts\python.exe -m metals.cli gemini-check
.\.venv\Scripts\python.exe -m metals.cli sentiment --limit 5
```

Abre **Noticias y minería** y selecciona **Analizadas con Gemini**. Cada resultado muestra score, relevancia, explicación, evidencia, modelo y fecha. Un artículo ya analizado con el mismo modelo/prompt no se vuelve a consumir.

También puedes seleccionar una fuente guardada para analizar un lote temático:

```powershell
.\.venv\Scripts\python.exe -m metals.cli sentiment --limit 2 --source "Mining Weekly seleccion verificada"
```

Un error HTTP 503 indica saturación temporal del servicio. Las noticias completadas se conservan; vuelve a ejecutar el comando más tarde para las pendientes. `gemini-check` confirma acceso y catálogo, pero no garantiza capacidad de generación en ese momento.

El score contextual no cambia automáticamente una predicción cuyo origen es anterior a las noticias. Cuando se incorporen nuevas variables a un mes cerrado, vuelve a ejecutar:

```powershell
.\.venv\Scripts\python.exe -m metals.cli train
.\.venv\Scripts\python.exe -m metals.cli predict
```

## Comprobar el trabajo

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m metals.cli status
```

Pruebas aprobadas significan que los comportamientos verificados funcionan; no garantizan exactitud financiera. Consulta la matriz de confusión y compara las referencias.

## Entregar al docente

```powershell
.\.venv\Scripts\python.exe scripts/package_space.py --target docente --include-data
```

Entrega el ZIP que aparece en `dist`, junto con la indicación de abrir `LEEME_DOCENTE.md`. La guía detallada está en `09_INSTALACION_DOCENTE.md`. El paquete conserva resultados y excluye tu clave.

## Detener la aplicación

Selecciona la terminal que ejecuta Streamlit y pulsa `Ctrl+C`. Tus archivos y datos quedan guardados; para volver a abrir sólo repite el comando de inicio.
