# Instalar y ejecutar en la computadora del docente

Esta guía permite entregar un ZIP y mostrar la aplicación en otra computadora Windows. La instalación será local. No exige cuenta de Hugging Face, Docker, Gemini ni VS Code para ver la demostración.

## Qué debe tener la otra computadora

- Windows de 64 bits y Python **3.13 de 64 bits**. Esta es la versión comprobada con el archivo de dependencias fijadas.
- Un navegador actualizado, como Edge, Chrome o Firefox.
- Conexión a internet durante la primera instalación de dependencias. Después, el paquete con datos y modelos se puede consultar sin nuevas descargas de fuentes ni llamadas a Gemini.
- Como recomendación práctica: 8 GB de RAM y unos 2 GB libres para Python, bibliotecas y archivos. No requiere GPU.

Si la computadora pertenece a la universidad, la instalación de Python puede requerir que el docente o soporte la habilite. El proyecto no cambia permisos del equipo.

## A. Preparar el ZIP en tu computadora

Desde la terminal de VS Code, situada en la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m metals.cli predict
.\.venv\Scripts\python.exe scripts/package_space.py --target docente --include-data
```

Si `predict` informa que cambió el dataset, ejecuta `train` y luego `predict` antes de empaquetar. Si quieres entregar noticias adicionales analizadas, ejecuta primero `sentiment --limit 5` y reconstruye los resultados.

El programa muestra el nombre del ZIP creado en `dist`. Incluye los datos reales guardados en SQLite, los modelos, los reportes, las noticias y los resultados de sentimiento. También incluye el código y las guías para que el docente pueda revisar el trabajo.

No incluye `.env`, tu clave, `.venv` ni `.git`. **No copies toda tu carpeta manualmente**: usa este ZIP para evitar trasladar credenciales o un entorno Python con rutas de tu equipo. Puedes transportar el ZIP por USB o por el medio de entrega que indique tu curso. La herramienta no lo envía por ti.

## B. Extraer y preparar Python en la computadora del docente

1. Copia el ZIP al equipo del docente.
2. Clic derecho → **Extraer todo**. Usa, por ejemplo, `Documentos\AurumDocente`.
3. Abre la carpeta extraída. Deben verse directamente `app.py`, `INSTALAR.cmd`, `INICIAR.cmd`, `metals`, `data` y `artifacts`.
4. Abre una terminal y escribe `python --version`. Debe indicar Python 3.13.x. Si no existe, instala esa versión de 64 bits desde [Python.org](https://www.python.org/downloads/). Añade Python al PATH y abre de nuevo la terminal.

## C. Instalación sencilla

Haz doble clic en **INSTALAR.cmd**. El archivo crea su propio entorno `.venv` en la nueva computadora e instala `requirements-lock.txt`. La primera vez puede tardar varios minutos. Espera **Instalacion completada**.

Después haz doble clic en **INICIAR.cmd**. Abre [http://localhost:8501](http://localhost:8501). Mantén la ventana de comandos abierta mientras se usa la app. Para finalizar, presiona `Ctrl+C` en esa ventana.

Los scripts son texto y pueden abrirse en Bloc de notas o VS Code para revisar exactamente qué ejecutan. No se necesita cambiar la política de PowerShell ni usar privilegios de administrador para el proyecto.

## D. Alternativa con terminal de VS Code

Si el docente desea ejecutar y revisar el código:

1. Abre VS Code → **Archivo → Abrir carpeta** → selecciona la carpeta extraída.
2. Abre **Terminal → Nueva terminal**.
3. Ejecuta, uno por uno:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

No ejecutes `run` para la primera demostración si deseas conservar exactamente las métricas entregadas. El ZIP ya contiene un experimento calculado con fuentes reales.

## E. Comprobar que funciona

| Acción | Resultado esperado |
|---|---|
| Abrir la página | Título del proyecto y tarjetas de precio/fecha/tendencia/riesgo |
| Cambiar Oro por Plata | Cambian precios, métricas y señal |
| Descriptivo | Serie de precios, media móvil, retornos, comparación y macro |
| Predictivo | Mes objetivo, probabilidades, baselines, matriz de confusión y validación |
| Prescriptivo | Señal y razones de riesgo/abstención |
| Noticias y minería | Noticias con enlace, score, relevancia, explicación y evidencia Gemini cuando existen |
| Descargar un reporte | Archivo JSON con las mismas métricas mostradas |

Que aparezca **Abstenerse** no significa que la aplicación falle. Es una salida prevista cuando no hay evidencia suficiente o el resultado está vencido. Tampoco debe aparecer una credencial en ningún panel.

## F. Actualizar o analizar nuevas noticias

`ACTUALIZAR.cmd` descarga las fuentes y vuelve a entrenar. Necesita internet y puede cambiar las métricas. Para preservar la entrega original, conserva el ZIP y los reportes originales.

Los análisis Gemini ya guardados se consultan sin clave. Para generar nuevos análisis en el equipo del docente, él debe configurar **su propia clave** en un nuevo `.env`, siguiendo `.env.example`, y ejecutar:

```powershell
.\.venv\Scripts\python.exe -m metals.cli gemini-check
.\.venv\Scripts\python.exe -m metals.cli sentiment --limit 2
```

No es necesario compartir tu clave para presentar el proyecto. El nombre del modelo puede cambiar; el diagnóstico informa disponibilidad y el análisis informa errores de servicio sin mostrar la credencial.

## G. Errores habituales

- **No se encuentra Python:** instalar 3.13 de 64 bits y reabrir VS Code.
- **No module named streamlit:** ejecutar `INSTALAR.cmd` y esperar a que termine.
- **Puerto ocupado:** cierra la otra instancia o añade `--server.port 8502` y abre `http://localhost:8502`.
- **No aparecen datos/modelos:** se entregó un paquete sin `--include-data` o no se extrajo completo. Regenera el ZIP correcto.
- **Dataset cambiado:** reentrena y predice en la misma computadora antes de comparar los resultados.
- **Sin internet al instalar:** primero instala las dependencias con conexión. Este paquete no incluye ruedas offline de Python.

La instalación se verificó en la computadora de desarrollo. El archivo puede inspeccionarse y probarse extraído; la comprobación en el equipo físico del docente se completa al seguir esta guía. Para publicar un enlace en internet, usa la guía independiente `05_HUGGING_FACE.md`.
