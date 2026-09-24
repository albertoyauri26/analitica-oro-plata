# Despliegue posterior en Hugging Face Spaces

Esta etapa empieza cuando hayas abierto la aplicación local, revisado ambos metales y entendido sus limitaciones. Preparar un paquete no crea un Space ni publica archivos.

La vía de este proyecto es **Streamlit dentro de Docker**. Hugging Face documenta Docker para alojar aplicaciones fuera de sus SDK específicos. `README.md` ya indica `sdk: docker` y `app_port: 7860`; `Dockerfile` inicia Streamlit en ese puerto y en `0.0.0.0`.

Consulta antes de crear el Space la [documentación de Spaces](https://huggingface.co/docs/hub/spaces-overview) y los requisitos/precios que muestre tu cuenta. No se presupone que el cómputo sea gratuito ni se contrata capacidad automáticamente.

## 1. Congelar una demostración local

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m metals.cli train
.\.venv\Scripts\python.exe -m metals.cli predict
.\.venv\Scripts\python.exe scripts/package_space.py --include-data
```

El ZIP de `dist` contiene aplicación, código, documentación, dependencias y una copia consistente de SQLite junto con modelos/reportes. Su manifiesto enumera archivos y huellas. **No contiene `.env`, `.venv`, `.git`, claves, ni originales masivos de `data/raw`.** Revisa que los datos/noticias que vas a mostrar puedan redistribuirse según las condiciones de sus fuentes.

Esta copia permite presentar la app sin descargar ni entrenar al arrancar. Conserva fechas originales; si la predicción vence, la interfaz lo señalará. Para una nueva presentación actualiza localmente y genera otro paquete.

## 2. Prueba Docker opcional

Docker Desktop no se detectó en la revisión inicial de este equipo. No es necesario para ejecutar el proyecto local con Python. Si decides instalarlo, prueba desde la carpeta del proyecto:

```powershell
docker build -t metals-academic .
docker run --rm -p 7860:7860 metals-academic
```

Abre `http://localhost:7860`. `Ctrl+C` detiene el contenedor. La disponibilidad de la imagen y bibliotecas Linux se comprueba durante la construcción; la instalación Windows no demuestra por sí sola que el contenedor ya haya sido probado.

## 3. Crear el Space

Entra en [Hugging Face Spaces](https://huggingface.co/spaces) con tu cuenta. Selecciona **Create new Space**. Elige nombre, visibilidad y SDK **Docker**, opción de Dockerfile propio/Blank cuando esté disponible. Selecciona capacidad de CPU apropiada y revisa cualquier coste que se muestre.

Descomprime el ZIP en una carpeta separada. En la pestaña **Files** sube su contenido conservando carpetas. `Dockerfile`, `README.md` y `app.py` deben quedar en la raíz, no dentro de otra carpeta llamada `metals-space`.

Puedes usar Git más adelante para actualizaciones. El flujo manual evita pedirte credenciales antes de tener un resultado local comprobado. No compartas tokens en el chat ni los incluyas en archivos.

## 4. Verificar la construcción y la app

Observa **Build logs** hasta que termine la instalación. Luego observa **Container logs** y abre **App**. Verifica:

- Ambos metales muestran precios reales, mes de referencia y procedencia.
- El dataset coincide con el modelo. No hay aviso de huella incompatible.
- El panel predictivo muestra probabilidades y métricas, y el prescriptivo explica sus razones.
- La falta de Gemini o de producción numérica se muestra con claridad.
- Las descargas de JSON/CSV funcionan.

Un error de `libgomp.so.1` se atiende con la dependencia `libgomp1`, incluida en el Dockerfile. Un fallo de puerto se revisa comparando 7860 en README, Dockerfile y logs. Un error de paquetes Linux se resuelve revisando el lock y probando una nueva construcción, no borrando datos locales.

## 5. Secretos y persistencia

La demo pública funciona sin Gemini en tiempo real. El análisis se realiza localmente y se publica su resultado cuando proceda. Si en una ampliación se habilita backend privado con APIs, usa **Settings → Secrets**, nunca código fuente. [Hugging Face documenta variables y secretos](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces).

El disco de ejecución de un Docker Space puede perder sus cambios al reiniciar. La copia incluida en el repositorio se restaura al reconstruir. El modelo actual de despliegue es una **demo con snapshot**, no una base de datos productiva persistente. [Referencia Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

## 6. Presentar el enlace

Después de comprobar estos pasos, copia la URL del Space para tu docente. Conserva una copia local funcional y los reportes de la misma ejecución. La publicación quedará pendiente hasta realizarla en tu cuenta y verificar su estado real; un Dockerfile creado no equivale a un despliegue completado.
