# Publicación gratuita en Render

La aplicación es Python + Streamlit con XGBoost, SQLite y un LLM (Mistral) por API. Necesita un proceso de Python siempre activo, por eso no encaja en hosting estático ni en Vercel. Render la ejecuta con el `Dockerfile` del repositorio y entrega una URL pública con HTTPS.

**Plan gratuito (comprobar en render.com antes de usarlo):** 512 MB de memoria, la instancia se suspende tras unos 15 minutos sin visitas (la primera carga posterior tarda cerca de 50 segundos) y el disco no es persistente: al reiniciar vuelve a la base `data/metals.sqlite` del repositorio. La app medida en local usa unos 350 MB incluido el análisis en vivo.

## 1. Subir el código a GitHub (desde Visual Studio Code)

En la terminal PowerShell, dentro de la carpeta del proyecto:

```powershell
git status
git add -A
git commit -m "Descripción del cambio"
git push origin master
```

`.env` (claves) está en `.gitignore` y nunca se sube. Sí se suben `data/metals.sqlite` y los modelos de `artifacts/gold` y `artifacts/silver`, porque Render construye la app desde GitHub.

## 2. Crear el servicio en Render

1. Entrar en [render.com](https://render.com) con la cuenta de GitHub.
2. **New + → Web Service** y elegir el repositorio (al instalar la app de Render en GitHub, conviene **Only select repositories**).
3. Configuración:

| Campo | Valor |
|---|---|
| Language | Docker |
| Branch | master |
| Root Directory | vacío |
| Instance Type | Free |

4. **Environment Variables** (los valores se escriben sólo en Render):

| Key | Value |
|---|---|
| `MISTRAL_API_KEY` | clave de [console.mistral.ai](https://console.mistral.ai), sin fecha de vencimiento próxima |
| `SENTIMENT_PROVIDER` | `mistral` |
| `APP_PASSWORD` | contraseña para el botón "Probar el análisis en vivo" |

5. **Deploy Web Service**. Al terminar, el registro muestra `Your service is live` y la URL `https://<nombre>.onrender.com`.

El `Dockerfile` usa el puerto que asigna Render (`$PORT`) y 7860 en local.

## 3. Actualizar la versión publicada

Cada `git push` a `master` despliega una versión nueva. Si no ocurre, en Render: **Manual Deploy → Deploy latest commit**. En **Events** debe aparecer `Deploy live for <commit>`.

## 4. Comprobar la publicación

- Abrir la URL y la pestaña **Noticias y minería**.
- Escribir `APP_PASSWORD` y pulsar **Actualizar noticias y analizar**. Ambos pasos deben terminar con `código 0`.
- En `"providers"` sólo debe aparecer el modelo de Mistral configurado.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| 502 Bad Gateway | Despliegue en curso o memoria agotada | Esperar a `Live` en Events; revisar Logs |
| `HTTP 401` de Mistral | Clave vencida o incorrecta | Crear otra clave y actualizar `MISTRAL_API_KEY` en Environment |
| Sigue apareciendo una versión vieja | No se desplegó el último commit | Manual Deploy → Deploy latest commit |
| Primera carga lenta | Instancia gratuita suspendida | Normal en el plan gratuito; abrirla unos minutos antes de presentar |
