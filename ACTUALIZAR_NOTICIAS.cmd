@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
echo Recopila cinco noticias por metal y analiza con el LLM de .env (SENTIMENT_PROVIDER: mistral o gemini).
".venv\Scripts\python.exe" -m metals.cli news-refresh --per-metal 5 --analyze
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m metals.cli predict
if errorlevel 1 goto :error
echo Noticias y escenario actualizados. Pulsa Actualizar vista en la aplicacion.
pause
exit /b 0
:error
echo Se conservaron los datos completados. Consulta la guia de errores o de sentimiento.
pause
exit /b 1
