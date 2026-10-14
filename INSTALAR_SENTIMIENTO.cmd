@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
echo Instala la API local FinBERT como respaldo de Gemini.
echo La primera vez descarga PyTorch CPU y el modelo. Necesita internet y espacio libre.
".venv\Scripts\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements-sentiment.txt
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m metals.local_sentiment_api --prepare
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :error
echo Respaldo local instalado. Usa ACTUALIZAR_NOTICIAS.cmd.
pause
exit /b 0
:error
echo No se completo la instalacion. Revisa docs\14_PROVEEDORES_SENTIMIENTO.md.
pause
exit /b 1
