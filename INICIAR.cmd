@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
echo Abre http://localhost:8501 en el navegador.
echo Mantener esta ventana abierta. Ctrl+C para detener.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 pause
