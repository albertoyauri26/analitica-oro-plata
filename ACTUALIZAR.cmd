@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Ejecuta INSTALAR.cmd primero.
 pause
 exit /b 1
)
echo Actualiza fuentes, dataset, modelos y senales. Requiere internet.
echo No realiza llamadas a Gemini. Conserva aparte los reportes usados en una presentacion.
".venv\Scripts\python.exe" -m metals.cli run
if errorlevel 1 (
 echo La actualizacion no termino correctamente. Revisa el mensaje y la guia de errores.
) else (
 echo Actualizacion completada. Pulsa Actualizar vista en el dashboard.
)
pause
