@echo off
setlocal
cd /d "%~dp0"
echo Instalacion local de Aurum - Python 3.13 de 64 bits
py -3.13 --version >nul 2>&1
if errorlevel 1 (set "METALS_PYTHON=python") else (set "METALS_PYTHON=py -3.13")
%METALS_PYTHON% -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
 echo Instala Python 3.13 de 64 bits desde python.org y vuelve a ejecutar este archivo.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 %METALS_PYTHON% -m venv .venv
 if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :error
echo Instalacion completada. Abre INICIAR.cmd.
pause
exit /b 0
:error
echo La instalacion no termino. Consulta docs\03_SOLUCION_ERRORES.md.
pause
exit /b 1
