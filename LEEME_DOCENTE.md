# Abrir la demostración en otra computadora

1. Extraiga **todo el ZIP** en una carpeta local de Windows. No ejecute archivos dentro del ZIP.
2. Instale **Python 3.13 de 64 bits** desde https://www.python.org/downloads/ si no está instalado. Marque la opción para añadir Python al PATH.
3. Abra `INSTALAR.cmd`. Este paso necesita internet para instalar las bibliotecas. Espere el mensaje de instalación completada.
4. Abra `INICIAR.cmd`.
5. Abra http://localhost:8501 en el navegador. Mantenga abierta la ventana de comandos.
6. Recorra las cinco pestañas y cambie entre Oro y Plata. Los datos, modelos y análisis Gemini incluidos permiten abrir la demostración sin proporcionar una clave de API.
7. Para detener el servidor, presione `Ctrl+C` en la ventana de comandos.

La aplicación muestra las fechas reales de sus datos. Si un resultado mensual ya venció, se conserva como histórico y se indica abstención. Use `ACTUALIZAR.cmd` sólo si desea recalcular los resultados con datos nuevos.

**Guía detallada:** [Instalación en la computadora del docente](docs/09_INSTALACION_DOCENTE.md).

**Uso desde VS Code:** [Guía local](docs/01_GUIA_LOCAL.md).

El paquete no incluye claves de API ni el entorno Python del autor. Los scripts y sus comandos pueden revisarse antes de ejecutarlos. La aplicación escucha sólo en la propia computadora, no se publica en internet.
