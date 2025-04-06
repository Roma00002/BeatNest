#!/bin/bash

# Nota: Este script necesita permisos de ejecución
# Ejecutar: chmod +x iniciar.sh

# Colores para los mensajes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mostrar mensajes de error y salir
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Verificar si el directorio venv existe
if [ ! -d "venv" ]; then
    error_exit "No se encontró el entorno virtual en ./venv/
Por favor, cree el entorno virtual con:
python -m venv venv
pip install -r requirements.txt"
fi

# Verificar si run_gui.py existe
if [ ! -f "run_gui.py" ]; then
    error_exit "No se encontró el archivo run_gui.py"
fi

# Activar el entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# Verificar si la activación fue exitosa
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✔ Entorno virtual activado correctamente${NC}"
else
    error_exit "No se pudo activar el entorno virtual"
fi

# Crear directorios necesarios si no existen
mkdir -p checkpoints generated

# Ejecutar la aplicación
echo -e "${GREEN}Iniciando AI Music Beat Generator...${NC}"
python run_gui.py

# Capturar el código de salida de la aplicación
exit_code=$?

# Desactivar el entorno virtual
deactivate

# Verificar si la aplicación se ejecutó correctamente
if [ $exit_code -ne 0 ]; then
    error_exit "La aplicación terminó con errores (código: $exit_code)"
fi 