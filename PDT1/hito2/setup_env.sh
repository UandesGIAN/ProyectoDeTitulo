#!/bin/bash
# setup_env.sh

# Crear entorno virtual
python3 -m venv venv

# Activar venv
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

echo "¡Entorno listo! Activar con: source venv/bin/activate"
