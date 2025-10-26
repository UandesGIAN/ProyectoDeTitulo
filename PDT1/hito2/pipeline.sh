echo "Ejecutando procesar_encuesta.py..."
python procesar_encuesta.py
if [ $? -ne 0 ]; then
    echo "Error en procesar_encuesta.py. Abortando."
    exit 1
fi

echo "Ejecutando procesar_datos_encuesta.py..."
python procesar_datos_encuesta.py
if [ $? -ne 0 ]; then
    echo "Error en procesar_datos_encuesta.py. Abortando."
    exit 1
fi

echo "Ejecutando obtener_querys.py..."
python obtener_querys.py
if [ $? -ne 0 ]; then
    echo "Error en obtener_querys.py. Abortando."
    exit 1
fi

echo "Ejecutando generar_reportes.py..."
python generar_reportes.py
if [ $? -ne 0 ]; then
    echo "Error en generar_reportes.py."
    exit 1
fi

echo "Todos los scripts ejecutados correctamente."