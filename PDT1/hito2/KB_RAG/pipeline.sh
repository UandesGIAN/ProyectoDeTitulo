echo "Ejecutando ingest.py..."
python3 ingest.py
if [ $? -ne 0 ]; then
    echo "Error en ingest.py. Abortando."
    exit 1
fi

echo "Ejecutando semantic_enrichment.py..."
python3 semantic_enrichment.py
if [ $? -ne 0 ]; then
    echo "Error en semantic_enrichment.py. Abortando."
    exit 1
fi

echo "Ejecutando vector_store.py..."
python3 vector_store.py
if [ $? -ne 0 ]; then
    echo "Error en vector_store.py. Abortando."
    exit 1
fi

echo "Ejecutando tester.py..."
python3 tester.py
if [ $? -ne 0 ]; then
    echo "Error en tester.py."
    exit 1
fi

echo "Todos los scripts ejecutados correctamente."