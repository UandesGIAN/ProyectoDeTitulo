# Hito 2: Prototipo de la KB con RAG y generación de reportes primitiva

## El pipeline

1. **Procesar muestra encuesta**
- [procesar_encuesta.py](procesar_encuesta.py)  lee los datos puros del imech y la muestra de la encuesta para crear un JSON que combina ambos datos.

2. **Analizar datos encuesta**
- [procesar_datos_encuesta.py](procesar_datos_encuesta.py) Lee los datos de la encuesta puros y evalúa las dimensiones más débiles del usuario.
- Asigna un puntaje a cada respuesta del usuario y la compara con una respuesta promedio. Si  está por debajo del 65% considera que está en riesgo.
- Genera JSON de salida con un análisis por encuestado.

3. **Obtener recomendaciones**
- Se accede a los datos de las encuestas procesados y se hacen consultas a los embeddings del vector store (KB) para recuperar las recomendaciones.
- Guarda las recomendaciones en la carpeta `/recomendaciones`

4. **Generar reportes**
- Usar Gemini AI para generar markdown con 10 recomendaciones de todas las que se fueron recomendadas para el encuestado.
- Se puede editar el prompt, es provisional [generar_reportes.py](generar_reportes.py)


## Para ejecutar
1. 
```bash
./setup_env.sh
```

2.
Crear `.env` y poner dentro `GOOGLE_API_KEY=api_key`

3. 
Si se desea ejecutar todo el proceso desde procesar la encuesta hasta la generación de reportes, se puede usar el siguiente script:
```bash
./pipeline.sh
```


## Para borrar
```bash
deactivate
rm -rf venv
```
