# Hito 2: Prototipo de la KB con RAG y generación de reportes primitiva

## El pipeline

1. **Procesar muestra encuesta**
- LangChain para leer documentos: [ingest.py](ingest.py)
- Se guardan en JSON, uno por documento. En `/extracted_texts_jsons`.
- Se extrae metadata como fecha, autor, fuente y se divide el texto en chunks.

2. **Analizar datos encuesta**
- Usa Gemini AI para enriquecer los chunks de los JSON en `/extracted_texts_jsons` y extraer roles, riesgos, dimensiones, recomendación, etc. [semantic_enrichment.py](semantic_enrichment.py)
- Genera JSON para cada fragmento con las recomendaciones, se guardan en `/processed_jsons`.

3. **Obtener recomendaciones**
- LangChain Chroma para Vector Store, usa Gemini Embeddings para crear los vectores de las recomendaciones. [vector_store.py](vector_store.py)
- Se almacenan en `/vectorstore_chroma`.
- Cada embedding posee la metadata necesaria para poder obternerse al consultar con querys.

4. **Generar reportes**
- Usar Gemini AI para probar prompts que recuperen chunks relevantes y generen la respuesta usando esos fragmentos como contexto.
- Para ello se dispone del archivo [tester.py](tester.py).


## Para ejecutar
1. 
```bash
./setup_env.sh
```

2.
Crear `.env` y poner dentro `GOOGLE_API_KEY=api_key`

3. 
Si se desea ejecutar todo el proceso desde la ingesta hasta el tester para regenerar el vector_store, se puede usar el siguiente script:
```bash
./pipeline.sh
```

Pero de normal se espera que se utilice la KB disponible en vectorstore_chroma con el [tester.py](tester.py).


## Para borrar
```bash
deactivate
rm -rf venv
```
