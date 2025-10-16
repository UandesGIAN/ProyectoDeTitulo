# Hito 2: Prototipo de la KB con RAG y generación de reportes primitiva

## El pipeline

1. **Ingesta de documentos**
- LangChain para leer documentos: [ingest.py](ingest.py)
- Se guardan en JSON, uno por documento. En `/extracted_texts_jsons`.
- Se extrae metadata como fecha, autor, fuente y se divide el texto en chunks.

2. **Enriquecimiento semántico**
- Usa Gemini AI para enriquecer los chunks de los JSON en `/extracted_texts_jsons` y extraer roles, riesgos, controles, etc. [semantic_enrichment.py](semantic_enrichment.py)
- Generar JSON para cada fragmento con las recomendaciones, se guardan en `/processed_jsons`.

3. **Indexado vectorial**
- LangChain Chroma para Vector Store, usa Gemini Embeddings para crear los vectores de las recomendaciones. [vector_store.py](vector_store.py)
- Se almacenan en `/vectorstore_chroma`.
- Tambien define el retriever para buscar los vectores más cercanos a partir de una consulta.

4. Consulta con RAG
- Usar Gemini AI para probar prompts que recuperen chunks relevantes y generen la respuesta usando esos fragmentos como contexto.


## Para ejecutar
1. 
```bash
./setup_env.sh
```

2.
Crear `.env` y poner dentro `GOOGLE_API_KEY=api_key`

3. 
```bash
(to do)
./pipeline.sh
```


## Para borrar
```bash
deactivate
rm -rf venv
```
