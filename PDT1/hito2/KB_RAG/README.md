# Hito 2: Prototipo de la KB con RAG y generación de reportes primitiva

## El pipeline

1. Ingesta de documentos
- LangChain para leer documentos: [ingest.py](ingest.py)
- Se guardan en JSON, uno por documento. En `/extracted_texts_jsons`.

2. Enriquecimiento semántico
- Primero split en chunks para cada documento.
- Luego se usa Gemini AI para leer cada documento y extraer roles, riesgos, controles, fuentes...
- Generar JSON para cada fragmento con las recomendaciones.

3. Indexado vectorial
- FAISS (local, rápido) o Qdrant (más avanzado, tolera metadatos y filtros).

4. Consulta con RAG
- Usar Gemini AI para probar prompts que recuperen chunks relevantes y generen la respuesta usando esos fragmentos como contexto.

## Para ejecutar
```bash
./setup_env.sh
```

- *ingest.py* no hace falta ejecutarlo, pues ya están los documentos en la carpeta `/extracted_texts_jsons`.


## Para borrar
```bash
deactivate
rm -rf venv
```
