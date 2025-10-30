import os
import time
from dotenv import load_dotenv
import google.genai as genai
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(__file__)
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore_chroma")


# GEMINI SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró GOOGLE_API_KEY en el entorno o .env")

client = genai.Client(api_key=API_KEY)


# FUNCIONES AUXILIARES
def get_gemini_embedding(text: str):
    if not text.strip():
        return [0.0] * 3072  # embedding dummy
    for attempt in range(5):
        try:
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return resp.embeddings[0].values
        except Exception as e:
                print(f"[ERROR] Intento {attempt+1}/{5} fallido: {e}")
                if attempt < 4:
                    time.sleep(5)
                else:
                    raise


# CARGAR VECTOR STORE
vector_store = Chroma(
    embedding_function=get_gemini_embedding,
    collection_name="kb_rag",
    persist_directory=VECTOR_DB_DIR
)
print("Vectores cargados:", len(vector_store._collection.get()["ids"]))

# RETRIEVER
def normalize_str(s):
    if not s:
        return ""
    return str(s).strip().lower()

def normalize_list(lst):
    return [normalize_str(x) for x in lst]

def semantic_query(query_text="", top_k=5, filters=None):
    """
    Busca recomendaciones por:
    - query_text: similitud semántica
    - filters: metadata (dimension, nivel, tags, tema...)
    """
    if query_text:
        print(f"[DEBUG] Buscando por query: {query_text}")
        query_emb = get_gemini_embedding(query_text)
        results = vector_store.similarity_search_by_vector(query_emb, k=top_k)
    else:
        raw = vector_store._collection.get()
        results = [
            Document(page_content=d, metadata=m)
            for d, m in zip(raw["documents"], raw["metadatas"])
        ]
    
    # Filtrado por metadata
    if filters:
        print(f"[DEBUG] Aplicando filtros: {filters}")
        filtered = []
        for i, r in enumerate(results):
            match = True
            for key, val in filters.items():
                if key not in r.metadata:
                    match = False
                    break
                
                doc_val = r.metadata.get(key)

                # Tags: coincidencia exacta dentro de la lista
                if key == "tags":
                    if isinstance(doc_val, str) and doc_val.startswith('[') and doc_val.endswith(']'):
                        doc_val = doc_val[1:-1].replace("'", "").replace('"','').split(',')
                        doc_val = [t.strip() for t in doc_val if t.strip()]
                    doc_tags = [normalize_str(t) for t in doc_val or []]
                    filter_tags = [normalize_str(f) for f in val]
                    if not any(f == t for f in filter_tags for t in doc_tags):
                        match = False
                        break

                # Tema: coincidencia parcial
                elif key == "tema":
                    doc_temas = doc_val if isinstance(doc_val, list) else [doc_val]
                    doc_temas = [normalize_str(t) for t in doc_temas]
                    if not any(normalize_str(val) in t for t in doc_temas):
                        match = False
                        break

                # Otros campos: coincidencia parcial (substring)
                else:
                    doc_val_str = normalize_str(doc_val)
                    filter_val_str = normalize_str(val)
                    if filter_val_str not in doc_val_str:
                        match = False
                        break

            if match:
                filtered.append(r)
        print(f"[DEBUG] Total documentos encontrados: {len(filtered)}")
        filtered.sort(key=lambda x: len(x.page_content), reverse=True)
        return filtered[:top_k]

    return results


# EJEMPLO DE USO
example_results = semantic_query(
    query_text="Usar autenticación multifactor",
    top_k=5,
    filters={}
)

for r in example_results:
    print(r.metadata["recomendacion"], "\n")

print("SIN QUERY:")
# Solo buscar por metadata (sin query_text)
example_results2 = semantic_query(
    query_text="",
    top_k=5,
    filters={"tags": ["MFA"], "nivel": "promedio", "dimension": "AUC"}
)

for r in example_results2:
    print(r.metadata["recomendacion"], "\n")
    print("FUENTE: ", r.metadata["fuente"], "\n")
