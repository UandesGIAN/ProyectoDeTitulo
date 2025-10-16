import os
import time
from dotenv import load_dotenv
import google.genai as genai
from langchain_chroma import Chroma

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
    """Obtiene embedding de Gemini para un texto"""
    if not text.strip():
        return [0.0] * 1536  # embedding dummy
    for attempt in range(5):
        try:
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return resp.embeddings[0].values
        except Exception as e:
                print(f"[ERROR] Intento {attempt+1}/{5} fallido: {e}")
                if attempt < 5 - 1:
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
def semantic_query(query_text, top_k=5):
    query_emb = get_gemini_embedding(query_text)
    results = vector_store.similarity_search_by_vector(query_emb, k=top_k)
    return results


# EJEMPLO DE USO
query = "Buenas prácticas de contraseñas"
results = semantic_query(query, top_k=3)
print(f"\n[RESULTADOS] Top {len(results)} documentos relevantes:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}\n   METADATA: {r.metadata}\n")
