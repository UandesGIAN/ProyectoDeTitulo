import os
import json
import uuid
import time
from dotenv import load_dotenv
import google.genai as genai
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_jsons")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore_chroma")

os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# GEMINI SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró GOOGLE_API_KEY en el entorno o .env")
client = genai.Client(api_key=API_KEY)


# FUNCIONES AUXILIARES
def get_gemini_embedding(text: str, retries=5, delay=5):
    """Obtiene embedding de Gemini para un texto con reintentos"""
    for attempt in range(retries):
        try:
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return resp.embeddings[0].values
        except Exception as e:
            print(f"[ERROR] Intento {attempt+1}/{retries} fallido: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

def clean_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name)

def clean_metadata(meta: dict):
    """Convierte valores None a string vacío y fuerza tipos válidos"""
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, (int, float, bool, str)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)  # por si es lista, dict, etc.
    return cleaned


# CARGAR CHUNKS ENRIQUECIDOS
docs = []
metadatas = []

json_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".json")]
for file in json_files:
    path = os.path.join(PROCESSED_DIR, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for chunk in data.get("chunks_enriquecidos", []):
        if not isinstance(chunk, dict):
            continue
        text = (chunk.get("accion_recomendada") or chunk.get("tema") or "").strip()
        if not text:
            continue  # ignorar chunks vacíos
        metadata = clean_metadata({
            "fuente": data.get("fuente", ""),
            "rol_afectado": chunk.get("rol_afectado", ""),
            "riesgo": chunk.get("riesgo", ""),
            "participante": chunk.get("triplete", ["", "", ""])[0],
            "amenaza": chunk.get("triplete", ["", "", ""])[1],
            "mitigacion": chunk.get("triplete", ["", "", ""])[2],
            "tipo_documento": data.get("metadata_original", {}).get("tipo", ""),
            "fecha": data.get("metadata_original", {}).get("fecha", "")
        })
        docs.append(Document(page_content=text, metadata=metadata))

print(f"[SETUP] Total chunks válidos: {len(docs)}\n")


# GENERAR EMBEDDINGS POR LOTES
docs = docs[:100]
print(f"[DEBUG] Solo se procesarán {len(docs)} chunks para testeo\n")

texts = [d.page_content for d in docs]
metadatas_list = [d.metadata for d in docs]

embeddings = []
batch_size = 50  # procesar en batches para no saturar
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_emb = [get_gemini_embedding(t) for t in batch_texts]
    embeddings.extend(batch_emb)
    print(f"[EMBEDDINGS] Procesados {i+len(batch_texts)}/{len(texts)} chunks")


# VECTOR STORE (Chroma)
class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [get_gemini_embedding(t) for t in texts]

    def embed_query(self, text):
        return get_gemini_embedding(text)

vector_store = Chroma(
    embedding_function=GeminiEmbeddings(),  # no se usarán embeddings automáticos
    collection_name="kb_rag",
    persist_directory=VECTOR_DB_DIR
)

vector_store.add_documents(docs)
print("\n[VECTOR STORE] Guardado en:", VECTOR_DB_DIR)


# RETRIEVER SIMPLE
def semantic_query(query_text, top_k=5):
    query_emb = get_gemini_embedding(query_text)
    results = vector_store.similarity_search_by_vector(query_emb, k=top_k)
    return results