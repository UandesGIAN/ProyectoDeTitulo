import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
import google.genai as genai
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(BASE_DIR, "analisis_encuesta/resumen_participantes.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "recomendaciones")
OUTPUT_JSON= os.path.join(OUTPUT_DIR, "resumen_participantes.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

dimensiones = {
    "DAI": "Dispositivos y almacenamiento de información",
    "TRI": "Transmisión de la información",
    "CRS": "Comportamiento en las redes sociales",
    "AUC": "Autenticación y uso de credenciales",
    "MCE": "Mensajería y correo electrónico"
}

VECTOR_DB_DIR = os.path.join(BASE_DIR, "KB_RAG/vectorstore_chroma")

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

                # Nivel: coincidencia exacta dentro de la lista
                elif key == "nivel":
                    filter_niveles = [normalize_str(f) for f in val] if isinstance(val, list) else [normalize_str(val)]
                    if normalize_str(doc_val) not in filter_niveles:
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
        print(f"[DEBUG] Total documentos encontrados después de filtros: {len(filtered)}")
        filtered.sort(key=lambda x: len(x.page_content), reverse=True)
        return filtered[:top_k]
    return results[:top_k]


# Función para obtener recomendaciones únicas
def get_unique_recommendations(docs, existing=set(), max_rec=5):
    recs = []
    for d in docs:
        r_text = d.metadata.get("recomendacion", "")
        if r_text not in existing:
            rec = {
                "recomendacion": r_text,
                "fuente": d.metadata.get("fuente", ""),
                "nivel": d.metadata.get("nivel", ""),
                "dimension": d.metadata.get("dimension", ""),
                "tags": d.metadata.get("tags", []),
                "esfuerzo": d.metadata.get("esfuerzo", d.metadata.get("riesgo", "")),
                "impacto": d.metadata.get("impacto", d.metadata.get("nivel", "")),
                "texto_original": d.metadata.get("original_text", ""),
                "fecha": d.metadata.get("fecha", "")
            }
            recs.append(rec)
            existing.add(r_text)
        if len(recs) >= max_rec:
            break
    return recs, existing

def niveles_alternativos(nivel):
    nivel = nivel.lower()
    if nivel == "básico":
        return ["promedio", "técnico", "administrador"]
    elif nivel == "promedio":
        return ["técnico", "básico", "administrador"]
    elif nivel == "técnico":
        return ["promedio", "básico", "administrador"]
    elif nivel == "administrador":
        return ["técnico", "promedio", "básico"]
    return []


# Cargar JSON de participantes
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
participantes = data if isinstance(data, list) else [data]


# Procesar cada participante
recomendaciones_final = []
for participante in participantes:
    datos_personales = participante.get("Datos_personales", {})
    nivel_usuario = datos_personales.get("Nivel_expertis_ciberseguridad", "promedio").lower()
    analisis = participante.get("Análisis_datos", {})
    dim_criticas = analisis.get("Dimensiones_criticas", [])
    items_criticos = analisis.get("Items_criticos_personales", []) + analisis.get("Items_criticos_debajo_percentil35", [])

    recs_por_dimension = []
    recs_por_item = []

    # Recomendaciones por dimensión crítica
    for dim in dim_criticas:
        seen_recs_dim = set()
        niveles = [nivel_usuario] + niveles_alternativos(nivel_usuario)
        recs_dim = []
        for nivel in niveles:
            docs = semantic_query(
                query_text=dimensiones[dim],
                top_k=10,
                filters={"dimension": dim, "nivel": nivel}
            )
            recs, seen_recs_dim = get_unique_recommendations(docs, existing=seen_recs_dim, max_rec=10)
            recs_dim.extend(recs)
            if len(recs_dim) >= 10:
                break
        # Actualizamos cada recomendación con metadata extendida
        for r in recs_dim:
            r.update({
                "tipo": "dimension",
                "dimension_asociada": dim,
                "esfuerzo": r.get("esfuerzo", ""),
                "impacto": r.get("impacto", ""),
                "texto_original": r.get("original_text", ""),
                "fecha": r.get("fecha", "")
            })
        recs_por_dimension.extend(recs_dim)

    # Recomendaciones por ítem crítico
    seen_items = set()
    for i in items_criticos:
        item_code = i.get("Item")
        if item_code in seen_items:
            continue
        seen_items.add(item_code)

        que_mide = i.get("Que_mide", "")
        dimension = i.get("Dimension", "")
        seen_recs_item = set()
        
        niveles_prueba = [nivel_usuario] + niveles_alternativos(nivel_usuario)
        docs = []
        for nivel in niveles_prueba:
            docs = semantic_query(
                query_text=que_mide,
                top_k=5,
                filters={"dimension": dimension, "nivel": [nivel]}
            )
            if len(docs) > 0:
                break
        
        recs, _ = get_unique_recommendations(docs, existing=seen_recs_item, max_rec=5)
        # Cada item no lleva 'dimension_asociada'
        for r in recs:
            r.update({
                "tipo": "item",
                "item_asociado": item_code,
                "esfuerzo": r.get("esfuerzo", ""),
                "impacto": r.get("impacto", ""),
                "texto_original": r.get("original_text", ""),
                "fecha": r.get("fecha", "")
            })
        recs_por_item.extend(recs)

    recomendaciones_final.append({
        "Participante": participante.get("Participante"),
        "Datos_personales": datos_personales,
        "Recomendaciones": {
            "dimension": recs_por_dimension,
            "item": recs_por_item
        }
    })


# Guardar JSON de salida
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(recomendaciones_final, f, indent=2, ensure_ascii=False)

print(f"Recomendaciones generadas en: {OUTPUT_JSON}")