import os
import json
import re
from dotenv import load_dotenv
import google.genai as genai

# DIRECTORIOS
BASE_DIR = os.path.dirname(__file__)
IN_DIR = os.path.join(BASE_DIR, "extracted_texts_jsons")
OUT_DIR = os.path.join(BASE_DIR, "processed_jsons")

os.makedirs(OUT_DIR, exist_ok=True)


# GEMINI AI
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("[GEMINI AI CONFIG] No se encontró GOOGLE_API_KEY en el entorno o .env")

client = genai.Client(api_key=API_KEY)
print("[GEMINI AI CONFIG] PASS: Gemini configurado correctamente\n")


# FUNCIONES AUXILIARES
def clean_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name)

def safe_parse_json(text: str):
    """
    Intenta limpiar y parsear un JSON devuelto por Gemini.
    """
    try:
        clean_text = re.sub(r"^```json|```$", "", text.strip())
        return json.loads(clean_text)
    except Exception as e:
        print(f"[GEMINI AI] ERROR al parsear JSON: {e}")
        return {
            "tema": "",
            "rol_afectado": "",
            "accion_recomendada": "",
            "riesgo": "",
            "tipo_vulnerabilidad": "",
            "nivel_impacto": "",
            "triplete": ["", "", ""]
        }

def enrich_chunk_with_gemini(chunk_text: str, source_file: str):
    """
    Envía un fragmento al LLM y devuelve el JSON semántico enriquecido.
    """
    prompt = f"""
    Analiza el siguiente texto sobre ciberhigiene y devuelve SOLO un JSON con estrictamente estos campos:

    {{
      "tema": "",
      "rol_afectado": "",
      "accion_recomendada": "",
      "riesgo": "",
      "tipo_vulnerabilidad": "",
      "nivel_impacto": "",
      "triplete": ["actor", "accion", "control"]
    }}

    Texto:
    {chunk_text[:3000]}

    Devuelve solo el JSON, sin explicaciones ni texto adicional.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt]
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        enriched = safe_parse_json(text)
        return enriched
    except Exception as e:
        print(f"[GEMINI AI] ERROR enriqueciendo chunk de {source_file}: {e}")
        return {
            "tema": "",
            "rol_afectado": "",
            "accion_recomendada": "",
            "riesgo": "",
            "tipo_vulnerabilidad": "",
            "nivel_impacto": "",
            "triplete": ["", "", ""]
        }


# PROCESAR TODOS LOS JSONS
json_files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".json")]
print(f"[SETUP] Archivos JSON detectados para enriquecimiento:\n{json_files}\n")

for file in json_files:
    path = os.path.join(IN_DIR, file)
    print(f"\n[PROCESS] Procesando: {file}\n")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("texto", [])
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        print(f" - Enriqueciendo chunk {i+1}/{len(chunks)} de {file}")
        enriched = enrich_chunk_with_gemini(chunk, file)
        enriched_chunks.append(enriched)

    output_data = {
        "fuente": file,
        "metadata_original": data.get("metadata", {}),
        "chunks_enriquecidos": enriched_chunks
    }

    base_name = clean_filename(file)
    out_path = os.path.join(OUT_DIR, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)

    print(f"[OUTPUT] Enriquecimiento guardado en: {out_path}\n")
