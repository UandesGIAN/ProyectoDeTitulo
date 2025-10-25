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
    Limpia y parsea el JSON que devuelve Gemini.
    Maneja casos donde hay múltiples objetos o texto adicional.
    """
    try:
        clean_text = re.sub(r"^```json|```$", "", text.strip())
        # Buscar primer '{' y último '}'
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        if start == -1 or end == -1:
            return {}
        json_text = clean_text[start:end]
        return json.loads(json_text)
    except Exception as e:
        print(f"[GEMINI AI] ERROR al parsear JSON: {e}")
        return {}

def normalize_text(s: str) -> str:
    s = re.sub(r"[^a-z0-9áéíóúñ ]", "", s.lower())
    return s.strip()

def remove_duplicates(chunks):
    """
    Elimina chunks con acciones o temas duplicados.
    Prefiere los más largos/completos.
    """
    unique = []
    seen = set()

    for ch in chunks:
        if not ch.get("recomendacion") or not ch.get("dimension_IMECH"):
            continue
        key = normalize_text(ch["recomendacion"])
        if key not in seen:
            seen.add(key)
            unique.append(ch)
        else:
            existing = next((u for u in unique if normalize_text(u["recomendacion"]) == key), None)
            if existing and len(json.dumps(ch)) > len(json.dumps(existing)):
                unique.remove(existing)
                unique.append(ch)
    return unique

def is_valid_chunk(ch):
    """
    Valida que el JSON generado por Gemini use categorías correctas.
    """
    if not ch or not isinstance(ch, dict):
        return False

    valid_roles = {"enfermería", "médico", "TI", "administrativo"}
    valid_dim = {"DAI", "TRI", "CRS", "AUC", "MCE"}
    valid_niveles = {"básico", "promedio", "elevado", "técnico", "administrador"}
    valid_esfuerzo = {"bajo", "medio", "alto"}
    valid_impacto = {"bajo", "medio", "alto"}

    if ch.get("dimension_IMECH") not in valid_dim:
        return False
    if "nivel" in ch and ch["nivel"] and ch["nivel"] not in valid_niveles:
        return False
    if "rol" in ch and ch["rol"] and ch["rol"] not in valid_roles:
        return False
    if "esfuerzo" in ch and ch["esfuerzo"] and ch["esfuerzo"] not in valid_esfuerzo:
        return False
    if "impacto" in ch and ch["impacto"] and ch["impacto"] not in valid_impacto:
        return False

    return True

def flatten_chunks(chunks):
    """
    Asegura que todos los elementos sean diccionarios.
    Si un chunk es lista, extrae sus dicts individuales.
    """
    flat = []
    for ch in chunks:
        if isinstance(ch, dict):
            flat.append(ch)
        elif isinstance(ch, list):
            flat.extend([c for c in ch if isinstance(c, dict)])
    return flat

def enrich_chunk_with_gemini(chunk_text: str, source_file: str):
    """
    Envía un fragmento al LLM y devuelve el JSON semántico enriquecido con rol, dimensión IMECH, esfuerzo, impacto, etc.
    """
    imech_context = """
    MODELO IMECH (Instrumento de Medición de Ciberhigiene)
    -----------------------------------------------------
    El IMECH evalúa prácticas de ciberhigiene en instituciones críticas (como hospitales, puertos, o servicios públicos).
    Se compone de 5 dimensiones basadas en el modelo de Vishwanath et al. (2020):

    1. DAI - Higiene de Dispositivos y Almacenamiento de Información:
       Prácticas para proteger la integridad física y digital de dispositivos personales o laborales.
       Incluye: actualizaciones, respaldos, bloqueo de pantalla, separación de cuentas, instalación segura de apps.

    2. TRI - Higiene de la Transmisión de Información:
       Acciones para reducir riesgos al compartir o transferir información sensible.
       Incluye: evitar Wi-Fi público, usar canales seguros, no enviar contraseñas por correo o chat.

    3. CRS - Higiene del Comportamiento en Redes Sociales:
       Conductas seguras en redes sociales y plataformas públicas.
       Incluye: no compartir información sensible, configurar privacidad, verificar fuentes de contenido.

    4. AUC - Higiene de la Autenticación y Uso de Credenciales:
       Gestión segura de contraseñas y autenticaciones.
       Incluye: uso de contraseñas únicas y complejas, MFA, no reutilizar claves, cambio periódico.

    5. MCE - Higiene del Correo Electrónico y la Mensajería:
       Prácticas seguras en la comunicación digital.
       Incluye: detectar correos sospechosos, no abrir adjuntos dudosos, verificar remitentes y solicitudes.
    -----------------------------------------------------
    """

    prompt = f"""
    {imech_context}

    Analiza el siguiente texto sobre CIBERHIGIENE institucional y devuelve SOLO un JSON con los siguientes campos:

    {{
      "original_text": "",       # Cita textual del fragmento procesado (máx. 300 caracteres)
      "tema": "",                # Tema general (ej. contraseñas seguras, actualización de software, phishing)
      "nivel": "",               # Nivel de conocimiento que requiere el usurio, debe ser uno de: ["básico", "promedio", "elevado", "técnico", "administrador"]
      "dimension_IMECH": "",     # Una de: ["DAI", "TRI", "CRS", "AUC", "MCE"]
      "recomendacion": "",       # Recomendación clara, precisa y aplicable en distintos contextos, entre más completa y precisa mejor.
      "riesgo": "",              # Qué riesgo o problema se mitiga
      "tipo_vulnerabilidad": "", # Breve descripción (phishing, credenciales débiles, etc.)
      "esfuerzo": "",            # Uno de: ["bajo", "medio", "alto"]
      "impacto": "",             # Uno de: ["bajo", "medio", "alto"]
      "tags": []                 # Palabras clave: ["passwords", "MFA", "phishing", "actualizaciones", "backups", "RDP", "BYOD", "EHR", "correo"]
    }}

    Reglas:
    - Usa SOLO las categorías válidas anteriores.
    - Si el texto no se relaciona con ciberhigiene o ninguna dimensión IMECH, devuelve un JSON vacío ({{}}).
    - La acción recomendada debe ser práctica, clara y aplicable (no técnica).
    - No repitas el texto original completo: resume la cita en "original_text" (300 caracteres máx).

    Texto a analizar:
    {chunk_text[:2500]}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt]
        )
        text = response.text.strip()
        enriched = safe_parse_json(text)
        return enriched if enriched else {}
    except Exception as e:
        print(f"[GEMINI AI] ERROR enriqueciendo chunk de {source_file}: {e}")
        return {}



# PROCESAR JSONS
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
        if enriched:
            enriched_chunks.append(enriched)

    # Filtrar duplicados y vacíos
    enriched_chunks = flatten_chunks(enriched_chunks)
    enriched_chunks = remove_duplicates(enriched_chunks)

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