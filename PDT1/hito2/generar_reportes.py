import os
import json
from dotenv import load_dotenv
import google.genai as genai
from markdown2 import markdown
from weasyprint import HTML

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(__file__)
INPUT_JSON = os.path.join(BASE_DIR, "recomendaciones/resumen_participantes.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print()

# GEMINI AI SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("[GEMINI SETUP] ERROR: No se encontró GOOGLE_API_KEY en el entorno o .env\n")

client = genai.Client(api_key=API_KEY)
print("[GEMINI SETUP] PASS: Gemini configurado correctamente.\n")

def safe_parse_json(text):
    """Intenta parsear JSON de manera segura"""
    try:
        return json.loads(text)
    except:
        return None

def get_next_available_filename(base_path, base_name, ext):
    """Evita sobrescribir archivos, agrega _i si ya existe"""
    candidate = f"{base_name}.{ext}"
    i = 1
    while os.path.exists(os.path.join(base_path, candidate)):
        candidate = f"{base_name}_{i}.{ext}"
        i += 1
    return os.path.join(base_path, candidate)

def generate_report_md(participant_data):
    """
    Genera el contenido MD del reporte usando Gemini AI
    """
    prompt = f"""
    Eres un asistente experto en ciberseguridad que redacta informes en formato Markdown estilo receta médica, breve, en lenguaje chileno, amigable, no culpabilizador. 
    Usa el siguiente contexto JSON sobre el participante:

    {json.dumps(participant_data, indent=2, ensure_ascii=False)}

    Redacta un informe de recomendación completo en formato markdown, debe indicar las dimensiones más debiles del usuario, con su descripción completa e indicar máximo 10 recomendaciones en un punteo
    que sean claramente distintas, para que el usuario que lea el reporte sepa qué hacer para mejorar su ciberhigiene. No debe exceder las 600 palabras. Y debe incluir al final un apartado de fuentes, con citas textuales del texto original de cada recomendación dada.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt]
        )
        return response.text.strip()
    except Exception as e:
        print(f"[GEMINI] ERROR generando reporte: {e}")
        return ""

def save_md_and_pdf(md_content, base_name):
    """Guarda MD y PDF en la carpeta OUTPUT_DIR"""
    md_file = get_next_available_filename(OUTPUT_DIR, base_name, "md")
    pdf_file = get_next_available_filename(OUTPUT_DIR, base_name, "pdf")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] Reporte MD generado: {md_file}")

    # Convertir MD a HTML y luego a PDF
    html_content = markdown(md_content)
    HTML(string=html_content).write_pdf(pdf_file)
    print(f"[INFO] Reporte PDF generado: {pdf_file}")



with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Tomar el primer participante
participant = data[0] if isinstance(data, list) else data

base_name = participant.get("Participante", "reporte_participante").replace(" ", "_")

# Generar reporte
md_report = generate_report_md(participant)

# Guardar MD y PDF
save_md_and_pdf(md_report, base_name)