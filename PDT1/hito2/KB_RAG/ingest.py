import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
import google.genai as genai


# VARIABLES GLOBALES
BASE_DIR = os.path.join(os.path.dirname(__file__), "./docs")
PDF_DIR = os.path.join(BASE_DIR, "pdf")
HTML_DIR = os.path.join(BASE_DIR, "html")
OUT_DIR = os.path.join(os.path.dirname(__file__), "./extracted_texts_jsons")

os.makedirs(OUT_DIR, exist_ok=True)

print()

# GEMINI AI SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("[GEMINI SETUP] ERROR: No se encontró GOOGLE_API_KEY en el entorno o .env\n")

client = genai.Client(api_key=API_KEY)
print("[GEMINI SETUP] PASS: Gemini configurado correctamente.\n")

# FUNCIONES AUXILIARES
def clean_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name)

def save_doc_json(chunks, metadata, base_name):
    file_path = os.path.join(OUT_DIR, f"{base_name}.json")
    out_data = {
        "texto": chunks,
        "metadata": metadata
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"[JSON OUTPUT] Guardado: {file_path} (chunks: {len(chunks)})\n")

def extract_date_from_text(text):
    """
    Busca fechas en formato DD Month YYYY o YYYY-MM-DD dentro del texto.
    Devuelve la primera encontrada en formato DD-MM-AAAA.
    """
    # Patrones comunes
    patterns = [
        r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(\d{4})-(\d{2})-(\d{2})'
    ]
    months = {
        'January':'01','February':'02','March':'03','April':'04','May':'05','June':'06',
        'July':'07','August':'08','September':'09','October':'10','November':'11','December':'12'
    }

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if len(m.groups()) == 3:
                if pat.startswith(r'(\d{1,2})'):
                    # Formato: 17 December 2018
                    d, month, y = m.groups()
                    return f"{int(d):02d}-{months[month.capitalize()]}-{y}"
                else:
                    # Formato: YYYY-MM-DD
                    y, mo, d = m.groups()
                    return f"{d}-{mo}-{y}"
    # Si no se encuentra nada, pone el inicio de año actual
    return datetime(datetime.today().year,1,1).strftime("%d-%m-%Y")

def safe_parse_json(text: str):
    try:
        clean_text = re.sub(r"^```json|```$", "", text.strip())
        parsed = json.loads(clean_text)
        # Normalizar valores vacíos
        for key in ["fecha", "entidad_emisora", "titulo"]:
            if key not in parsed or parsed[key] in [None, "", "null", "desconocida"]:
                parsed[key] = None
        return parsed
    except:
        return {"fecha": None, "entidad_emisora": None, "titulo": None}


# METADATA
def get_pdf_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    info = reader.metadata

    fecha = "desconocida"
    autor = "desconocida"
    titulo = None

    # Metadata del PDF
    if info:
        if info.get('/CreationDate'):
            raw = info['/CreationDate']
            match = re.search(r'D:(\d{4})(\d{2})(\d{2})', raw)
            if match:
                fecha = f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
        if info.get('/Author'):
            autor = info['/Author']
        if info.get('/Title'):
            titulo = info['/Title']

    # Si no hay título, usar nombre del archivo
    if not titulo or titulo.strip() == "":
        titulo = pdf_path.split(os.sep)[-1].replace(".pdf", "")

    # Convertir pages a lista para poder manipular slices
    pages = list(reader.pages)

    # Extraer texto completo
    full_text = "\n".join([page.extract_text() or "" for page in pages])

    # Si fecha sigue desconocida, intentar extraer del texto
    if fecha in ["desconocida", None, "null"]:
        fecha = extract_date_from_text(full_text)

    # Si autor o fecha aún son desconocidos, usar Gemini
    if autor in ["desconocida", None, "null"] or fecha in ["desconocida", None, "null"]:
        # Tomar primeros 3 y últimos 3 fragmentos del PDF
        first_pages = [pages[i].extract_text() or "" for i in range(min(3, len(pages)))]
        last_pages = [pages[i].extract_text() or "" for i in range(max(len(pages)-3, 0), len(pages))]
        snippet = "\n".join(first_pages + last_pages)

        gemini_meta = infer_pdf_metadata_with_gemini(snippet, pdf_path)

        # Validar y actualizar fecha y autor si Gemini devuelve datos válidos
        fecha_candidato = gemini_meta.get("fecha", "").strip()
        if re.match(r"^\d{2}-\d{2}-\d{4}$", fecha_candidato):
            fecha = fecha_candidato

        autor_candidato = gemini_meta.get("entidad_emisora", "").strip()
        if autor_candidato:
            autor = autor_candidato

    return fecha, autor, titulo

def infer_pdf_metadata_with_gemini(pdf_text, pdf_path):
    """
    Usa Gemini para inferir metadata de un PDF (fecha, autor, título)
    pdf_text: texto de los primeros y últimos fragmentos del PDF
    """
    prompt = f"""
    Analiza el siguiente texto extraído de un PDF y devuelve SOLO un JSON con estos campos:

    {{
      "fecha": "",             # Formato DD-MM-AAAA
      "entidad_emisora": "",   # Autor o institución responsable del PDF
      "titulo": ""             # Título del PDF
    }}

    Reglas:
    1. Busca primero cualquier fecha exacta mencionada en el texto (ejemplo: "Published 17 December 2018").
    2. Si hay varias fechas, usa la más probable de publicación original.
    3. Si no encuentras ninguna fecha, inventa una fecha coherente, nunca la dejes vacía.
    4. Devuelve SIEMPRE valores válidos.
    5. Devuelve solo JSON limpio, sin explicaciones.

    Texto del PDF (solo fragmentos relevantes, máximo 6000 caracteres):
    {pdf_text[:6000]}

    PDF path: {pdf_path}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt]
        )
        meta = safe_parse_json(response.text)

        # Validar fecha
        if not meta.get("fecha") or not re.match(r"^\d{2}-\d{2}-\d{4}$", meta["fecha"]):
            meta["fecha"] = datetime(datetime.today().year, 1, 1).strftime("%d-%m-%Y")

        # Validar autor
        if not meta.get("entidad_emisora"):
            meta["entidad_emisora"] = "Entidad desconocida"

        # Validar título
        if not meta.get("titulo"):
            meta["titulo"] = pdf_path.split(os.sep)[-1].replace(".pdf", "")

        return meta
    except Exception as e:
        print(f"[GEMINI AI] Error infiriendo metadata del PDF: {e}")
        return {
            "fecha": datetime(datetime.today().year, 1, 1).strftime("%d-%m-%Y"),
            "entidad_emisora": "Entidad desconocida",
            "titulo": pdf_path.split(os.sep)[-1].replace(".pdf", "")
        }

def get_html_metadata(html_path):
    """Obtiene fecha, autor y URL del HTML; usa Gemini si hace falta"""
    fecha = "desconocida"
    autor = "desconocida"
    url = None

    with open(html_path, "r", encoding="utf-8") as f:
        text = f.read()
        soup = BeautifulSoup(text, "html.parser")

        # Extraer URL
        match = re.search(r'<!-- saved from url=\(\d+\)(.*?) -->', text)
        if match:
            url = match.group(1).strip()

        # Extraer meta tags
        meta_author = soup.find("meta", attrs={"name": "author"}) or \
                      soup.find("meta", attrs={"property": "author"})
        if meta_author and meta_author.get("content"):
            autor = meta_author["content"]

        meta_date = soup.find("meta", attrs={"name": "date"}) or \
                    soup.find("meta", attrs={"property": "article:published_time"})
        if meta_date and meta_date.get("content"):
            fecha = meta_date["content"]
            # normalizar DD-MM-AAAA
            m = re.search(r'(\d{4})-(\d{2})-(\d{2})', fecha)
            if m:
                fecha = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # Si falta info, usar Gemini
    if fecha == "desconocida" or autor == "desconocida":
        print(f"[HTML METADATA] Usando Gemini para {html_path}")
        gemini_meta = infer_html_metadata_with_gemini(text, url if url else "desconocida")
        if fecha == "desconocida":
            fecha = gemini_meta.get("fecha", "desconocida")
        if autor == "desconocida":
            autor = gemini_meta.get("entidad_emisora", "desconocida")

    return fecha, autor, url

def infer_html_metadata_with_gemini(html_text, url):
    prompt = f"""
    Analiza el siguiente HTML y devuelve SOLO un JSON con estos campos:

    {{
      "fecha": "",             # Formato DD-MM-AAAA
      "entidad_emisora": ""    # Autor o institución responsable de la página
    }}

    Reglas:
    1. Busca primero cualquier fecha exacta mencionada en el HTML (ejemplo: "Published 17 December 2018").
    2. Si hay varias fechas, usa la más probable de publicación original.
    3. Si no encuentras ninguna fecha, puedes inventar una fecha aproximada coherente, nunca la dejes vacía o pongas N/A.
    4. Devuelve SIEMPRE un valor válido en "fecha" en formato DD-MM-AAAA.
    5. Devuelve solo JSON válido, sin explicaciones adicionales.

    HTML:
    {html_text}

    URL: {url}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt]
        )
        meta = safe_parse_json(response.text)

        # Verificar que la fecha cumpla DD-MM-AAAA
        fecha_valida = False
        if meta.get("fecha"):
            if re.match(r"^\d{2}-\d{2}-\d{4}$", meta["fecha"]):
                fecha_valida = True

        if not fecha_valida:
            inicio_anio = datetime(datetime.today().year, 1, 1)
            meta["fecha"] = inicio_anio.strftime("%d-%m-%Y")

        if meta.get("entidad_emisora") in [None, "", "desconocida", "null"]:
            meta["entidad_emisora"] = "desconocida"

        return meta
    except Exception as e:
        print(f"[GEMINI AI] Error infiriendo metadatos: {e}")
        inicio_anio = datetime(datetime.today().year, 1, 1)
        return {
            "fecha": inicio_anio.strftime("%d-%m-%Y"),
            "entidad_emisora": "desconocida"
        }


# SPLITTER DE TEXTO
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)


# PROCESAR PDFS
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
print(f"[SETUP] Archivos PDF detectados:\n {pdf_files}\n")

for file in pdf_files:
    path = os.path.join(PDF_DIR, file)
    print(f"\n[PDF] Procesando PDF: {file}\n")
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f" - Páginas cargadas: {len(docs)}")

    full_text = "\n".join([d.page_content for d in docs])
    idioma = detect(full_text)
    print(f" - Idioma detectado: {idioma}")

    chunks = text_splitter.split_text(full_text)
    print(f" - Número de chunks generados: {len(chunks)}")

    sample_text = " ".join(chunks[:2])

    fecha_doc, autor_doc, titulo_doc = get_pdf_metadata(path)
    metadata = {
        "fuente": titulo_doc,
        "tipo": "pdf",
        "idioma": idioma,
        "fecha": fecha_doc,
        "entidad_emisora": autor_doc,
    }

    base_name = clean_filename(file)
    save_doc_json(chunks, metadata, base_name)


# PROCESAR HTML
html_files = [f for f in os.listdir(HTML_DIR) if f.lower().endswith(".html")]
print(f"\n[SETUP] Archivos HTML detectados:\n {html_files}\n")

for file in html_files:
    path = os.path.join(HTML_DIR, file)
    print(f"\n[HTML] Procesando HTML: {file}\n")
    loader = UnstructuredHTMLLoader(path)
    docs = loader.load()
    print(f" - Fragmentos cargados: {len(docs)}")

    full_text = "\n".join([d.page_content for d in docs])
    idioma = detect(full_text)
    print(f" - Idioma detectado: {idioma}")

    chunks = text_splitter.split_text(full_text)
    print(f" - Número de chunks generados: {len(chunks)}")

    sample_text = " ".join(chunks[:2])

    fecha_doc, autor_doc, url_doc = get_html_metadata(path)
    metadata = {
        "fuente": url_doc,
        "tipo": "html",
        "idioma": idioma,
        "fecha": fecha_doc,
        "entidad_emisora": autor_doc,
    }

    base_name = clean_filename(file)
    save_doc_json(chunks, metadata, base_name)