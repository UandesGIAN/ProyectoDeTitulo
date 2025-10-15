import os
import json
from langchain.document_loaders import PyPDFLoader, UnstructuredHTMLLoader

# Carpetas base
BASE_DIR = os.path.join(os.path.dirname(__file__), "./docs")
PDF_DIR = os.path.join(BASE_DIR, "pdf")
HTML_DIR = os.path.join(BASE_DIR, "html")
OUT_DIR = os.path.join(os.path.dirname(__file__), "./extracted_texts_jsons")

os.makedirs(OUT_DIR, exist_ok=True)

def save_doc_json(texts, metadata, base_name):
    file_path = os.path.join(OUT_DIR, f"{base_name}.json")
    out_data = {
        "texto": texts,  # Lista de textos
        "metadata": metadata
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Guardado: {file_path}")

def clean_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name)

# Procesar PDFs
for file in os.listdir(PDF_DIR):
    if file.lower().endswith(".pdf"):
        path = os.path.join(PDF_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        texts = [d.page_content for d in docs]
        metadata = {"fuente": file, "tipo": "pdf"}
        base_name = clean_filename(file)
        save_doc_json(texts, metadata, base_name)

# Procesar HTML
for file in os.listdir(HTML_DIR):
    if file.lower().endswith(".html"):
        path = os.path.join(HTML_DIR, file)
        loader = UnstructuredHTMLLoader(path)
        docs = loader.load()
        texts = [d.page_content for d in docs]
        metadata = {"fuente": file, "tipo": "html"}
        base_name = clean_filename(file)
        save_doc_json(texts, metadata, base_name)
