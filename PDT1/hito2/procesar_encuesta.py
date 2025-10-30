import pandas as pd
import os
import json

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(BASE_DIR, "datos_encuesta/items_imech.csv")
RESPUESTAS_CSV = os.path.join(BASE_DIR, "datos_encuesta/muestra_encuesta.csv")
RESULTADO_JSON = os.path.join(BASE_DIR, "datos_encuesta/muestra_encuesta_procesado.json")

# Lee items del IMECH
items_df = pd.read_csv(ITEMS_CSV)

# Adapta a diccionario
items_dict = {}
for _, row in items_df.iterrows():
    items_dict[row["Items"]] = {
        "item": row["Items"],
        "dimension": row["Dimensión"],
        "indica_riesgo": row["Comportamiento de Riesgo"],
        "enunciado": row["Enunciado"],
        "que_mide": row["Aspecto de la variable latente medido"] if pd.notna(row["Aspecto de la variable latente medido"]) else ""
    }

print(f"Items cargados: {len(items_dict)}")

# Lee respuestas
respuestas_df = pd.read_csv(RESPUESTAS_CSV)
respuestas_por_participante = []

for idx, row in respuestas_df.iterrows():
    participante = {
        "Participante": idx + 1,
        "informacion_personal": {
            "Sexo": row["p01"],
            "Rango_etario": row["p02"],
            "Area_servicio_salud": row["p03"],
            "Responsabilidad": row["p11"],
            "Posee_capacitacion": row["p12"]
        },
        "items": []
    }
    
    # Recorre todos los items
    for col in respuestas_df.columns:
        if col.startswith("i"):
            respuesta_val = row[col]
            item_info = items_dict.get(col, {})
            participante["items"].append({
                "Item": col,
                "Respuesta": respuesta_val,
                "Nombre_item": item_info.get("item"),
                "Dimension": item_info.get("dimension"),
                "Indica_riesgo": item_info.get("indica_riesgo"),
                "Enunciado": item_info.get("enunciado"),
                "Que_mide": item_info.get("que_mide")
            })
    
    respuestas_por_participante.append(participante)

# Guarda en JSON
with open(RESULTADO_JSON, "w", encoding="utf-8") as f:
    json.dump(respuestas_por_participante, f, ensure_ascii=False, indent=2)

print(f"JSON procesado guardado en: {RESULTADO_JSON}")