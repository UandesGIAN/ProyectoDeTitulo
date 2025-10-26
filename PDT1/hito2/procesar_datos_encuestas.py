import pandas as pd
import os
import json

# RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADO_JSON = os.path.join(BASE_DIR, "datos_encuesta/muestra_encuesta_procesado.json")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "analisis_encuesta")
OUTPUT_JSON = os.path.join(OUTPUT_JSON_DIR, "resumen_participantes.json")

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# Datos sobre el IMECH
dimensiones = [
    {"nombre": "Dispositivos y almacenamiento de información", "codigo": "DAI", "cantidad": 10},
    {"nombre": "Transmisión de la información", "codigo": "TRI", "cantidad": 8},
    {"nombre": "Comportamiento en las redes sociales", "codigo": "CRS", "cantidad": 7},
    {"nombre": "Autenticación y uso de credenciales", "codigo": "AUC", "cantidad": 9},
    {"nombre": "Mensajería y correo electrónico", "codigo": "MCE", "cantidad": 8}
]

respuestas_likert = {
    1: "No me describe para nada",
    2: "Me describe ligeramente",
    3: "Me describe moderadamente",
    4: "Me describe bastante",
    5: "Me describe completamente"
}

informacion_personal = {
    "Rango etario": ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    "Género": ["Femenino", "Masculino", "Otro", "Prefiero no informarlo"],
    "Área en el Servicio de Salud": [
        "Dirección",
        "Subdirección de Servicios Clínicos",
        "Subdirección de Operaciones",
        "Subdirección de Administración y Finanzas",
        "Subdirección de Personas",
        "Comités, Consejos, Unidades"
    ],
    "Nivel de Responsabilidad": [
        "Dirección o Subdirección",
        "Jefatura",
        "Colaborador(a) individual",
        "Externo"
    ],
    "Capacitación en seguridad/ciberseguridad": ["Sí", "No"]
}

# Qué tanto riesgo indica cada pregunta. 3: Bastante, 4: Mucho, 5: Máximo.
pesos_items = {
    "i01": 5, "i02": 4, "i03": 3, "i04": 4, "i05": 4,
    "i06": 5, "i07": 3, "i08": 4, "i09": 5, "i10": 5,
    "i11": 5, "i12": 4, "i13": 5, "i14": 3, "i15": 4,
    "i16": 4, "i17": 4, "i18": 4, "i19": 5, "i20": 3,
    "i21": 4, "i22": 4, "i23": 3, "i24": 3, "i25": 4,
    "i26": 5, "i27": 4, "i28": 3, "i29": 5, "i30": 3,
    "i31": 5, "i32": 4, "i33": 4, "i34": 3, "i35": 4,
    "i36": 4, "i37": 3, "i38": 3, "i39": 4, "i40": 4,
    "i41": 4, "i42": 4
}

# Cargar JSON procesado
with open(RESULTADO_JSON, "r", encoding="utf-8") as f:
    participantes = json.load(f)


# OBTENER DATOS GLOBALES PROMEDIO
dimension_base = {}
item_to_dimension = {}
item_to_indica = {}

for p in participantes:
    for it in p.get("items", []):
        code = it.get("Item")
        if not code:
            continue
        if code not in item_to_dimension:
            item_to_dimension[code] = it.get("Dimension", "") or ""
            item_to_indica[code] = it.get("Indica_riesgo", "") or ""

# Calcular puntaje promedio total por dimensión
for item_code, dim in item_to_dimension.items():
    peso = pesos_items.get(item_code, 1)
    indica = item_to_indica.get(item_code, "")
    if indica == "Sí":
        base_item = 3 * peso * 2
    else:
        base_item = 3 * peso
    dimension_base[dim] = dimension_base.get(dim, 0) + base_item

base_global_total = sum(dimension_base.values())
umbral_critico_pct = 0.65
umbral_por_dimension = {dim: val * umbral_critico_pct for dim, val in dimension_base.items()}

print("DATOS PROMEDIO...")
print("Puntaje_promedio_total:", round(base_global_total, 2))
print("Puntaje_promedio_por_dimension:", {k: round(v, 2) for k, v in dimension_base.items()})
print("Umbral_critico_por_dimension", {k: round(v, 2) for k, v in umbral_por_dimension.items()})
print()


# ANÁLISIS POR PARTICIPANTE
resumen_participantes = []

for participante in participantes:
    items_analisis = []
    dimension_scores = {}

    for it in participante.get("items", []):
        item_code = it.get("Item")
        try:
            respuesta = float(it.get("Respuesta")) if it.get("Respuesta") not in (None, "") else None
        except Exception:
            respuesta = None

        if respuesta is None or (isinstance(respuesta, float) and pd.isna(respuesta)):
            riesgo_ponderado = None
        else:
            peso = pesos_items.get(item_code, 1)
            indica = (it.get("Indica_riesgo") == "Sí")
            if indica:
                riesgo_ponderado = respuesta * peso * 2 # Si es un ítem que indica riesgo vale x2
            else:
                riesgo_ponderado = (6 - respuesta) * peso

            dim = it.get("Dimension") or ""
            if dim:
                dimension_scores[dim] = dimension_scores.get(dim, 0) + riesgo_ponderado

        items_analisis.append({
            "Item": item_code,
            "Dimension": it.get("Dimension", ""),
            "Indica_riesgo": it.get("Indica_riesgo", ""),
            "Enunciado": it.get("Enunciado", "") or "",
            "Que_mide": it.get("Que_mide", "") or "",
            "Respuesta": respuesta,
            "Riesgo_ponderado": riesgo_ponderado
        })

    # Calcular porcentajes
    dimension_percentual = {}
    dimensiones_criticas = []
    for dim, base_val in dimension_base.items():
        real_val = dimension_scores.get(dim, 0)
        pct = (real_val / base_val * 100) if base_val else 0.0
        dimension_percentual[dim] = pct
        if pct <= umbral_critico_pct * 100:
            dimensiones_criticas.append(dim)

    real_global = sum(dimension_scores.values())
    promedio_relativo_global_pct = (real_global / base_global_total * 100) if base_global_total else 0.0

    # ítems con valores que indican peligro
    items_validos = [i for i in items_analisis if i["Respuesta"] is not None]

    items_criticos = []
    for i in items_validos:
        resp = i["Respuesta"]
        indica = i["Indica_riesgo"] == "Sí"

        # Criterios: bajo (1 o 2) en positivos, alto (5) en negativos
        if (indica and resp >= 5) or (not indica and resp <= 2):
            items_criticos.append(i)

    # Ordenar los ítems críticos por riesgo ponderado (mayor = más riesgoso)
    items_criticos_ordenados = sorted(
        items_criticos,
        key=lambda x: (x["Riesgo_ponderado"] if x["Riesgo_ponderado"] is not None else 0),
        reverse=True
    )

    # Si hay más de 5 ítems, tomar el valor del 5
    if len(items_criticos_ordenados) >= 5:
        limite_riesgo = items_criticos_ordenados[4]["Riesgo_ponderado"]
        items_criticos_top5 = [
            i for i in items_criticos_ordenados
            if i["Riesgo_ponderado"] is not None and i["Riesgo_ponderado"] >= limite_riesgo
        ]
    else:
        # Si hay menos de 5, tomar todos
        items_criticos_top5 = items_criticos_ordenados

    salida = {
        "Participante": participante.get("Participante"),
        "Datos_personales": participante.get("informacion_personal", {}),
        "Análisis_datos": {
            "Puntaje_total": real_global,
            "Percentil_relativo_al_promedio": round(promedio_relativo_global_pct, 2),
            "Puntaje_por_dimension": {k: round(dimension_scores.get(k, 0), 2) for k in dimension_base.keys()},
            "Percentil_relativo_al_promedio_por_dimension": {k: round(v, 2) for k, v in dimension_percentual.items()},
            "Dimensiones_criticas": dimensiones_criticas,
            "Items_criticos": items_criticos,
            "Items_criticos_por_puntaje": items_criticos_top5
        }
    }

    json_individual = os.path.join(OUTPUT_JSON_DIR, f"participante_{participante.get('Participante')}.json")
    with open(json_individual, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    resumen_participantes.append(salida)

# Guardar JSON agregado
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(resumen_participantes, f, indent=2, ensure_ascii=False)

print(f"JSON individuales guardados en: {OUTPUT_JSON_DIR}")
print(f"JSON agregado guardado en: {OUTPUT_JSON}")