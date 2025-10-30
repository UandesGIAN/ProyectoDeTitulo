import pandas as pd
import os
import json
import numpy as np

# VARIABLES GLOBALES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(BASE_DIR, "datos_encuesta/muestra_encuesta_procesado.json")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "analisis_encuesta")
OUTPUT_JSON = os.path.join(OUTPUT_JSON_DIR, "resumen_participantes.json")
OUTPUT_GLOBAL = os.path.join(OUTPUT_JSON_DIR, "analisis_global.json")

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


def asignar_nivel_expertis(datos):
    rango = datos.get("Rango_etario", "")
    nivel_resp = datos.get("Responsabilidad", "")
    area = datos.get("Area_servicio_salud", "")
    capacitado = datos.get("Posee_capacitacion", "No") == "Sí"

    mayores_45 = ["46 a 59", "60 a 65", "65+"]

    if capacitado and rango not in mayores_45:
        return "Técnico"
    if rango in ["18 a 25", "26 a 35"] and nivel_resp not in ["Colaborador(a) individual", "Jefatura"]:
        return "Promedio"
    if area in ["Subdirección de Gestión Clínica", "Subdirección de Servicios Clínicos",
                "Subdirección de Operaciones", "Subdirección de Administración y Finanzas",
                "Subdirección de Personas", "Subdirección Administrativa"]:
        if nivel_resp in ["Colaborador(a) individual", "Externo"]:
            if rango not in ["18 a 25", "26 a 35"] and nivel_resp != "Externo":
                return "Promedio"
            else:
                return "Básico"
        return "Administrador"
    return "Básico"


# Cargar JSON procesado
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    participantes = json.load(f)


# ANALISIS DATOS PROMEDIO GLOBALES SEGUN LA MUESTRA
item_to_dimension = {}
item_to_indica = {}
item_to_enunciado = {}
respuestas_por_item = {}
for p in participantes:
    for it in p.get("items", []):
        code = it.get("Item")
        if not code:
            continue

        item_to_dimension[code] = it.get("Dimension", "") or ""
        item_to_indica[code] = it.get("Indica_riesgo", "") or ""
        item_to_enunciado[code] = it.get("Enunciado", "") or ""

        val = it.get("Respuesta", None)
        if val is None or val == "":
            continue
        try:
            val = float(val)
        except Exception:
            continue

        # Invertir si el ítem indica riesgo
        if item_to_indica[code] == "Sí":
            val = 6 - val

        respuestas_por_item.setdefault(code, []).append(val)

# Promedio para cada item y su puntaje equivalente al percentil 35
promedio_por_item = {}
percentil35_por_item = {}
for code, vals in respuestas_por_item.items():
    if len(vals) == 0:
        continue
    promedio_por_item[code] = round(np.mean(vals), 2)
    percentil35_por_item[code] = round(np.percentile(vals, 35), 2)

# Promedio por dimensión (a partir de promedios de ítem)
dimension_items_vals = {}
for item_code, prom in promedio_por_item.items():
    dim = item_to_dimension.get(item_code, "")
    dimension_items_vals.setdefault(dim, []).append(prom)

promedio_por_dimension = {dim: round(np.mean(vals), 2) for dim, vals in dimension_items_vals.items() if vals}

# Promedio global (promedio de las dimensiones)
promedio_global = round(np.mean(list(promedio_por_dimension.values())), 2) if promedio_por_dimension else 0.0

print("DATOS PROMEDIO...")
print("Puntaje_promedio_total:", promedio_global)
print("Puntaje_promedio_por_dimension:", promedio_por_dimension)
print()


# ANÁLISIS POR PARTICIPANTE
resumen_participantes = []
for participante in participantes:
    items_analisis = []
    dimension_scores = {}

    for it in participante.get("items", []):
        item_code = it.get("Item")
        indica = it.get("Indica_riesgo", "")
        dim = it.get("Dimension", "")
        try:
            respuesta = float(it.get("Respuesta")) if it.get("Respuesta") not in (None, "") else None
        except Exception:
            respuesta = None

        # Invertir si es ítem de riesgo
        if respuesta is not None:
            if indica == "Sí":
                respuesta_invertida = 6 - respuesta
            else:
                respuesta_invertida = respuesta
        else:
            respuesta_invertida = None

        # Registrar valores por dimensión
        if dim and respuesta_invertida is not None:
            dimension_scores.setdefault(dim, []).append(respuesta_invertida)

        items_analisis.append({
            "Item": item_code,
            "Dimension": dim,
            "Indica_riesgo": indica,
            "Enunciado": it.get("Enunciado", "") or "",
            "Respuesta": respuesta,
            "respuesta_normalizada": respuesta_invertida, # Realmente a cuanto equivale del 1 al 5 si 1 es lo peor y 5 lo ideal
            "Promedio_item_global": promedio_por_item.get(item_code), # Para comparar
            "Percentil35_item_global": percentil35_por_item.get(item_code) # Para comparar
        })

    # Puntajes por dimensión del participante
    puntaje_por_dimension = {
        k: round(np.mean(v), 2) for k, v in dimension_scores.items() if v
    }

    # Puntaje promedio total del participante
    puntaje_total = (
        round(np.mean(list(puntaje_por_dimension.values())), 2)
        if puntaje_por_dimension else 0
    )

    # Items criticos vs si mismo
    items_criticos_personales = []
    for i in items_analisis:
        r = i["Respuesta"]
        if r is None:
            continue
        if i["Indica_riesgo"] == "Sí" and r >= 4:
            items_criticos_personales.append({**i})
        elif i["Indica_riesgo"] == "No" and r <= 2:
            items_criticos_personales.append({**i})

    # Dimensiones críticas: aquellas con peor puntaje o debajo de la media
    conteo_extremos_por_dimension = {}
    for i in items_criticos_personales:
        dim = i.get("Dimension")
        if not dim:
            continue
        conteo_extremos_por_dimension[dim] = conteo_extremos_por_dimension.get(dim, 0) + 1

    # Identificar la(s) dimensión(es) con mayor cantidad de respuestas extremas
    max_extremos = max(conteo_extremos_por_dimension.values(), default=0)
    dimensiones_por_respuestas_extremas = [
        dim for dim, count in conteo_extremos_por_dimension.items()
        if count == max_extremos and count > 0
    ]

    # Dimensiones con promedio por debajo del promedio global
    dimensiones_bajo_promedio_global = [
        d for d, val in puntaje_por_dimension.items()
        if val < promedio_global
    ]

    # Unir ambas condiciones
    dimensiones_criticas = sorted(
        set(dimensiones_por_respuestas_extremas + dimensiones_bajo_promedio_global)
    )


    # Ítems críticos vs percentil 35
    items_criticos_vs_media = []
    for i in items_analisis:
        prom_item = i.get("Promedio_item_global")
        perc35 = i.get("Percentil35_item_global")
        if prom_item is None or perc35 is None:
            continue

        if i["Indica_riesgo"] == "Sí":
            desempeño = 6 - (i["respuesta_normalizada"] or 0)
        else:
            desempeño = i["respuesta_normalizada"] or 0

        if desempeño <= perc35:
            items_criticos_vs_media.append(i)

    datos_personales = participante.get("informacion_personal", {})
    datos_personales["Nivel_expertis_ciberseguridad"] = asignar_nivel_expertis(datos_personales)

    salida = {
        "Participante": participante.get("Participante"),
        "Datos_personales": datos_personales,
        "Datos globales": {
            "Puntaje_promedio_global": promedio_global,
            "Puntaje_global_por_dimension": promedio_por_dimension
        },
        "Análisis_datos": {
            "Puntaje_promedio_total": puntaje_total,
            "Puntaje_promedio_por_dimension": puntaje_por_dimension,
            "Dimensiones_criticas": dimensiones_criticas,
            "Items_criticos_personales": items_criticos_personales, # Aquellos con puntaje 1,2 o 4,5 segun si son de riesgo o no
            "Items_criticos_debajo_percentil35": items_criticos_vs_media  # Debajo percentil 35
        }
    }

    # Guardar JSON individual
    json_individual = os.path.join(OUTPUT_JSON_DIR, f"participante_{participante.get('Participante')}.json")
    with open(json_individual, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    resumen_participantes.append(salida)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(resumen_participantes, f, indent=2, ensure_ascii=False)

print(f"JSON individuales guardados en: {OUTPUT_JSON_DIR}")
print(f"JSON agregado guardado en: {OUTPUT_JSON}")