"""
Diccionarios y constantes maestras del proyecto SBC Demanda.

Todos los mapeos categóricos viven aquí: colores, familias de género,
eventos eclesiásticos, tamaños, etc. Si algo se ve mal, se cambia acá
y se vuelve a correr el feature store.
"""
from __future__ import annotations
import datetime as dt
from typing import Dict, List, Tuple

# =========================================================================
# COLORES → FAMILIA DE GÉNERO
# =========================================================================
# Mapeo de tokens de color (en mayúsculas, sin tildes) a la familia
# del producto. La familia de género se infiere por el color dominante
# de la biblia, NO por la nomenclatura del nombre (que no está
# estandarizada). Orden de prioridad importa: si la descripción tiene
# múltiples colores, gana el primero que aparezca en este diccionario.

COLOR_TO_FAMILIA = {
    # FEMENINO — rosados y derivados
    "FUSCIA": "femenino",
    "FUCSIA": "femenino",
    "FUSC": "femenino",
    "FUCS": "femenino",
    "ROSADO": "femenino",
    "ROSA": "femenino",
    "ROS-": "femenino",
    "ROS": "femenino",  # abreviatura SBC
    "PALO ROSA": "femenino",
    "PALO": "femenino",
    "LILA": "femenino",
    "MORADO": "femenino",
    "MOR": "femenino",
    "VIOLETA": "femenino",
    "MAGENTA": "femenino",
    "CORAL": "femenino",
    "PURPURA": "femenino",

    # MASCULINO — oscuros y tonos tierra
    "AZUL": "masculino",
    "AZ": "masculino",  # abreviatura SBC
    "NAVY": "masculino",
    "NEGRA": "masculino",
    "NEGRO": "masculino",
    "NG": "masculino",  # abreviatura SBC
    "CAFE": "masculino",
    "CF": "masculino",  # abreviatura SBC
    "MARRON": "masculino",
    "MARR": "masculino",  # abreviatura SBC
    "MARO": "masculino",
    "MA-TE": "masculino",
    "MA": "masculino",
    "VINOTINTO": "masculino",
    "V.TINTO": "masculino",
    "VINO": "masculino",
    "BORDO": "masculino",
    "BURDEOS": "masculino",
    "GRIS": "masculino",
    "TE": "masculino",  # tabaco/teja oscuro
    "BRON": "masculino",  # bronce

    # JUVENIL — vibrantes
    "NARANJA": "juvenil",
    "NA-V": "juvenil",
    "NA": "juvenil",
    "AMARILLO": "juvenil",
    "VERDE MENTA": "juvenil",
    "MENTA": "juvenil",
    "VERDE": "juvenil",
    "VER": "juvenil",  # abreviatura SBC
    "OLIVO": "juvenil",
    "TURQUESA": "juvenil",
    "CELESTE": "juvenil",
    "COLORES": "juvenil",  # multicolor
    "FLORES": "femenino",
    "FLORAL": "femenino",

    # NEUTRO — universal/familiar
    "DORADO": "neutro",
    "DORADA": "neutro",
    "PLATEADO": "neutro",
    "PLATEADA": "neutro",
    "PLATA": "neutro",
    "BLANCO": "neutro",
    "BLANCA": "neutro",
    "BEIGE": "neutro",
    "CREMA": "neutro",
    "PERLA": "neutro",
    "MIEL": "neutro",
    "ROJO": "neutro",  # rojo es ambiguo, lo dejamos neutro
    "ROJA": "neutro",
    "CHERRY": "neutro",
    "CASTAÑO": "neutro",
    "CASTANO": "neutro",
    "UVA": "neutro",
}

# Lista ordenada de tokens (para detección por prioridad)
# Tokens más específicos PRIMERO (ej. "PALO ROSA" antes que "PALO")
COLOR_TOKENS_ORDENADOS: List[str] = sorted(
    COLOR_TO_FAMILIA.keys(),
    key=lambda x: (-len(x), x)  # Más largos primero
)


# =========================================================================
# TAMAÑO DE BIBLIA — extraído de la nomenclatura del SKU
# =========================================================================
# La SBC usa un código de 3 dígitos en la posición tras el versión:
#   RVR020, RVR040, RVR060, RVR080, RVR086, etc.
# El segundo dígito indica el tamaño general:
#   020, 022, 025, 026 → tamaño 02 (bolsillo/pequeña)
#   040, 045, 046     → tamaño 04 (mediana)
#   050, 055, 056     → tamaño 05 (mediana grande)
#   060, 065, 066     → tamaño 06 (grande/letra grande)
#   080, 085, 086     → tamaño 08 (super grande)
#   010               → tamaño 01 (mini bolsillo)
#   030               → tamaño 03 (cheque)

TAMANO_FAMILIA = {
    "01": "mini",
    "02": "bolsillo",
    "03": "cheque",
    "04": "mediana",
    "05": "estandar",
    "06": "letra_grande",
    "08": "super_gigante",
}


# =========================================================================
# TIPO DE LETRA — extraído de tokens en la nomenclatura
# =========================================================================
TIPO_LETRA_TOKENS = {
    "LSG": "letra_super_gigante",
    "LG": "letra_grande",
    "cLG": "letra_grande_compacta",
    "MN": "mini",
    "P": "pequena",
}


# =========================================================================
# ENCUADERNACIÓN — extraído de la nomenclatura
# =========================================================================
ENCUADERNACION_TOKENS = {
    "PJR": "tapa_dura",          # pasta dura
    "ZPJR": "con_cierre",        # cremallera/zipper
    "ZTIPJR": "con_cierre_indice", # cremallera + índice
    "TI": "con_indice",
    "IMI": "imitacion_cuero",
    "PIEL": "piel",
    "VINILO": "vinilo",
    "CART": "carton",            # cartoné
    "CN PL": "canto_plata",
    "CN DOR": "canto_dorado",
    "CN BL": "canto_blanco",
    "CN PIN": "canto_pintado",
}


# =========================================================================
# EVENTOS ECLESIÁSTICOS — REGRESORES PARA PROPHET
# =========================================================================
# Cada evento es un shock predecible de demanda, NO una anomalía.
# Para Prophet se modelan como "holidays" con ventana de influencia.
#
# Lógica:
# - Semana Santa: fecha móvil. La iglesia compra material 4-6 semanas antes.
# - Congresos julio: el mes completo es pico (ventana mes entero).
# - Mes de la Biblia (octubre): pico mayor. Ventana = todo octubre.
# - Pre-abasto diciembre: últimos 15 días para eventos primera semana enero.

# Fechas de Semana Santa (Domingo de Resurrección) por año
SEMANA_SANTA = {
    2018: dt.date(2018, 4, 1),
    2019: dt.date(2019, 4, 21),
    2020: dt.date(2020, 4, 12),
    2021: dt.date(2021, 4, 4),
    2022: dt.date(2022, 4, 17),
    2023: dt.date(2023, 4, 9),
    2024: dt.date(2024, 3, 31),
    2025: dt.date(2025, 4, 20),
    2026: dt.date(2026, 4, 5),
    2027: dt.date(2027, 3, 28),
    2028: dt.date(2028, 4, 16),
    2029: dt.date(2029, 4, 1),
    2030: dt.date(2030, 4, 21),
}


def construir_eventos_eclesiasticos() -> "pd.DataFrame":
    """
    Devuelve DataFrame con holidays personalizados para Prophet.
    Columnas: holiday, ds, lower_window, upper_window, prior_scale.

    prior_scale alto (>10) = Prophet le da más peso al evento.
    """
    import pandas as pd
    rows = []

    for year, fecha in SEMANA_SANTA.items():
        # Pre-Semana Santa: compra anticipada 6 semanas antes hasta el día
        rows.append({
            "holiday": "semana_santa",
            "ds": pd.Timestamp(fecha),
            "lower_window": -42,
            "upper_window": 7,
            "prior_scale": 12.0,
        })

    for year in range(2018, 2031):
        # Congresos anuales (julio completo)
        rows.append({
            "holiday": "congresos_julio",
            "ds": pd.Timestamp(year, 7, 15),
            "lower_window": -14,
            "upper_window": 16,
            "prior_scale": 10.0,
        })

        # Mes de la Biblia (octubre completo, pico el segundo domingo)
        rows.append({
            "holiday": "mes_biblia",
            "ds": pd.Timestamp(year, 10, 15),
            "lower_window": -14,
            "upper_window": 16,
            "prior_scale": 15.0,  # El más fuerte
        })

        # Pre-abastecimiento eventos enero (últimos 15 días dic)
        rows.append({
            "holiday": "pre_abasto_enero",
            "ds": pd.Timestamp(year, 12, 22),
            "lower_window": -7,
            "upper_window": 9,
            "prior_scale": 8.0,
        })

    return pd.DataFrame(rows)


# =========================================================================
# CATEGORÍAS INTERNAS DE PRECIO (5 niveles)
# =========================================================================
# Se calculan como QUINTILES sobre el precio promedio del ISBN, pero
# se exponen también como reglas de negocio para el modelo hedónico.
CATEGORIAS_PRECIO = ["economica", "semi_economica", "media", "semi_fina", "fina"]


# =========================================================================
# MERCADOS
# =========================================================================
# Colombia = nacional. Resto = internacional.
# Un ISBN puede ser solo nacional, solo internacional, o ambos.
PAISES_NACIONAL = {"COLOMBIA", "Colombia", "colombia"}


# =========================================================================
# CICLO DE VIDA DE ISBN
# =========================================================================
# Default: 24-36 meses. Editable desde la app.
CICLO_VIDA_DEFAULT_MESES = 30  # Punto medio
CICLO_VIDA_MIN = 24
CICLO_VIDA_MAX = 36


# =========================================================================
# CAPACIDAD DE LANZAMIENTO DE NOVEDADES
# =========================================================================
# Restricción operativa de Publicaciones:
# - 5 conceptos nuevos por año
# - Cada concepto = 1 TACO con 3 cubiertas distintas (mismo bloque, diferente
#   color/diseño exterior)
# - Las 3 cubiertas del mismo concepto se lanzan EL MISMO mes
# - Total máximo: 15 SKUs nuevos por año
# Esto es un techo HARD para el sugerido automático del Sprint 3.

CAPACIDAD_NOVEDADES = {
    "conceptos_por_ano": 5,
    "cubiertas_por_concepto": 3,
    "skus_por_ano": 15,  # = 5 × 3
}


# =========================================================================
# DESCUENTOS POR CATEGORÍA — DECISIÓN ESTRUCTURAL AL LANZAR SKU
# =========================================================================
# El descuento NO es endógeno (output del canal): es decisión estructural
# de política comercial al diseñar el SKU. Las económicas/misioneras
# tienen tope hard porque su margen no lo permite; las finas pueden
# absorber descuentos altos sin destruir el margen.

DESCUENTO_DEFAULT_POR_CATEGORIA = {
    "economica":      23.0,   # tope normal (25% excepcional)
    "semi_economica": 28.0,
    "media":          33.0,
    "semi_fina":      38.0,
    "fina":           42.0,
}

DESCUENTO_MAX_POR_CATEGORIA = {
    "economica":      25.0,   # excepcional
    "semi_economica": 32.0,
    "media":          38.0,
    "semi_fina":      43.0,
    "fina":           47.0,
}


def categoria_por_precio(precio: float) -> str:
    """
    Asigna categoría comercial a partir del precio sugerido COP.
    Umbrales calibrados con la distribución histórica del catálogo SBC.
    """
    if precio < 30_000:
        return "economica"
    if precio < 60_000:
        return "semi_economica"
    if precio < 100_000:
        return "media"
    if precio < 180_000:
        return "semi_fina"
    return "fina"


# =========================================================================
# CODIFICACIÓN SBU — Sociedades Bíblicas Unidas (v3.10)
# =========================================================================
# Convención canónica de codificación de Biblias y productos derivados,
# documentada en CODIFICACION_BIBLIAS.pdf (SBU/SBC). Estructura del código:
#
#   [VERSIÓN] [FAMILIA] [TAMAÑO] [PASTA] [SÍMBOLOS]
#
# Ejemplo: RVR035ZTI
#   RVR = Reina Valera 1960
#   0   = Biblia con referencias
#   3   = Tamaño 3 (8,5 × 17,2 cm)
#   5   = Imitación cuero
#   ZTI = Cierre + Índice
#
# El análisis del histórico de ventas (mayo 2026) confirma que 99,3% del
# volumen de la clase BIBLIAS sigue esta codificación (76,8% formato largo
# con 3 dígitos + 22,5% formato corto con 2 dígitos donde se omite la
# familia=0 por convención).

# ---- Versiones / traducciones ------------------------------------------
# Orden: las más largas primero para que el parser regex no confunda
# RVR con RVR95 al hacer match.
SBU_VERSIONES = [
    ("RVR95", "Reina Valera 1995"),
    ("RVR",   "Reina Valera 1960"),
    ("RVC",   "Reina Valera Contemporánea"),
    ("DHH",   "Dios Habla Hoy"),
    ("TLA",   "Traducción Lenguaje Actual"),
    ("VR",    "Reina Valera 1909"),
    # Traducciones SBU/SBC adicionales que aparecen en el histórico
    # pero no están en el folleto formal. Conservadas para compatibilidad
    # con el parser; el modelo las trata como "version" categórica.
    ("NTV",   "Nueva Traducción Viviente"),
    ("NVI",   "Nueva Versión Internacional"),
    ("NBLA",  "Nueva Biblia de las Américas"),
    ("BLP",   "Biblia La Palabra"),
]

# ---- Familia de productos (posición 4 del código) -----------------------
# El dígito puede ser "." (punto) para biblias sin referencias o un dígito.
# El modelo hedónico está entrenado SOLO sobre BIBLIAS (familia 0 o .).
# Para otras familias el constructor advierte y el R² del modelo cae.
SBU_FAMILIA_PRODUCTO = {
    "0": "Biblia con referencias",
    ".": "Biblia sin referencias",
    "2": "Nuevo Testamento",
    "3": "Porciones",
    "4": "Porciones nuevos lectores",
    "5": "Selecciones",
    "6": "Selecciones nuevos lectores",
    "7": "Libros y otros",
}
# Familias que el modelo hedónico cubre con confianza
SBU_FAMILIAS_MODELADAS = {"0", "."}


# ---- Tamaño del taco (posición 5) --------------------------------------
SBU_TAMANO = {
    "1": ("6,5 × 10,5 cm",  "Mini"),
    "2": ("10,5 × 14,5 cm", "Cheque"),
    "3": ("8,5 × 17,2 cm",  "Largo"),
    "4": ("11,5 × 16 cm",   "Bolsillo"),
    "5": ("13,5 × 19,5 cm", "Media"),
    "6": ("13,5 × 21 cm",   "Estándar (más vendida)"),
    "8": ("17 × 23 cm",     "Grande"),
}

# Mapeo del código SBU de tamaño a la familia textual usada por el
# pipeline anterior (compatibilidad con la columna `tamano_familia`).
SBU_TAMANO_A_FAMILIA_TEXTUAL = {
    "1": "mini",
    "2": "cheque",       # antes era 'bolsillo' con código 02 — la clave era ambigua
    "3": "bolsillo",     # tamaño 3 (8,5x17,2) según el folleto SBU
    "4": "bolsillo",
    "5": "estandar",
    "6": "letra_grande",
    "8": "super_gigante",
}


# ---- Tipo de pasta o tapa (posición 6) ---------------------------------
SBU_PASTA = {
    "0": "Rústica",
    "2": "Vinilo / tapaflex",
    "3": "Dura",
    "4": "Jean / jacquard",
    "5": "Imitación cuero",
    "6": "PU",
    "9": "Cuero genuino / piel",
}
# Códigos legacy detectados en el histórico que no están en el folleto
# oficial pero suman volumen pequeño (1, 7). Los tratamos como "otro".
SBU_PASTA_LEGACY = {"1": "Otro (legacy)", "7": "Otro (legacy)"}


# ---- Símbolos de características específicas ----------------------------
# Convención case-sensitive: c (concordancia breve) ≠ C (concordancia amplia).
# Cada entrada: (codigo, etiqueta_interna, descripcion, rango_pts_opcional,
#                familia_simbolo, frecuente_en_historico)
#
# `familia_simbolo` agrupa los símbolos para resolver conflictos cuando un
# producto activa varios del mismo grupo. Por ejemplo, tipo_letra solo
# admite uno de LM/LG/LGi/LSGi.
SBU_SIMBOLOS = [
    # Tamaño de letra (mutuamente excluyentes — el constructor obliga a 1)
    ("LM",   "letra_mediana",       "Letra Mediana",         "6-11 pt",  "letra", True),
    ("LG",   "letra_grande",        "Letra Grande",          "12-14 pt", "letra", True),
    ("LGi",  "letra_gigante",       "Letra Gigante",         "15-17 pt", "letra", True),
    ("LSGi", "letra_super_gigante", "Letra Súper Gigante",   "18+ pt",   "letra", True),
    # Características de encuadernación
    ("Z",    "cierre",              "Cierre / cremallera",   None,       "encuad", True),
    ("TI",   "indice",              "Índice",                None,       "encuad", True),
    # Notas / aparato crítico
    ("PJR",  "palabras_jesus",      "Palabras de Jesús resaltadas", None, "notas",  True),
    ("C",    "concordancia_amplia", "Concordancia Amplia (276 p)",  None, "notas",  True),
    ("c",    "concordancia_breve",  "Concordancia Breve (128 p)",   None, "notas",  True),
    ("DK",   "deuteros_alej",       "Deuteros cánon Alejandrino",   None, "notas",  True),
    ("D",    "deuteros_sep",        "Deuteros cánon separado",      None, "notas",  False),
    # Acabados y formato
    ("EE",   "estudio_economica",   "Edición de estudio económica", None, "edicion", False),
    ("P",    "ilustrado",           "Ilustrado",                    None, "edicion", False),
    ("T",    "tematica",            "Temática",                     None, "edicion", False),
    ("a",    "acolchado",           "Acolchado",                    None, "edicion", False),
    ("e",    "economica",           "Económica",                    None, "edicion", True),
    ("ue",   "ultraeconomica",      "Ultraeconómica",               None, "edicion", True),
]

# Lookups útiles para consumidores
SBU_SIMBOLO_TO_ETIQUETA   = {s[0]: s[1] for s in SBU_SIMBOLOS}
SBU_ETIQUETA_TO_SIMBOLO   = {s[1]: s[0] for s in SBU_SIMBOLOS}
SBU_SIMBOLOS_FRECUENTES   = {s[1] for s in SBU_SIMBOLOS if s[5]}  # set de etiquetas
SBU_SIMBOLOS_POCO_FRECUENTES = {s[1] for s in SBU_SIMBOLOS if not s[5]}
SBU_SIMBOLOS_LETRA        = {s[1] for s in SBU_SIMBOLOS if s[4] == "letra"}

# Orden de prioridad para tokenización greedy: más largos primero.
# CRÍTICO: case-sensitive (c ≠ C). El parser respeta esto.
SBU_SIMBOLOS_ORDEN = sorted(
    [s[0] for s in SBU_SIMBOLOS],
    key=lambda x: (-len(x), x)
)
