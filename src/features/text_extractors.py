"""
Extractor de features desde texto libre de la descripción del ISBN.

A partir de v3.10, dos parsers conviven:

1. EXTRACTOR LEGACY (extraer_features_isbn):
   - Detecta color, género, encuadernación, tamaño y tipo de letra
     desde tokens libres en la descripción.
   - Sigue siendo la fuente del color/familia_genero (la SBU no codifica
     color) y del flag es_imitacion_cuero retro-compatible.

2. PARSER SBU (parsear_sbu):
   - Tokeniza la descripción siguiendo la convención canónica del folleto
     CODIFICACION_BIBLIAS.pdf (Sociedades Bíblicas Unidas).
   - Estructura: [VERSIÓN] [FAMILIA] [TAMAÑO] [PASTA] [SÍMBOLOS]
   - Cobertura medida sobre el histórico de ventas BIBLIAS: 99,3% del
     volumen (76,8% formato largo + 22,5% formato corto).
   - Output: dict con sbu_version, sbu_familia, sbu_tamano, sbu_pasta,
     sbu_simbolos (set de etiquetas internas), sbu_codigo_canonico,
     sbu_formato ('largo' | 'corto' | 'ninguno').

Ambos extractores se invocan por separado y sus salidas se unen en
build_features.py.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Optional, List, Dict, Set, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dictionaries import (
    COLOR_TO_FAMILIA,
    COLOR_TOKENS_ORDENADOS,
    TAMANO_FAMILIA,
    TIPO_LETRA_TOKENS,
    ENCUADERNACION_TOKENS,
    SBU_VERSIONES,
    SBU_FAMILIA_PRODUCTO,
    SBU_TAMANO,
    SBU_PASTA,
    SBU_PASTA_LEGACY,
    SBU_SIMBOLOS_ORDEN,
    SBU_SIMBOLO_TO_ETIQUETA,
)


def _normalizar(texto: str) -> str:
    """Sube a mayúsculas y quita tildes para matching robusto."""
    if not isinstance(texto, str):
        return ""
    # Quitar tildes
    nfkd = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sin_tildes.upper().strip()


def detectar_color(descripcion: str) -> tuple[Optional[str], Optional[str]]:
    """
    Detecta el color dominante y la familia de género.

    Estrategia: encuentra TODAS las ocurrencias de tokens de color en el
    texto y devuelve el que aparece primero (el color que el comprador
    ve primero en la descripción visual). Si dos colores empatan en
    posición, gana el más largo (más específico).

    Devuelve: (color, familia_genero) o (None, None) si no encuentra.
    """
    texto = _normalizar(descripcion)
    if not texto:
        return None, None

    candidatos = []  # (posicion, -longitud, token)
    for token in COLOR_TOKENS_ORDENADOS:
        pattern = r"(?:^|[^A-Z])" + re.escape(token) + r"(?:[^A-Z]|$)"
        match = re.search(pattern, texto)
        if match:
            # Posición del token (no del separador)
            pos = match.start() + (1 if match.group()[0] != texto[0] or
                                       not texto.startswith(token) else 0)
            # Buscar la posición exacta del token dentro del match
            token_pos = texto.find(token, match.start())
            candidatos.append((token_pos, -len(token), token))

    if not candidatos:
        return None, None

    # Ordenar: primero por posición ascendente, luego por longitud descendente
    candidatos.sort()
    _, _, ganador = candidatos[0]
    return ganador, COLOR_TO_FAMILIA[ganador]


def detectar_version(descripcion: str) -> Optional[str]:
    """
    Detecta la versión bíblica: RVR, RVR60, RVR95, RVC, DHH, TLA, NTV, etc.
    """
    texto = _normalizar(descripcion)
    # Versiones más comunes en SBC
    for version in ["RVR9508", "RVR95", "RVR60", "RVR086", "RVR085", "RVR066",
                    "RVR065", "RVR056", "RVR055", "RVR046", "RVR045",
                    "RVR026", "RVR025", "RVR022", "RVR016", "RVR",
                    "RVC", "DHH", "TLA", "NTV", "BLP", "DK", "NVI"]:
        if version in texto:
            # Normalizar a familia
            if version.startswith("RVR"):
                return "RVR"
            return version
    return None


def detectar_tamano(descripcion: str) -> tuple[Optional[str], Optional[str]]:
    """
    Detecta el código de tamaño y la familia.
    Pattern: versión seguida de 3 dígitos. RVR086, RVR065, etc.
    """
    texto = _normalizar(descripcion)
    # Buscar 3 dígitos después de RVR/DHH/TLA/RVC
    match = re.search(r"(?:RVR|DHH|TLA|RVC|NTV)(\d{3})", texto)
    if not match:
        return None, None
    codigo3 = match.group(1)
    codigo2 = codigo3[:2]
    return codigo3, TAMANO_FAMILIA.get(codigo2)


def detectar_tipo_letra(descripcion: str) -> Optional[str]:
    """
    Detecta el tipo de letra desde tokens en la descripción.
    Los tokens pueden estar pegados al SKU (ej. "RVR065cLGZPJR" → cLG).
    Orden de prioridad: LSG > cLG > LG > MN.
    """
    texto = _normalizar(descripcion)
    # Importante: probar más específicos primero
    # LSG = letra super gigante (3 letras pegadas)
    if re.search(r"LSG", texto):
        return "letra_super_gigante"
    # CLG = letra grande compacta (2 letras pegadas a c minúscula)
    if re.search(r"CLG", texto):
        return "letra_grande_compacta"
    # LG = letra grande (debe ir después de CLG)
    if re.search(r"(?:^|[^A-Z])LG(?:[^A-Z]|$)", texto) or re.search(r"\dLG", texto):
        return "letra_grande"
    if re.search(r"(?:^|[^A-Z])MN(?:[^A-Z]|$)", texto):
        return "mini"
    return None


def detectar_encuadernacion(descripcion: str) -> List[str]:
    """
    Detecta atributos de encuadernación. Puede haber varios simultáneos.
    """
    texto = _normalizar(descripcion)
    encontrados = []
    for token, etiqueta in ENCUADERNACION_TOKENS.items():
        token_up = token.upper()
        if token_up in texto and etiqueta not in encontrados:
            encontrados.append(etiqueta)
    return encontrados


def extraer_features_isbn(descripcion: str, isbn: str = "") -> Dict:
    """
    Aplica todos los detectores y devuelve un diccionario con features.

    Args:
        descripcion: texto libre del campo "Descripcion ISBN"
        isbn: opcional, para detectar patrones del ID

    Returns:
        dict con: version, tamano_codigo, tamano_familia, color_dominante,
                  familia_genero, tipo_letra, encuadernacion (lista),
                  tiene_cierre, tiene_indice
    """
    color, familia_g = detectar_color(descripcion)
    tamano_cod, tamano_fam = detectar_tamano(descripcion)
    encuad = detectar_encuadernacion(descripcion)

    return {
        "version": detectar_version(descripcion),
        "tamano_codigo": tamano_cod,
        "tamano_familia": tamano_fam,
        "color_dominante": color,
        "familia_genero": familia_g if familia_g else "no_clasificado",
        "tipo_letra": detectar_tipo_letra(descripcion),
        "encuadernacion_lista": encuad,
        "tiene_cierre": "con_cierre" in encuad or "con_cierre_indice" in encuad,
        "tiene_indice": any("indice" in e for e in encuad),
        "es_imitacion_cuero": "imitacion_cuero" in encuad,
        "tiene_canto_dorado": "canto_dorado" in encuad,
    }


# =========================================================================
# PARSER SBU — codificación canónica Sociedades Bíblicas Unidas (v3.10)
# =========================================================================
# Estructura del código:
#   [VERSIÓN] [FAMILIA opt] [TAMAÑO] [PASTA] [SÍMBOLOS pegados]
#
# Formato LARGO: 3 dígitos después de la versión (familia + tamaño + pasta).
#   Ejemplo: RVR065ZTI → RVR | 0 | 6 | 5 | ZTI
# Formato CORTO: 2 dígitos (familia=0 implícita, asume biblia con referencias).
#   Ejemplo: RVR65ZTI  → RVR | 0 | 6 | 5 | ZTI
#
# Lista de versiones reconocidas — ordenadas por longitud descendente
# para que "RVR95" no sea capturado como "RVR" + "95".
_VERSIONES_SBU = [v[0] for v in SBU_VERSIONES]
_VERSIONES_REGEX = "|".join(sorted(_VERSIONES_SBU, key=lambda x: (-len(x), x)))

# Permitimos prefijo "BIBLIA " o "BIBLIAS " opcional (común en descripciones
# antiguas como "BIBLIA RVR060e PERS AZUL").
_PAT_LARGO = re.compile(
    r"^(?:BIBLIAS?\s*)?(?P<ver>" + _VERSIONES_REGEX +
    r")(?P<fam>[0-9\.])(?P<tam>[0-9])(?P<pasta>[0-9])(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_PAT_CORTO = re.compile(
    r"^(?:BIBLIAS?\s*)?(?P<ver>" + _VERSIONES_REGEX +
    r")(?P<tam>[0-9])(?P<pasta>[0-9])(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _tokenizar_simbolos_sbu(sufijo_pegado: str) -> Set[str]:
    """
    Tokeniza el sufijo (parte case-sensitive después del último dígito).
    Estrategia greedy: en cada posición intenta el símbolo más largo posible.

    IMPORTANTE: case-sensitive. 'c' (concordancia breve) ≠ 'C' (amplia).

    El sufijo solo va hasta el primer espacio (los nombres comerciales y
    descripciones de color vienen después de espacio, no son parte del
    código SBU).
    """
    if not sufijo_pegado:
        return set()
    # Cortar en primer espacio
    pegado = sufijo_pegado.split(" ", 1)[0]
    # Quitar punto separador si lo trae (ej. "RVR.46LM" → familia=., resto=LM)
    pegado = pegado.lstrip(".")

    encontrados: Set[str] = set()
    i = 0
    n = len(pegado)
    while i < n:
        matched = False
        for tok in SBU_SIMBOLOS_ORDEN:
            L = len(tok)
            if pegado[i:i+L] == tok:  # case-sensitive
                encontrados.add(SBU_SIMBOLO_TO_ETIQUETA[tok])
                i += L
                matched = True
                break
        if not matched:
            i += 1  # consumir char desconocido (ej. dígitos extra, separadores)
    return encontrados


def _construir_codigo_canonico(
    version: str,
    familia: str,
    tamano: str,
    pasta: str,
    simbolos_etiquetas: Set[str],
) -> str:
    """
    Reconstruye el código SBU canónico a partir de los componentes.
    Útil para que el constructor del simulador genere el código en vivo.

    El orden interno de los símbolos sigue una convención editorial:
    primero notas/aparato (C, c, DK, D, PJR, EE), luego encuadernación
    (Z, TI), luego letra (LM/LG/LGi/LSGi), luego edición (P, T, a, e, ue).

    NOTA: la convención SBU formal no fija orden estricto entre símbolos,
    y el histórico muestra variaciones (cLGZPJR, LGZTIPJR, ZTIPJR cLG...).
    Usamos un orden estable y legible.
    """
    orden_canonico = [
        "concordancia_amplia", "concordancia_breve",
        "estudio_economica",
        "deuteros_alej", "deuteros_sep",
        "letra_mediana", "letra_grande", "letra_gigante", "letra_super_gigante",
        "cierre", "indice",
        "palabras_jesus",
        "ilustrado", "tematica", "acolchado",
        "economica", "ultraeconomica",
    ]
    etiqueta_a_codigo = {v: k for k, v in SBU_SIMBOLO_TO_ETIQUETA.items()}
    # Para los símbolos que no aparecen en orden_canonico, los anexamos
    # al final ordenados por longitud descendente (mismo criterio que el
    # parser greedy).
    en_orden = [e for e in orden_canonico if e in simbolos_etiquetas]
    resto = [e for e in simbolos_etiquetas if e not in orden_canonico]
    resto.sort(key=lambda e: (-len(etiqueta_a_codigo[e]), etiqueta_a_codigo[e]))

    sufijo = "".join(etiqueta_a_codigo[e] for e in (en_orden + resto))
    # En formato canónico SIEMPRE incluimos los 3 dígitos (familia + tamaño + pasta)
    # aunque la familia sea "." (sin referencias).
    return f"{version}{familia}{tamano}{pasta}{sufijo}"


def parsear_sbu(descripcion: str) -> Dict:
    """
    Parsea una descripción siguiendo la convención SBU.

    Returns:
        dict con claves:
        - sbu_version           : 'RVR' | 'RVR95' | 'RVC' | 'DHH' | 'TLA' | ...
                                  None si no se detecta versión.
        - sbu_familia           : '0' | '.' | '2' | '3' | ... | None
                                  En formato corto se asume '0'.
        - sbu_tamano            : '1'..'6' | '8' | None
        - sbu_pasta             : '0','2','3','4','5','6','9' (oficiales)
                                  | '1','7' (legacy) | None
        - sbu_simbolos          : set de etiquetas internas (puede estar vacío)
        - sbu_codigo_canonico   : string reconstruido (ej. 'RVR065ZTI')
        - sbu_formato           : 'largo' | 'corto' | 'ninguno'
        - sbu_tipo_letra        : 'letra_mediana' | 'letra_grande' |
                                  'letra_gigante' | 'letra_super_gigante' | None
                                  Single value (no se permite más de uno).
    """
    if not isinstance(descripcion, str):
        descripcion = ""
    texto = descripcion.strip()
    if not texto:
        return _resultado_vacio()

    texto_up = texto.upper()

    # Probar largo primero (más específico)
    m = _PAT_LARGO.match(texto_up)
    formato = "largo"
    if not m:
        m = _PAT_CORTO.match(texto_up)
        formato = "corto"
    if not m:
        return _resultado_vacio()

    version = m.group("ver").upper()
    if formato == "largo":
        familia = m.group("fam")
        # Si capturamos "." mantenemos el punto (sin referencias)
    else:
        familia = "0"  # convención: formato corto asume familia=0

    tamano = m.group("tam")
    pasta  = m.group("pasta")

    # El sufijo case-sensitive lo tomamos del texto ORIGINAL en la misma
    # posición — preservamos mayúsculas/minúsculas (c vs C es crítico).
    start_resto = m.end("pasta")
    resto_original = texto[start_resto:]
    simbolos = _tokenizar_simbolos_sbu(resto_original)

    # Tipo de letra: resolver a un único valor (en histórico solo 2 SKUs
    # tienen más de una etiqueta de letra, y son errores de codificación).
    simbolos_letra = [
        s for s in ["letra_super_gigante", "letra_gigante",
                    "letra_grande", "letra_mediana"]
        if s in simbolos
    ]
    sbu_tipo_letra = simbolos_letra[0] if simbolos_letra else None

    codigo_canonico = _construir_codigo_canonico(
        version, familia, tamano, pasta, simbolos
    )

    return {
        "sbu_version":         version,
        "sbu_familia":         familia,
        "sbu_tamano":          tamano,
        "sbu_pasta":           pasta,
        "sbu_simbolos":        simbolos,
        "sbu_codigo_canonico": codigo_canonico,
        "sbu_formato":         formato,
        "sbu_tipo_letra":      sbu_tipo_letra,
    }


def _resultado_vacio() -> Dict:
    return {
        "sbu_version":         None,
        "sbu_familia":         None,
        "sbu_tamano":          None,
        "sbu_pasta":           None,
        "sbu_simbolos":        set(),
        "sbu_codigo_canonico": "",
        "sbu_formato":         "ninguno",
        "sbu_tipo_letra":      None,
    }


def construir_codigo_sbu_desde_inputs(
    version: str,
    familia: str,
    tamano: str,
    pasta: str,
    simbolos_etiquetas: List[str],
) -> str:
    """
    Wrapper público para que el constructor del simulador (página 2) genere
    el código SBU en vivo desde los selects del formulario.

    Args:
        version: 'RVR', 'RVR95', 'RVC', ...
        familia: '0' | '.' | '2' | ...
        tamano:  '1'..'8'
        pasta:   '0','2','3','4','5','6','9'
        simbolos_etiquetas: lista de etiquetas internas seleccionadas
            (ej. ['cierre', 'indice', 'letra_grande', 'palabras_jesus'])

    Returns:
        Código SBU canónico (string), por ejemplo 'RVR065ZTILGPJR'.
    """
    return _construir_codigo_canonico(
        version, familia, tamano, pasta, set(simbolos_etiquetas or [])
    )


# =========================================================================
# Test rápido en línea de comandos
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EXTRACTOR LEGACY (color/encuadernación libre)")
    print("=" * 70)
    ejemplos = [
        "RVR065cLGZPJR IMI FUSC. CN PIN FUCS",
        "RVR086cLGiZTIPJR AZ AGUA.MNA CN PL",
        "RVR025cZPJR IMI LILA CN PIN",
        "TLA066LGPJR VALIOSA ROS- ROJA CN PL",
        "RVR066LGPJR EL CAMINO AZ CN PL",
        "RVR086CLSGiZTIPJR FUSCIA CN PL",
        "RVR046ZPJRLM UVA CN BL",
        "RVR065cLGZTIPJR IMI CF CN DOR",
    ]
    for d in ejemplos:
        feats = extraer_features_isbn(d)
        print(f"\n{d}")
        for k, v in feats.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("PARSER SBU (codificación canónica SBU)")
    print("=" * 70)
    for d in ejemplos:
        sbu = parsear_sbu(d)
        print(f"\n{d}")
        for k, v in sbu.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("TESTS DE CONSTRUCCIÓN INVERSA")
    print("=" * 70)
    # Caso 1: RVR 0 6 5 + cierre + índice + letra grande + palabras jesús
    codigo = construir_codigo_sbu_desde_inputs(
        "RVR", "0", "6", "5",
        ["cierre", "indice", "letra_grande", "palabras_jesus"]
    )
    print(f"  RVR + 065 + ZTI+LG+PJR → {codigo}")
    assert codigo.startswith("RVR065"), f"Esperaba RVR065..., got {codigo}"

    # Caso 2: TLA . 4 6 + letra mediana
    codigo = construir_codigo_sbu_desde_inputs(
        "TLA", ".", "4", "6", ["letra_mediana"]
    )
    print(f"  TLA + .46 + LM    → {codigo}")
    assert codigo == "TLA.46LM"

    # Caso 3: round-trip parser → constructor → parser debe ser idempotente
    for d_test in [
        "RVR065ZTILGPJR",
        "TLA046LM",
        "RVR65cLGZPJR",   # formato corto
        "DHH065ZTIcDK",
    ]:
        r = parsear_sbu(d_test)
        if r["sbu_formato"] != "ninguno":
            recons = construir_codigo_sbu_desde_inputs(
                r["sbu_version"], r["sbu_familia"],
                r["sbu_tamano"], r["sbu_pasta"],
                list(r["sbu_simbolos"]),
            )
            # Parsear el reconstruido y comparar features (no string idéntico)
            r2 = parsear_sbu(recons)
            assert r["sbu_version"] == r2["sbu_version"]
            assert r["sbu_familia"] == r2["sbu_familia"]
            assert r["sbu_tamano"]  == r2["sbu_tamano"]
            assert r["sbu_pasta"]   == r2["sbu_pasta"]
            assert r["sbu_simbolos"] == r2["sbu_simbolos"]
            print(f"  Round-trip OK: {d_test:<25} → {recons}")
    print("\n✅ Todos los tests pasaron")
