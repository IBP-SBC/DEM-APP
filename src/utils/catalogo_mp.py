"""
Catálogo de TACOs MP (Materias Primas Semielaboradas) de la SBC.

A partir de v3.11, la app tiene un catálogo completo de los TACOs MP con:
  - Código (ej. SE1130100000, M2036000)
  - Descripción larga (ej. "TACO RVR060cLG PJR CLARIDAD AYUDAS DIGITALES")
  - Stock actual y comprometido
  - Valor unitario en COP
  - Tamaño físico (cm) y gramaje
  - Atributos derivados (versión bíblica, familia/tamaño/pasta SBU)

Jerarquía de fuentes (la primera que exista gana):
  1. data/state/catalogo_mp.parquet
     - Persistido por el usuario tras cargar el Excel desde la app.
  2. data/raw/INVENTARIO_MP*.xlsx (el más reciente por mtime)
     - Excel oficial de inventario MP que Alberto recibe periódicamente.
     - Lo parseamos automáticamente y persistimos como parquet.
  3. src/utils/seed_data/catalogo_mp_seed.parquet
     - Datos semilla incluidos en el paquete (snapshot 2026-05-11).
     - Permite que la app funcione antes de que el usuario cargue el Excel real.

El catálogo es un DataFrame con columnas estándar; cualquier consumidor debe
usar las funciones de este módulo, no hardcodear nombres.
"""
from __future__ import annotations
import os
import re
import glob
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# --- Paths -----------------------------------------------------------------
_THIS_DIR     = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # src/utils/.. -> src/.. -> root
_SEED_PATH    = _THIS_DIR / "seed_data" / "catalogo_mp_seed.parquet"
_STATE_PATH   = _PROJECT_ROOT / "data" / "state" / "catalogo_mp.parquet"
_RAW_DIR      = _PROJECT_ROOT / "data" / "raw"

# Columnas estándar del catálogo (todas presentes en el output)
COLUMNAS_CATALOGO = [
    "codigo",
    "descripcion",
    "grupo",
    "subgrupo",
    "stock_unidades",
    "comprometido",
    "valor_unidad_cop",
    "tamano_ancho_cm",
    "tamano_alto_cm",
    "tamano_grosor_cm",
    "gramaje",
    "bodega",
    "ubicacion",
    "version_biblica",
    "familia_sbu",
    "tamano_sbu",
    "pasta_sbu",
]


# =========================================================================
# CARGA
# =========================================================================
_CACHE: Optional[pd.DataFrame] = None


def cargar_catalogo(force_reload: bool = False) -> pd.DataFrame:
    """
    Devuelve el catálogo de TACOs MP. Cachea en memoria entre llamadas.

    Si en data/raw/ hay un Excel de inventario MÁS RECIENTE que el parquet
    persistido, lo re-parsea y guarda automáticamente.
    """
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    # 1. Estrategia: ¿hay Excel más reciente que el parquet persistido?
    excel_path = _excel_mp_mas_reciente()
    if excel_path is not None:
        mtime_excel = excel_path.stat().st_mtime
        mtime_state = _STATE_PATH.stat().st_mtime if _STATE_PATH.exists() else 0
        if mtime_excel > mtime_state:
            # Excel es más reciente → parsear y persistir
            try:
                df = parsear_excel_inventario_mp(excel_path)
                _persistir_catalogo(df)
                _CACHE = df
                return _CACHE
            except Exception as e:
                print(f"⚠️ Error parseando {excel_path.name}: {e}. "
                      f"Usando parquet o seed.")

    # 2. Si hay parquet persistido, usar ese
    if _STATE_PATH.exists():
        _CACHE = pd.read_parquet(_STATE_PATH)
        return _CACHE

    # 3. Fallback al seed embebido
    if _SEED_PATH.exists():
        _CACHE = pd.read_parquet(_SEED_PATH)
        return _CACHE

    # 4. Catálogo vacío (no debería pasar; el seed siempre está)
    _CACHE = pd.DataFrame(columns=COLUMNAS_CATALOGO)
    return _CACHE


def _excel_mp_mas_reciente() -> Optional[Path]:
    """Busca data/raw/INVENTARIO_MP*.xlsx más reciente. None si no hay."""
    if not _RAW_DIR.exists():
        return None
    archivos = list(_RAW_DIR.glob("INVENTARIO_MP*.xlsx")) + \
               list(_RAW_DIR.glob("inventario_mp*.xlsx"))
    if not archivos:
        return None
    return max(archivos, key=lambda p: p.stat().st_mtime)


def _persistir_catalogo(df: pd.DataFrame) -> None:
    """Guarda el catálogo en data/state/catalogo_mp.parquet."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_STATE_PATH, index=False)


# =========================================================================
# PARSER DEL EXCEL OFICIAL
# =========================================================================
def parsear_excel_inventario_mp(path) -> pd.DataFrame:
    """
    Parsea el Excel de "INVENTARIO DE MATERIAS PRIMAS E INSUMOS" de la SBC.

    Estructura conocida:
    - Primera fila: título mergeado "INVENTARIO DE MATERIAS PRIMAS E INSUMOS"
    - Segunda fila: encabezados reales
    - A partir de la tercera fila: datos

    Filtramos solo el Subgrupo TACO (Prod. Semielaborado) que es lo que la
    app de proyección de demanda usa.
    """
    df = pd.read_excel(path, header=1)
    df = df[df["Subgrupo"].astype(str).str.startswith("TACO")].copy()
    df = df[df["Código"].notna()]  # Drop filas sin código

    # Parseador robusto de tamaño físico (formato '13.5x21x0.00', 'x0x', etc.)
    def parse_tamano(s):
        if not isinstance(s, str) or not s.strip():
            return None, None, None
        parts = s.strip().replace("X", "x").split("x")
        def to_f(p):
            p = p.strip().replace(",", ".")
            try:
                return float(p) if p else None
            except Exception:
                return None
        a = to_f(parts[0]) if len(parts) > 0 else None
        b = to_f(parts[1]) if len(parts) > 1 else None
        c = to_f(parts[2]) if len(parts) > 2 else None
        return a, b, c

    tamanos = df["Tamaño"].apply(parse_tamano)
    df["tamano_ancho_cm"]  = [t[0] for t in tamanos]
    df["tamano_alto_cm"]   = [t[1] for t in tamanos]
    df["tamano_grosor_cm"] = [t[2] for t in tamanos]

    out = pd.DataFrame({
        "codigo":             df["Código"].astype(str).str.strip(),
        "descripcion":        df["Material"].astype(str).str.strip(),
        "grupo":              df["Grupo"].astype(str).str.strip(),
        "subgrupo":           df["Subgrupo"].astype(str).str.strip(),
        "stock_unidades":     pd.to_numeric(df["Stock"], errors="coerce").fillna(0).astype(int),
        "comprometido":       pd.to_numeric(df["Comprometido"], errors="coerce").fillna(0).astype(int),
        "valor_unidad_cop":   pd.to_numeric(df["Vr. Unidad Actual"], errors="coerce"),
        "tamano_ancho_cm":    df["tamano_ancho_cm"],
        "tamano_alto_cm":     df["tamano_alto_cm"],
        "tamano_grosor_cm":   df["tamano_grosor_cm"],
        "gramaje":            pd.to_numeric(df["Gramaje"], errors="coerce"),
        "bodega":             df["Bodega"].astype(str).str.strip() if "Bodega" in df.columns else "",
        "ubicacion":          df["Ubicacion"].astype(str).str.strip() if "Ubicacion" in df.columns else "",
    })

    # Atributos derivados de la descripción (versión, familia/tamaño/pasta SBU)
    atts = out["descripcion"].apply(_detectar_atributos_sbu)
    out = pd.concat([out, atts], axis=1)

    # Reordenar a columnas estándar
    cols_existentes = [c for c in COLUMNAS_CATALOGO if c in out.columns]
    return out[cols_existentes].reset_index(drop=True)


def _detectar_atributos_sbu(desc: str) -> pd.Series:
    """Extrae versión bíblica + dígitos SBU (familia, tamaño, pasta) de la
    descripción comercial del TACO."""
    if not isinstance(desc, str):
        desc = ""
    desc_up = desc.upper()
    # Versión: probar las más largas primero
    versiones = ["RVR95", "RVR", "RVC", "DHH", "TLA", "NTV", "NVI"]
    version = next(
        (v for v in versiones if re.search(rf"\b{v}\b|\b{v}[0-9]", desc_up)),
        None,
    )
    # Después de la versión vienen 2 o 3 dígitos (familia + tamaño + pasta)
    m = re.search(
        r"(RVR95|RVR|RVC|DHH|TLA|NTV|NVI)\.?([0-9])([0-9]{1,2})",
        desc_up.replace(" ", ""),
    )
    familia_sbu = tamano_sbu = pasta_sbu = None
    if m:
        if len(m.group(3)) == 2:
            familia_sbu = m.group(2)
            tamano_sbu  = m.group(3)[0]
            pasta_sbu   = m.group(3)[1]
        else:
            # Formato corto: familia=0 implícita
            familia_sbu = "0"
            tamano_sbu  = m.group(2)
            pasta_sbu   = m.group(3)
    return pd.Series({
        "version_biblica": version,
        "familia_sbu":     familia_sbu,
        "tamano_sbu":      tamano_sbu,
        "pasta_sbu":       pasta_sbu,
    })


# =========================================================================
# CONSULTAS Y FORMATEO
# =========================================================================
def info_taco_mp(codigo: str) -> Optional[Dict[str, Any]]:
    """Devuelve dict con toda la info del TACO o None si no se encuentra."""
    if not codigo or not isinstance(codigo, str):
        return None
    cat = cargar_catalogo()
    fila = cat[cat["codigo"].astype(str).str.strip() == str(codigo).strip()]
    if len(fila) == 0:
        return None
    return fila.iloc[0].to_dict()


def descripcion_taco_mp(codigo: str, fallback: str = "") -> str:
    """Devuelve la descripción larga del TACO. Si no la encuentra, devuelve
    el fallback (o el código mismo si fallback está vacío)."""
    info = info_taco_mp(codigo)
    if info and info.get("descripcion"):
        return str(info["descripcion"])
    return fallback or str(codigo or "")


def label_taco_mp(codigo: str, max_len: int = 80) -> str:
    """Devuelve 'codigo — descripción' para usar en selects/multiselects.

    Si la descripción no se encuentra, muestra solo el código.
    """
    if not codigo or pd.isna(codigo):
        return ""
    desc = descripcion_taco_mp(codigo)
    if desc and desc != str(codigo):
        full = f"{codigo} — {desc}"
        if max_len and len(full) > max_len:
            full = full[: max_len - 1] + "…"
        return full
    return str(codigo)


def labels_de_lista(codigos) -> Dict[str, str]:
    """Devuelve dict {codigo: label} para una colección. Útil con
    st.selectbox(..., format_func=lambda c: labels[c])."""
    if codigos is None:
        return {}
    cat = cargar_catalogo()
    lookup = dict(zip(cat["codigo"].astype(str).str.strip(),
                       cat["descripcion"].astype(str).str.strip()))
    out = {}
    for c in codigos:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        c_str = str(c).strip()
        desc = lookup.get(c_str, "")
        out[c] = f"{c_str} — {desc}" if desc and desc != c_str else c_str
    return out


def listar_codigos() -> list:
    """Devuelve lista de todos los códigos de TACOs MP en el catálogo."""
    cat = cargar_catalogo()
    return sorted(cat["codigo"].astype(str).str.strip().unique().tolist())


# =========================================================================
# Test en línea de comandos
# =========================================================================
if __name__ == "__main__":
    cat = cargar_catalogo()
    print(f"Catálogo de TACOs MP: {len(cat)} registros")
    print(f"Columnas: {list(cat.columns)}")
    print()
    print("Primeros 5:")
    print(cat.head().to_string())
    print()
    print("Ejemplos de queries:")
    for cod in ["M2015101", "SE1130100000", "INEXISTENTE"]:
        print(f"  {cod:>15}  →  {label_taco_mp(cod)}")
