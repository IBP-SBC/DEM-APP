"""
Ingesta de ventas ejecutadas desde el archivo SIESA y catálogo de novedades
(v3.13).

Por qué:
  La app de demanda proyecta a partir de un histórico (historico_ventas.xlsx,
  2018→abr-2026). Para mantener las proyecciones frescas, Alberto carga
  periódicamente el archivo SIESA (la misma fuente que alimenta la app de
  ventas) y la plantilla de novedades del año. Este módulo:
    1. Procesa el SIESA con la MISMA política que la app de ventas
       (solo Aprobadas, sin distribución gratuita, sin servicios, cajas
        convertidas a unidades equivalentes).
    2. Lo mapea al formato del histórico (mismas columnas que
       historico_ventas.xlsx) para que build_features lo combine.
    3. Toma SOLO del año de corte en adelante (default 2026), porque el
       histórico ya cubre lo anterior y el SIESA es la fuente viva del año
       en curso.
    4. Procesa la plantilla de novedades (ISBN × fecha de lanzamiento) para
       marcar esos ISBNs como novedad/reciente en el feature store.

Política de empalme (IMPORTANTE):
  Para los años >= ANIO_CORTE, manda el SIESA (reemplaza lo que el histórico
  tuviera de esos años, evitando doble conteo). Para años < ANIO_CORTE, manda
  el histórico. Así, cargar el SIESA de mayo actualiza ene-may 2026 completo.

Persistencia:
  - data/state/ventas_ejecutadas_siesa.parquet  (ventas SIESA mapeadas)
  - data/state/ingesta_meta.json                 (año de corte, fechas, conteos)
  - data/state/novedades_catalogo_anio.parquet   (catálogo de novedades del año)
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
try:
    from core.fast_io import leer_excel
except Exception:
    leer_excel = pd.read_excel

_ROOT = Path(__file__).parent.parent.parent
_DATA_STATE = _ROOT / "data" / "state"
_DATA_PROC = _ROOT / "data" / "processed"

PATH_EJECUTADAS = _DATA_STATE / "ventas_ejecutadas_siesa.parquet"
PATH_META       = _DATA_STATE / "ingesta_meta.json"
PATH_NOVEDADES  = _DATA_STATE / "novedades_catalogo_anio.parquet"

ANIO_CORTE_DEFAULT = 2026

# --- Política replicada de la app de ventas (core/config.py + loader.py) ----
ESTADOS_INGRESO_VALIDOS = ["Aprobada"]
CANAL_PRINCIPAL_PRIORIDAD = ["B2B", "B2C", "MINISTERIAL"]
SIN_CANAL_LABEL = "Sin asignar"
FAMILIA_BIBLIAS = {"Con referencias", "Biblias sin referencias"}

# Cajas → unidades equivalentes (idéntico a la app de ventas v1.4.5)
MAPEO_CAJAS_UNIDADES = {
    "CAJA X16 9789587450033": 16, "CAJA X16 9789587450040": 16,
    "CAJA X16 9789587450057": 16, "CAJA X20 9786287781832": 20,
    "CAJA MISIONERA AZUL X20": 20, "CAJA MISIONERA ROJA X20": 20,
    "CAJA X24 9786287781474": 24, "CAJA X24 9786287781467": 24,
    "CAJA X24 9786287781481": 24, "CAJA X24 9789587457285": 24,
    "CAJA X24 9789587457377": 24, "CAJA X24 9789587457384": 24,
    "CAJA X24 9789587457391": 24, "CAJA X24 9789587457599": 24,
    "CAJA X24 9789587457605": 24, "CAJA X24 9789587457612": 24,
    "CAJA X24 9789587458381": 24, "CAJA X24 7899938428085": 24,
    "CAJA X24 7899938427163": 24, "CAJA X24 7899938428078": 24,
    "CAJA X32 9789587457308": 32, "CAJA X32 9789587457605": 32,
    "CAJA X32 9789587458374": 32, "CAJA X32 9789587458381": 32,
    "CAJA X32 9789587458398": 32, "CAJA X32 9781598778175": 32,
}

# Servicios / fletes que NO son productos vendibles (excluidos del ingreso)
PALABRAS_SERVICIO = ["FLETE", "SERVICIO LOGISTICO", "SERVICIOS LOGISTICOS",
                     "ARRIENDO", "WEB APP", "RECORRIDO", "SALA DE JUNTAS"]


# =========================================================================
# PROCESAMIENTO DEL SIESA
# =========================================================================
def _asignar_canal(row) -> str:
    """Deriva el canal (B2B/B2C/MINISTERIAL) de las 3 columnas flag."""
    for canal in CANAL_PRINCIPAL_PRIORIDAD:
        val = row.get(canal, "")
        if isinstance(val, str) and val.strip():
            return canal
        if pd.notna(val) and str(val).strip() and str(val).strip() != "nan":
            return canal
    return SIN_CANAL_LABEL


def _es_servicio(desc: str) -> bool:
    d = str(desc).upper()
    return any(p in d for p in PALABRAS_SERVICIO)


def procesar_siesa(
    ruta,
    anio_corte: int = ANIO_CORTE_DEFAULT,
    mapa_taco_clase: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Lee el SIESA, aplica la política de ingreso, filtra desde `anio_corte`,
    y devuelve (df_formato_historico, resumen).

    df_formato_historico tiene las MISMAS columnas que historico_ventas.xlsx,
    para que build_features lo combine sin fricción.

    mapa_taco_clase: DataFrame opcional con columnas [isbn, taco_mp, clase]
        para heredar TACO MP y CLASE de ISBNs ya conocidos. Para ISBNs nuevos
        (novedades), CLASE se deriva del SIESA y TACO MP queda vacío.
    """
    df = leer_excel(ruta, sheet_name=0, dtype={"Referencia": str})

    resumen = {"filas_crudas": len(df)}

    # Tipos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df[df["Fecha"].notna()].copy()
    for c in ["Cantidad", "Total conversion", "Precio unit."]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # --- Política de ingreso (idéntica a la app de ventas) ---
    # 1) solo Aprobadas
    if "Estado" in df.columns:
        df = df[df["Estado"].isin(ESTADOS_INGRESO_VALIDOS)]
    # 2) excluir servicios/fletes
    df = df[~df["Desc. item"].apply(_es_servicio)]
    # 3) excluir distribución gratuita (valor 0 con lista gratuita o Total=0)
    desc_lista = df.get("Desc. lista de precios", pd.Series("", index=df.index)).astype(str)
    es_gratuita = desc_lista.str.upper().str.contains("DISTRIBUCION GRATUITA", na=False)
    df = df[~es_gratuita]
    resumen["filas_tras_politica"] = len(df)

    # --- Cajas → unidades equivalentes ---
    df["factor_caja"] = (
        df["Referencia"].astype(str).str.strip()
        .map(MAPEO_CAJAS_UNIDADES).fillna(1).astype(int)
    )
    df["unidades_equivalentes"] = df["Cantidad"].astype(float) * df["factor_caja"]

    # --- Filtro temporal: SOLO desde el año de corte ---
    df["anio"] = df["Fecha"].dt.year
    df = df[df["anio"] >= anio_corte].copy()
    resumen["anio_corte"] = anio_corte
    resumen["filas_desde_corte"] = len(df)
    if len(df) == 0:
        resumen["fecha_min"] = resumen["fecha_max"] = None
        return _df_historico_vacio(), resumen
    resumen["fecha_min"] = str(df["Fecha"].min().date())
    resumen["fecha_max"] = str(df["Fecha"].max().date())

    # --- Derivar canal ---
    df["canal_calc"] = df.apply(_asignar_canal, axis=1)

    # --- Derivar CLASE (para ISBNs nuevos sin histórico) ---
    es_v2 = "FAMILA DE PRODUCTO" in df.columns or "FAMILIA DE PRODUCTO" in df.columns
    fam_col = "FAMILA DE PRODUCTO" if "FAMILA DE PRODUCTO" in df.columns else (
              "FAMILIA DE PRODUCTO" if "FAMILIA DE PRODUCTO" in df.columns else None)
    def _clase_derivada(row):
        fam = str(row.get(fam_col, "")).strip() if fam_col else ""
        if fam in FAMILIA_BIBLIAS:
            return "BIBLIAS"
        # heurística por descripción
        desc = str(row.get("Desc. item", "")).upper()
        if "NUEVO TESTAMENTO" in desc or desc.startswith("NT "):
            return "NT"
        if "PORCION" in desc:
            return "PORCIONES"
        if "SELECCION" in desc:
            return "SELECCIONES"
        return "OTROS"
    df["clase_derivada"] = df.apply(_clase_derivada, axis=1)

    # --- Heredar TACO MP y CLASE de ISBNs conocidos ---
    if mapa_taco_clase is not None and len(mapa_taco_clase) > 0:
        m = mapa_taco_clase.copy()
        m["isbn"] = m["isbn"].astype(str).str.strip()
        taco_lookup  = dict(zip(m["isbn"], m["taco_mp"]))
        clase_lookup = dict(zip(m["isbn"], m["clase"]))
    else:
        taco_lookup = {}
        clase_lookup = {}

    ref = df["Referencia"].astype(str).str.strip()
    df["taco_mp_final"] = ref.map(taco_lookup).fillna("")
    df["clase_final"] = ref.map(clase_lookup)
    # Donde no hay clase conocida, usar la derivada
    df["clase_final"] = df["clase_final"].fillna(df["clase_derivada"])

    # --- Construir DataFrame en formato del histórico (18 columnas) ---
    out = pd.DataFrame({
        "CANAL":                        df["canal_calc"],
        "Fecha":                        df["Fecha"],
        "AÑO":                          df["anio"],
        "Razón social cliente factura": df.get("Razon social cliente factura", ""),
        "Desc. sucursal factura":       df.get("Desc. sucursal factura", ""),
        "PAIS":                         df.get("Desc. país", ""),
        "DEPARTAMENTO":                 df.get("Desc. depto", ""),
        "CIUDAD":                       df.get("Desc. ciudad", ""),
        "Lista de precios cliente":     df.get("Lista de precios cliente", ""),
        "Nombre vendedor":              df.get("Nombre vendedor", ""),
        "ISBN":                         ref,
        "Descripcion ISBN":             df.get("Desc. item", ""),
        "TACO MP":                      df["taco_mp_final"],
        "CLASE":                        df["clase_final"],
        # Cantidad = unidades equivalentes (cajas ya convertidas)
        "Cantidad":                     df["unidades_equivalentes"],
        "Precio unit.":                 df.get("Precio unit.", 0),
        "Descuento porcentaje":         df.get("Dscto. promedio %", 0),
        "Valor venta":                  df["Total conversion"],
    })
    # Quitar filas sin unidades positivas (devoluciones netas se mantienen aparte)
    out = out[out["Cantidad"] != 0]
    resumen["filas_finales"] = len(out)
    resumen["isbns_unicos"] = int(out["ISBN"].nunique())
    resumen["unidades_total"] = int(out["Cantidad"].sum())
    resumen["valor_total"] = float(out["Valor venta"].sum())
    # ISBNs nuevos (no estaban en el histórico/feature)
    if taco_lookup:
        conocidos = set(taco_lookup.keys())
        resumen["isbns_nuevos"] = int(out[~out["ISBN"].isin(conocidos)]["ISBN"].nunique())
    else:
        resumen["isbns_nuevos"] = resumen["isbns_unicos"]
    return out, resumen


def _df_historico_vacio() -> pd.DataFrame:
    cols = ["CANAL", "Fecha", "AÑO", "Razón social cliente factura",
            "Desc. sucursal factura", "PAIS", "DEPARTAMENTO", "CIUDAD",
            "Lista de precios cliente", "Nombre vendedor", "ISBN",
            "Descripcion ISBN", "TACO MP", "CLASE", "Cantidad",
            "Precio unit.", "Descuento porcentaje", "Valor venta"]
    return pd.DataFrame(columns=cols)


# =========================================================================
# PLANTILLA DE NOVEDADES DEL AÑO
# =========================================================================
def procesar_novedades(ruta) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lee la plantilla de novedades (hoja 'Novedades') y la normaliza.

    Devuelve (df, resumen) con columnas:
      isbn, descripcion, anio_lanzamiento, mes_lanzamiento,
      es_personalizacion, cliente_personalizado, observaciones
    """
    MESES = {"ENE":1,"FEB":2,"MAR":3,"ABR":4,"MAY":5,"JUN":6,
             "JUL":7,"AGO":8,"SEP":9,"OCT":10,"NOV":11,"DIC":12}
    try:
        df = leer_excel(ruta, sheet_name="Novedades", dtype={"ISBN": str})
    except Exception:
        df = leer_excel(ruta, sheet_name=0, dtype={"ISBN": str})

    def _mes_num(v):
        if pd.isna(v):
            return None
        s = str(v).strip().upper()
        if s.isdigit():
            return int(s)
        return MESES.get(s[:3], None)

    out = pd.DataFrame({
        "isbn":                  df["ISBN"].astype(str).str.strip(),
        "descripcion":           df.get("Descripción", "").astype(str).str.strip(),
        "anio_lanzamiento":      pd.to_numeric(df.get("Año lanzamiento"), errors="coerce"),
        "mes_lanzamiento":       df.get("Mes lanzamiento").apply(_mes_num),
        "es_personalizacion":    df.get("Es personalización", "No").astype(str).str.strip().str.upper().isin(["SI", "SÍ", "YES", "TRUE"]),
        "cliente_personalizado": df.get("Cliente personalizado", "").astype(str).str.strip(),
        "observaciones":         df.get("Observaciones", "").astype(str).str.strip(),
    })
    out = out[out["isbn"].str.strip() != ""]
    out = out[out["isbn"].str.lower() != "nan"]
    resumen = {
        "n_novedades": len(out),
        "isbns_unicos": int(out["isbn"].nunique()),
        "n_personalizaciones": int(out["es_personalizacion"].sum()),
        "anios": sorted(out["anio_lanzamiento"].dropna().astype(int).unique().tolist()),
    }
    return out, resumen


# =========================================================================
# PERSISTENCIA
# =========================================================================
def persistir_ejecutadas(df: pd.DataFrame, resumen: Dict[str, Any]) -> None:
    _DATA_STATE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH_EJECUTADAS, index=False)
    meta = dict(resumen)
    meta["actualizado"] = datetime.now().isoformat(timespec="seconds")
    with open(PATH_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def persistir_novedades(df: pd.DataFrame) -> None:
    _DATA_STATE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH_NOVEDADES, index=False)


def cargar_ejecutadas() -> Optional[pd.DataFrame]:
    if PATH_EJECUTADAS.exists():
        return pd.read_parquet(PATH_EJECUTADAS)
    return None


def cargar_meta() -> Optional[Dict[str, Any]]:
    if PATH_META.exists():
        with open(PATH_META, encoding="utf-8") as f:
            return json.load(f)
    return None


def cargar_novedades_catalogo() -> Optional[pd.DataFrame]:
    if PATH_NOVEDADES.exists():
        return pd.read_parquet(PATH_NOVEDADES)
    return None


def mapa_taco_clase_desde_feature() -> Optional[pd.DataFrame]:
    """Devuelve [isbn, taco_mp, clase] desde el feature_isbn ya construido,
    para heredar TACO MP y CLASE al ingerir el SIESA. None si no existe."""
    fpath = _DATA_PROC / "feature_isbn.parquet"
    if not fpath.exists():
        return None
    f = pd.read_parquet(fpath, columns=["isbn", "taco_mp", "clase"])
    f["isbn"] = f["isbn"].astype(str).str.strip()
    return f


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        mapa = mapa_taco_clase_desde_feature()
        df, res = procesar_siesa(ruta, mapa_taco_clase=mapa)
        print("Resumen ingesta SIESA:")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print(df.head().to_string())
