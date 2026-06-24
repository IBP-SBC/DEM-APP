"""
Constructor del Feature Store de la SBC.

INPUT:
  data/raw/historico_ventas.xlsx
  data/raw/tacos_por_sustituir.csv

OUTPUT (en data/processed/):
  feature_isbn.parquet       — un registro por ISBN con todas las features
  feature_cliente.parquet    — RFM por cliente
  feature_canal.parquet      — descuentos y volúmenes por canal/lista
  ventas_mensual_isbn.parquet — serie mensual por ISBN (input al modelo)
  eventos_eclesiasticos.parquet — calendario de eventos para Prophet

Uso:
  python src/features/build_features.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
try:
    from core.fast_io import leer_excel
except Exception:
    leer_excel = pd.read_excel
import numpy as np
from datetime import datetime

# Permitir imports relativos
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.text_extractors import extraer_features_isbn, parsear_sbu
from utils.dictionaries import (
    PAISES_NACIONAL,
    CATEGORIAS_PRECIO,
    CICLO_VIDA_DEFAULT_MESES,
    construir_eventos_eclesiasticos,
    SBU_SIMBOLOS,
)

# =========================================================================
# RUTAS
# =========================================================================
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"
DATA_PROC.mkdir(parents=True, exist_ok=True)

PATH_HIST = DATA_RAW / "historico_ventas.xlsx"
PATH_CLARIDAD = DATA_RAW / "tacos_por_sustituir.csv"


# =========================================================================
# 1. CARGA Y LIMPIEZA DEL HISTÓRICO
# =========================================================================
def _empalmar_ejecutadas_siesa(df_hist: pd.DataFrame) -> pd.DataFrame:
    """v3.13: combina el histórico del xlsx con las ventas ejecutadas del SIESA.

    Para años >= ANIO_CORTE (lo que persistió la ingesta), manda el SIESA:
    se eliminan esos años del histórico y se reemplazan por las filas del SIESA.
    Para años anteriores, se conserva el histórico tal cual.

    df_hist ya viene en snake_case (isbn, fecha, mes, cantidad, valor_venta...).
    Las ejecutadas vienen en formato del histórico (CANAL, Fecha, ...), así que
    se normalizan igual antes de concatenar.
    """
    path_ejec = DATA_STATE / "ventas_ejecutadas_siesa.parquet"
    path_meta = DATA_STATE / "ingesta_meta.json"
    if not path_ejec.exists():
        return df_hist  # nada cargado, histórico tal cual

    try:
        ejec_raw = pd.read_parquet(path_ejec)
    except Exception as e:
        print(f"   ⚠️ No se pudo leer ventas ejecutadas SIESA: {e}")
        return df_hist
    if len(ejec_raw) == 0:
        return df_hist

    # Año de corte (de la meta; default 2026)
    anio_corte = 2026
    if path_meta.exists():
        try:
            import json
            with open(path_meta, encoding="utf-8") as f:
                anio_corte = int(json.load(f).get("anio_corte", 2026))
        except Exception:
            pass

    # Normalizar las ejecutadas al mismo snake_case que df_hist
    ejec = ejec_raw.rename(columns={
        "CANAL": "canal", "Fecha": "fecha", "AÑO": "anio",
        "Razón social cliente factura": "cliente",
        "Desc. sucursal factura": "sucursal", "PAIS": "pais",
        "DEPARTAMENTO": "departamento", "CIUDAD": "ciudad",
        "Lista de precios cliente": "lista_precios", "Nombre vendedor": "vendedor",
        "ISBN": "isbn", "Descripcion ISBN": "descripcion", "TACO MP": "taco_mp",
        "CLASE": "clase", "Cantidad": "cantidad", "Precio unit.": "precio_unitario",
        "Descuento porcentaje": "descuento_pct", "Valor venta": "valor_venta",
    })
    ejec["fecha"] = pd.to_datetime(ejec["fecha"], errors="coerce")
    ejec = ejec[ejec["fecha"].notna()]
    ejec = ejec[ejec["cantidad"] > 0]
    ejec["mes"] = ejec["fecha"].dt.to_period("M").dt.to_timestamp()
    ejec["mercado"] = ejec["pais"].apply(
        lambda x: "nacional" if str(x).strip() in PAISES_NACIONAL else "internacional"
    )
    ejec["isbn"] = ejec["isbn"].astype(str).str.strip()
    ejec["es_posible_importado"] = ejec["taco_mp"].astype(str).str.upper().str.contains(
        "POSIBLE IMPORTADO", na=False
    )
    # Asegurar todas las columnas de df_hist
    for col in df_hist.columns:
        if col not in ejec.columns:
            ejec[col] = np.nan if df_hist[col].dtype.kind in "fi" else ""
    ejec = ejec[df_hist.columns]

    # Empalme: histórico < corte  +  SIESA >= corte
    hist_antes = df_hist[df_hist["fecha"].dt.year < anio_corte]
    n_reemplazadas = len(df_hist[df_hist["fecha"].dt.year >= anio_corte])
    combinado = pd.concat([hist_antes, ejec], ignore_index=True)

    print(f"   🔗 Empalme SIESA: histórico <{anio_corte} ({len(hist_antes):,} filas) "
          f"+ SIESA >={anio_corte} ({len(ejec):,} filas). "
          f"Reemplazadas {n_reemplazadas:,} filas del histórico original.")
    print(f"   ✓ Combinado: {len(combinado):,} filas | "
          f"{combinado['fecha'].min().date()} → {combinado['fecha'].max().date()}")
    return combinado


def cargar_historico() -> pd.DataFrame:
    """Lee el histórico de ventas y aplica limpiezas básicas."""
    print(f"📥 Cargando {PATH_HIST.name}...")
    df = leer_excel(PATH_HIST)

    # Normalizar tipos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df[df["Fecha"].notna()]
    df = df[df["Fecha"].dt.year >= 2018]
    df = df[df["Cantidad"] > 0]

    # Renombrar a snake_case (más fácil de manejar)
    df = df.rename(columns={
        "CANAL": "canal",
        "Fecha": "fecha",
        "AÑO": "anio",
        "Razón social cliente factura": "cliente",
        "Desc. sucursal factura": "sucursal",
        "PAIS": "pais",
        "DEPARTAMENTO": "departamento",
        "CIUDAD": "ciudad",
        "Lista de precios cliente": "lista_precios",
        "Nombre vendedor": "vendedor",
        "ISBN": "isbn",
        "Descripcion ISBN": "descripcion",
        "TACO MP": "taco_mp",
        "CLASE": "clase",
        "Cantidad": "cantidad",
        "Precio unit.": "precio_unitario",
        "Descuento porcentaje": "descuento_pct",
        "Valor venta": "valor_venta",
    })

    # Mes calendario (para series mensuales)
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()

    # Mercado (nacional/internacional)
    df["mercado"] = df["pais"].apply(
        lambda x: "nacional" if str(x).strip() in PAISES_NACIONAL else "internacional"
    )

    # ISBN como string (puede venir mixto numérico/alfa)
    df["isbn"] = df["isbn"].astype(str).str.strip()

    # =====================================================================
    # v3.13: EMPALME CON VENTAS EJECUTADAS DEL SIESA (año en curso)
    # =====================================================================
    # Si el usuario cargó el SIESA desde el Home, sus ventas (desde el año de
    # corte, default 2026) REEMPLAZAN lo que el histórico tuviera de esos años.
    # Así las proyecciones se alimentan de los datos frescos sin doble conteo.
    df = _empalmar_ejecutadas_siesa(df)

    # Limpieza de strings clave
    for col in ["taco_mp", "canal", "lista_precios", "clase"]:
        df[col] = df[col].astype(str).str.strip()

    # Flag de TACO MP "POSIBLE IMPORTADO" (= biblia importada, no producida en planta)
    df["es_posible_importado"] = df["taco_mp"].str.upper().str.contains(
        "POSIBLE IMPORTADO", na=False
    )

    print(f"   ✓ {len(df):,} filas | {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    return df


# =========================================================================
# 2. MAPEO CLARIDAD
# =========================================================================
def cargar_mapeo_claridad() -> pd.DataFrame:
    """Lee el CSV de sustitución de TACOs por familia CLARIDAD."""
    df = pd.read_csv(PATH_CLARIDAD, sep=";", encoding="utf-8")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.rename(columns={
        "SKU TACO MP": "taco_mp",
        "Descripcion TACO MP": "descripcion_taco",
        "SUSTITUTO": "taco_destino_claridad",
    })
    df["taco_mp"] = df["taco_mp"].astype(str).str.strip()
    df["taco_destino_claridad"] = df["taco_destino_claridad"].astype(str).str.strip()
    df["migra_a_claridad"] = df["taco_destino_claridad"].str.upper() != "NO"
    return df


# Cronograma de inicio CLARIDAD (de v3.2 — no cambiar sin consultar)
CLARIDAD_INICIO = {
    "CLARIDAD 060":                       "2027-01",
    "CLARIDAD 060 EL CAMINO":             "2027-01",
    "CLARIDAD 060 MUJER VIRTUOSA":        "2027-01",
    "CLARIDAD 060 SIRVE Y LIDERA":        "2027-01",
    "CLARIDAD 060 VIVE LA PALABRA":       "2027-01",
    "CLARIDAD 060 EVANGELISTICO":         "2027-01",
    "CLARIDAD 020":                       "2027-07",
    "CLARIDAD 020 AXS":                   "2027-07",
    "CLARIDAD 040":                       "2028-01",
    "CLARIDAD 050":                       "2029-01",
    "CLARIDAD 050 ORACIONES":             "2029-01",
    "CLARIDAD 080 E IPUC":                "2029-01",
    "CLARIDAD 080 LETRA SUPER GIGANTE":   "2029-01",
}


# =========================================================================
# 3. FEATURE STORE POR ISBN
# =========================================================================
def construir_feature_isbn(df: pd.DataFrame, claridad: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un registro por ISBN con todas las features:
    - Identificación: ISBN, descripción, taco_mp, clase
    - Texto: versión, color, familia_genero, tamaño, tipo_letra, encuadernación
    - Económicas: precio promedio, descuento promedio, ticket promedio
    - Comerciales: unidades_total, valor_total, n_facturas, primer/última venta
    - Mercado: nacional, internacional, ambos, share_internacional
    - Categoría: por quintil de precio
    - CLARIDAD: si migra, a qué taco, en qué mes
    - Estado: ACTIVO / RECIENTE / DESCONTINUADO
    """
    print("🏗  Construyendo feature_isbn...")

    # --- Agregaciones por ISBN ---
    def safe_mode(x):
        """Devuelve el valor más frecuente, o '' si la serie está vacía."""
        s = x.dropna()
        if len(s) == 0:
            return ""
        vc = s.value_counts()
        return vc.index[0] if len(vc) else ""

    g = df.groupby("isbn").agg(
        descripcion=("descripcion", lambda x: x.dropna().iloc[0] if x.notna().any() else ""),
        taco_mp=("taco_mp", safe_mode),
        clase=("clase", safe_mode),
        primera_venta=("fecha", "min"),
        ultima_venta=("fecha", "max"),
        unidades_total=("cantidad", "sum"),
        valor_total=("valor_venta", "sum"),
        n_facturas=("fecha", "count"),
        n_clientes=("cliente", "nunique"),
        n_meses_con_venta=("mes", "nunique"),
        precio_p25=("precio_unitario", lambda x: x.quantile(0.25)),
        precio_p50=("precio_unitario", "median"),
        precio_p75=("precio_unitario", lambda x: x.quantile(0.75)),
        precio_promedio=("precio_unitario", "mean"),
        descuento_promedio=("descuento_pct", "mean"),
        share_importado=("es_posible_importado", "mean"),
    ).reset_index()

    # --- Mercado ---
    mer = df.groupby(["isbn", "mercado"])["cantidad"].sum().unstack(fill_value=0)
    mer.columns = [f"unidades_{c}" for c in mer.columns]
    if "unidades_nacional" not in mer.columns:
        mer["unidades_nacional"] = 0
    if "unidades_internacional" not in mer.columns:
        mer["unidades_internacional"] = 0
    mer["share_internacional"] = mer["unidades_internacional"] / (
        mer["unidades_nacional"] + mer["unidades_internacional"]
    ).clip(lower=1)
    mer["mercado_principal"] = np.where(
        mer["share_internacional"] >= 0.5, "internacional",
        np.where(mer["share_internacional"] > 0, "ambos", "nacional")
    )
    g = g.merge(mer.reset_index(), on="isbn", how="left")

    # --- Canal y lista de precios principales ---
    canal_top = df.groupby(["isbn", "canal"])["cantidad"].sum().reset_index()
    canal_top = canal_top.sort_values(["isbn", "cantidad"], ascending=[True, False])
    canal_top = canal_top.drop_duplicates("isbn", keep="first")[["isbn", "canal"]]
    canal_top.columns = ["isbn", "canal_principal"]
    g = g.merge(canal_top, on="isbn", how="left")

    lista_top = df.groupby(["isbn", "lista_precios"])["cantidad"].sum().reset_index()
    lista_top = lista_top.sort_values(["isbn", "cantidad"], ascending=[True, False])
    lista_top = lista_top.drop_duplicates("isbn", keep="first")[["isbn", "lista_precios"]]
    lista_top.columns = ["isbn", "lista_precios_principal"]
    g = g.merge(lista_top, on="isbn", how="left")

    # --- Features de texto (color, género, tamaño, etc.) ---
    print("   • Extrayendo features de texto (color, tamaño, encuadernación)...")
    feats = g["descripcion"].apply(extraer_features_isbn).apply(pd.Series)
    g = pd.concat([g, feats], axis=1)

    # --- Features SBU (codificación canónica Sociedades Bíblicas Unidas) ---
    # Añadidas en v3.10. Permiten al modelo hedónico distinguir explícitamente
    # tipo de pasta (vinilo, dura, jean, PU, cuero genuino), tamaño de letra
    # con granularidad SBU (LM/LG/LGi/LSGi) y características específicas
    # (PJR, C, c, DK, D, etc.) que antes el extractor legacy no separaba.
    print("   • Parseando codificación SBU (v3.10)...")
    sbu_feats = g["descripcion"].apply(parsear_sbu).apply(pd.Series)
    g = pd.concat([g, sbu_feats], axis=1)

    # Convertir el set sbu_simbolos a dummies por símbolo, para que el
    # modelo LightGBM las consuma como features booleanas.
    print("   • Generando dummies SBU por símbolo...")
    for codigo, etiqueta, _desc, _pts, _fam, _frec in SBU_SIMBOLOS:
        col = f"sbu_sym_{etiqueta}"
        g[col] = g["sbu_simbolos"].apply(
            lambda s: int(etiqueta in s) if isinstance(s, set) else 0
        )

    # Cobertura SBU sobre BIBLIAS (diagnóstico)
    mask_b = (g["clase"] == "BIBLIAS")
    if mask_b.sum() > 0:
        n_bibl = mask_b.sum()
        n_parse = (mask_b & g["sbu_formato"].isin(["largo", "corto"])).sum()
        print(f"   • Cobertura SBU en BIBLIAS: {n_parse}/{n_bibl} ({n_parse/n_bibl*100:.1f}%)")

    # --- Categoría de precio (quintiles) — solo BIBLIAS para que tenga sentido ---
    print("   • Categorizando por precio (quintiles solo en BIBLIAS)...")
    g["categoria_precio"] = "no_aplica"
    mask_b = (g["clase"] == "BIBLIAS") & (g["precio_promedio"] > 1000)
    if mask_b.sum() > 5:
        g.loc[mask_b, "categoria_precio"] = pd.qcut(
            g.loc[mask_b, "precio_promedio"], q=5, labels=CATEGORIAS_PRECIO
        ).astype(str)

    # --- Estado del ISBN ---
    fecha_corte = df["fecha"].max()
    g["meses_desde_ultima_venta"] = (
        (fecha_corte - g["ultima_venta"]).dt.days / 30
    ).round(1)
    g["meses_vida"] = (
        (g["ultima_venta"] - g["primera_venta"]).dt.days / 30
    ).round(1)

    def clasificar_estado(row):
        if row["meses_desde_ultima_venta"] <= 3:
            if row["meses_vida"] < 6:
                return "RECIENTE"
            return "ACTIVO"
        elif row["meses_desde_ultima_venta"] <= 12:
            return "DECLINANDO"
        return "DESCONTINUADO"
    g["estado"] = g.apply(clasificar_estado, axis=1)

    # --- CLARIDAD: ¿migra? ¿a cuál? ¿cuándo? ---
    g = g.merge(
        claridad[["taco_mp", "taco_destino_claridad", "migra_a_claridad"]],
        on="taco_mp", how="left"
    )
    g["migra_a_claridad"] = g["migra_a_claridad"].fillna(False)
    g["mes_inicio_claridad"] = g["taco_destino_claridad"].map(CLARIDAD_INICIO).fillna("")

    # --- Anualización: ventas anuales típicas (madura) ---
    # Solo años con venta plena (>=8 meses) y excluyendo el actual incompleto
    fecha_max_periodo = df["fecha"].max().to_period("M")
    df_anu = df.copy()
    df_anu["anio_v"] = df_anu["fecha"].dt.year
    venta_anio = df_anu.groupby(["isbn", "anio_v"]).agg(
        unidades=("cantidad", "sum"),
        meses=("mes", "nunique"),
    ).reset_index()
    venta_anio = venta_anio[venta_anio["meses"] >= 8]  # solo años "completos"
    if len(venta_anio):
        venta_anio["unidades_anualizadas"] = venta_anio["unidades"] * (12 / venta_anio["meses"])
        anu = venta_anio.groupby("isbn")["unidades_anualizadas"].mean().reset_index()
        anu.columns = ["isbn", "demanda_anual_madura"]
        g = g.merge(anu, on="isbn", how="left")
    else:
        g["demanda_anual_madura"] = np.nan

    # --- Ciclo de vida observado ---
    g["ciclo_meses_observado"] = g["n_meses_con_venta"]

    # --- Limpiezas finales ---
    for col in ["share_internacional", "demanda_anual_madura"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    # Ordenar columnas
    sbu_dummies = [f"sbu_sym_{etiq}" for _c, etiq, *_ in SBU_SIMBOLOS]
    cols_orden = [
        "isbn", "descripcion", "clase", "taco_mp",
        "version", "tamano_codigo", "tamano_familia",
        "color_dominante", "familia_genero",
        "tipo_letra", "encuadernacion_lista",
        "tiene_cierre", "tiene_indice", "es_imitacion_cuero", "tiene_canto_dorado",
        # --- Features SBU (v3.10) ---
        "sbu_version", "sbu_familia", "sbu_tamano", "sbu_pasta",
        "sbu_codigo_canonico", "sbu_formato", "sbu_tipo_letra",
        "sbu_simbolos",
        *sbu_dummies,
        # ---
        "categoria_precio",
        "precio_p25", "precio_p50", "precio_p75", "precio_promedio",
        "descuento_promedio", "share_importado",
        "canal_principal", "lista_precios_principal",
        "mercado_principal", "share_internacional",
        "unidades_nacional", "unidades_internacional",
        "unidades_total", "valor_total", "n_facturas", "n_clientes", "n_meses_con_venta",
        "primera_venta", "ultima_venta", "meses_vida", "meses_desde_ultima_venta",
        "ciclo_meses_observado", "demanda_anual_madura",
        "estado",
        "migra_a_claridad", "taco_destino_claridad", "mes_inicio_claridad",
    ]
    cols_orden = [c for c in cols_orden if c in g.columns]
    g = g[cols_orden]

    print(f"   ✓ {len(g):,} ISBNs · {len(g.columns)} features")
    return g


# =========================================================================
# 4. SERIE MENSUAL POR ISBN (input al modelo Prophet)
# =========================================================================
def construir_mix_cliente_isbn(df: pd.DataFrame) -> pd.DataFrame:
    """v3.16: mix histórico cliente × ISBN (para desagregar la proyección del
    cliente entre sus ISBNs — Camino B).

    Una fila por (cliente, isbn) con unidades y valor históricos totales y la
    descripción del producto. La participación (share) se calcula al desagregar.
    """
    print("🧩 Construyendo mix cliente × ISBN...")
    desc_map = (df.groupby("isbn")["descripcion"].agg(
        lambda s: s.dropna().iloc[0] if len(s.dropna()) else "") )
    mix = df.groupby(["cliente", "isbn"]).agg(
        unidades_hist=("cantidad", "sum"),
        valor_hist=("valor_venta", "sum"),
        ultima_compra=("fecha", "max"),
    ).reset_index()
    mix = mix[mix["unidades_hist"] > 0]
    mix["descripcion"] = mix["isbn"].map(desc_map).fillna("")
    print(f"   ✓ {len(mix):,} pares cliente-isbn")
    return mix


def construir_serie_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (isbn, mes) con cantidades y valores agregados."""
    print("📈 Construyendo serie mensual por ISBN...")
    serie = df.groupby(["isbn", "mes"]).agg(
        unidades=("cantidad", "sum"),
        valor=("valor_venta", "sum"),
        n_facturas=("fecha", "count"),
        n_clientes=("cliente", "nunique"),
        precio_promedio_mes=("precio_unitario", "mean"),
        descuento_promedio_mes=("descuento_pct", "mean"),
        share_internacional_mes=("mercado", lambda x: (x == "internacional").mean()),
    ).reset_index()
    print(f"   ✓ {len(serie):,} filas isbn-mes")
    return serie


# =========================================================================
# 5. RFM POR CLIENTE
# =========================================================================
def construir_feature_cliente(df: pd.DataFrame) -> pd.DataFrame:
    print("👥 Construyendo feature_cliente (RFM)...")

    def safe_mode(x):
        s = x.dropna()
        if len(s) == 0:
            return ""
        vc = s.value_counts()
        return vc.index[0] if len(vc) else ""

    fecha_corte = df["fecha"].max()
    cli = df.groupby("cliente").agg(
        primera_compra=("fecha", "min"),
        ultima_compra=("fecha", "max"),
        n_facturas=("fecha", "count"),
        unidades_total=("cantidad", "sum"),
        valor_total=("valor_venta", "sum"),
        n_isbns=("isbn", "nunique"),
        canal_top=("canal", safe_mode),
        lista_top=("lista_precios", safe_mode),
        mercado=("mercado", safe_mode),
        descuento_promedio=("descuento_pct", "mean"),
    ).reset_index()
    cli["recencia_dias"] = (fecha_corte - cli["ultima_compra"]).dt.days
    cli["frecuencia_meses"] = (
        (cli["ultima_compra"] - cli["primera_compra"]).dt.days / 30
    ).clip(lower=1)
    cli["compras_por_mes"] = cli["n_facturas"] / cli["frecuencia_meses"]
    cli["unidades_por_compra"] = cli["unidades_total"] / cli["n_facturas"]
    cli["ticket_promedio"] = cli["valor_total"] / cli["n_facturas"]

    # v3.15: último vendedor que le facturó a cada cliente (registro más
    # reciente por fecha). Útil para el CSV de presupuesto por cliente.
    if "vendedor" in df.columns:
        df_ord = df[["cliente", "fecha", "vendedor"]].dropna(subset=["cliente"]).copy()
        df_ord = df_ord.sort_values("fecha")
        ultimo_vend = (
            df_ord.groupby("cliente")["vendedor"]
            .last().reset_index().rename(columns={"vendedor": "ultimo_vendedor"})
        )
        cli = cli.merge(ultimo_vend, on="cliente", how="left")
    else:
        cli["ultimo_vendedor"] = ""
    cli["ultimo_vendedor"] = cli["ultimo_vendedor"].fillna("").astype(str)

    print(f"   ✓ {len(cli):,} clientes")
    return cli


def construir_ventas_mensual_cliente_clase(
    df: pd.DataFrame,
    feat_isbn: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Serie mensual agregada por cliente × clase × version × categoria_precio × mes.

    v3.9: ahora incluye `version` y `categoria_precio` (vienen de feature_isbn)
    para soportar los filtros encadenados del Explorador por Cliente.

    Output: cliente, clase, version, categoria_precio, mes,
            unidades, valor, n_facturas, n_isbns.

    Si `feat_isbn` no se entrega o no tiene esas columnas, se sustituyen por
    "(sin clasificar)" y la app sigue funcionando con el filtro por clase
    nada más.
    """
    print("👥📅 Construyendo ventas_mensual_cliente_clase (con version y categoria_precio)...")
    df_ok = df[df["cliente"].notna() & (df["cliente"].astype(str).str.strip() != "")].copy()
    if "clase" not in df_ok.columns:
        df_ok["clase"] = "SIN_CLASE"
    df_ok["clase"] = df_ok["clase"].fillna("SIN_CLASE").astype(str)
    df_ok["mes"] = pd.to_datetime(df_ok["fecha"]).dt.to_period("M").dt.to_timestamp()
    df_ok["isbn"] = df_ok["isbn"].astype(str).str.strip()

    # ───────────────────────────────────────────────────────────────────
    # Join con feature_isbn para traer version y categoria_precio.
    # Si no llega feat_isbn (uso aislado de esta función), las columnas se
    # rellenan con "(sin clasificar)" y los filtros downstream lo manejan.
    # ───────────────────────────────────────────────────────────────────
    if feat_isbn is not None and {"isbn", "version", "categoria_precio"}.issubset(feat_isbn.columns):
        mapa = feat_isbn[["isbn", "version", "categoria_precio"]].copy()
        mapa["isbn"] = mapa["isbn"].astype(str).str.strip()
        mapa["version"] = mapa["version"].fillna("(sin versión)").replace("", "(sin versión)").astype(str)
        mapa["categoria_precio"] = (
            mapa["categoria_precio"].fillna("no_aplica").replace("", "no_aplica").astype(str)
        )
        df_ok = df_ok.merge(mapa, on="isbn", how="left")
        df_ok["version"] = df_ok["version"].fillna("(sin versión)")
        df_ok["categoria_precio"] = df_ok["categoria_precio"].fillna("no_aplica")
    else:
        df_ok["version"] = "(sin clasificar)"
        df_ok["categoria_precio"] = "(sin clasificar)"

    agg = df_ok.groupby(
        ["cliente", "clase", "version", "categoria_precio", "mes"],
        observed=True,  # evita warnings con categóricas
    ).agg(
        unidades=("cantidad", "sum"),
        valor=("valor_venta", "sum"),
        n_facturas=("fecha", "count"),
        n_isbns=("isbn", "nunique"),
    ).reset_index()

    print(f"   ✓ {len(agg):,} filas (cliente × clase × version × cat_precio × mes)")
    return agg


# =========================================================================
# 6. FEATURE CANAL/LISTA
# =========================================================================
def construir_feature_canal(df: pd.DataFrame) -> pd.DataFrame:
    print("📊 Construyendo feature_canal...")
    canal = df.groupby(["canal", "lista_precios"]).agg(
        n_facturas=("fecha", "count"),
        unidades_total=("cantidad", "sum"),
        valor_total=("valor_venta", "sum"),
        precio_promedio=("precio_unitario", "mean"),
        descuento_promedio=("descuento_pct", "mean"),
        n_clientes=("cliente", "nunique"),
        n_isbns=("isbn", "nunique"),
    ).reset_index()
    print(f"   ✓ {len(canal):,} pares canal-lista")
    return canal


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 70)
    print("CONSTRUCCIÓN DEL FEATURE STORE — SBC DEMANDA")
    print("=" * 70)
    print(f"⏱  Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    df = cargar_historico()
    claridad = cargar_mapeo_claridad()

    feat_isbn = construir_feature_isbn(df, claridad)
    serie = construir_serie_mensual(df)
    feat_cli = construir_feature_cliente(df)
    serie_cli_clase = construir_ventas_mensual_cliente_clase(df, feat_isbn=feat_isbn)
    mix_cli_isbn = construir_mix_cliente_isbn(df)
    feat_canal = construir_feature_canal(df)
    eventos = construir_eventos_eclesiasticos()

    # Guardar
    print("\n💾 Guardando outputs en data/processed/...")
    # sbu_simbolos viene como set de Python; convertir a lista para que
    # pyarrow lo serialice (parquet no soporta el tipo set).
    if "sbu_simbolos" in feat_isbn.columns:
        feat_isbn["sbu_simbolos"] = feat_isbn["sbu_simbolos"].apply(
            lambda s: sorted(s) if isinstance(s, set) else (list(s) if s else [])
        )
    feat_isbn.to_parquet(DATA_PROC / "feature_isbn.parquet", index=False)
    serie.to_parquet(DATA_PROC / "ventas_mensual_isbn.parquet", index=False)
    feat_cli.to_parquet(DATA_PROC / "feature_cliente.parquet", index=False)
    serie_cli_clase.to_parquet(DATA_PROC / "ventas_mensual_cliente_clase.parquet", index=False)
    mix_cli_isbn.to_parquet(DATA_PROC / "mix_cliente_isbn.parquet", index=False)
    feat_canal.to_parquet(DATA_PROC / "feature_canal.parquet", index=False)
    eventos.to_parquet(DATA_PROC / "eventos_eclesiasticos.parquet", index=False)

    print(f"   ✓ feature_isbn.parquet         ({len(feat_isbn):>7,} ISBNs)")
    print(f"   ✓ ventas_mensual_isbn.parquet  ({len(serie):>7,} filas)")
    print(f"   ✓ feature_cliente.parquet      ({len(feat_cli):>7,} clientes)")
    print(f"   ✓ feature_canal.parquet        ({len(feat_canal):>7,} canal-listas)")
    print(f"   ✓ eventos_eclesiasticos.parquet({len(eventos):>7,} eventos)")
    print(f"\n⏱  Fin: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # Resumen rápido
    print("\n📋 RESUMEN DEL FEATURE STORE:")
    print(f"   BIBLIAS:                {(feat_isbn['clase']=='BIBLIAS').sum():,} ISBNs")
    print(f"   ISBNs ACTIVOS:          {(feat_isbn['estado']=='ACTIVO').sum():,}")
    print(f"   ISBNs migran a CLARIDAD:{feat_isbn['migra_a_claridad'].sum():,}")
    print(f"   Color femenino:         {(feat_isbn['familia_genero']=='femenino').sum():,}")
    print(f"   Color masculino:        {(feat_isbn['familia_genero']=='masculino').sum():,}")
    print(f"   Color neutro:           {(feat_isbn['familia_genero']=='neutro').sum():,}")
    print(f"   Color juvenil:          {(feat_isbn['familia_genero']=='juvenil').sum():,}")
    print(f"   Sin color detectado:    {(feat_isbn['familia_genero']=='no_clasificado').sum():,}")

    return feat_isbn, serie, feat_cli, feat_canal, eventos


if __name__ == "__main__":
    main()
