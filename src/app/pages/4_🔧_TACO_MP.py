"""
Página 4 — Corrección y exploración de TACO MP
==============================================
Mejoras v3.4:
- Tabla de modificaciones realizadas con botón "Revertir" individual o múltiple
- Carga masiva de correcciones vía CSV
- Lista de ISBNs ya corregidos visible al final
- Las correcciones se persisten en data/state/correcciones_taco_mp.json
  y se reflejan en TODAS las demás páginas + descarga CSV/JSON
"""
from __future__ import annotations
import sys
import json
import io
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)
CORRECCIONES_PATH = DATA_STATE / "correcciones_taco_mp.json"

st.set_page_config(page_title="Corrección TACO MP", page_icon="🔧", layout="wide")
st.title("🔧 Corrección y exploración de TACO MP")
st.markdown("""
Filtra por TACO MP, ISBN, año o mes para revisar la distribución de demanda
y reasignar TACOs marcados como **POSIBLE IMPORTADO** que en realidad
deberían ser producidos en planta. Las correcciones se aplican
automáticamente en las demás páginas y en el CSV de demanda final.
""")


@st.cache_data(max_entries=1)
def cargar():
    isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")
    serie = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    serie["anio"] = serie["mes"].dt.year
    serie["mes_num"] = serie["mes"].dt.month
    return isbn, serie


def cargar_correcciones() -> dict:
    if not CORRECCIONES_PATH.exists():
        return {}
    try:
        with open(CORRECCIONES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def guardar_correcciones(correcciones: dict):
    with open(CORRECCIONES_PATH, "w", encoding="utf-8") as f:
        json.dump(correcciones, f, indent=2, ensure_ascii=False, default=str)
    try:
        from core import cloud_storage as _cloud
        if _cloud.nube_activa():
            _cloud.subir_archivo(CORRECCIONES_PATH, "correcciones_taco_mp.json", subcarpeta="state")
    except Exception:
        pass


# Defensa en profundidad: si faltan los artefactos (ej. nube antes de
# hidratar), degradar con gracia en vez de tronar.
if not (DATA_PROC / "feature_isbn.parquet").exists():
    st.title("🔧 Corrección de TACO MP")
    st.warning(
        "⚠️ Aún no están disponibles los datos del feature store. "
        "En la nube, espera a que termine la sincronización con Supabase "
        "(panel ☁️ en el Home). En escritorio, corre "
        "`uv run python src/models/run_all.py` para generarlos."
    )
    st.stop()

isbn, serie = cargar()
correcciones = cargar_correcciones()

isbn["taco_mp_corregido"] = isbn["isbn"].map(
    lambda x: correcciones.get(x, {}).get("taco_mp_nuevo", "")
)
isbn["taco_mp_efectivo"] = isbn.apply(
    lambda r: r["taco_mp_corregido"] if r["taco_mp_corregido"] else r["taco_mp"],
    axis=1,
)

# Filtros globales (v3.12) — sidebar compartido con todas las páginas
from app.filtros_globales import (
    render_filtros_globales, aplicar_filtros_isbn,
    aplicar_filtros_temporal_serie, label_periodo_actual,
)
filtros_g = render_filtros_globales(isbn, serie)
isbn_global = aplicar_filtros_isbn(isbn, filtros_g)
label_periodo_global = label_periodo_actual(filtros_g)

# Etiquetas de descripción de TACO (catálogo MP)
try:
    from utils.catalogo_mp import labels_de_lista as _labels_mp_taco_fn, descripcion_taco_mp
    _label_taco_full = lambda c: _labels_mp_taco_fn([c]).get(c, str(c))
except Exception:
    descripcion_taco_mp = lambda c, fallback="": fallback or str(c)
    _label_taco_full = lambda c: str(c)

# =========================================================================
# FILTROS
# =========================================================================
st.divider()
st.subheader("Filtros")
st.caption(
    "La **clase, categoría, género, mercado y el período temporal** se controlan "
    "desde el sidebar (filtros globales, afectan toda la app). Aquí solo eliges "
    "qué TACO(s) MP analizar."
)

tacos_disponibles = sorted(isbn_global["taco_mp_efectivo"].dropna().unique())
try:
    from utils.catalogo_mp import labels_de_lista as _labels_mp
    _labels_mp_taco = _labels_mp(tacos_disponibles)
except Exception:
    _labels_mp_taco = {t: str(t) for t in tacos_disponibles}
taco_sel = st.multiselect(
    "TACO MP (multi-select)",
    options=tacos_disponibles,
    default=[],
    format_func=lambda c: _labels_mp_taco.get(c, str(c))[:90],
    placeholder="Vacío = todos · busca por código o descripción",
)

# Período viene de los filtros globales del sidebar
anios_sel = filtros_g["anios"]
meses_sel = filtros_g["meses"]
meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Filtrado: parte del universo global ya filtrado por dashboard
isbn_filt = isbn_global.copy()
if taco_sel:
    isbn_filt = isbn_filt[isbn_filt["taco_mp_efectivo"].isin(taco_sel)]

serie_filt = serie[serie["isbn"].isin(isbn_filt["isbn"])].copy()
if anios_sel:
    serie_filt = serie_filt[serie_filt["anio"].isin(anios_sel)]
if meses_sel and len(meses_sel) < 12:
    serie_filt = serie_filt[serie_filt["mes_num"].isin(meses_sel)]

# Métricas
st.divider()
st.subheader("Resumen del filtro")
col1, col2, col3, col4, col5 = st.columns(5)
n_isbns = isbn_filt["isbn"].nunique()
n_tacos = isbn_filt["taco_mp_efectivo"].nunique()
unidades = serie_filt["unidades"].sum()
valor = serie_filt["valor"].sum()
n_importados = (isbn_filt["taco_mp_efectivo"].str.upper().str.contains(
    "POSIBLE IMPORTADO", na=False)).sum()

col1.metric("ISBNs", f"{n_isbns:,}")
col2.metric("TACOs únicos", f"{n_tacos:,}")
col3.metric("Unidades", f"{unidades:,.0f}")
col4.metric("Valor", f"${valor/1e9:.2f} B")
col5.metric("POSIBLE IMPORTADO", f"{n_importados}")

# =========================================================================
# TABLA AGREGADA POR TACO
# =========================================================================
st.divider()
st.subheader("Vista agregada por TACO MP")
st.caption(f"📅 Período del filtro temporal (sidebar): **{label_periodo_global}**")

agregado_taco = (
    serie_filt.merge(isbn_filt[["isbn", "taco_mp_efectivo"]], on="isbn")
    .groupby("taco_mp_efectivo")
    .agg(unidades=("unidades", "sum"), valor=("valor", "sum"),
         n_meses=("mes", "nunique"))
    .reset_index()
)
n_isbns_taco = isbn_filt.groupby("taco_mp_efectivo")["isbn"].nunique().reset_index()
n_isbns_taco.columns = ["taco_mp_efectivo", "n_isbns"]
agregado_taco = agregado_taco.merge(n_isbns_taco, on="taco_mp_efectivo", how="left")
agregado_taco["es_importado"] = agregado_taco["taco_mp_efectivo"].str.upper().str.contains(
    "POSIBLE IMPORTADO", na=False
)
# v3.12: descripción del código de cada TACO MP desde el catálogo
agregado_taco["descripcion_mp"] = agregado_taco["taco_mp_efectivo"].map(
    lambda c: descripcion_taco_mp(c, fallback="")
)
agregado_taco = agregado_taco.sort_values("unidades", ascending=False)

agregado_view = agregado_taco.copy()
agregado_view["unidades"] = agregado_view["unidades"].apply(lambda x: f"{int(x):,}")
agregado_view["valor"] = agregado_view["valor"].apply(lambda x: f"${int(x):,}")
agregado_view["es_importado"] = agregado_view["es_importado"].map({True: "🌐 SÍ", False: ""})
# Reordenar: código, descripción, luego métricas
agregado_view = agregado_view[["taco_mp_efectivo", "descripcion_mp", "unidades",
                                 "valor", "n_meses", "n_isbns", "es_importado"]]
agregado_view.columns = ["TACO MP", "Descripción", "Unidades", "Valor",
                          "Meses con venta", "# ISBNs", "Importado?"]
st.dataframe(agregado_view, use_container_width=True, hide_index=True, height=380)

# =========================================================================
# DETALLE ISBN A ISBN + EDITOR INDIVIDUAL
# =========================================================================
st.divider()
st.subheader("Detalle por ISBN")

col_search, col_select = st.columns([1, 3])
with col_search:
    busqueda_isbn = st.text_input(
        "🔎 Buscar ISBN/descripción",
        placeholder="ej: mujer virtuosa, 9789587...",
    )
with col_select:
    isbns_filtrados = isbn_filt.copy()
    if busqueda_isbn:
        mask = (
            isbns_filtrados["isbn"].str.contains(busqueda_isbn, case=False, na=False)
            | isbns_filtrados["descripcion"].fillna("").str.contains(busqueda_isbn, case=False, na=False)
        )
        isbns_filtrados = isbns_filtrados[mask]

    isbns_sel = st.multiselect(
        f"Selecciona ISBN(s) ({len(isbns_filtrados)} disponibles)",
        options=isbns_filtrados["isbn"].tolist(),
        format_func=lambda x: f"{x} — {isbn[isbn['isbn']==x]['descripcion'].iloc[0][:60]}",
        placeholder="Selecciona uno o más ISBNs",
    )

if isbns_sel:
    detalle = isbn_filt[isbn_filt["isbn"].isin(isbns_sel)][[
        "isbn", "descripcion", "clase", "taco_mp", "taco_mp_efectivo",
        "color_dominante", "familia_genero", "categoria_precio",
        "mercado_principal", "estado", "unidades_total", "valor_total",
        "primera_venta", "ultima_venta"
    ]].copy()
    detalle["unidades_total"] = detalle["unidades_total"].apply(lambda x: f"{int(x):,}")
    detalle["valor_total"] = detalle["valor_total"].apply(lambda x: f"${int(x):,}")
    detalle.columns = ["ISBN", "Descripción", "Clase", "TACO MP original",
                        "TACO MP efectivo", "Color", "Género", "Categoría precio",
                        "Mercado", "Estado", "Unidades totales", "Valor total",
                        "Primera venta", "Última venta"]
    st.dataframe(detalle, use_container_width=True, hide_index=True)

    serie_isbn_sel = serie[serie["isbn"].isin(isbns_sel)]
    if len(serie_isbn_sel):
        st.markdown("**Serie mensual agregada de los ISBNs seleccionados**")
        agg = serie_isbn_sel.groupby("mes")["unidades"].sum().reset_index()
        fig = px.bar(agg, x="mes", y="unidades",
                     labels={"unidades": "Unidades", "mes": "Mes"},
                     color_discrete_sequence=["#1f4e79"])
        fig.update_layout(height=320, margin=dict(t=20, b=30))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# EDITOR INDIVIDUAL
# =========================================================================
st.divider()
st.subheader("Editor de correcciones — corrección individual")
st.caption(
    "Asigna manualmente el TACO MP correcto a un ISBN. La corrección se persiste y "
    "se aplica al regenerar el feature store y en las demás páginas del tablero."
)

if isbns_sel:
    isbn_a_corregir = st.selectbox(
        "ISBN a corregir",
        options=isbns_sel,
        format_func=lambda x: f"{x} — TACO actual: {isbn[isbn['isbn']==x]['taco_mp_efectivo'].iloc[0]}",
    )
    col1, col2 = st.columns(2)
    with col1:
        opciones_tacos = sorted(isbn["taco_mp"].dropna().unique())
        nuevo_taco = st.selectbox(
            "Asignar a TACO MP",
            options=[""] + opciones_tacos,
            help="Selecciona el TACO MP nacional correcto. Vacío = sin corrección.",
        )
    with col2:
        nota = st.text_input("Nota (opcional)", placeholder="Razón de la corrección...")

    col_apply, col_clear = st.columns(2)
    with col_apply:
        if st.button("✅ Aplicar corrección", type="primary", use_container_width=True,
                      disabled=not nuevo_taco):
            correcciones[isbn_a_corregir] = {
                "taco_mp_nuevo": nuevo_taco,
                "taco_mp_original": isbn[isbn["isbn"]==isbn_a_corregir]["taco_mp"].iloc[0],
                "nota": nota,
                "fecha": datetime.now().isoformat(),
            }
            guardar_correcciones(correcciones)
            st.success(f"✓ Corrección guardada para {isbn_a_corregir}")
            st.cache_data.clear()
            st.rerun()
    with col_clear:
        if isbn_a_corregir in correcciones:
            if st.button("🗑️ Eliminar corrección de este ISBN", use_container_width=True):
                del correcciones[isbn_a_corregir]
                guardar_correcciones(correcciones)
                st.success("✓ Corrección eliminada")
                st.cache_data.clear()
                st.rerun()
else:
    st.info("Selecciona uno o más ISBNs arriba para corregir su TACO MP.")

# =========================================================================
# CARGA MASIVA POR CSV
# =========================================================================
st.divider()
st.subheader("Corrección masiva por CSV")
st.caption(
    "Sube un CSV con columnas mínimas **`isbn`** y **`taco_mp_nuevo`** (opcional: `nota`). "
    "Las correcciones se aplican en bloque. ISBNs que ya tengan corrección se actualizan."
)

# Plantilla descargable
plantilla = pd.DataFrame({
    "isbn": ["9789587450040", "9789587457285"],
    "taco_mp_nuevo": ["SE1103008010", "MISIONERAS RVR"],
    "nota": ["Reasignación a TACO de planta", "Misionera RVR confirmada"],
})
plantilla_csv = plantilla.to_csv(index=False).encode("utf-8")

col1, col2 = st.columns([1, 3])
with col1:
    st.download_button(
        "📄 Descargar plantilla",
        data=plantilla_csv,
        file_name="plantilla_correcciones_taco_mp.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col2:
    archivo_csv = st.file_uploader(
        "Sube CSV con correcciones (UTF-8)", type=["csv"], key="upload_corr_csv",
    )

if archivo_csv is not None:
    try:
        df_csv = pd.read_csv(archivo_csv, dtype={"isbn": str})
        # Validar columnas
        cols_req = {"isbn", "taco_mp_nuevo"}
        if not cols_req.issubset(df_csv.columns):
            st.error(f"❌ Faltan columnas requeridas: {cols_req - set(df_csv.columns)}")
        else:
            st.info(f"📋 Archivo leído: {len(df_csv)} filas")
            st.dataframe(df_csv.head(20), use_container_width=True, hide_index=True)

            # Validar ISBNs existentes
            isbns_validos = set(isbn["isbn"].astype(str).tolist())
            df_csv["isbn"] = df_csv["isbn"].astype(str)
            df_validas = df_csv[df_csv["isbn"].isin(isbns_validos)]
            df_invalidas = df_csv[~df_csv["isbn"].isin(isbns_validos)]

            if len(df_invalidas):
                st.warning(f"⚠️ {len(df_invalidas)} ISBN(s) no existen en el catálogo y serán ignorados.")
                st.dataframe(df_invalidas.head(10), use_container_width=True, hide_index=True)

            if st.button(
                f"✅ Aplicar {len(df_validas)} correcciones",
                type="primary", use_container_width=True,
                disabled=(len(df_validas) == 0),
            ):
                aplicadas = 0
                for _, fila in df_validas.iterrows():
                    isbn_id = str(fila["isbn"])
                    correcciones[isbn_id] = {
                        "taco_mp_nuevo": str(fila["taco_mp_nuevo"]),
                        "taco_mp_original": isbn[isbn["isbn"]==isbn_id]["taco_mp"].iloc[0] if isbn_id in isbns_validos else "",
                        "nota": str(fila.get("nota", "")) if pd.notna(fila.get("nota", "")) else "",
                        "fecha": datetime.now().isoformat(),
                        "origen": "carga_masiva_csv",
                    }
                    aplicadas += 1
                guardar_correcciones(correcciones)
                st.success(f"✓ Aplicadas {aplicadas} correcciones desde CSV.")
                st.cache_data.clear()
                st.rerun()
    except Exception as e:
        st.error(f"❌ Error al procesar CSV: {e}")

# =========================================================================
# HISTORIAL DE MODIFICACIONES (con reversión)
# =========================================================================
st.divider()
st.subheader(f"📋 Historial de correcciones vigentes ({len(correcciones)})")
st.caption(
    "Marca con ✓ las correcciones que quieras **revertir** (eliminar). "
    "También puedes exportar el historial completo. Las correcciones se reflejan en "
    "el Explorador de pronóstico, en el CSV final y en el JSON de Guardar/Cargar."
)

if not correcciones:
    st.info("Aún no hay correcciones aplicadas.")
else:
    historial = []
    for isbn_id, c in correcciones.items():
        desc = isbn[isbn["isbn"] == isbn_id]["descripcion"].iloc[0] if len(isbn[isbn["isbn"] == isbn_id]) else "?"
        taco_orig = c.get("taco_mp_original") or (isbn[isbn["isbn"] == isbn_id]["taco_mp"].iloc[0] if len(isbn[isbn["isbn"] == isbn_id]) else "?")
        historial.append({
            "Revertir": False,
            "ISBN": isbn_id,
            "Descripción": str(desc)[:60],
            "TACO original": str(taco_orig)[:30],
            "TACO corregido": c.get("taco_mp_nuevo", ""),
            "Nota": c.get("nota", "")[:60],
            "Fecha": c.get("fecha", "")[:10],
            "Origen": c.get("origen", "manual"),
        })
    df_hist = pd.DataFrame(historial).sort_values("Fecha", ascending=False)

    edited_hist = st.data_editor(
        df_hist,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "Revertir": st.column_config.CheckboxColumn("✓", default=False, width="small"),
        },
        disabled=[c for c in df_hist.columns if c != "Revertir"],
        key="editor_hist_taco",
    )
    isbns_revertir = edited_hist[edited_hist["Revertir"]]["ISBN"].tolist()

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button(
            f"↩️ Revertir {len(isbns_revertir)} marcada(s)",
            disabled=(len(isbns_revertir) == 0),
            use_container_width=True,
            type="primary" if len(isbns_revertir) > 0 else "secondary",
        ):
            for isbn_id in isbns_revertir:
                if isbn_id in correcciones:
                    del correcciones[isbn_id]
            guardar_correcciones(correcciones)
            st.success(f"✓ Revertidas {len(isbns_revertir)} correcciones.")
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Revertir TODAS", use_container_width=True):
            if st.session_state.get("confirma_revertir_todas"):
                guardar_correcciones({})
                st.session_state["confirma_revertir_todas"] = False
                st.success("✓ Todas las correcciones revertidas.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.session_state["confirma_revertir_todas"] = True
                st.warning("⚠️ Vuelve a hacer clic para confirmar revertir TODAS las correcciones.")
    with col3:
        if correcciones:
            csv_hist = df_hist.drop(columns=["Revertir"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Exportar historial a CSV",
                data=csv_hist,
                file_name=f"historial_correcciones_taco_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )
