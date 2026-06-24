"""
Página 5 — Descargar demanda 2027-2030
========================================
Genera el CSV final con todos los componentes del pronóstico:
- Proyecciones Prophet por ISBN (catálogo actual)
- Decay de ISBNs declinantes
- Novedades aprobadas (manuales + sugerencias automáticas aprobadas)
- Migración CLARIDAD aplicada (TACOs viejos se cortan en mes de migración)

El CSV se descarga con todas las columnas relevantes para planeación.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
from io import BytesIO
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models import novedades_store, overrides_store

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"

st.set_page_config(page_title="Descargar demanda", page_icon="📥", layout="wide")
st.title("📥 Descargar demanda 2027-2030")
st.markdown(
    "Genera el CSV final consolidado para planeación operativa. "
    "Incluye TODAS las fuentes: proyecciones Prophet del catálogo, decay de "
    "declinantes, novedades aprobadas (manuales + sugerencias automáticas), "
    "y migración CLARIDAD aplicada."
)

proy_path = DATA_STATE / "proyecciones_prophet.parquet"
isbn_path = DATA_PROC / "feature_isbn.parquet"

if not proy_path.exists() or not isbn_path.exists():
    st.warning(
        "⚠️ Para generar el CSV necesitas correr primero:\n\n"
        "```bash\nuv run python src/models/run_all.py --with-forecasts\n```"
    )
    st.stop()


# =========================================================================
# CARGA DE DATOS
# =========================================================================
@st.cache_data(max_entries=1)
def cargar_base():
    proy = pd.read_parquet(proy_path)
    isbn = pd.read_parquet(isbn_path)
    return proy, isbn


proy, isbn = cargar_base()
novedades = novedades_store.cargar()

# Filtros globales (v3.12) — sidebar consistente entre páginas.
try:
    _serie_fg = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    from app.filtros_globales import render_filtros_globales
    render_filtros_globales(isbn, _serie_fg)
except Exception:
    pass

# Aplicar overrides de proyección (escala y ciclo por categoría / ISBN)
# Esto asegura que el CSV final refleje los ajustes manuales del Explorador.
overrides_actuales = overrides_store.cargar()
proy = overrides_store.aplicar_overrides_a_proyecciones(proy, isbn, overrides_actuales)
n_ov_cat = len(overrides_actuales.get("categorias", {}))
n_ov_isbn = len(overrides_actuales.get("isbns", {}))
if n_ov_cat > 0 or n_ov_isbn > 0:
    st.info(
        f"📌 Aplicando overrides activos: **{n_ov_cat}** categorías, "
        f"**{n_ov_isbn}** ISBNs específicos. El CSV reflejará estos ajustes."
    )

# =========================================================================
# OPCIONES DE EXPORTACIÓN
# =========================================================================
st.divider()
st.subheader("Opciones de exportación")

c1, c2, c3 = st.columns(3)
with c1:
    año_desde = st.selectbox(
        "Año desde", options=[2026, 2027, 2028, 2029, 2030],
        index=1,  # default 2027
    )
with c2:
    año_hasta = st.selectbox(
        "Año hasta", options=[2026, 2027, 2028, 2029, 2030],
        index=4,  # default 2030
    )
with c3:
    incluir_bandas = st.checkbox(
        "Incluir bandas p10/p90", value=True,
        help="Incluye las columnas de incertidumbre p10 (pesimista) y p90 (optimista)."
    )

c4, c5, c6 = st.columns(3)
with c4:
    incluir_novedades = st.checkbox(
        "Incluir novedades aprobadas", value=True,
        help="Suma las novedades aprobadas (manuales + automáticas) al CSV."
    )
with c5:
    incluir_solo_biblias = st.checkbox(
        "Solo BIBLIAS", value=False,
        help="Si se desmarca, incluye también LITERATURA, MISCELÁNEOS, etc."
    )
with c6:
    formato = st.selectbox(
        "Formato del CSV",
        options=["Largo (1 fila por ISBN-mes)", "Ancho (1 fila por ISBN, columnas por mes)"],
        help="Largo: fácil para análisis; Ancho: fácil para planeación operativa."
    )

# =========================================================================
# CONSTRUIR EL CSV
# =========================================================================
st.divider()


def construir_csv() -> pd.DataFrame:
    """Construye el dataframe final combinando todas las fuentes."""
    filas = []

    # Cargar correcciones TACO MP para aplicar (si existen)
    corr_path = DATA_STATE / "correcciones_taco_mp.json"
    correcciones_taco = {}
    if corr_path.exists():
        try:
            import json
            with open(corr_path, "r", encoding="utf-8") as f:
                correcciones_taco = json.load(f)
        except Exception:
            correcciones_taco = {}

    # 1. Proyecciones del catálogo
    p = proy.copy()
    p["anio"] = pd.to_datetime(p["ds"]).dt.year
    p = p[(p["anio"] >= año_desde) & (p["anio"] <= año_hasta)]

    # Filtrar clase si solo BIBLIAS
    if incluir_solo_biblias:
        biblias_ids = isbn[isbn["clase"] == "BIBLIAS"]["isbn"].tolist()
        p = p[p["isbn"].isin(biblias_ids)]

    # Enriquecer con metadata del ISBN
    meta_cols = ["isbn", "descripcion", "clase", "familia_genero",
                  "mercado_principal", "categoria_precio", "estado", "version"]
    cols_existentes = [c for c in meta_cols if c in isbn.columns]
    p_full = p.merge(isbn[cols_existentes], on="isbn", how="left")

    for _, row in p_full.iterrows():
        # Aplicar corrección TACO MP si existe
        isbn_id = row["isbn"]
        taco_efectivo = row.get("taco_mp_proyectado", row.get("taco_mp", ""))
        if isbn_id in correcciones_taco:
            taco_efectivo = correcciones_taco[isbn_id].get("taco_mp_nuevo", taco_efectivo)
        fila = {
            "isbn": isbn_id,
            "descripcion": row.get("descripcion", ""),
            "mes": row["ds"].strftime("%Y-%m"),
            "anio": int(row["anio"]),
            "mes_num": int(row["ds"].month),
            "prediccion": round(row["yhat"], 1),
            "clase": row.get("clase", ""),
            "familia_genero": row.get("familia_genero", ""),
            "mercado": row.get("mercado_principal", ""),
            "categoria_precio": row.get("categoria_precio", ""),
            "estado": row.get("estado", ""),
            "version": row.get("version", ""),
            "taco_mp_proyectado": taco_efectivo,
            "fuente": row["fuente"],
            "tipo_origen": "catalogo",
        }
        if incluir_bandas:
            fila["p10"] = round(row["yhat_lower"], 1)
            fila["p90"] = round(row["yhat_upper"], 1)
        filas.append(fila)

    # 2. Novedades aprobadas
    if incluir_novedades:
        for nov in novedades:
            if nov.get("estado") != "aprobado":
                continue
            origen = nov.get("origen", "manual")
            fuera = nov.get("fuera_capacidad", False)
            # Tipo_origen detallado distinguiendo origen y capacidad:
            #   novedad_simulada                      → del simulador, dentro de cap
            #   novedad_simulada_fuera_capacidad      → del simulador, fuera de cap
            #   sugerencia_aprobada                   → del sugerido auto, dentro de cap
            #   sugerencia_aprobada_fuera_capacidad   → del sugerido auto, fuera de cap
            if origen == "sugerencia_automatica":
                tipo_origen = "sugerencia_aprobada_fuera_capacidad" if fuera else "sugerencia_aprobada"
            else:
                tipo_origen = "novedad_simulada_fuera_capacidad" if fuera else "novedad_simulada"

            features = nov.get("features", {})
            for entrada in nov.get("curva_mensual", []):
                if isinstance(entrada, dict):
                    try:
                        ds = pd.Timestamp(entrada["ds"])
                    except (ValueError, TypeError, KeyError):
                        continue
                    anio_e = ds.year
                    if anio_e < año_desde or anio_e > año_hasta:
                        continue
                    pred = entrada.get("prediccion", 0)
                    fila = {
                        "isbn": f"NOV_{nov.get('id', '')[:18]}",
                        "descripcion": nov.get("nombre", ""),
                        "mes": ds.strftime("%Y-%m"),
                        "anio": int(anio_e),
                        "mes_num": int(ds.month),
                        "prediccion": round(pred, 1),
                        "clase": "BIBLIAS",
                        "familia_genero": features.get("familia_genero", ""),
                        "mercado": features.get("mercado_principal", ""),
                        "categoria_precio": "",  # se infiere del precio si se necesita
                        "estado": "NOVEDAD",
                        "version": features.get("version", "RVR"),
                        "taco_mp_proyectado": nov.get("taco_destino", ""),
                        "fuente": "novedad",
                        "tipo_origen": tipo_origen,
                    }
                    if incluir_bandas:
                        fila["p10"] = round(entrada.get("p10", 0), 1)
                        fila["p90"] = round(entrada.get("p90", 0), 1)
                    filas.append(fila)

    return pd.DataFrame(filas)


with st.spinner("Construyendo el CSV..."):
    df_csv = construir_csv()

# =========================================================================
# RESUMEN
# =========================================================================
st.subheader("Resumen del CSV generado")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas totales", f"{len(df_csv):,}")
c2.metric("ISBNs únicos", f"{df_csv['isbn'].nunique():,}")
c3.metric("Meses cubiertos", f"{df_csv['mes'].nunique()}")
c4.metric("Suma anual", f"{df_csv['prediccion'].sum():,.0f}")

# Composición por tipo_origen
st.markdown("**Composición por origen**")
comp = df_csv.groupby("tipo_origen").agg(
    filas=("isbn", "count"),
    suma=("prediccion", "sum"),
).reset_index()
comp["suma"] = comp["suma"].apply(lambda x: f"{x:,.0f}")
st.dataframe(comp, use_container_width=True, hide_index=True)

# Resumen anual
st.markdown("**Resumen anual de la demanda proyectada**")
resumen_anual = df_csv.pivot_table(
    index="anio", columns="tipo_origen",
    values="prediccion", aggfunc="sum", fill_value=0,
).round(0)
resumen_anual["TOTAL"] = resumen_anual.sum(axis=1)
for c in resumen_anual.columns:
    resumen_anual[c] = resumen_anual[c].apply(lambda x: f"{int(x):,}")
st.dataframe(resumen_anual, use_container_width=True)

# =========================================================================
# DESCARGA
# =========================================================================
st.divider()
st.subheader("Descargar archivos")

# Formato largo
if formato.startswith("Largo"):
    df_export = df_csv.sort_values(["isbn", "mes"])
else:
    # Formato ancho: pivot por mes
    cols_meta = ["isbn", "descripcion", "clase", "familia_genero", "mercado",
                  "categoria_precio", "estado", "version", "taco_mp_proyectado",
                  "fuente", "tipo_origen"]
    pivoted = df_csv.pivot_table(
        index=[c for c in cols_meta if c in df_csv.columns],
        columns="mes",
        values="prediccion",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    df_export = pivoted

# CSV bytes
csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")

# Excel bytes
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df_export.to_excel(writer, sheet_name="demanda_mensual", index=False)
    # Hoja adicional: resumen
    resumen_x = df_csv.groupby(["anio", "tipo_origen"])["prediccion"].sum().reset_index()
    resumen_x.to_excel(writer, sheet_name="resumen_anual", index=False)
excel_bytes = excel_buffer.getvalue()

stamp = datetime.now().strftime("%Y%m%d_%H%M")
nombre_base = f"sbc_demanda_{año_desde}_{año_hasta}_{stamp}"

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Descargar CSV",
        data=csv_bytes,
        file_name=f"{nombre_base}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
with c2:
    st.download_button(
        "⬇️ Descargar Excel (con resumen)",
        data=excel_bytes,
        file_name=f"{nombre_base}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# =========================================================================
# PREVIEW
# =========================================================================
st.divider()
st.subheader("Vista previa del CSV")
st.caption(f"Mostrando las primeras 200 filas. El archivo completo tiene {len(df_export):,} filas.")
st.dataframe(df_export.head(200), use_container_width=True, height=440)
