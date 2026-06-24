"""
SBC Demanda — App Streamlit
============================
Página principal: Dashboard ejecutivo.

Para correr:
    cd ~/sbc_demanda
    uv run streamlit run src/app/Home.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# =========================================================================
# CONFIGURACIÓN DE LA APP
# =========================================================================
st.set_page_config(
    page_title="SBC Demanda",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PROC = ROOT / "data" / "processed"

# =========================================================================
# NUBE (v3.17): login + hidratación desde Supabase al iniciar sesión
# =========================================================================
# Defensivo: si no hay secrets de Supabase/usuarios, todo degrada a no-op y la
# app corre 100% en local (modo escritorio).
from core import auth, cloud_storage as cloud

# 1) Login (solo se exige si hay usuarios configurados en st.secrets)
if not auth.gate():
    st.stop()

# 2) Hidratar artefactos + estado desde la nube UNA vez por sesión
if cloud.nube_activa() and not st.session_state.get("_hidratado"):
    with st.spinner("Sincronizando datos desde la nube..."):
        res = cloud.hidratar_desde_nube(ROOT, solo_si_falta=True)
    st.session_state["_hidratado"] = True
    st.session_state["_hidratacion_res"] = res

MODO_NUBE = cloud.nube_activa()

# Metas conservadoras 2027-2030 (inmutables)
METAS_BIBLIAS = {2027: 1_673_298, 2028: 2_139_351, 2029: 2_666_471, 2030: 3_243_508}


# =========================================================================
# CARGA DE DATOS (cached)
# =========================================================================
@st.cache_data(ttl=600, max_entries=1)
def cargar_feature_store():
    """Carga los parquets del feature store. Cacheado 10 min."""
    archivos = {
        "isbn": "feature_isbn.parquet",
        "serie": "ventas_mensual_isbn.parquet",
        "cliente": "feature_cliente.parquet",
        "canal": "feature_canal.parquet",
        "eventos": "eventos_eclesiasticos.parquet",
    }
    data = {}
    for k, f in archivos.items():
        path = DATA_PROC / f
        if not path.exists():
            return None, f"❌ Falta {f}. Corre primero: python src/features/build_features.py"
        data[k] = pd.read_parquet(path)
    return data, None


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.title("📖 SBC Demanda")
    st.caption("Sistema de Planeación 2027-2030")
    st.divider()
    st.markdown("### Estado del modelo")

data, err = cargar_feature_store()

if err:
    st.error(err)
    if MODO_NUBE:
        # En la nube NO se corre build_features: los artefactos vienen de
        # Supabase por hidratación. Mostrar diagnóstico en vez de mandar a
        # un paso que no aplica.
        st.markdown("### ☁️ Diagnóstico de la nube")
        ok, msg = cloud.probar_conexion()
        if ok:
            st.success(f"Conexión a Supabase: ✓ {msg}")
        else:
            st.error(f"Conexión a Supabase: ✗ {msg}")
        archivos_proc = cloud.listar("processed")
        archivos_state = cloud.listar("state")
        res = st.session_state.get("_hidratacion_res", {})
        st.write(f"**Archivos en el bucket** · processed: {len(archivos_proc)} · "
                 f"state: {len(archivos_state)}")
        if archivos_proc:
            st.caption("processed/: " + ", ".join(archivos_proc[:10]))
        st.write(f"**Hidratación al iniciar:** {res.get('processed',0)} artefactos + "
                 f"{res.get('modelos',0)} modelos + {res.get('estado',0)} estado descargados.")
        st.info(
            "**Qué hacer:**\n\n"
            "1. Si la conexión falla → revisá los *Secrets* en Streamlit "
            "(sección `[supabase]` con url, key y `bucket = \"sbc-demanda\"`).\n"
            "2. Si la conexión está OK pero el bucket está vacío → en tu Mac corré "
            "**`5-Subir a la nube`** (antes asegurate de tener los artefactos: "
            "corré **`4-Reentrenar modelo`** si `data/processed/` está vacío).\n"
            "3. Si el bucket tiene archivos pero no bajaron → puede ser un tema de "
            "políticas del bucket; avisame y ajustamos los permisos."
        )
        if st.button("🔄 Reintentar sincronización"):
            st.session_state.pop("_hidratado", None)
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("""
        **Cómo arreglar esto (escritorio):**

        Abre la terminal en la carpeta del proyecto y ejecuta:
        ```bash
        uv run python src/features/build_features.py
        ```
        Eso construye el feature store. Después recarga esta página.
        """)
    st.stop()

isbn = data["isbn"]
serie = data["serie"]
eventos = data["eventos"]

with st.sidebar:
    st.success(f"✅ {len(isbn):,} ISBNs cargados")
    st.caption(f"Última venta: {serie['mes'].max():%Y-%m}")
    st.caption(f"Eventos eclesiásticos: {len(eventos)}")
    st.divider()

# Filtros globales (v3.12): viven en el sidebar de TODAS las páginas y
# persisten entre ellas vía session_state.
from app.filtros_globales import (
    render_filtros_globales, aplicar_filtros_isbn,
    aplicar_filtros_temporal_serie, label_periodo_actual,
)

serie["anio"] = serie["mes"].dt.year
filtros_g = render_filtros_globales(isbn, serie)

clase_sel    = filtros_g["clase"]
estado_sel   = filtros_g["estado"]
mercado_sel  = filtros_g["mercado"]
anios_sel    = filtros_g["anios"]
meses_sel    = filtros_g["meses"]
meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# =========================================================================
# FILTRADO
# =========================================================================
isbn_f = aplicar_filtros_isbn(isbn, filtros_g)

# Filtrado temporal de la serie
serie_filt_temp = aplicar_filtros_temporal_serie(serie, filtros_g)
serie_filt_temp["mes_num"] = serie_filt_temp["mes"].dt.month

# Label del período aplicado
label_periodo = label_periodo_actual(filtros_g)

# =========================================================================
# HEADER
# =========================================================================
st.title("📖 Dashboard ejecutivo · SBC Demanda")
st.caption(
    f"Período: **{label_periodo}** | "
    f"Histórico {serie['mes'].min():%Y-%m} → {serie['mes'].max():%Y-%m} | "
    f"Última corrida: {pd.Timestamp.now():%Y-%m-%d %H:%M}"
)

# =========================================================================
# PANEL DE ESTADO DE LA NUBE (v3.17)
# =========================================================================
auth.logout_boton()
if MODO_NUBE:
    with st.expander("☁️ Estado de la nube (Supabase)", expanded=False):
        ok, msg = cloud.probar_conexion()
        if ok:
            st.success(f"✓ {msg}")
        else:
            st.error(f"✗ {msg}")
        res = st.session_state.get("_hidratacion_res", {})
        if res:
            st.caption(
                f"Hidratación al iniciar: {res.get('processed',0)} artefactos · "
                f"{res.get('modelos',0)} modelos · {res.get('estado',0)} archivos de estado "
                f"descargados desde la nube."
            )
        if st.button("🔄 Re-sincronizar desde la nube"):
            st.session_state.pop("_hidratado", None)
            st.cache_data.clear()
            st.rerun()

# =========================================================================
# CARGA DE DATOS FRESCOS — SIESA + NOVEDADES (v3.13)
# =========================================================================
# En ESCRITORIO: cargar SIESA + reentrenar. En NUBE: el reentrenamiento no
# corre (cálculo pesado), así que el cargador queda informativo.
from utils import ingesta_siesa as _ing

if MODO_NUBE:
    with st.expander("📥 Ventas SIESA y novedades del año (paso de escritorio)",
                     expanded=False):
        st.info(
            "🖥️ **Este paso se hace en escritorio.** La ingesta del SIESA y el "
            "reentrenamiento del modelo (Prophet + hedónico, 8-11 min) corren en "
            "tu Mac, no en la nube. Allí cargás el SIESA, reentrenás con "
            "`4-Reentrenar modelo`, y los nuevos artefactos se suben a Supabase; "
            "esta app en la nube los toma al sincronizar."
        )
        _meta_ing = _ing.cargar_meta()
        if _meta_ing:
            st.caption(
                f"Última ingesta SIESA (desde escritorio): "
                f"{_meta_ing.get('fecha_min','?')} → {_meta_ing.get('fecha_max','?')} · "
                f"{_meta_ing.get('isbns_unicos',0):,} ISBNs."
            )
else:
  with st.expander("📥 Cargar ventas SIESA y novedades del año (alimentar ejecutadas 2026+)",
                 expanded=False):
    _meta_ing = _ing.cargar_meta()
    if _meta_ing:
        st.success(
            f"✓ Ventas SIESA cargadas el {_meta_ing.get('actualizado','?')[:16]} · "
            f"{_meta_ing.get('filas_finales',0):,} líneas desde {_meta_ing.get('anio_corte','?')} "
            f"({_meta_ing.get('fecha_min','?')} → {_meta_ing.get('fecha_max','?')}) · "
            f"{_meta_ing.get('isbns_unicos',0):,} ISBNs ({_meta_ing.get('isbns_nuevos',0)} nuevos) · "
            f"{_meta_ing.get('unidades_total',0):,} unidades."
        )
    else:
        st.info(
            "Aún no has cargado el SIESA en la app de demanda. El histórico actual "
            "llega hasta " + f"{serie['mes'].max():%Y-%m}. "
            "Carga el SIESA para alimentar las ventas ejecutadas del año en curso."
        )

    st.markdown(
        "**¿Cómo funciona?** El archivo SIESA trae 2025 y 2026, pero solo se toma "
        "**desde el año de corte en adelante** (default 2026), porque el histórico "
        "ya cubre lo anterior. Al reentrenar, ese tramo del histórico se **reemplaza** "
        "con los datos frescos del SIESA (mismo criterio que la app de ventas: solo "
        "Aprobadas, sin distribución gratuita ni servicios, cajas convertidas a unidades)."
    )

    col_ay, col_a1, col_a2 = st.columns([1, 1, 1])
    with col_ay:
        anio_corte_sel = st.number_input(
            "Año de corte", min_value=2020, max_value=2030,
            value=int(_meta_ing.get("anio_corte", 2026)) if _meta_ing else 2026,
            step=1, help="Se toma el SIESA desde este año en adelante.",
        )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("##### 1️⃣ Archivo de ventas SIESA")
        up_siesa = st.file_uploader(
            "VENTAS_SIESA.xlsx", type=["xlsx"], key="up_siesa",
            help="El export de ventas del ERP (formato V2, 47 columnas).",
        )
        if up_siesa is not None:
            if st.button("⚙️ Procesar e integrar SIESA", type="primary",
                         use_container_width=True, key="btn_proc_siesa"):
                with st.spinner("Procesando SIESA con la política de ingreso..."):
                    try:
                        mapa = _ing.mapa_taco_clase_desde_feature()
                        df_ej, res = _ing.procesar_siesa(
                            up_siesa, anio_corte=int(anio_corte_sel),
                            mapa_taco_clase=mapa,
                        )
                        _ing.persistir_ejecutadas(df_ej, res)
                        st.cache_data.clear()
                        st.success(
                            f"✓ Procesadas **{res['filas_finales']:,} líneas** desde "
                            f"{res['anio_corte']} ({res['fecha_min']} → {res['fecha_max']}). "
                            f"{res['isbns_unicos']:,} ISBNs, {res['isbns_nuevos']} nuevos, "
                            f"{res['unidades_total']:,} unidades, "
                            f"${res['valor_total']/1e9:.2f}B."
                        )
                        st.warning(
                            "📌 Las ventas quedaron guardadas. Para que entren al "
                            "modelo y las proyecciones, **reentrena** con "
                            "`4-Reentrenar modelo` o `python src/models/run_all.py`."
                        )
                    except Exception as e:
                        st.error(f"No se pudo procesar el SIESA: {e}")

    with col_s2:
        st.markdown("##### 2️⃣ Plantilla de novedades del año")
        up_nov = st.file_uploader(
            "plantilla_novedades_AAAA.xlsx", type=["xlsx"], key="up_nov",
            help="ISBN × fecha de lanzamiento de las novedades del año.",
        )
        _cat_nov = _ing.cargar_novedades_catalogo()
        if _cat_nov is not None and len(_cat_nov):
            st.caption(f"✓ Catálogo de novedades cargado: {len(_cat_nov)} ISBNs.")
        if up_nov is not None:
            if st.button("⚙️ Cargar catálogo de novedades", type="primary",
                         use_container_width=True, key="btn_proc_nov"):
                try:
                    df_nov, resn = _ing.procesar_novedades(up_nov)
                    _ing.persistir_novedades(df_nov)
                    st.cache_data.clear()
                    st.success(
                        f"✓ Cargadas {resn['n_novedades']} novedades "
                        f"({resn['isbns_unicos']} ISBNs, "
                        f"{resn['n_personalizaciones']} personalizaciones) "
                        f"de {', '.join(map(str, resn['anios']))}."
                    )
                except Exception as e:
                    st.error(f"No se pudo procesar la plantilla: {e}")

# =========================================================================
# KPIs PRINCIPALES
# =========================================================================
st.divider()
# Aplicar filtros del ISBN sobre la serie filtrada temporalmente
serie_filt = serie_filt_temp[serie_filt_temp["isbn"].isin(isbn_f["isbn"])]
serie_filt = serie_filt.merge(isbn[["isbn", "clase"]], on="isbn", how="left")

col1, col2, col3, col4, col5 = st.columns(5)
unidades_total = serie_filt["unidades"].sum()
valor_total = serie_filt["valor"].sum()
n_isbns = serie_filt["isbn"].nunique() if len(serie_filt) else 0
n_isbns_activos_filtro = (isbn_f["estado"] == "ACTIVO").sum()
# Share internacional sobre el filtro
unid_int = serie_filt["unidades"] * serie_filt["share_internacional_mes"] if "share_internacional_mes" in serie_filt.columns else 0
share_int = (
    unid_int.sum() / max(unidades_total, 1) * 100 if len(serie_filt) else 0
)
col1.metric("ISBNs con venta", f"{n_isbns:,}")
col2.metric("ISBNs ACTIVOS (filtro)", f"{n_isbns_activos_filtro:,}")
col3.metric("Unidades del período", f"{unidades_total:,.0f}")
col4.metric("Valor del período", f"${valor_total/1e9:.2f} B")
col5.metric("% Internacional", f"{share_int:.1f}%")

# =========================================================================
# BIBLIAS POR AÑO vs METAS
# =========================================================================
st.divider()
st.subheader("BIBLIAS por año vs metas conservadoras")

isbn_b = isbn[isbn["clase"] == "BIBLIAS"]
serie_b = serie[serie["isbn"].isin(isbn_b["isbn"])].copy()
serie_b["anio"] = serie_b["mes"].dt.year

biblias_anio = serie_b.groupby("anio")["unidades"].sum().reset_index()
biblias_anio.columns = ["anio", "unidades"]

# Anualizar 2026 si está incompleto
ultimo_mes_anio = serie_b[serie_b["anio"] == 2026]["mes"].max()
if ultimo_mes_anio is not pd.NaT and ultimo_mes_anio.month < 12:
    n_meses = ultimo_mes_anio.month
    val_2026 = biblias_anio.loc[biblias_anio["anio"] == 2026, "unidades"].iloc[0]
    biblias_anio.loc[biblias_anio["anio"] == 2026, "unidades"] = val_2026 * 12 / n_meses
    biblias_anio.loc[biblias_anio["anio"] == 2026, "anio_label"] = (
        f"2026 (anualizado, base {n_meses}m)"
    )

# Agregar metas
metas_df = pd.DataFrame({
    "anio": list(METAS_BIBLIAS.keys()),
    "unidades": list(METAS_BIBLIAS.values()),
})
metas_df["tipo"] = "Meta"
biblias_anio["tipo"] = "Histórico"

fig = go.Figure()
fig.add_trace(go.Bar(
    x=biblias_anio["anio"], y=biblias_anio["unidades"],
    name="Histórico",
    marker_color="#1f4e79",
    text=biblias_anio["unidades"].apply(lambda x: f"{x:,.0f}"),
    textposition="outside",
))
fig.add_trace(go.Bar(
    x=metas_df["anio"], y=metas_df["unidades"],
    name="Meta conservadora",
    marker_color="#d97706",
    text=metas_df["unidades"].apply(lambda x: f"{x:,.0f}"),
    textposition="outside",
))
fig.update_layout(
    height=420,
    barmode="group",
    yaxis_title="Unidades de BIBLIAS",
    xaxis_title="Año",
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=30, b=30),
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# DISTRIBUCIÓN POR FAMILIA DE GÉNERO Y CATEGORÍA DE PRECIO
# =========================================================================
st.divider()
col1, col2 = st.columns(2)

# Calcular agregados con filtros temporales aplicados
agregado_isbn = (
    serie_filt.groupby("isbn")
    .agg(unidades_periodo=("unidades", "sum"), valor_periodo=("valor", "sum"))
    .reset_index()
)
isbn_agregado = isbn_f.merge(agregado_isbn, on="isbn", how="left").fillna(
    {"unidades_periodo": 0, "valor_periodo": 0}
)

with col1:
    st.subheader(f"Unidades por familia de género · {label_periodo}")
    if len(isbn_agregado):
        gen = (
            isbn_agregado.groupby("familia_genero")["unidades_periodo"]
            .sum()
            .reset_index()
        )
        gen = gen.sort_values("unidades_periodo", ascending=True)
        colores_map = {
            "femenino": "#ec4899",
            "masculino": "#1e40af",
            "juvenil": "#10b981",
            "neutro": "#a78bfa",
            "no_clasificado": "#9ca3af",
        }
        fig_g = px.bar(
            gen, x="unidades_periodo", y="familia_genero",
            orientation="h",
            color="familia_genero",
            color_discrete_map=colores_map,
            text=gen["unidades_periodo"].apply(lambda x: f"{x:,.0f}"),
        )
        fig_g.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        fig_g.update_traces(textposition="outside")
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("Detección de género por color dominante de la biblia.")

with col2:
    st.subheader(f"Unidades por categoría de precio · {label_periodo}")
    if len(isbn_agregado):
        cat = isbn_agregado[isbn_agregado["categoria_precio"] != "no_aplica"]
        if len(cat):
            cat_g = cat.groupby("categoria_precio")["unidades_periodo"].sum().reset_index()
            orden = ["economica", "semi_economica", "media", "semi_fina", "fina"]
            cat_g["categoria_precio"] = pd.Categorical(
                cat_g["categoria_precio"], categories=orden, ordered=True
            )
            cat_g = cat_g.sort_values("categoria_precio")
            fig_c = px.bar(
                cat_g, x="categoria_precio", y="unidades_periodo",
                color="categoria_precio",
                color_discrete_sequence=px.colors.sequential.Blues,
                text=cat_g["unidades_periodo"].apply(lambda x: f"{x:,.0f}"),
            )
            fig_c.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
            fig_c.update_traces(textposition="outside")
            st.plotly_chart(fig_c, use_container_width=True)
            st.caption("Quintiles sobre precio promedio.")

# =========================================================================
# TOP 20 ISBNs DEL PERÍODO (Pareto)
# =========================================================================
st.divider()
st.subheader(f"Top 20 ISBNs por unidades · {label_periodo}")
st.caption(
    "Ranking dinámico que respeta los filtros temporales. Útil para identificar "
    "el Pareto del período seleccionado (qué ISBNs concentran la mayoría de las ventas)."
)

top = (
    isbn_agregado.sort_values("unidades_periodo", ascending=False)
    .head(20)
    [["isbn", "descripcion", "clase", "taco_mp", "color_dominante",
      "familia_genero", "categoria_precio", "mercado_principal",
      "unidades_periodo", "valor_periodo", "estado"]]
).copy()

# Calcular acumulado de Pareto
total_periodo = isbn_agregado["unidades_periodo"].sum()
if total_periodo > 0:
    top["pct"] = (top["unidades_periodo"] / total_periodo * 100).round(1)
    top["pct_acum"] = top["pct"].cumsum().round(1)
else:
    top["pct"] = 0
    top["pct_acum"] = 0

top["unidades_periodo"] = top["unidades_periodo"].apply(lambda x: f"{x:,.0f}")
top["valor_periodo"] = top["valor_periodo"].apply(lambda x: f"${x:,.0f}")
top["pct"] = top["pct"].apply(lambda x: f"{x:.1f}%")
top["pct_acum"] = top["pct_acum"].apply(lambda x: f"{x:.1f}%")

st.dataframe(top, use_container_width=True, height=440, hide_index=True)

# =========================================================================
# CALENDARIO DE EVENTOS ECLESIÁSTICOS
# =========================================================================
st.divider()
st.subheader("Eventos eclesiásticos como regresores del modelo")
st.caption(
    "Estos eventos NO se tratan como anomalías — son shocks predecibles que el modelo "
    "Prophet usa como variables exógenas para anticipar los picos."
)

eventos_view = eventos.copy()
eventos_view["año"] = eventos_view["ds"].dt.year
eventos_view["mes_evento"] = eventos_view["ds"].dt.strftime("%Y-%m-%d")
eventos_view["ventana"] = (
    eventos_view["lower_window"].abs().astype(str) + "d antes / " +
    eventos_view["upper_window"].astype(str) + "d después"
)
eventos_view = eventos_view[["año", "holiday", "mes_evento", "ventana", "prior_scale"]]
eventos_view.columns = ["Año", "Evento", "Fecha pivote", "Ventana", "Peso (prior_scale)"]
eventos_view = eventos_view[eventos_view["Año"] >= 2024].sort_values(["Año", "Fecha pivote"])
st.dataframe(eventos_view, use_container_width=True, height=380, hide_index=True)

# =========================================================================
# ESTADO DEL MODELO HEDÓNICO
# =========================================================================
st.divider()
st.subheader("Estado del modelo hedónico")

modelo_path = ROOT / "data" / "state" / "modelo_hedonico.joblib"
metricas_path = ROOT / "data" / "state" / "metricas_hedonico.json"

if modelo_path.exists() and metricas_path.exists():
    import json
    with open(metricas_path) as f:
        metricas = json.load(f)
    cM1, cM2, cM3, cM4 = st.columns(4)
    cM1.metric("ISBNs entrenamiento", f"{metricas['n_train']:,}")
    cM2.metric("R² (log)", f"{metricas['r2_log']:.3f}")
    cM3.metric("MAE (log)", f"{metricas['mae_log']:.3f}")
    cM4.metric("MAE (unidades)", f"{metricas['mae_real']:,.0f}")
    st.caption(
        f"Features usadas: {len(metricas['features_usadas'])} | "
        f"Folds MAE log: {[round(s,3) for s in metricas['fold_scores']]}"
    )
    st.info("👉 Ve a la página **Simulador novedades** para usar el modelo.")
else:
    from core import cloud_storage as _cl
    if _cl.nube_activa():
        if modelo_path.exists() and not metricas_path.exists():
            st.info(
                "El modelo hedónico está cargado y el Simulador de novedades "
                "funciona. Solo faltan las métricas (R²/MAE) para mostrarlas "
                "aquí: en tu Mac corré **5-Subir a la nube** otra vez y luego "
                "**Re-sincronizar** en el panel ☁️ de arriba."
            )
        else:
            st.warning(
                "⚠️ El modelo hedónico aún no está disponible en la nube. "
                "Subilo desde tu Mac con **5-Subir a la nube** y luego "
                "**Re-sincronizar** en el panel ☁️."
            )
    else:
        st.warning(
            "⚠️ Modelo hedónico no entrenado. Corre desde terminal:\n\n"
            "```bash\nuv run python src/models/run_all.py\n```"
        )

# =========================================================================
# ESTADO DE LAS PROYECCIONES PROPHET (Sprint 3a)
# =========================================================================
st.divider()
st.subheader("Estado de las proyecciones Prophet")

proy_path = ROOT / "data" / "state" / "proyecciones_prophet.parquet"
if proy_path.exists():
    proy = pd.read_parquet(proy_path)
    n_isbns = proy["isbn"].nunique()
    fuentes = proy["fuente"].value_counts().to_dict()
    cP1, cP2, cP3 = st.columns(3)
    cP1.metric("ISBNs proyectados", f"{n_isbns:,}")
    cP2.metric("Prophet", f"{fuentes.get('prophet', 0)}")
    cP3.metric("Decay", f"{fuentes.get('decay', 0)}")
    st.info("👉 Ve a la página **Explorador de pronóstico** para ver el gap a metas y explorar las proyecciones por ISBN.")
else:
    st.warning(
        "⚠️ Proyecciones Prophet no generadas (Sprint 3a). Corre desde terminal:\n\n"
        "```bash\nuv run python src/models/run_forecasts.py\n```\n\n"
        "Toma ~8 min en Mac M1/M2/M3 (paraleliza Prophet sobre los 354 ISBNs activos)."
    )

st.divider()
st.caption(
    "💡 Tablero completo: 6 páginas operativas. **Sugerencias automáticas** te dice qué "
    "novedades lanzar para cerrar el gap. **Descargar demanda** te da el CSV final 2027-2030 "
    "consolidado con todas las fuentes."
)
