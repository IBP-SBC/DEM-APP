"""
Página 1 — Explorador de pronóstico
====================================
Renombrado de "Editor" a "Explorador" (es un visor, no editor todavía).

Mejoras v3.4:
- Curva total con líneas estilo (verde para novedades, azul brillante para total)
- Gap a metas distingue novedades DENTRO y FUERA de capacidad (morado)
- Métricas separan aprobaciones por origen (simulador vs sugeridos)

Mejoras v3.9:
- UX del filtro de clases: popover compacto en lugar de multiselect plano
  (no se "abulta la línea" cuando hay muchas clases seleccionadas)
- Filtros nuevos por VERSIÓN y CATEGORÍA DE PRECIO en los exploradores
  por ISBN y por Cliente. Filtros encadenados: las opciones de cada uno
  se restringen a las combinaciones que sí existen en los datos.
- Tabla de ESTADÍSTICOS DESCRIPTIVOS mensuales debajo de las tablas
  anuales (en ambos exploradores), con columnas para unidades y valor:
  N, total, media, mediana, mínimo, máximo, desv. estándar, varianza,
  P25, P75, rango intercuartílico, coeficiente de variación (CoV),
  asimetría (skewness).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models import novedades_store, overrides_store
from utils.dictionaries import CAPACIDAD_NOVEDADES

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"

st.set_page_config(page_title="Explorador de pronóstico", page_icon="📈", layout="wide")
st.title("📈 Explorador de pronóstico")

proy_path = DATA_STATE / "proyecciones_prophet.parquet"
anc_path = DATA_STATE / "anclaje_metas.csv"
isbn_path = DATA_PROC / "feature_isbn.parquet"
serie_path = DATA_PROC / "ventas_mensual_isbn.parquet"

if not proy_path.exists():
    st.warning(
        "⚠️  Aún no hay proyecciones generadas. Corre desde terminal:\n\n"
        "```bash\nuv run python src/models/run_forecasts.py\n```"
    )
    st.stop()

st.warning(
    "⚠️ **Alerta estratégica de capacidad**\n\n"
    f"La capacidad actual de Publicaciones es **{CAPACIDAD_NOVEDADES['skus_por_ano']} SKUs nuevos/año**. "
    "Las metas conservadoras BIBLIAS 2029-2030 implican un gap por encima de "
    "esta capacidad. **Es necesario fortalecer la capacidad del equipo de publicaciones** "
    "para alcanzar las metas 2029-2030. Más abajo verás el gap cuantitativo año por año."
)

with st.expander("📖 Apuntes PATMOS sobre el portafolio actual"):
    st.markdown("""
    El gap a metas no se cierra solo con más volumen — el **mix del catálogo** también
    debe alinearse con la oportunidad de mercado identificada por el estudio PATMOS:

    - **Brecha femenina**: solo 20% de la demanda histórica es femenino, pero el
      segmento S1 PATMOS (Activos) tiene 60% mujeres. El sugerido automático empuja
      mínimo 30% femenino/año.
    - **Brecha juvenil (S6)**: catálogo apenas 7% juvenil. El sugerido garantiza
      al menos 1 SKU juvenil cada 2 años.
    - **Misioneras al Clúster 4 (S4-S5-S6)**: la fuerza histórica de SBC. Iglesias
      las regalan sin distinguir género, por eso el sugerido las clasifica como
      **universal**, no masculino.
    """)


@st.cache_data(max_entries=1)
def cargar_datos():
    # Nota de memoria: optimizamos (category) solo los DataFrames que se usan
    # para agregación/filtro (.isin, groupby), NO los que el código muta con
    # asignaciones de valores nuevos (isbn/serie), para no arriesgar estabilidad.
    from core.optimize import optimizar_memoria
    proy = pd.read_parquet(proy_path)
    isbn = pd.read_parquet(isbn_path)
    serie = pd.read_parquet(serie_path)
    # ventas mensuales por cliente × clase (el más pesado; solo se agrega/filtra)
    cli_clase_path = DATA_PROC / "ventas_mensual_cliente_clase.parquet"
    if cli_clase_path.exists():
        serie_cli = optimizar_memoria(pd.read_parquet(cli_clase_path))
    else:
        serie_cli = None
    if anc_path.exists():
        anc = pd.read_csv(anc_path)
    else:
        anc = None
    return proy, isbn, serie, anc, serie_cli


proy, isbn, serie, anclaje, serie_cli = cargar_datos()
novedades = novedades_store.cargar()
aporte_novedades_mensual = novedades_store.obtener_aporte_mensual_novedades()
aporte_novedades_anual = novedades_store.obtener_aporte_anual_novedades()
aporte_desglosado = novedades_store.obtener_aporte_anual_desglosado()

# =========================================================================
# AJUSTES MANUALES — PANEL LATERAL POR CATEGORÍA
# =========================================================================
# Sistema de overrides: el usuario puede mover toda la curva de una categoría
# arriba/abajo (escala) o alargar/acortar el ciclo de vida (ciclo).
#
# Flujo: los sliders mantienen valores PENDIENTES (no se aplican aún).
# Al darle al botón "💾 Aplicar overrides" se persisten a disco
# (data/state/overrides_proyeccion.json) y se hace rerun para que TODAS
# las gráficas y tablas de la app reflejen los cambios.
overrides_actuales = overrides_store.cargar()

# =========================================================================
# FILTROS GLOBALES (v3.12) — viven en TODAS las páginas
# =========================================================================
from app.filtros_globales import (
    render_filtros_globales, aplicar_filtros_isbn,
    aplicar_filtros_temporal_serie, label_periodo_actual,
    resumen_filtros_dashboard,
)
filtros_g = render_filtros_globales(isbn, serie)
# Universo de ISBNs que pasan el filtro de dashboard (afecta TODA la página)
isbn_filtrado_global = aplicar_filtros_isbn(isbn, filtros_g)
isbns_validos_global = set(isbn_filtrado_global["isbn"])

with st.sidebar:
    st.markdown("## 🎚️ Ajustes manuales de proyecciones")
    st.caption(
        "Mueve los sliders abajo y luego haz clic en **💾 Aplicar overrides**. "
        "Los cambios se reflejarán en TODAS las páginas (Explorador, Sugerencias, "
        "Descargar demanda) y en el CSV final."
    )

    clases_disponibles = sorted(isbn["clase"].dropna().unique().tolist())

    # Diccionario de valores pendientes (en session_state local)
    valores_pendientes_cat = {}

    with st.expander("📂 Por categoría (clase)", expanded=True):
        st.caption("⚠️ Los sliders muestran tus cambios pendientes. Hasta no aplicar, no se guardan.")
        for clase in clases_disponibles:
            ov_cat = overrides_actuales.get("categorias", {}).get(clase, {})
            esc_guardado = float(ov_cat.get("escala", 1.0))
            ciclo_guardado = float(ov_cat.get("ciclo", 1.0))

            # Header con indicador de si está modificado
            estado_guardado = (esc_guardado != 1.0 or ciclo_guardado != 1.0)
            etiqueta = f"**{clase}**" + (" 🟢" if estado_guardado else "")
            st.markdown(etiqueta)
            if estado_guardado:
                st.caption(f"_Aplicado actualmente: escala={esc_guardado:.2f}, ciclo={ciclo_guardado:.2f}_")

            col_a, col_b = st.columns(2)
            with col_a:
                nuevo_esc = st.slider(
                    "Escala", min_value=0.5, max_value=2.0,
                    value=esc_guardado, step=0.05,
                    key=f"esc_cat_{clase}",
                    help="1.0 = sin cambio. >1 mueve la curva arriba. <1 la mueve abajo.",
                )
            with col_b:
                nuevo_ciclo = st.slider(
                    "Ciclo", min_value=0.5, max_value=2.0,
                    value=ciclo_guardado, step=0.05,
                    key=f"ciclo_cat_{clase}",
                    help="1.0 = ciclo normal. >1 alarga el ciclo. <1 lo acorta.",
                )
            valores_pendientes_cat[clase] = {
                "escala": nuevo_esc,
                "ciclo": nuevo_ciclo,
                "esc_guardado": esc_guardado,
                "ciclo_guardado": ciclo_guardado,
            }

    # Detectar si hay cambios pendientes vs lo guardado en disco
    cambios_pendientes = any(
        abs(v["escala"] - v["esc_guardado"]) > 1e-6 or abs(v["ciclo"] - v["ciclo_guardado"]) > 1e-6
        for v in valores_pendientes_cat.values()
    )

    if cambios_pendientes:
        st.warning("⚠️ Tienes cambios pendientes sin aplicar")

    col_aplicar, col_reset = st.columns(2)
    with col_aplicar:
        boton_aplicar = st.button(
            "💾 Aplicar overrides",
            type="primary",
            use_container_width=True,
            disabled=not cambios_pendientes,
            help="Guarda los cambios de categoría y los refleja en TODA la app",
        )
    with col_reset:
        boton_reset = st.button(
            "🗑️ Reset categorías",
            use_container_width=True,
            help="Elimina SOLO los overrides de categoría hechos desde estos "
                 "sliders. Los overrides por ISBN del explorador NO se tocan.",
        )

    if boton_aplicar:
        # Persistir todos los valores en disco
        for clase, vals in valores_pendientes_cat.items():
            overrides_store.set_categoria(clase, vals["escala"], vals["ciclo"])
        st.success("✓ Overrides de categoría aplicados a TODA la app. Refrescando...")
        st.cache_data.clear()
        st.rerun()

    if boton_reset:
        n_cats = overrides_store.reset_categorias()
        # Limpia SOLO el session_state de los sliders de categoría
        # (NO toca esc_isbn_/ciclo_isbn_ que son overrides individuales)
        for k in list(st.session_state.keys()):
            if k.startswith("esc_cat_") or k.startswith("ciclo_cat_"):
                del st.session_state[k]
        st.cache_data.clear()
        st.success(
            f"✓ {n_cats} override(s) de categoría eliminados. "
            f"Los overrides por ISBN se conservan."
        )
        st.rerun()

    # Resumen de overrides activos en disco
    resumen_ov = overrides_store.resumen()
    if resumen_ov["n_categorias"] > 0 or resumen_ov["n_isbns"] > 0:
        st.info(
            f"📌 Aplicados en disco: **{resumen_ov['n_categorias']}** categorías, "
            f"**{resumen_ov['n_isbns']}** ISBNs específicos"
        )

# Aplicar los overrides GUARDADOS (no los pendientes) a las proyecciones
proy = overrides_store.aplicar_overrides_a_proyecciones(proy, isbn, overrides_store.cargar())

# RESUMEN
st.subheader("Resumen del catálogo")
col1, col2, col3, col4, col5 = st.columns(5)
n_isbns_proyectados = proy["isbn"].nunique()
fuentes = proy["fuente"].value_counts().to_dict()
n_simulador = len(novedades_store.filtrar_por_origen("manual"))
n_sugerencias = len(novedades_store.filtrar_por_origen("sugerencia_automatica"))
col1.metric("ISBNs proyectados", f"{n_isbns_proyectados:,}")
col2.metric("Prophet", f"{fuentes.get('prophet', 0)}")
col3.metric("Decay", f"{fuentes.get('decay', 0)}")
col4.metric("Aprobadas simulador", f"{n_simulador}")
col5.metric("Aprobadas sugeridos", f"{n_sugerencias}")

# GAP A METAS con distinción de novedades dentro/fuera de capacidad
if anclaje is not None:
    st.divider()
    st.subheader("Gap a metas conservadoras 2027-2030")
    st.caption(
        "**Verde claro** = novedades aprobadas dentro de capacidad (manuales + sugeridos). "
        "**Violeta** = novedades aprobadas FUERA de capacidad (déficit estructural que requiere "
        "fortalecer Publicaciones). **Naranja** = gap aún no cubierto."
    )

    # Recalcular aporte_catalogo desde proy (que YA tiene overrides aplicados)
    # en lugar de usar el del CSV anclaje (que es precalculado).
    # Las metas son de BIBLIAS, por eso filtramos.
    isbn_biblias_lista = isbn[isbn["clase"] == "BIBLIAS"]["isbn"].tolist()
    proy_biblias_year = proy[proy["isbn"].isin(isbn_biblias_lista)].copy()
    proy_biblias_year["anio"] = pd.to_datetime(proy_biblias_year["ds"]).dt.year
    aporte_catalogo_dinamico = (
        proy_biblias_year.groupby("anio")["yhat"].sum().astype(int).to_dict()
    )

    anclaje_v2 = anclaje.copy()
    # Reemplazar la columna estática por la dinámica con overrides aplicados
    anclaje_v2["aporte_catalogo"] = anclaje_v2["anio"].map(aporte_catalogo_dinamico).fillna(0).astype(int)
    anclaje_v2["nov_dentro"] = anclaje_v2["anio"].apply(
        lambda a: aporte_desglosado.get(a, {}).get("manual_dentro", 0)
                 + aporte_desglosado.get(a, {}).get("sugerencia_dentro", 0)
    ).astype(int)
    anclaje_v2["nov_fuera"] = anclaje_v2["anio"].apply(
        lambda a: aporte_desglosado.get(a, {}).get("manual_fuera", 0)
                 + aporte_desglosado.get(a, {}).get("sugerencia_fuera", 0)
    ).astype(int)
    anclaje_v2["aporte_total"] = (
        anclaje_v2["aporte_catalogo"] + anclaje_v2["nov_dentro"] + anclaje_v2["nov_fuera"]
    )
    anclaje_v2["gap_restante"] = (anclaje_v2["meta"] - anclaje_v2["aporte_total"]).clip(lower=0).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=anclaje_v2["anio"], y=anclaje_v2["aporte_catalogo"],
        name="Catálogo (Prophet+Decay)",
        marker_color="#1f4e79",
        text=[f"{x:,}" for x in anclaje_v2["aporte_catalogo"]],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=anclaje_v2["anio"], y=anclaje_v2["nov_dentro"],
        name="Novedades (dentro capacidad)",
        marker_color="#10b981",
        text=[f"{x:,}" if x > 0 else "" for x in anclaje_v2["nov_dentro"]],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=anclaje_v2["anio"], y=anclaje_v2["nov_fuera"],
        name="Novedades (FUERA capacidad)",
        marker_color="#8b5cf6",
        text=[f"{x:,}" if x > 0 else "" for x in anclaje_v2["nov_fuera"]],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=anclaje_v2["anio"], y=anclaje_v2["gap_restante"],
        name="Gap restante",
        marker_color="#f59e0b",
        text=[f"{x:,}" for x in anclaje_v2["gap_restante"]],
        textposition="inside",
    ))
    fig.add_trace(go.Scatter(
        x=anclaje_v2["anio"], y=anclaje_v2["meta"],
        mode="lines+markers+text",
        line=dict(color="#dc2626", width=3, dash="dash"),
        marker=dict(size=12, symbol="diamond"),
        name="Meta conservadora",
        text=[f"{x:,}" for x in anclaje_v2["meta"]],
        textposition="top center",
    ))
    fig.update_layout(
        barmode="stack",
        height=440,
        yaxis_title="Unidades BIBLIAS",
        xaxis_title="Año",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    anclaje_view = anclaje_v2.copy()
    for c in ["meta", "aporte_catalogo", "nov_dentro", "nov_fuera", "aporte_total", "gap_restante"]:
        anclaje_view[c] = anclaje_view[c].apply(lambda x: f"{x:,}")
    anclaje_view = anclaje_view[["anio", "meta", "aporte_catalogo", "nov_dentro",
                                  "nov_fuera", "aporte_total", "gap_restante", "gap_pct_de_meta"]]
    anclaje_view.columns = ["Año", "Meta", "Catálogo", "Nov. dentro cap",
                              "Nov. FUERA cap", "Total", "Gap restante", "Gap inicial %"]
    st.dataframe(anclaje_view, use_container_width=True, hide_index=True)

# CURVA TOTAL — estilo líneas
st.divider()
st.subheader("Curva total proyectada del catálogo")
st.caption(
    "**Azul oscuro** = histórico real. **Naranja** = proyección catálogo (Prophet+Decay). "
    "**Verde** = aporte mensual de novedades aprobadas. "
    "**Azul brillante** = total proyectado (catálogo + novedades), como continuación del histórico."
)

# La curva respeta el filtro global de dashboard. Si el usuario filtró por
# categoría/género/mercado, la curva muestra solo ese subconjunto de BIBLIAS.
_chips_g = resumen_filtros_dashboard(filtros_g)
if _chips_g != "sin filtros de dashboard":
    st.caption(f"🎚️ Filtros activos: {_chips_g}")

isbn_biblias = isbn_filtrado_global[isbn_filtrado_global["clase"] == "BIBLIAS"]["isbn"].tolist()
# Si el filtro de clase excluye BIBLIAS por completo, caer a todas las del filtro
if len(isbn_biblias) == 0:
    isbn_biblias = isbn_filtrado_global["isbn"].tolist()
serie_biblias = serie[serie["isbn"].isin(isbn_biblias)]
hist_total = serie_biblias.groupby("mes")["unidades"].sum().reset_index()

proy_biblias = proy[proy["isbn"].isin(isbn_biblias)]
proy_total = proy_biblias.groupby("ds").agg(
    yhat=("yhat", "sum"),
    yhat_lower=("yhat_lower", "sum"),
    yhat_upper=("yhat_upper", "sum"),
).reset_index()

# Calcular total = catálogo + novedades (mensual)
if len(aporte_novedades_mensual):
    proy_y_nov = proy_total.merge(
        aporte_novedades_mensual[["ds", "prediccion"]].rename(columns={"prediccion": "novedades"}),
        on="ds", how="left",
    )
    proy_y_nov["novedades"] = proy_y_nov["novedades"].fillna(0)
    proy_y_nov["total"] = proy_y_nov["yhat"] + proy_y_nov["novedades"]
else:
    proy_y_nov = proy_total.copy()
    proy_y_nov["novedades"] = 0
    proy_y_nov["total"] = proy_y_nov["yhat"]

fig_total = go.Figure()

# Histórico real (azul oscuro, sólido)
fig_total.add_trace(go.Scatter(
    x=hist_total["mes"], y=hist_total["unidades"],
    mode="lines+markers",
    line=dict(color="#1f4e79", width=2.5), marker=dict(size=5),
    name="Histórico real",
))
# Banda p10-p90 catálogo (sin nombre, fondo)
fig_total.add_trace(go.Scatter(
    x=proy_total["ds"], y=proy_total["yhat_upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_total.add_trace(go.Scatter(
    x=proy_total["ds"], y=proy_total["yhat_lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(245,158,11,0.15)",
    name="Banda p10-p90 catálogo",
))
# Proyección catálogo (naranja punteada)
fig_total.add_trace(go.Scatter(
    x=proy_total["ds"], y=proy_total["yhat"],
    mode="lines+markers",
    line=dict(color="#f59e0b", width=2.5, dash="dot"), marker=dict(size=5),
    name="Proyección catálogo",
))
# Novedades aprobadas (verde, mismo estilo que catálogo)
if (proy_y_nov["novedades"] > 0).any():
    fig_total.add_trace(go.Scatter(
        x=proy_y_nov["ds"], y=proy_y_nov["novedades"],
        mode="lines+markers",
        line=dict(color="#10b981", width=2.5, dash="dot"), marker=dict(size=5),
        name="Novedades aprobadas",
    ))
    # Total proyectado (azul brillante sólido, continuación visual del histórico)
    fig_total.add_trace(go.Scatter(
        x=proy_y_nov["ds"], y=proy_y_nov["total"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3), marker=dict(size=6),
        name="Total proyectado (catálogo + novedades)",
    ))

fig_total.update_layout(
    height=480,
    yaxis_title="Unidades BIBLIAS (mes)",
    xaxis_title="Mes",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12),
    margin=dict(t=50, b=30),
)
st.plotly_chart(fig_total, use_container_width=True)

# Resumen anual combinado
proy_total_anio = proy_total.copy()
proy_total_anio["anio"] = pd.to_datetime(proy_total_anio["ds"]).dt.year
resumen_anio = proy_total_anio.groupby("anio")["yhat"].sum().reset_index()
hist_total_anio = hist_total.copy()
hist_total_anio["anio"] = pd.to_datetime(hist_total_anio["mes"]).dt.year
hist_anio_agg = hist_total_anio.groupby("anio")["unidades"].sum().reset_index()

filas = []
for año in sorted(set(hist_anio_agg["anio"]).union(resumen_anio["anio"])):
    hist = hist_anio_agg[hist_anio_agg["anio"] == año]["unidades"].sum()
    proy_a = resumen_anio[resumen_anio["anio"] == año]["yhat"].sum()
    nov = aporte_novedades_anual.get(año, 0)
    total = proy_a + nov
    filas.append({
        "Año": año,
        "Histórico real": int(hist) if hist > 0 else "—",
        "Proyección catálogo": int(proy_a) if proy_a > 0 else "—",
        "Novedades aprobadas": int(nov) if nov > 0 else "—",
        "Total proyectado": int(total) if total > 0 else "—",
    })
st.markdown("**Resumen anual**")
st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

# EXPLORADOR POR ISBN con búsqueda
st.divider()

# =========================================================================
# HELPERS COMUNES PARA AMBOS EXPLORADORES (ISBN y CLIENTE) — v3.9
# =========================================================================
def calcular_estadisticos_mensuales(
    serie_df: pd.DataFrame,
    col_unidades: str = "unidades",
    col_valor: str = "valor",
) -> pd.DataFrame:
    """
    Calcula estadísticos descriptivos sobre una serie mensual.

    Recibe un DataFrame con una fila por mes (ya agregado) y devuelve
    una tabla larga con dos columnas: 'Unidades' y 'Valor (COP)',
    cubriendo:
      - N (meses con observación)
      - Total acumulado
      - Media (promedio aritmético)
      - Mediana (P50)
      - Mínimo / Máximo
      - Desviación estándar (muestral, n-1)
      - Varianza (muestral, n-1)
      - P25 / P75 / Rango intercuartílico (P75-P25)
      - Coeficiente de variación (σ / μ) — adimensional
      - Asimetría (skewness, Fisher-Pearson)

    Si la columna de valor no existe o está vacía, esa columna queda en "—".
    """
    if serie_df is None or len(serie_df) == 0:
        return pd.DataFrame()

    def _stats(serie: pd.Series, es_valor: bool) -> dict:
        s = pd.to_numeric(serie, errors="coerce").dropna()
        if len(s) == 0:
            return {k: "—" for k in [
                "N (meses)", "Total acumulado", "Media (promedio)", "Mediana (P50)",
                "Mínimo", "Máximo", "Desv. estándar (σ)", "Varianza (σ²)",
                "Percentil 25", "Percentil 75", "Rango intercuartílico (P75-P25)",
                "Coef. variación (σ/μ)", "Asimetría (skewness)",
            ]}
        n = int(len(s))
        total = float(s.sum())
        media = float(s.mean())
        mediana = float(s.median())
        mn = float(s.min())
        mx = float(s.max())
        # Desv. estándar y varianza muestrales (n-1) → pandas default ddof=1
        sd = float(s.std()) if n > 1 else 0.0
        var = float(s.var()) if n > 1 else 0.0
        p25 = float(s.quantile(0.25))
        p75 = float(s.quantile(0.75))
        iqr = p75 - p25
        cov = (sd / media) if media != 0 else float("nan")
        sk = float(s.skew()) if n > 2 else float("nan")

        # Formato amigable: enteros con coma; CoV y skewness con 2 decimales
        def fmt_num(x: float) -> str:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "—"
            if es_valor:
                return f"${int(round(x)):,}"
            return f"{int(round(x)):,}"

        def fmt_ratio(x: float) -> str:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "—"
            return f"{x:.2f}"

        return {
            "N (meses)": f"{n:,}",
            "Total acumulado": fmt_num(total),
            "Media (promedio)": fmt_num(media),
            "Mediana (P50)": fmt_num(mediana),
            "Mínimo": fmt_num(mn),
            "Máximo": fmt_num(mx),
            "Desv. estándar (σ)": fmt_num(sd),
            "Varianza (σ²)": fmt_num(var * 1) if not es_valor else (
                # Para valor, varianza está en (COP)² — número enorme. Mostrar resumido.
                fmt_num(var)
            ),
            "Percentil 25": fmt_num(p25),
            "Percentil 75": fmt_num(p75),
            "Rango intercuartílico (P75-P25)": fmt_num(iqr),
            "Coef. variación (σ/μ)": fmt_ratio(cov),
            "Asimetría (skewness)": fmt_ratio(sk),
        }

    stats_u = _stats(serie_df[col_unidades], es_valor=False) if col_unidades in serie_df.columns else {}
    stats_v = _stats(serie_df[col_valor], es_valor=True) if col_valor in serie_df.columns else {}

    # Asegurar mismo orden de filas y unir ambas columnas
    orden = [
        "N (meses)", "Total acumulado", "Media (promedio)", "Mediana (P50)",
        "Mínimo", "Máximo", "Desv. estándar (σ)", "Varianza (σ²)",
        "Percentil 25", "Percentil 75", "Rango intercuartílico (P75-P25)",
        "Coef. variación (σ/μ)", "Asimetría (skewness)",
    ]
    filas = []
    for k in orden:
        filas.append({
            "Estadístico": k,
            "Unidades": stats_u.get(k, "—"),
            "Valor (COP)": stats_v.get(k, "—"),
        })
    return pd.DataFrame(filas)


def _opciones_efectivas(
    df: pd.DataFrame,
    col: str,
    filtros_aplicados: dict,
) -> list:
    """
    Devuelve los valores únicos de `col` que existen DESPUÉS de aplicar
    todos los filtros en `filtros_aplicados` (dict col → lista de valores).
    Útil para que los selectores encadenados solo muestren combinaciones
    que sí existen en los datos.
    """
    sub = df
    for c, vals in filtros_aplicados.items():
        if c == col:  # no auto-filtrarse
            continue
        if vals is None or len(vals) == 0:
            continue
        sub = sub[sub[c].isin(vals)]
    if col not in sub.columns:
        return []
    return sorted(sub[col].dropna().astype(str).unique().tolist())


def filtro_popover_multi(
    label_corto: str,
    icono: str,
    opciones: list,
    key: str,
    ayuda: str = "",
) -> list:
    """
    Filtro multi-select dentro de un popover para evitar abultar la línea
    cuando hay muchas opciones. Devuelve la selección actual.

    Estado:
      - Si el key no está en session_state, default = todas las opciones.
      - Botones "Todas" y "Ninguna" para reset rápido sin scroll.
      - El título del popover muestra cuántas opciones están activas.
    """
    # Inicializar estado
    if key not in st.session_state:
        st.session_state[key] = list(opciones)
    # Limpieza: quitar valores que ya no existen en opciones
    st.session_state[key] = [v for v in st.session_state[key] if v in opciones]

    n_sel = len(st.session_state[key])
    n_tot = len(opciones)
    if n_sel == n_tot and n_tot > 0:
        resumen = f"Todas ({n_tot})"
    elif n_sel == 0:
        resumen = "ninguna"
    else:
        resumen = f"{n_sel} de {n_tot}"
    titulo = f"{icono} {label_corto}: {resumen}"

    with st.popover(titulo, use_container_width=True):
        if ayuda:
            st.caption(ayuda)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Todas", key=f"{key}_todas", use_container_width=True):
                st.session_state[key] = list(opciones)
                st.rerun()
        with c2:
            if st.button("✗ Ninguna", key=f"{key}_ninguna", use_container_width=True):
                st.session_state[key] = []
                st.rerun()
        seleccion = st.multiselect(
            f"Marca las {label_corto.lower()} que quieres incluir",
            options=opciones,
            default=st.session_state[key],
            key=f"{key}_ms",
            label_visibility="collapsed",
        )
        # Guardar selección actual en el key "estable"
        st.session_state[key] = seleccion

    return st.session_state[key]


# =========================================================================
# EXPLORADOR POR ISBN
# =========================================================================
st.subheader("Explorador de proyecciones por ISBN")

# ─────────────────────────────────────────────────────────────────────────
# RESUMEN DE OVERRIDES INDIVIDUALES POR ISBN
# ─────────────────────────────────────────────────────────────────────────
# Solo aparece si hay al menos un ISBN con override propio. Permite ver de un
# vistazo cuáles ISBNs tienen ajustes individuales y eliminar los que ya no
# quieras (ya sea uno por uno o varios a la vez).
ov_isbns_actuales = overrides_actuales.get("isbns", {})
if ov_isbns_actuales:
    with st.expander(
        f"📋 ISBNs con override individual ({len(ov_isbns_actuales)}) — clic para gestionar",
        expanded=True,
    ):
        st.caption(
            "Estos ISBNs tienen un ajuste propio que **anula** el de su categoría. "
            "Marca los que quieras eliminar y haz clic en _Eliminar overrides seleccionados_."
        )
        # Construir tabla con descripción + clase + valores del override
        filas_ov = []
        for isbn_id, vals in ov_isbns_actuales.items():
            meta = isbn[isbn["isbn"] == isbn_id]
            if len(meta) == 0:
                descripcion = "(ISBN no encontrado en feature_isbn)"
                clase = "?"
            else:
                descripcion = str(meta["descripcion"].iloc[0])[:80]
                clase = str(meta["clase"].iloc[0])
            filas_ov.append({
                "Eliminar": False,
                "ISBN": isbn_id,
                "Descripción": descripcion,
                "Clase": clase,
                "Escala": float(vals.get("escala", 1.0)),
                "Ciclo": float(vals.get("ciclo", 1.0)),
            })
        df_ov = pd.DataFrame(filas_ov).sort_values("ISBN").reset_index(drop=True)

        df_ov_edit = st.data_editor(
            df_ov,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Eliminar": st.column_config.CheckboxColumn(
                    "Eliminar", default=False, width="small",
                ),
                "ISBN": st.column_config.TextColumn(width="medium"),
                "Descripción": st.column_config.TextColumn(width="large"),
                "Escala": st.column_config.NumberColumn(format="%.2f"),
                "Ciclo": st.column_config.NumberColumn(format="%.2f"),
            },
            disabled=["ISBN", "Descripción", "Clase", "Escala", "Ciclo"],
            key="tabla_overrides_isbn",
        )

        ids_a_eliminar = df_ov_edit[df_ov_edit["Eliminar"]]["ISBN"].tolist()

        col_del, col_del_all = st.columns(2)
        with col_del:
            if st.button(
                f"🗑️ Eliminar overrides seleccionados ({len(ids_a_eliminar)})",
                disabled=(len(ids_a_eliminar) == 0),
                use_container_width=True,
                type="primary",
                key="btn_del_ov_isbn_sel",
            ):
                for isbn_id in ids_a_eliminar:
                    overrides_store.eliminar_isbn(isbn_id)
                    # También limpia el session_state de sliders de ese ISBN
                    for k in [f"esc_isbn_{isbn_id}", f"ciclo_isbn_{isbn_id}"]:
                        if k in st.session_state:
                            del st.session_state[k]
                st.cache_data.clear()
                st.success(f"✓ Eliminados {len(ids_a_eliminar)} overrides de ISBN")
                st.rerun()
        with col_del_all:
            if st.button(
                f"🗑️ Eliminar TODOS los overrides de ISBN ({len(ov_isbns_actuales)})",
                use_container_width=True,
                key="btn_del_ov_isbn_all",
                help="Elimina todos los overrides individuales. Los overrides por categoría NO se tocan.",
            ):
                for isbn_id in list(ov_isbns_actuales.keys()):
                    overrides_store.eliminar_isbn(isbn_id)
                    for k in [f"esc_isbn_{isbn_id}", f"ciclo_isbn_{isbn_id}"]:
                        if k in st.session_state:
                            del st.session_state[k]
                st.cache_data.clear()
                st.success(f"✓ Eliminados todos los overrides individuales por ISBN")
                st.rerun()

isbn_disponibles = proy["isbn"].unique()
# Antes: solo ISBNs con proyección Prophet (filtrado por proy)
# Ahora: TODOS los ISBNs del feature store (BIBLIAS + LITERATURA + PORCIONES + ...)
# Si el ISBN tiene proyección, mostramos histórico + pronóstico + overrides.
# Si no, solo histórico.
isbns_proy = isbn[isbn["isbn"].notna()].copy()
isbns_proy["tiene_proyeccion"] = isbns_proy["isbn"].isin(isbn_disponibles)
isbns_proy["clase"] = isbns_proy["clase"].fillna("(sin clase)").astype(str)
# Limpiar la columna estado por si tiene NaN
if "estado" in isbns_proy.columns:
    isbns_proy["estado"] = isbns_proy["estado"].fillna("(sin estado)").astype(str)
else:
    isbns_proy["estado"] = "(sin estado)"
# v3.9: normalizar version y categoria_precio para los nuevos filtros
if "version" in isbns_proy.columns:
    isbns_proy["version"] = (
        isbns_proy["version"].fillna("(sin versión)").replace("", "(sin versión)").astype(str)
    )
else:
    isbns_proy["version"] = "(sin versión)"
if "categoria_precio" in isbns_proy.columns:
    isbns_proy["categoria_precio"] = (
        isbns_proy["categoria_precio"].fillna("no_aplica").replace("", "no_aplica").astype(str)
    )
else:
    isbns_proy["categoria_precio"] = "no_aplica"

isbns_proy["label"] = (
    isbns_proy["isbn"].astype(str) + " — " +
    isbns_proy["descripcion"].fillna("").astype(str).str[:70] +
    isbns_proy["tiene_proyeccion"].map({True: "", False: "  [solo histórico]"})
)
isbns_proy = isbns_proy.sort_values(
    ["tiene_proyeccion", "unidades_total"], ascending=[False, False]
)

# ─── Filtros LOCALES a esta sección (no afectan otras páginas) ───
# v3.9: filtros encadenados clase ↔ version ↔ categoria_precio para no
# mostrar combinaciones que no existen en los datos.
st.markdown("**Filtros locales** (solo afectan esta sección)")

# Fila A: clase, versión, categoría de precio — los 3 como popovers compactos
fa1, fa2, fa3 = st.columns(3)
clases_all_isbn = sorted(isbns_proy["clase"].unique().tolist())

with fa1:
    clases_isbn_sel = filtro_popover_multi(
        label_corto="Clase",
        icono="📚",
        opciones=clases_all_isbn,
        key="filtro_clase_isbn_v39",
        ayuda="Filtra por clase de producto (BIBLIAS, LITERATURA, etc.). Solo se muestran ISBNs que cumplen las clases marcadas.",
    )

# Encadenar: versiones disponibles dependen de las clases seleccionadas
versiones_disp = _opciones_efectivas(
    isbns_proy, "version",
    filtros_aplicados={"clase": clases_isbn_sel},
)
with fa2:
    versiones_sel = filtro_popover_multi(
        label_corto="Versión",
        icono="📖",
        opciones=versiones_disp,
        key="filtro_version_isbn_v39",
        ayuda="Versión bíblica (RVR, RVC, DHH, TLA, NTV, BLP, NVI...). Las opciones cambian según la clase elegida.",
    )

# Encadenar: categorías de precio dependen de clase + versión
categorias_disp = _opciones_efectivas(
    isbns_proy, "categoria_precio",
    filtros_aplicados={"clase": clases_isbn_sel, "version": versiones_sel},
)
with fa3:
    categorias_sel = filtro_popover_multi(
        label_corto="Categoría precio",
        icono="💲",
        opciones=categorias_disp,
        key="filtro_cat_precio_isbn_v39",
        ayuda="Quintiles de precio para BIBLIAS (economica → fina). 'no_aplica' = clases no biblias.",
    )

# Fila B: solo proyectados + toggle unidades/valor
fb1, fb2 = st.columns([1, 1])
with fb1:
    solo_proyectados = st.checkbox(
        "Solo con proyección Prophet",
        value=False,
        key="filtro_solo_proy_isbn",
        help="Marca para ver únicamente los ISBNs con pronóstico Prophet calculado.",
    )
with fb2:
    metrica_isbn = st.radio(
        "Métrica de visualización",
        options=["Unidades", "Valor ($)"],
        index=0,
        horizontal=True,
        key="toggle_metrica_isbn",
        help="Cambia entre unidades vendidas y valor monetario (COP).",
    )

# Aplicar filtros (todos los popovers + el checkbox)
isbns_filt = isbns_proy[
    isbns_proy["clase"].isin(clases_isbn_sel)
    & isbns_proy["version"].isin(versiones_sel)
    & isbns_proy["categoria_precio"].isin(categorias_sel)
]
if solo_proyectados:
    isbns_filt = isbns_filt[isbns_filt["tiene_proyeccion"]]

col_search, col_select = st.columns([1, 3])
with col_search:
    busqueda = st.text_input(
        "🔎 Buscar ISBN / descripción / NOVEDAD",
        placeholder="ej: mujer virtuosa, 9789587, NOVEDAD, RVR066...",
        help="Filtra la lista por ISBN, palabra clave en la descripción, "
             "o escribe 'NOVEDAD' para ver solo las novedades aprobadas."
    )
with col_select:
    if busqueda:
        mask = (
            isbns_filt["isbn"].str.contains(busqueda, case=False, na=False)
            | isbns_filt["descripcion"].fillna("").str.contains(busqueda, case=False, na=False)
        )
        opciones_filtradas = isbns_filt[mask]
    else:
        opciones_filtradas = isbns_filt

    # v3.12 (fix): construir las opciones de NOVEDAD ANTES de validar vacío,
    # para que el buscador también las encuentre (incluida la palabra "NOVEDAD").
    OPCION_TODAS = "🌐 TODAS (vista agregada del filtro actual)"
    _todas_novedades = [
        n for n in novedades_store.cargar() if n.get("estado") == "aprobado"
    ]
    _nov_labels = {}
    for n in _todas_novedades:
        nid = n.get("id", "")
        origen_tag = ("🤖 sugerida" if n.get("origen") == "sugerencia_automatica"
                      else "✍️ manual")
        cap_tag = " · 📈 fuera cap." if n.get("fuera_capacidad") else ""
        id_prod = n.get("id_producto_manual") or ""
        nombre_nov = n.get("nombre", nid)
        etiqueta = f"🆕 NOVEDAD [{origen_tag}] {nombre_nov[:50]}"
        if id_prod:
            etiqueta += f"  ·  {id_prod}"
        etiqueta += cap_tag
        _nov_labels[f"__NOV__{nid}"] = etiqueta

    # Filtrar novedades por la misma búsqueda (sobre su etiqueta + id de producto)
    _nov_labels_filtradas = dict(_nov_labels)
    if busqueda:
        b = busqueda.lower()
        _nov_labels_filtradas = {
            k: v for k, v in _nov_labels.items()
            if b in v.lower() or b in k.lower()
        }

    # Si la búsqueda no dejó NI ISBNs NI novedades, avisar y mostrar todo
    if busqueda and len(opciones_filtradas) == 0 and len(_nov_labels_filtradas) == 0:
        st.warning("Sin coincidencias en los filtros actuales. Mostrando todos.")
        opciones_filtradas = isbns_filt
        _nov_labels_filtradas = dict(_nov_labels)

    if len(opciones_filtradas) == 0 and len(_nov_labels_filtradas) == 0:
        st.error(
            "No hay ISBNs que cumplan los filtros. Abre los popovers de Clase/Versión/Categoría "
            "de precio y vuelve a marcar las opciones, o desmarca 'Solo con proyección'."
        )
        st.stop()

    opciones_select = (
        [OPCION_TODAS]
        + opciones_filtradas["label"].tolist()
        + list(_nov_labels_filtradas.keys())
    )
    label_sel = st.selectbox(
        f"Selecciona ISBN o NOVEDAD "
        f"({len(opciones_filtradas)} ISBNs · {len(_nov_labels_filtradas)} novedades · o vista agregada)",
        options=opciones_select,
        index=0,  # TODAS es el default al abrir
        format_func=lambda v: _nov_labels.get(v, v),
        key="isbn_explorer_select",
    )
    if _todas_novedades and not busqueda:
        st.caption(
            f"🆕 Hay **{len(_todas_novedades)} novedad(es) aprobada(s)** en la lista "
            f"(al final del desplegable, con prefijo 🆕 NOVEDAD). Escribe **NOVEDAD** "
            f"en el buscador para ver solo esas, o selecciona una para ajustarla."
        )

es_vista_todas = (label_sel == OPCION_TODAS)
es_novedad_sel = isinstance(label_sel, str) and label_sel.startswith("__NOV__")

if es_vista_todas:
    # ───────────── VISTA AGREGADA DE TODOS LOS ISBNS DEL FILTRO ─────────────
    isbns_agg = opciones_filtradas["isbn"].tolist()
    st.markdown(f"#### Vista agregada · {len(isbns_agg)} ISBNs del filtro actual")

    c1, c2, c3, c4 = st.columns(4)
    n_con_proy = opciones_filtradas["tiene_proyeccion"].sum()
    c1.metric("ISBNs en vista", f"{len(isbns_agg):,}")
    c2.metric("Con proyección", f"{int(n_con_proy):,}")
    clases_en_vista = opciones_filtradas["clase"].nunique()
    c3.metric("Clases distintas", f"{clases_en_vista}")
    unid_hist = serie[serie["isbn"].isin(isbns_agg)]["unidades"].sum()
    c4.metric("Unidades históricas", f"{int(unid_hist):,}")

    # Histórico + proyección agregados
    serie_agg = serie[serie["isbn"].isin(isbns_agg)]
    hist_agg = serie_agg.groupby("mes")["unidades"].sum().reset_index()
    proy_agg_src = proy[proy["isbn"].isin(isbns_agg)]
    proy_agg = proy_agg_src.groupby("ds").agg(
        yhat=("yhat", "sum"),
        yhat_lower=("yhat_lower", "sum"),
        yhat_upper=("yhat_upper", "sum"),
    ).reset_index()

    fig_agg = go.Figure()
    if len(hist_agg) > 0:
        fig_agg.add_trace(go.Scatter(
            x=hist_agg["mes"], y=hist_agg["unidades"],
            mode="lines", line=dict(color="#1f4e79", width=2),
            name="Histórico (suma)",
        ))
    if len(proy_agg) > 0:
        fig_agg.add_trace(go.Scatter(
            x=proy_agg["ds"], y=proy_agg["yhat_upper"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_agg.add_trace(go.Scatter(
            x=proy_agg["ds"], y=proy_agg["yhat_lower"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)", showlegend=False, hoverinfo="skip"))
        fig_agg.add_trace(go.Scatter(
            x=proy_agg["ds"], y=proy_agg["yhat"], mode="lines",
            line=dict(color="#3b82f6", width=3), name="Proyección (suma)"))
    if len(hist_agg) > 0:
        fig_agg.add_vline(x=hist_agg["mes"].max(), line_width=1,
                          line_dash="dash", line_color="#94a3b8")
    fig_agg.update_layout(height=380, yaxis_title="Unidades / mes",
                          hovermode="x unified", margin=dict(t=10, b=0))
    st.plotly_chart(fig_agg, use_container_width=True)

    # Resumen anual agregado
    if len(proy_agg) > 0:
        proy_agg["anio"] = pd.to_datetime(proy_agg["ds"]).dt.year
        resumen_agg = proy_agg.groupby("anio")["yhat"].sum().reset_index()
        resumen_agg["yhat"] = resumen_agg["yhat"].astype(int)
        resumen_agg.columns = ["Año", "Proyección total (u)"]
        st.dataframe(resumen_agg, use_container_width=True, hide_index=True)

    st.info(
        "💡 Selecciona un ISBN específico en el desplegable de arriba para ver "
        "su detalle y aplicarle un override individual."
    )

elif es_novedad_sel:
    # ───────────── DETALLE DE UNA NOVEDAD APROBADA ─────────────
    # Las novedades viven en novedades_aprobadas.json (no en feature_isbn).
    # Aquí mostramos su curva proyectada (curva_mensual) y permitimos override.
    nid_sel = label_sel.replace("__NOV__", "")
    nov_obj = next((n for n in _todas_novedades if n.get("id") == nid_sel), None)
    if nov_obj is None:
        st.warning("No se encontró la novedad seleccionada.")
    else:
        origen_txt = ("Sugerencia automática" if nov_obj.get("origen") == "sugerencia_automatica"
                      else "Manual (simulador)")
        st.markdown(f"#### 🆕 {nov_obj.get('nombre', nid_sel)}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Origen", origen_txt.split()[0])
        m2.metric("Tipo TACO", nov_obj.get("tipo_taco", "—"))
        m3.metric("Lanzamiento", nov_obj.get("mes_lanzamiento", "—"))
        m4.metric("Ciclo (m)", nov_obj.get("ciclo_vida_meses", "—"))
        esc_nov_actual = float(nov_obj.get("override_escala", 1.0))
        cap_txt = "📈 FUERA" if nov_obj.get("fuera_capacidad") else "✅ dentro"
        m5.metric("Capacidad", cap_txt)

        id_prod = nov_obj.get("id_producto_manual")
        sbu = nov_obj.get("sbu_codigo", "")
        st.caption(
            (f"**ID producto (ISBN/SKU):** {id_prod}  ·  " if id_prod else "")
            + (f"**Código SBU:** `{sbu}`  ·  " if sbu else "")
            + (f"**Marca:** {nov_obj.get('marca')}" if nov_obj.get("marca") else "")
            + (f"  ·  **TACO destino:** {nov_obj.get('taco_destino','')}")
        )

        # Curva mensual de la novedad (aplicando su override)
        curva = nov_obj.get("curva_mensual", [])
        if curva:
            df_curva = pd.DataFrame(curva)
            df_curva["ds"] = pd.to_datetime(df_curva["ds"])
            for c in ["prediccion", "p10", "p90"]:
                if c in df_curva.columns:
                    df_curva[c] = pd.to_numeric(df_curva[c], errors="coerce").fillna(0) * esc_nov_actual

            fig_nov = go.Figure()
            if "p90" in df_curva.columns:
                fig_nov.add_trace(go.Scatter(
                    x=df_curva["ds"], y=df_curva["p90"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_nov.add_trace(go.Scatter(
                    x=df_curva["ds"], y=df_curva["p10"], mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(139,92,246,0.15)", showlegend=False, hoverinfo="skip"))
            fig_nov.add_trace(go.Scatter(
                x=df_curva["ds"], y=df_curva["prediccion"], mode="lines+markers",
                line=dict(color="#8b5cf6", width=2.5), name="Proyección novedad"))
            fig_nov.update_layout(
                height=360, yaxis_title="Unidades / mes",
                hovermode="x unified", margin=dict(t=10, b=20))
            st.plotly_chart(fig_nov, use_container_width=True)

            # Resumen anual
            df_curva["anio"] = df_curva["ds"].dt.year
            res_nov = df_curva.groupby("anio")["prediccion"].sum().reset_index()
            res_nov["prediccion"] = res_nov["prediccion"].astype(int)
            res_nov.columns = ["Año", "Demanda proyectada (u)"]
            st.dataframe(res_nov, use_container_width=True, hide_index=True)
        else:
            st.info("Esta novedad no tiene curva mensual registrada.")

        # Override de la novedad (multiplicador)
        st.markdown("**🎚️ Override de esta novedad**")
        st.caption(
            "Multiplica la curva de la novedad. **1.00** = sin cambio. "
            "Se refleja en el gap a metas, la curva total y el CSV final."
        )
        col_o1, col_o2 = st.columns([2, 1])
        with col_o1:
            nueva_esc_nov = st.number_input(
                "Multiplicador de demanda",
                min_value=0.0, max_value=3.0, value=esc_nov_actual, step=0.05,
                key=f"nov_detalle_ovr_{nid_sel}",
            )
        with col_o2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if abs(nueva_esc_nov - esc_nov_actual) > 1e-6:
                if st.button("💾 Aplicar override", key=f"nov_det_btn_{nid_sel}",
                             type="primary", use_container_width=True):
                    novedades_store.set_override_escala(nid_sel, nueva_esc_nov)
                    st.cache_data.clear()
                    st.rerun()
            elif esc_nov_actual != 1.0:
                if st.button("↩️ Quitar override", key=f"nov_det_rst_{nid_sel}",
                             use_container_width=True):
                    novedades_store.set_override_escala(nid_sel, 1.0)
                    st.cache_data.clear()
                    st.rerun()
        if esc_nov_actual != 1.0:
            st.info(f"📌 Override activo: ×{esc_nov_actual:.2f}")

else:
    isbn_sel = label_sel.split(" — ")[0].strip()
    # Tomamos la fila desde opciones_filtradas (que tiene tiene_proyeccion)
    row = isbns_filt[isbns_filt["isbn"] == isbn_sel].iloc[0]
    tiene_proy = bool(row["tiene_proyeccion"])

    # Etiqueta legible de categoría de precio
    _label_cat = {
        "economica": "Económica", "semi_economica": "Semi-económica",
        "media": "Media", "semi_fina": "Semi-fina", "fina": "Fina",
        "no_aplica": "No aplica",
    }
    cat_precio_isbn = str(row.get("categoria_precio", "no_aplica"))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clase", row["clase"])
    col2.metric("Estado", row["estado"])
    col3.metric("Categoría precio", _label_cat.get(cat_precio_isbn, cat_precio_isbn))
    col4.metric("Familia género", str(row.get("familia_genero", "—")))
    col5.metric("Mercado", str(row.get("mercado_principal", "—")))

    # Confirmar precio promedio si existe
    if "precio_promedio" in row and pd.notna(row.get("precio_promedio")):
        st.caption(
            f"💲 Precio promedio histórico: **${row['precio_promedio']:,.0f}** "
            f"· Categoría: **{_label_cat.get(cat_precio_isbn, cat_precio_isbn)}**"
        )

    if not tiene_proy:
        st.info(
            f"📌 Este ISBN NO tiene proyección Prophet calculada. Solo se muestra el "
            f"histórico. La proyección por categorías distintas a BIBLIAS está en backlog "
            f"(no hay aún suficiente data). Los ajustes manuales (overrides) no aplican aquí."
        )

# Detalle individual del ISBN (solo cuando NO es la vista agregada TODAS
# ni una novedad — las novedades tienen su propio bloque arriba)
if not es_vista_todas and not es_novedad_sel:
    # ─────────────────────────────────────────────────────────────────────────
    # AJUSTES MANUALES PARA ESTE ISBN (solo si tiene proyección)
    # ─────────────────────────────────────────────────────────────────────────
    # Permite override puntual a este ISBN. Si tiene override propio, ANULA el
    # de su categoría. Los cambios se aplican a TODAS las gráficas, tablas y CSV.
    if tiene_proy:
        with st.expander("🎚️ Ajustes manuales para este ISBN", expanded=False):
            st.caption(
                "El override de ISBN específico **anula** el de su categoría. "
                "Útil para ajustar ISBNs puntuales que se comportan distinto al promedio. "
                "Mueve los sliders y haz clic en **💾 Aplicar para este ISBN** para guardar."
            )
            ov_isbn = overrides_actuales.get("isbns", {}).get(isbn_sel, {})
            esc_isbn_guardado = float(ov_isbn.get("escala", 1.0))
            ciclo_isbn_guardado = float(ov_isbn.get("ciclo", 1.0))

            # Indicador del estado guardado (solo si hay override aplicado)
            if esc_isbn_guardado != 1.0 or ciclo_isbn_guardado != 1.0:
                st.caption(
                    f"_Aplicado actualmente para este ISBN: escala={esc_isbn_guardado:.2f}, "
                    f"ciclo={ciclo_isbn_guardado:.2f}_"
                )

            col_e, col_c = st.columns(2)
            with col_e:
                nuevo_esc_isbn = st.slider(
                    "Escala (multiplicador)",
                    min_value=0.3, max_value=3.0,
                    value=esc_isbn_guardado, step=0.05,
                    key=f"esc_isbn_{isbn_sel}",
                    help="1.0 = sin cambio. >1 mueve la curva arriba. <1 la mueve abajo.",
                )
            with col_c:
                nuevo_ciclo_isbn = st.slider(
                    "Ciclo de vida",
                    min_value=0.3, max_value=3.0,
                    value=ciclo_isbn_guardado, step=0.05,
                    key=f"ciclo_isbn_{isbn_sel}",
                    help="1.0 = ciclo normal. >1 alarga el ciclo. <1 lo acorta.",
                )

            # Detectar si hay cambios pendientes
            cambios_pend_isbn = (
                abs(nuevo_esc_isbn - esc_isbn_guardado) > 1e-6
                or abs(nuevo_ciclo_isbn - ciclo_isbn_guardado) > 1e-6
            )
            if cambios_pend_isbn:
                st.warning("⚠️ Tienes cambios pendientes para este ISBN sin aplicar")

            col_ap, col_rs = st.columns(2)
            with col_ap:
                if st.button(
                    "💾 Aplicar para este ISBN",
                    type="primary",
                    key=f"aplicar_isbn_{isbn_sel}",
                    use_container_width=True,
                    disabled=not cambios_pend_isbn,
                    help="Guarda el override y lo refleja en TODA la app",
                ):
                    overrides_store.set_isbn(isbn_sel, nuevo_esc_isbn, nuevo_ciclo_isbn)
                    st.cache_data.clear()
                    st.success(f"✓ Override guardado para {isbn_sel}. Refrescando...")
                    st.rerun()
            with col_rs:
                if st.button(
                    "🗑️ Quitar override de este ISBN",
                    key=f"reset_isbn_{isbn_sel}",
                    use_container_width=True,
                    disabled=(isbn_sel not in overrides_actuales.get("isbns", {})),
                    help="Elimina el override propio. El ISBN volverá a heredar el de su categoría (si lo hay).",
                ):
                    overrides_store.eliminar_isbn(isbn_sel)
                    for k in [f"esc_isbn_{isbn_sel}", f"ciclo_isbn_{isbn_sel}"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.cache_data.clear()
                    st.rerun()

            # Mostrar override efectivo (qué se está aplicando realmente al ISBN)
            eff = overrides_store.get_efectivo(isbn_sel, row["clase"])
            if eff["escala"] != 1.0 or eff["ciclo"] != 1.0:
                fuente = (
                    "ISBN específico" if isbn_sel in overrides_actuales.get("isbns", {})
                    else f"categoría '{row['clase']}'"
                )
                st.info(
                    f"📌 Override efectivo aplicado: escala={eff['escala']:.2f}, ciclo={eff['ciclo']:.2f} "
                    f"(viene de: {fuente})"
                )

    # ─── Gráfica ISBN con toggle Unidades vs Valor ───
    serie_isbn = serie[serie["isbn"] == isbn_sel].sort_values("mes").copy()
    proy_isbn = proy[proy["isbn"] == isbn_sel].sort_values("ds").copy() if tiene_proy else pd.DataFrame()

    usar_valor = (metrica_isbn == "Valor ($)")
    if usar_valor:
        col_hist_y = "valor"
        titulo_y = "Valor mensual (COP)"
        formato_y = ",.0f"
        # Para proyecciones, estimar valor usando el precio promedio del histórico
        if tiene_proy and len(serie_isbn) > 0:
            precio_avg = (serie_isbn["valor"].sum() / serie_isbn["unidades"].sum()) if serie_isbn["unidades"].sum() > 0 else 0
            proy_isbn["yhat_val"] = proy_isbn["yhat"] * precio_avg
            proy_isbn["yhat_lower_val"] = proy_isbn["yhat_lower"] * precio_avg
            proy_isbn["yhat_upper_val"] = proy_isbn["yhat_upper"] * precio_avg
            proy_y, proy_y_lo, proy_y_hi = "yhat_val", "yhat_lower_val", "yhat_upper_val"
        else:
            proy_y, proy_y_lo, proy_y_hi = None, None, None
    else:
        col_hist_y = "unidades"
        titulo_y = "Unidades mes"
        formato_y = ",.0f"
        proy_y, proy_y_lo, proy_y_hi = "yhat", "yhat_lower", "yhat_upper"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie_isbn["mes"], y=serie_isbn[col_hist_y],
        mode="lines+markers",
        line=dict(color="#1f4e79", width=2), marker=dict(size=6),
        name="Histórico real",
    ))
    if tiene_proy and proy_y is not None and len(proy_isbn) > 0:
        fig.add_trace(go.Scatter(
            x=proy_isbn["ds"], y=proy_isbn[proy_y_hi],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=proy_isbn["ds"], y=proy_isbn[proy_y_lo],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(245,158,11,0.2)",
            name="Banda p10-p90 (80%)",
        ))
        fig.add_trace(go.Scatter(
            x=proy_isbn["ds"], y=proy_isbn[proy_y],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dot"), marker=dict(size=6),
            name=f"Proyección ({proy_isbn['fuente'].iloc[0] if len(proy_isbn) else '?'})",
        ))
    fig.update_layout(
        height=420,
        yaxis_title=titulo_y,
        yaxis=dict(tickformat=formato_y),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Histórico anual ({'valor COP' if usar_valor else 'unidades'})**")
        if len(serie_isbn) > 0:
            s_year = serie_isbn.copy()
            s_year["año"] = s_year["mes"].dt.year
            hist_anual = s_year.groupby("año")[col_hist_y].sum().reset_index()
            hist_anual[col_hist_y] = hist_anual[col_hist_y].apply(lambda x: f"{int(x):,}")
            st.dataframe(hist_anual.rename(columns={col_hist_y: ("Valor (COP)" if usar_valor else "Unidades")}),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("Sin histórico disponible para este ISBN.")

    with col2:
        st.markdown(f"**Proyección anual ({'valor COP estimado' if usar_valor else 'unidades'})**")
        if tiene_proy and len(proy_isbn) > 0:
            p_year = proy_isbn.copy()
            p_year["año"] = pd.to_datetime(p_year["ds"]).dt.year
            if usar_valor:
                cols_agg = {"prediccion": (proy_y, "sum"), "p10": (proy_y_lo, "sum"), "p90": (proy_y_hi, "sum")}
            else:
                cols_agg = {"prediccion": ("yhat", "sum"), "p10": ("yhat_lower", "sum"), "p90": ("yhat_upper", "sum")}
            proy_anual = p_year.groupby("año").agg(**cols_agg).reset_index()
            for c in ["prediccion", "p10", "p90"]:
                proy_anual[c] = proy_anual[c].apply(lambda x: f"{int(x):,}")
            st.dataframe(proy_anual, hide_index=True, use_container_width=True)
        else:
            st.caption("Sin proyección Prophet para este ISBN.")

    # ─────────────────────────────────────────────────────────────────────────
    # v3.9 — TABLA DE ESTADÍSTICOS DESCRIPTIVOS MENSUALES
    # ─────────────────────────────────────────────────────────────────────────
    # Se calculan sobre la serie MENSUAL (no la anual) porque ahí está la
    # variabilidad útil para diagnóstico: estabilidad (CoV), asimetría,
    # concentración. Se muestran SIEMPRE las dos columnas (unidades y valor),
    # independientemente del toggle de visualización, para que Alberto vea el
    # panorama estadístico completo de una sola pasada.
    st.markdown("**📊 Estadísticos descriptivos — histórico mensual del ISBN**")
    st.caption(
        "Cálculos sobre la serie mensual histórica. Útil para clasificar "
        "estabilidad (CoV bajo = serie X de XYZ), detectar asimetría "
        "(skewness > 0 = pocos meses muy altos) y dimensionar dispersión "
        "(σ, IQR). Desv. estándar y varianza son muestrales (n-1)."
    )
    if len(serie_isbn) > 0:
        tabla_stats_isbn = calcular_estadisticos_mensuales(
            serie_isbn, col_unidades="unidades", col_valor="valor",
        )
        st.dataframe(tabla_stats_isbn, hide_index=True, use_container_width=True)
    else:
        st.caption("Sin histórico mensual disponible para este ISBN.")

    if tiene_proy and len(proy_isbn) > 0:
        st.caption(
            f"💡 ISBN clasificado como **{proy_isbn['nivel_madurez'].iloc[0]}** | "
            f"fuente: **{proy_isbn['fuente'].iloc[0]}**."
        )

# ─────────────────────────────────────────────────────────────────────────
# OVERRIDE DE NOVEDADES (v3.12) — sugeridas automáticas + aprobadas
# ─────────────────────────────────────────────────────────────────────────
# HISTÓRICO DE OVERRIDES MANUALES DE ESTE EXPLORADOR (v3.12)
# ─────────────────────────────────────────────────────────────────────────
# Consolida en un solo lugar todos los ajustes manuales hechos DESDE este
# explorador: overrides por ISBN específico y overrides de novedades. Permite
# ver de un vistazo qué se ha tocado y revertir cualquiera.
st.divider()
st.subheader("🧾 Histórico de overrides manuales (de este explorador)")

_ov_actuales = overrides_store.cargar()
_ov_isbns = _ov_actuales.get("isbns", {})
_novs_con_ovr = [
    n for n in novedades_store.cargar()
    if n.get("estado") == "aprobado" and abs(float(n.get("override_escala", 1.0)) - 1.0) > 1e-9
]

if not _ov_isbns and not _novs_con_ovr:
    st.caption(
        "Aún no has aplicado overrides manuales por ISBN ni a novedades desde "
        "este explorador. Cuando ajustes un ISBN o una novedad (seleccionándolos "
        "arriba), el cambio quedará registrado aquí para que puedas revisarlo o "
        "revertirlo."
    )
else:
    filas_hist = []
    # Overrides por ISBN
    for isbn_id, vals in _ov_isbns.items():
        meta_isbn = isbn[isbn["isbn"] == isbn_id]
        desc = (meta_isbn["descripcion"].iloc[0][:45]
                if len(meta_isbn) and pd.notna(meta_isbn["descripcion"].iloc[0]) else "")
        filas_hist.append({
            "Tipo": "📕 ISBN",
            "ID / Nombre": f"{isbn_id} · {desc}",
            "Escala": f"×{vals.get('escala', 1.0):.2f}",
            "Ciclo": f"×{vals.get('ciclo', 1.0):.2f}",
            "_key": f"isbn::{isbn_id}",
        })
    # Overrides de novedades
    for n in _novs_con_ovr:
        filas_hist.append({
            "Tipo": "🆕 Novedad",
            "ID / Nombre": f"{n.get('nombre', n.get('id',''))[:50]}",
            "Escala": f"×{float(n.get('override_escala', 1.0)):.2f}",
            "Ciclo": "—",
            "_key": f"nov::{n.get('id','')}",
        })

    st.caption(
        f"**{len(filas_hist)}** override(s) manual(es) activo(s). "
        f"Marca los que quieras revertir y haz clic en _Revertir seleccionados_."
    )
    df_hist = pd.DataFrame(filas_hist)
    df_hist_show = df_hist.drop(columns=["_key"]).copy()
    df_hist_show.insert(0, "Revertir", False)

    edited_hist = st.data_editor(
        df_hist_show,
        hide_index=True,
        use_container_width=True,
        column_config={"Revertir": st.column_config.CheckboxColumn("Revertir", default=False)},
        key="hist_overrides_editor",
    )
    if st.button("↩️ Revertir overrides seleccionados", key="btn_revertir_hist"):
        n_revertidos = 0
        for i, marcado in enumerate(edited_hist["Revertir"].tolist()):
            if marcado:
                key = df_hist.iloc[i]["_key"]
                tipo, ident = key.split("::", 1)
                if tipo == "isbn":
                    overrides_store.eliminar_isbn(ident)
                    for k in [f"esc_isbn_{ident}", f"ciclo_isbn_{ident}"]:
                        st.session_state.pop(k, None)
                    n_revertidos += 1
                elif tipo == "nov":
                    novedades_store.set_override_escala(ident, 1.0)
                    n_revertidos += 1
        if n_revertidos > 0:
            st.cache_data.clear()
            st.success(f"✓ {n_revertidos} override(s) revertido(s).")
            st.rerun()
        else:
            st.info("No marcaste ningún override para revertir.")

# =========================================================================
# EXPLORADOR DE PROYECCIONES POR TACO MP — v3.11
# =========================================================================
# Agrega los ISBNs por su TACO MP (materia prima semielaborada) y muestra:
#   - Card con info del catálogo (descripción larga, stock, valor, tamaño)
#   - Histórico + proyección agregada (suma de ISBNs del TACO)
#   - Días de cobertura (stock ÷ demanda mensual promedio)
#   - Tabla de ISBNs que componen el TACO con su proyección individual
#   - Alerta de migración si el TACO migra a CLARIDAD
st.divider()
st.subheader("Explorador de proyecciones por TACO MP")
st.caption(
    "Agrega los ISBNs según su materia prima semielaborada (taco impreso por "
    "el tercero) y proyecta la demanda total del taco. Útil para planeación "
    "de compras, programación de la planta de encuadernación y monitoreo de "
    "stock vs cobertura."
)

# Importar catálogo MP (con safe fallback)
try:
    from utils.catalogo_mp import (
        cargar_catalogo as cargar_catalogo_mp,
        info_taco_mp,
        descripcion_taco_mp,
        label_taco_mp,
        labels_de_lista,
    )
    _catalogo_mp_ok = True
except Exception as _e:
    cargar_catalogo_mp = info_taco_mp = descripcion_taco_mp = None
    label_taco_mp = lambda c, **k: str(c)
    labels_de_lista = lambda codigos: {c: str(c) for c in (codigos or [])}
    _catalogo_mp_ok = False

# Obtener lista de TACOs MP presentes en el catálogo de ISBNs
tacos_en_isbn = (
    isbn[isbn["taco_mp"].notna() & (isbn["taco_mp"].astype(str).str.strip() != "")]
    ["taco_mp"].astype(str).str.strip().unique().tolist()
)

# v3.12 (fix): incluir también los TACOs destino de las NOVEDADES aprobadas.
# Cuando creas una novedad sobre un TACO existente o creas un TACO nuevo, ese
# TACO debe aparecer aquí aunque todavía no tenga ISBNs reales en el feature store.
_novedades_taco = [
    n for n in novedades_store.cargar() if n.get("estado") == "aprobado"
]
# Mapa taco_destino → lista de novedades de ese taco
_novs_por_taco = {}
for n in _novedades_taco:
    td = str(n.get("taco_destino", "")).strip()
    if td:
        _novs_por_taco.setdefault(td, []).append(n)

# Unir ambas fuentes
tacos_set = set(tacos_en_isbn) | set(_novs_por_taco.keys())
tacos_en_isbn = sorted(t for t in tacos_set if t)

if len(tacos_en_isbn) == 0:
    st.warning(
        "No hay ISBNs con TACO MP asignado en el feature store. "
        "Verifica que `feature_isbn.parquet` tenga la columna `taco_mp` poblada."
    )
else:
    # Construir labels código — descripción larga
    labels_taco = labels_de_lista(tacos_en_isbn)
    # Inyectar también los que NO están en catálogo MP (sin descripción)
    for t in tacos_en_isbn:
        if t not in labels_taco:
            labels_taco[t] = str(t)

    # Búsqueda + selector
    col_busq, col_sel = st.columns([1, 2])
    with col_busq:
        busqueda_taco = st.text_input(
            "🔎 Filtrar TACO por código o descripción",
            value="",
            key="taco_mp_search",
            placeholder="ej: RVR060, CLARIDAD, SE1130, NOVEDAD",
        )

    tacos_filtrados = list(tacos_en_isbn)
    if busqueda_taco:
        b = busqueda_taco.upper()
        tacos_filtrados = [
            t for t in tacos_en_isbn
            if b in t.upper()
            or b in labels_taco.get(t, "").upper()
            # también permitir encontrar TACOs que solo tienen novedades
            or (t in _novs_por_taco and any(
                b in str(nv.get("nombre", "")).upper() for nv in _novs_por_taco[t]))
        ]
        if len(tacos_filtrados) == 0:
            st.info(f"Sin coincidencias para '{busqueda_taco}'. Mostrando todos los TACOs.")
            tacos_filtrados = list(tacos_en_isbn)

    with col_sel:
        if len(tacos_filtrados) > 0:
            TODAS_TACOS = "🌐 TODOS los TACOs (vista agregada)"
            opciones_taco = [TODAS_TACOS] + tacos_filtrados
            # Marca visual para TACOs que tienen novedades asociadas
            def _fmt_taco(c):
                if c == TODAS_TACOS:
                    return c
                base = labels_taco.get(c, str(c))[:88]
                if c in _novs_por_taco:
                    base = f"🆕 {base}"
                return base
            taco_sel_label = st.selectbox(
                "TACO MP",
                options=opciones_taco,
                index=0,  # TODOS es el default al abrir
                format_func=_fmt_taco,
                key="taco_mp_selected",
                label_visibility="collapsed",
            )
            es_todos_tacos = (taco_sel_label == TODAS_TACOS)
            taco_sel = None if es_todos_tacos else taco_sel_label
        else:
            taco_sel = None
            es_todos_tacos = False

    # ───────────── VISTA AGREGADA DE TODOS LOS TACOS ─────────────
    if es_todos_tacos:
        st.markdown(f"#### Vista agregada · {len(tacos_filtrados)} TACOs MP")
        filas_resumen = []
        for t in tacos_filtrados:
            isbns_t = isbn[isbn["taco_mp"].astype(str).str.strip() == str(t).strip()]
            proy_t = proy[proy["isbn"].isin(isbns_t["isbn"])]
            proy_t_anual = proy_t.groupby(pd.to_datetime(proy_t["ds"]).dt.year)["yhat"].sum() if len(proy_t) else {}
            # Sumar demanda de novedades de este taco (con override)
            nov_2027 = nov_2030 = 0.0
            for nv in _novs_por_taco.get(str(t).strip(), []):
                esc = float(nv.get("override_escala", 1.0))
                for e in nv.get("curva_mensual", []):
                    if isinstance(e, dict):
                        anio = pd.Timestamp(e.get("ds")).year
                        val = float(e.get("prediccion", 0)) * esc
                        if anio == 2027: nov_2027 += val
                        elif anio == 2030: nov_2030 += val
            info_t = info_taco_mp(t) if _catalogo_mp_ok else None
            n_novs_t = len(_novs_por_taco.get(str(t).strip(), []))
            filas_resumen.append({
                "TACO": ("🆕 " if n_novs_t > 0 else "") + str(t),
                "Descripción": (info_t.get("descripcion", "") if info_t else "")[:55],
                "ISBNs": len(isbns_t),
                "Novedades": n_novs_t,
                "Stock": int(info_t.get("stock_unidades", 0)) if info_t else 0,
                "Proy. 2027": int((proy_t_anual.get(2027, 0) if len(proy_t) else 0) + nov_2027),
                "Proy. 2030": int((proy_t_anual.get(2030, 0) if len(proy_t) else 0) + nov_2030),
            })
        df_resumen_tacos = pd.DataFrame(filas_resumen).sort_values("Proy. 2027", ascending=False)
        st.dataframe(df_resumen_tacos, use_container_width=True, hide_index=True,
                     height=min(38 + 35 * len(df_resumen_tacos), 500))
        st.info("💡 Selecciona un TACO específico arriba para ver su detalle, "
                "días de cobertura y el comportamiento de cada ISBN que lo usa. "
                "Los marcados con 🆕 tienen novedades asignadas.")

    if taco_sel is not None:
        # --- Card de info del catálogo MP ---
        info_mp = info_taco_mp(taco_sel) if _catalogo_mp_ok else None
        descripcion_mp = (info_mp.get("descripcion", "")
                          if info_mp else "(no en catálogo MP)")

        # ISBNs que pertenecen al TACO
        isbns_del_taco = (
            isbn[isbn["taco_mp"].astype(str).str.strip() == str(taco_sel).strip()]
        )
        # Histórico mensual del TACO (suma de sus ISBNs)
        hist_taco = serie[serie["isbn"].isin(isbns_del_taco["isbn"])]
        hist_mensual = (
            hist_taco.groupby("mes", as_index=False)
            .agg(unidades=("unidades", "sum"), valor=("valor", "sum"))
        )
        # Proyección del TACO (suma de proyección de sus ISBNs)
        proy_taco = proy[proy["isbn"].isin(isbns_del_taco["isbn"])]
        proy_taco_total = (
            proy_taco.groupby("ds", as_index=False)
            .agg(
                yhat=("yhat", "sum"),
                yhat_lower=("yhat_lower", "sum"),
                yhat_upper=("yhat_upper", "sum"),
            )
        )

        # v3.12 (fix): sumar las NOVEDADES asignadas a este TACO (taco_destino).
        # Su curva mensual (con override_escala) se agrega a la proyección del taco,
        # para que un TACO con novedades — o un TACO nuevo creado vía novedad —
        # muestre su demanda proyectada.
        _novs_de_este_taco = _novs_por_taco.get(str(taco_sel).strip(), [])
        if _novs_de_este_taco:
            filas_nov_curva = []
            for nv in _novs_de_este_taco:
                esc = float(nv.get("override_escala", 1.0))
                for e in nv.get("curva_mensual", []):
                    if isinstance(e, dict):
                        filas_nov_curva.append({
                            "ds": pd.Timestamp(e.get("ds")),
                            "yhat": float(e.get("prediccion", 0)) * esc,
                            "yhat_lower": float(e.get("p10", 0)) * esc,
                            "yhat_upper": float(e.get("p90", 0)) * esc,
                        })
            if filas_nov_curva:
                df_nov_curva = (
                    pd.DataFrame(filas_nov_curva)
                    .groupby("ds", as_index=False)
                    .agg(yhat=("yhat", "sum"), yhat_lower=("yhat_lower", "sum"),
                         yhat_upper=("yhat_upper", "sum"))
                )
                # Combinar con la proyección de ISBNs reales (sumar por mes)
                if len(proy_taco_total) > 0:
                    proy_taco_total = (
                        pd.concat([proy_taco_total, df_nov_curva], ignore_index=True)
                        .groupby("ds", as_index=False)
                        .agg(yhat=("yhat", "sum"), yhat_lower=("yhat_lower", "sum"),
                             yhat_upper=("yhat_upper", "sum"))
                    )
                else:
                    proy_taco_total = df_nov_curva

        # KPIs de stock vs demanda
        stock = (info_mp.get("stock_unidades", 0) or 0) if info_mp else 0
        comprometido = (info_mp.get("comprometido", 0) or 0) if info_mp else 0
        valor_unidad = (info_mp.get("valor_unidad_cop", 0) or 0) if info_mp else 0
        # Demanda promedio mensual últimos 12 meses
        if len(hist_mensual) >= 6:
            hist_last12 = hist_mensual.sort_values("mes").tail(12)
            dem_mens_prom = float(hist_last12["unidades"].mean())
        else:
            dem_mens_prom = 0.0
        # Días de cobertura: stock / (dem_mens / 30)
        if dem_mens_prom > 0:
            dias_cobertura = (stock - comprometido) / (dem_mens_prom / 30)
        else:
            dias_cobertura = None

        # ----- Tarjetas en pantalla -----
        # Header con descripción larga
        st.markdown(
            f"<div style='background:#f1f5f9;padding:14px;border-radius:8px;"
            f"border-left:4px solid #1f4e79;margin-bottom:14px;'>"
            f"<div style='color:#0f172a;font-size:13px;'>"
            f"<b>Código:</b> <code style='color:#0f172a'>{taco_sel}</code></div>"
            f"<div style='color:#0f172a;font-size:16px;font-weight:600;"
            f"margin-top:4px;'>{descripcion_mp}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("ISBNs en TACO", f"{len(isbns_del_taco)}")
        kpi_cols[1].metric("Stock actual", f"{stock:,.0f}")
        kpi_cols[2].metric("Comprometido", f"{comprometido:,.0f}")
        if dias_cobertura is not None:
            label_cob = "🟢" if dias_cobertura > 90 else ("🟡" if dias_cobertura > 30 else "🔴")
            kpi_cols[3].metric(f"Días cobertura {label_cob}",
                                f"{dias_cobertura:,.0f}d")
        else:
            kpi_cols[3].metric("Días cobertura", "—")
        if valor_unidad > 0:
            kpi_cols[4].metric("Valor unitario",
                                f"${valor_unidad:,.0f}")
        else:
            kpi_cols[4].metric("Valor unitario", "—")

        # v3.12: aviso si este TACO tiene novedades aprobadas asignadas
        if _novs_de_este_taco:
            nombres_nov = ", ".join(n.get("nombre", "")[:35] for n in _novs_de_este_taco[:5])
            extra = f" (+{len(_novs_de_este_taco)-5} más)" if len(_novs_de_este_taco) > 5 else ""
            es_taco_nuevo = len(isbns_del_taco) == 0
            if es_taco_nuevo:
                st.success(
                    f"🆕 **TACO nuevo creado vía novedad** — todavía sin ISBNs reales. "
                    f"La proyección que ves proviene de **{len(_novs_de_este_taco)} "
                    f"novedad(es)**: {nombres_nov}{extra}."
                )
            else:
                st.info(
                    f"🆕 Este TACO tiene **{len(_novs_de_este_taco)} novedad(es)** "
                    f"asignada(s) cuya demanda ya está sumada en la proyección: "
                    f"{nombres_nov}{extra}."
                )

        # Tamaño físico y atributos derivados (subline)
        if info_mp:
            atributos = []
            if info_mp.get("tamano_ancho_cm") and info_mp.get("tamano_alto_cm"):
                atributos.append(
                    f"📐 {info_mp['tamano_ancho_cm']:g} × {info_mp['tamano_alto_cm']:g} cm"
                )
            if info_mp.get("version_biblica"):
                atributos.append(f"📖 {info_mp['version_biblica']}")
            if info_mp.get("gramaje") and info_mp["gramaje"] > 0:
                atributos.append(f"⚖️ {info_mp['gramaje']:g} g")
            if info_mp.get("bodega"):
                atributos.append(f"🏭 Bodega {info_mp['bodega']}")
            if atributos:
                st.caption(" · ".join(atributos))

        # Alerta de migración a CLARIDAD (si aplica)
        migra = isbns_del_taco["migra_a_claridad"].sum() if "migra_a_claridad" in isbns_del_taco.columns else 0
        if migra > 0:
            taco_destino = (
                isbns_del_taco[isbns_del_taco["migra_a_claridad"] == True]
                ["taco_destino_claridad"].dropna().unique()
            )
            destino_str = ", ".join(taco_destino) if len(taco_destino) > 0 else "CLARIDAD"
            st.warning(
                f"⚠️ **{int(migra)} ISBN(s)** de este TACO migran al programa "
                f"**CLARIDAD** ({destino_str}). La proyección refleja este cambio "
                f"según el cronograma."
            )

        # ----- Gráfica histórico + proyección -----
        st.markdown("**Histórico + proyección 2027-2030**")
        fig_taco = go.Figure()
        if len(hist_mensual) > 0:
            fig_taco.add_trace(go.Scatter(
                x=hist_mensual["mes"], y=hist_mensual["unidades"],
                mode="lines",
                line=dict(color="#1f4e79", width=2),
                name="Histórico",
            ))
        if len(proy_taco_total) > 0:
            fig_taco.add_trace(go.Scatter(
                x=proy_taco_total["ds"], y=proy_taco_total["yhat_upper"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig_taco.add_trace(go.Scatter(
                x=proy_taco_total["ds"], y=proy_taco_total["yhat_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
                showlegend=False, hoverinfo="skip",
            ))
            fig_taco.add_trace(go.Scatter(
                x=proy_taco_total["ds"], y=proy_taco_total["yhat"],
                mode="lines",
                line=dict(color="#3b82f6", width=3),
                name="Proyección",
            ))
        # Línea vertical de corte histórico/proyección
        if len(hist_mensual) > 0:
            fecha_corte = hist_mensual["mes"].max()
            fig_taco.add_vline(
                x=fecha_corte, line_width=1,
                line_dash="dash", line_color="#94a3b8",
            )
        fig_taco.update_layout(
            height=350,
            yaxis_title="Unidades / mes",
            xaxis_title="Mes",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_taco, use_container_width=True)

        # ----- Resumen anual -----
        if len(proy_taco_total) > 0:
            proy_taco_total["anio"] = pd.to_datetime(proy_taco_total["ds"]).dt.year
            resumen_taco = (
                proy_taco_total.groupby("anio", as_index=False)
                .agg(
                    proyectado=("yhat", "sum"),
                    p10=("yhat_lower", "sum"),
                    p90=("yhat_upper", "sum"),
                )
            )
            # Histórico anual para comparar
            if len(hist_mensual) > 0:
                hist_anual = hist_mensual.copy()
                hist_anual["anio"] = pd.to_datetime(hist_anual["mes"]).dt.year
                hist_resumen = (
                    hist_anual.groupby("anio", as_index=False)
                    .agg(historico=("unidades", "sum"))
                )
                resumen_view = pd.concat([
                    hist_resumen.assign(fase="Histórico").rename(columns={"historico": "unidades"}),
                    resumen_taco.assign(fase="Proyección").rename(columns={"proyectado": "unidades"})[["anio", "unidades", "fase"]],
                ], ignore_index=True)
            else:
                resumen_view = resumen_taco.rename(columns={"proyectado": "unidades"})[["anio", "unidades"]]
                resumen_view["fase"] = "Proyección"

            col_tab_a, col_tab_b = st.columns(2)
            with col_tab_a:
                st.markdown("**Resumen anual (unidades)**")
                tabla_anual = resumen_view.pivot_table(
                    index="anio", columns="fase", values="unidades", aggfunc="sum"
                ).fillna(0).astype(int)
                st.dataframe(tabla_anual, use_container_width=True)
            with col_tab_b:
                st.markdown("**Intervalo 80% de proyección**")
                int_view = resumen_taco.copy()
                int_view["p10"] = int_view["p10"].astype(int)
                int_view["proyectado"] = int_view["proyectado"].astype(int)
                int_view["p90"] = int_view["p90"].astype(int)
                int_view.columns = ["Año", "p10 (bajo)", "Proyección", "p90 (alto)"]
                st.dataframe(int_view, use_container_width=True, hide_index=True)

        # ----- Tabla de ISBNs del TACO -----
        with st.expander(f"Ver los {len(isbns_del_taco)} ISBN(s) de este TACO"):
            cols_isbn_show = ["isbn", "descripcion", "categoria_precio",
                               "color_dominante", "demanda_anual_madura", "estado"]
            cols_isbn_show = [c for c in cols_isbn_show if c in isbns_del_taco.columns]
            isbns_view = isbns_del_taco[cols_isbn_show].copy()
            if "demanda_anual_madura" in isbns_view.columns:
                isbns_view["demanda_anual_madura"] = (
                    isbns_view["demanda_anual_madura"].fillna(0).astype(int)
                )
                isbns_view = isbns_view.sort_values("demanda_anual_madura", ascending=False)
            st.dataframe(isbns_view, use_container_width=True, hide_index=True, height=300)

        # ----- Gráfica de líneas: comportamiento de cada ISBN del TACO -----
        st.markdown("**Comportamiento individual de cada ISBN del TACO**")
        st.caption(
            "Cada línea es un ISBN que usa este taco. Pasa el cursor para ver "
            "ISBN y descripción. Útil para ver cuáles cubiertas jalan la demanda "
            "y cuáles están cayendo."
        )
        metr_taco_isbn = st.radio(
            "Métrica", options=["📦 Unidades", "💰 Valor ($)"],
            horizontal=True, key=f"metr_taco_isbn_{taco_sel}",
        )
        usar_valor_ti = (metr_taco_isbn == "💰 Valor ($)")
        col_y_ti = "valor" if usar_valor_ti else "unidades"

        isbns_lista_taco = isbns_del_taco["isbn"].tolist()
        # Mapa isbn → descripción corta para la leyenda
        desc_map = dict(zip(
            isbns_del_taco["isbn"].astype(str),
            isbns_del_taco["descripcion"].fillna("").astype(str),
        ))
        serie_por_isbn = serie[serie["isbn"].isin(isbns_lista_taco)]

        if len(serie_por_isbn) == 0:
            st.info("No hay histórico mensual para los ISBNs de este TACO.")
        else:
            # Paleta rotativa
            paleta = ['#3b82f6', '#10b981', '#f59e0b', '#dc2626', '#8b5cf6',
                      '#0891b2', '#7c3aed', '#059669', '#ea580c', '#1d4ed8',
                      '#1f4e79', '#be185d', '#0d9488', '#b45309', '#4f46e5']
            fig_lineas = go.Figure()
            # Ordenar ISBNs por volumen histórico para asignar colores estables
            orden_isbns = (
                serie_por_isbn.groupby("isbn")[col_y_ti].sum()
                .sort_values(ascending=False).index.tolist()
            )
            for i, isbn_t in enumerate(orden_isbns):
                s_t = serie_por_isbn[serie_por_isbn["isbn"] == isbn_t].sort_values("mes")
                if len(s_t) == 0:
                    continue
                desc_corta = desc_map.get(str(isbn_t), "")[:40]
                nombre_linea = f"{isbn_t} · {desc_corta}" if desc_corta else str(isbn_t)
                fig_lineas.add_trace(go.Scatter(
                    x=s_t["mes"], y=s_t[col_y_ti],
                    mode="lines",
                    line=dict(color=paleta[i % len(paleta)], width=1.8),
                    name=nombre_linea[:55],
                    hovertemplate=(
                        f"<b>{isbn_t}</b><br>{desc_corta}<br>"
                        "%{x|%Y-%m}: %{y:,.0f}<extra></extra>"
                    ),
                ))
            fig_lineas.update_layout(
                height=420,
                yaxis_title=("Valor (COP)" if usar_valor_ti else "Unidades / mes"),
                yaxis=dict(tickformat=",.0f"),
                hovermode="closest",
                legend=dict(orientation="v", x=1.02, y=1, font=dict(size=9)),
                margin=dict(t=10, b=30, r=10),
            )
            st.plotly_chart(fig_lineas, use_container_width=True)
            if len(orden_isbns) > 12:
                st.caption(
                    f"💡 Este TACO tiene {len(orden_isbns)} ISBNs. La leyenda "
                    f"puede quedar larga; haz clic en un nombre para ocultar/mostrar "
                    f"esa línea."
                )
# =========================================================================
# Permite explorar ventas históricas por cliente (Razón social cliente factura).
# Filtros LOCALES (no afectan otras secciones ni páginas).
# v3.9 agrega filtros encadenados por versión y categoría de precio
# (requiere parquet regenerado con build_features.py de v3.9).
# Aún no hay proyecciones por cliente — solo histórico.
st.divider()
st.subheader("Explorador de proyecciones por cliente")
st.caption(
    "Histórico de compras por cliente (Razón social) y, cuando aplique, "
    "proyección 2027-2030. El modelo de cliente (v3.11) clasifica cada uno "
    "por perfil y solo proyecta los que tienen historia suficiente."
)

# Cargar proyecciones y perfiles (si existen)
try:
    from models.cliente_forecast import (
        cargar_proyecciones_cliente,
        cargar_perfiles_cliente,
    )
    proy_cli_all = cargar_proyecciones_cliente()
    perfiles_cli = cargar_perfiles_cliente()
except Exception:
    proy_cli_all = None
    perfiles_cli = None

# Banner con la distribución de perfiles + dato impactante de inactivos
if perfiles_cli is not None and len(perfiles_cli) > 0:
    valor_total = perfiles_cli["valor_total"].sum()
    # v3.15: usar la columna `categoria` (negocio). Retrocompat si no existe.
    if "categoria" not in perfiles_cli.columns:
        _map_cat = {"RECURRENTE_GRANDE": "PROYECTABLE", "RECURRENTE_MEDIO": "PROYECTABLE",
                    "INACTIVO": "INACTIVO", "NUEVO_NO_PROYECTABLE": "NUEVO",
                    "ESPORADICO": "ESPORADICO", "ESPORADICO_GRANDE": "ESPORADICO",
                    "OTRO_NO_PROYECTABLE": "ESPORADICO"}
        perfiles_cli = perfiles_cli.copy()
        perfiles_cli["categoria"] = perfiles_cli["perfil"].map(_map_cat).fillna("NUEVO")

    def _cat_grp(cat):
        return perfiles_cli[perfiles_cli["categoria"] == cat]
    proyectables = _cat_grp("PROYECTABLE")
    inactivos    = _cat_grp("INACTIVO")
    nuevos       = _cat_grp("NUEVO")
    esporadicos  = _cat_grp("ESPORADICO")
    pct_proy = (proyectables["valor_total"].sum() / valor_total * 100) if valor_total > 0 else 0
    pct_inactivo = (inactivos["valor_total"].sum() / valor_total * 100) if valor_total > 0 else 0

    col_kpi_cli = st.columns(5)
    col_kpi_cli[0].metric("Total clientes", f"{len(perfiles_cli):,}")
    col_kpi_cli[1].metric("Proyectables", f"{len(proyectables):,}",
                           f"{pct_proy:.0f}% del valor")
    col_kpi_cli[2].metric("Inactivos (>12m)", f"{len(inactivos):,}",
                           f"{pct_inactivo:.0f}% del valor", delta_color="inverse")
    col_kpi_cli[3].metric("Nuevos (<12m histórico)", f"{len(nuevos):,}")
    col_kpi_cli[4].metric("Esporádicos", f"{len(esporadicos):,}")

    if pct_inactivo > 30:
        st.warning(
            f"⚠️ El **{pct_inactivo:.0f}%** del valor histórico viene de clientes "
            f"que NO han comprado en los últimos 12 meses (perfil INACTIVO). "
            f"Eso pesa sobre la base; las metas a futuro dependen casi por "
            f"completo de recuperar inactivos o ganar nuevos."
        )

    # =====================================================================
    # PRESUPUESTO TOTAL POR CLIENTE 2027-2030 (v3.14)
    # =====================================================================
    # Suma de la proyección de TODOS los clientes proyectables, con overrides
    # aplicados. Es el presupuesto en unidades e ingresos para los próximos años.
    from models import overrides_cliente_store as _ovc
    if proy_cli_all is not None and len(proy_cli_all) > 0:
        st.markdown("#### 🎯 Presupuesto proyectado por cliente · 2027-2030")
        _ov_cli_dict = _ovc.cargar()
        proy_pres = _ovc.aplicar_overrides(proy_cli_all)
        proy_pres["anio"] = pd.to_datetime(proy_pres["ds"]).dt.year
        _cv = "prediccion_valor" if "prediccion_valor" in proy_pres else "prediccion"
        _cu = "prediccion_unidades" if "prediccion_unidades" in proy_pres else None
        agg_dict = {"valor": (_cv, "sum")}
        if _cu:
            agg_dict["unidades"] = (_cu, "sum")
        pres_anual = proy_pres.groupby("anio").agg(**agg_dict).reset_index()

        # También la base (sin override) para mostrar el delta del presupuesto
        proy_base = proy_cli_all.copy()
        proy_base["anio"] = pd.to_datetime(proy_base["ds"]).dt.year
        base_anual_t = proy_base.groupby("anio").agg(**agg_dict).reset_index()

        ctot = st.columns(4)
        for i, a in enumerate([2027, 2028, 2029, 2030]):
            fila = pres_anual[pres_anual["anio"] == a]
            if len(fila):
                v = fila["valor"].iloc[0]
                u = fila["unidades"].iloc[0] if _cu else 0
                vbase = base_anual_t[base_anual_t["anio"] == a]["valor"].iloc[0]
                delta = (v / vbase - 1) * 100 if vbase > 0 else 0
                ctot[i].metric(
                    f"{a}", f"${v/1e9:.2f}B",
                    f"{delta:+.0f}% vs base" if abs(delta) > 0.5 else "= base",
                    delta_color="normal",
                )
                ctot[i].caption(f"{u:,.0f} u" if _cu else "")
        n_ov = len(_ov_cli_dict)
        st.caption(
            f"Suma de {proy_cli_all['cliente'].nunique()} clientes proyectables · "
            + (f"**{n_ov} con ajuste manual** aplicado." if n_ov
               else "sin ajustes manuales (proyección base del modelo).")
        )

        # Histórico de overrides de cliente
        if n_ov > 0:
            with st.expander(f"🧾 Ajustes de presupuesto activos ({n_ov} clientes)",
                             expanded=False):
                filas_ov = []
                for cli, o in _ov_cli_dict.items():
                    filas_ov.append({
                        "Cliente": cli[:50],
                        "Escala": f"×{o.get('escala',1):.2f}",
                        "Crec. anual": f"{o.get('crecimiento_anual_pct',0):+.0f}%",
                        "Nota": o.get("nota", ""),
                        "Actualizado": o.get("actualizado", "")[:16],
                    })
                st.dataframe(pd.DataFrame(filas_ov), hide_index=True,
                             use_container_width=True)
                if st.button("↩️ Quitar TODOS los ajustes de cliente",
                             key="reset_todos_ovc"):
                    n = _ovc.reset_todos()
                    st.cache_data.clear()
                    st.success(f"✓ {n} ajustes eliminados.")
                    st.rerun()
        st.divider()

if serie_cli is None or len(serie_cli) == 0:
    st.warning(
        "⚠️ El feature `ventas_mensual_cliente_clase.parquet` no existe. "
        "Vuelve a correr `uv run python src/models/run_all.py` para generarlo."
    )
else:
    # Compatibilidad hacia atrás: si el parquet aún no tiene version o
    # categoria_precio (porque fue generado con un build_features.py
    # anterior a v3.9), las creamos como "(sin clasificar)" y avisamos.
    parquet_v39_completo = ("version" in serie_cli.columns) and (
        "categoria_precio" in serie_cli.columns
    )
    if not parquet_v39_completo:
        st.warning(
            "ℹ️  El parquet de ventas por cliente no incluye aún las columnas "
            "**version** y **categoria_precio**. Para habilitar esos filtros, "
            "regenera el feature store:\n\n"
            "```bash\nuv run python src/features/build_features.py\n```\n\n"
            "Mientras tanto, solo el filtro por clase está activo."
        )
        # Crear columnas placeholder para que el resto del flujo no rompa
        serie_cli = serie_cli.copy()
        if "version" not in serie_cli.columns:
            serie_cli["version"] = "(sin clasificar)"
        if "categoria_precio" not in serie_cli.columns:
            serie_cli["categoria_precio"] = "(sin clasificar)"

    # Normalizar las columnas categóricas a string sin NaN
    for col in ["clase", "version", "categoria_precio"]:
        serie_cli[col] = serie_cli[col].fillna("(sin clasificar)").astype(str)

    st.caption(
        "Filtros locales (solo afectan esta sección). Las proyecciones por cliente están "
        "en backlog (aún no hay data suficiente). Por ahora solo histórico."
    )

    # ─── Filtros encadenados: Clase → Versión → Categoría de precio ───
    fca, fcb, fcc = st.columns(3)
    clases_all_cli = sorted(serie_cli["clase"].unique().tolist())

    with fca:
        clases_cli_sel = filtro_popover_multi(
            label_corto="Clase",
            icono="📚",
            opciones=clases_all_cli,
            key="filtro_clase_cli_v39",
            ayuda="Filtra las ventas del cliente por clase de producto.",
        )
    versiones_cli_disp = _opciones_efectivas(
        serie_cli, "version",
        filtros_aplicados={"clase": clases_cli_sel},
    )
    with fcb:
        if parquet_v39_completo:
            versiones_cli_sel = filtro_popover_multi(
                label_corto="Versión",
                icono="📖",
                opciones=versiones_cli_disp,
                key="filtro_version_cli_v39",
                ayuda="Versión bíblica. Las opciones cambian según la clase elegida.",
            )
        else:
            versiones_cli_sel = versiones_cli_disp
            st.caption("📖 Versión: _regenera el feature store para habilitar_")

    categorias_cli_disp = _opciones_efectivas(
        serie_cli, "categoria_precio",
        filtros_aplicados={"clase": clases_cli_sel, "version": versiones_cli_sel},
    )
    with fcc:
        if parquet_v39_completo:
            categorias_cli_sel = filtro_popover_multi(
                label_corto="Categoría precio",
                icono="💲",
                opciones=categorias_cli_disp,
                key="filtro_cat_precio_cli_v39",
                ayuda="Quintiles de precio para BIBLIAS. Las opciones cambian según clase y versión.",
            )
        else:
            categorias_cli_sel = categorias_cli_disp
            st.caption("💲 Categoría precio: _regenera el feature store para habilitar_")

    # Toggle de métrica
    fcd, fce = st.columns([1, 2])
    with fcd:
        metrica_cli = st.radio(
            "Métrica de visualización",
            options=["Unidades", "Valor ($)"],
            index=1,
            horizontal=True,
            key="toggle_metrica_cliente",
        )
    # v3.15: filtro por categoría de cliente (negocio)
    with fce:
        _CATS = ["PROYECTABLE", "INACTIVO", "NUEVO", "ESPORADICO"]
        _CATS_LABEL = {"PROYECTABLE": "Proyectables", "INACTIVO": "Inactivos (>12m)",
                       "NUEVO": "Nuevos (<12m)", "ESPORADICO": "Esporádicos"}
        cats_cli_sel = st.multiselect(
            "Categoría de cliente",
            options=_CATS, default=_CATS,
            format_func=lambda c: _CATS_LABEL.get(c, c),
            key="filtro_categoria_cliente",
            help="Proyectables: historia regular. Inactivos: +12m en base sin "
                 "comprar el último año. Nuevos: <12m de histórico. "
                 "Esporádicos: compran muy intermitentemente.",
        )
        if not cats_cli_sel:
            cats_cli_sel = _CATS

    # Mapa cliente → categoría y cliente → último vendedor (v3.15)
    if perfiles_cli is not None and "categoria" in perfiles_cli.columns:
        cat_por_cliente = dict(zip(perfiles_cli["cliente"], perfiles_cli["categoria"]))
    else:
        cat_por_cliente = {}
    try:
        _fc = pd.read_parquet(DATA_PROC / "feature_cliente.parquet",
                              columns=["cliente", "ultimo_vendedor"])
        vendedor_por_cliente = dict(zip(_fc["cliente"], _fc["ultimo_vendedor"]))
    except Exception:
        vendedor_por_cliente = {}

    # Aplicar todos los filtros
    serie_cli_filt = serie_cli[
        serie_cli["clase"].isin(clases_cli_sel)
        & serie_cli["version"].isin(versiones_cli_sel)
        & serie_cli["categoria_precio"].isin(categorias_cli_sel)
    ]
    # Filtrar por categoría de cliente
    clientes_de_cat = {c for c, cat in cat_por_cliente.items() if cat in cats_cli_sel}
    if clientes_de_cat:
        serie_cli_filt = serie_cli_filt[serie_cli_filt["cliente"].isin(clientes_de_cat)]
    if len(serie_cli_filt) == 0:
        st.error(
            "No hay datos con esa combinación de filtros. Abre los popovers de "
            "Clase / Versión / Categoría de precio y revisa lo seleccionado."
        )
    else:
        # Ranking de clientes según métrica activa (para ordenar el selector)
        col_metrica_total = "valor" if metrica_cli == "Valor ($)" else "unidades"
        clientes_rank = (
            serie_cli_filt.groupby("cliente")[col_metrica_total]
            .sum().sort_values(ascending=False).reset_index()
        )

        col_search_cli, col_select_cli = st.columns([1, 3])
        with col_search_cli:
            busqueda_cli = st.text_input(
                "🔎 Buscar cliente",
                placeholder="ej: OK BIENESTAR, LIBRERIA...",
                key="busqueda_cliente",
            )
        with col_select_cli:
            if busqueda_cli:
                mask_cli = clientes_rank["cliente"].str.contains(
                    busqueda_cli, case=False, na=False
                )
                clientes_opts = clientes_rank[mask_cli]
                if len(clientes_opts) == 0:
                    st.warning("Sin coincidencias. Mostrando todos.")
                    clientes_opts = clientes_rank
            else:
                clientes_opts = clientes_rank
            # Etiqueta con total al lado para guiar la búsqueda
            clientes_opts["label_cli"] = (
                clientes_opts["cliente"].str[:80]
                + " — "
                + clientes_opts[col_metrica_total].apply(
                    lambda x: f"${int(x):,}" if metrica_cli == "Valor ($)" else f"{int(x):,} u"
                )
            )
            label_cli_sel = st.selectbox(
                f"Selecciona cliente ({len(clientes_opts):,} disponibles)",
                options=["🌐 TODOS los clientes filtrados"] + clientes_opts["label_cli"].tolist(),
                key="select_cliente_label",
            )
            es_todos_cli = (label_cli_sel == "🌐 TODOS los clientes filtrados")
            cliente_sel = None if es_todos_cli else label_cli_sel.split(" — ")[0]

        # ============================================================
        # DESCARGA CSV de clientes (categoría + último vendedor) v3.15
        # ============================================================
        from models import overrides_cliente_store as _ovc_dl
        def _construir_csv_clientes(lista_clientes):
            """CSV con histórico, categoría, último vendedor y proyección
            anual (con overrides) 2027-2030 por cliente."""
            filas = []
            ov_dict = _ovc_dl.cargar()
            proy_src = proy_cli_all if proy_cli_all is not None else None
            if proy_src is not None:
                proy_aj_dl = _ovc_dl.aplicar_overrides(proy_src)
                proy_aj_dl["anio"] = pd.to_datetime(proy_aj_dl["ds"]).dt.year
                cv = "prediccion_valor" if "prediccion_valor" in proy_aj_dl else "prediccion"
                cu = "prediccion_unidades" if "prediccion_unidades" in proy_aj_dl else None
            for cli in lista_clientes:
                sub_h = serie_cli_filt[serie_cli_filt["cliente"] == cli]
                fila = {
                    "cliente": cli,
                    "categoria": cat_por_cliente.get(cli, "NUEVO"),
                    "ultimo_vendedor": vendedor_por_cliente.get(cli, ""),
                    "unidades_hist_total": int(sub_h["unidades"].sum()),
                    "valor_hist_total": int(sub_h["valor"].sum()),
                    "tiene_override": cli in ov_dict,
                }
                if proy_src is not None:
                    pcli = proy_aj_dl[proy_aj_dl["cliente"] == cli]
                    for a in [2027, 2028, 2029, 2030]:
                        pa = pcli[pcli["anio"] == a]
                        fila[f"proy_unid_{a}"] = int(pa[cu].sum()) if (cu and len(pa)) else 0
                        fila[f"proy_valor_{a}"] = int(pa[cv].sum()) if len(pa) else 0
                filas.append(fila)
            return pd.DataFrame(filas)

        # KPIs del cliente / vista agregada
        if es_todos_cli:
            lista_cli = clientes_opts["cliente"].tolist()
            st.markdown(f"#### 🌐 Vista agregada · {len(lista_cli):,} clientes filtrados")
            # KPIs agregados
            sub_all = serie_cli_filt[serie_cli_filt["cliente"].isin(lista_cli)]
            ka1, ka2, ka3, ka4 = st.columns(4)
            ka1.metric("Clientes", f"{len(lista_cli):,}")
            ka2.metric("Unidades históricas", f"{int(sub_all['unidades'].sum()):,}")
            ka3.metric("Valor histórico", f"${sub_all['valor'].sum()/1e9:.2f}B")
            n_proy_en_vista = sum(1 for c in lista_cli
                                  if cat_por_cliente.get(c) == "PROYECTABLE")
            ka4.metric("Proyectables en vista", f"{n_proy_en_vista:,}")

            usar_valor_agg = (metrica_cli == "Valor ($)")
            col_y_agg = "valor" if usar_valor_agg else "unidades"

            # Histórico agregado mensual
            hist_agg_cli = sub_all.groupby("mes")[col_y_agg].sum().reset_index()
            fig_agg_cli = go.Figure()
            fig_agg_cli.add_trace(go.Scatter(
                x=hist_agg_cli["mes"], y=hist_agg_cli[col_y_agg], mode="lines",
                line=dict(color="#1f4e79", width=2), name="Histórico (suma)",
                fill="tozeroy", fillcolor="rgba(31,78,121,0.12)"))
            # Proyección agregada (con overrides) de los clientes de la vista
            if proy_cli_all is not None:
                proy_v = _ovc_dl.aplicar_overrides(
                    proy_cli_all[proy_cli_all["cliente"].isin(lista_cli)])
                if len(proy_v) > 0:
                    cpred = ("prediccion_valor" if usar_valor_agg else "prediccion_unidades")
                    if cpred not in proy_v.columns:
                        cpred = "prediccion"
                    proy_v["ds"] = pd.to_datetime(proy_v["ds"])
                    proy_v_m = proy_v.groupby("ds")[cpred].sum().reset_index()
                    fig_agg_cli.add_trace(go.Scatter(
                        x=proy_v_m["ds"], y=proy_v_m[cpred], mode="lines",
                        line=dict(color="#10b981", width=2.8),
                        name="Proyección (suma, con ajustes)"))
                    if len(hist_agg_cli) > 0:
                        fig_agg_cli.add_vline(x=hist_agg_cli["mes"].max(), line_width=1,
                                              line_dash="dash", line_color="#94a3b8")
            fig_agg_cli.update_layout(
                height=380, hovermode="x unified",
                yaxis_title=("Valor (COP)" if usar_valor_agg else "Unidades"),
                yaxis=dict(tickformat=",.0f"), margin=dict(t=20, b=30))
            st.plotly_chart(fig_agg_cli, use_container_width=True)

            # Tabla por categoría
            st.markdown("**Resumen por categoría (clientes en la vista)**")
            df_cat = pd.DataFrame({"cliente": lista_cli})
            df_cat["categoria"] = df_cat["cliente"].map(cat_por_cliente).fillna("NUEVO")
            tot_cli = sub_all.groupby("cliente")[["unidades", "valor"]].sum()
            df_cat = df_cat.merge(tot_cli, on="cliente", how="left").fillna(0)
            resumen_cat = df_cat.groupby("categoria").agg(
                Clientes=("cliente", "count"),
                Unidades_hist=("unidades", "sum"),
                Valor_hist=("valor", "sum"),
            ).reset_index()
            resumen_cat["Unidades_hist"] = resumen_cat["Unidades_hist"].apply(lambda x: f"{int(x):,}")
            resumen_cat["Valor_hist"] = resumen_cat["Valor_hist"].apply(lambda x: f"${int(x):,}")
            st.dataframe(resumen_cat, hide_index=True, use_container_width=True)

            # Tabla top + descarga
            st.markdown("**Detalle por cliente (con proyección y vendedor)**")
            csv_df = _construir_csv_clientes(lista_cli)
            st.dataframe(csv_df.head(50), hide_index=True, use_container_width=True,
                         height=min(38 + 35 * min(len(csv_df), 50), 500))
            if len(csv_df) > 50:
                st.caption(f"Mostrando 50 de {len(csv_df):,}. El CSV trae todos.")
            st.download_button(
                "⬇️ Descargar CSV de todos los clientes filtrados",
                data=csv_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"presupuesto_clientes_{pd.Timestamp.now():%Y%m%d}.csv",
                mime="text/csv", use_container_width=True,
            )

            # Desagregación ISBN×cliente×año de los proyectables de la vista (Camino B)
            st.markdown("**📦 Proyección por ISBN × cliente (clientes proyectables de la vista)**")
            st.caption(
                "Reparte la proyección de cada cliente proyectable entre sus ISBNs "
                "según su mix histórico. Genera la matriz cliente × ISBN × año "
                "(unidades y valor) para presupuesto detallado."
            )
            proy_en_vista = [c for c in lista_cli if cat_por_cliente.get(c) == "PROYECTABLE"]
            if proy_en_vista:
                if st.button(f"🧩 Generar proyección ISBN×cliente ({len(proy_en_vista)} clientes)",
                             key="btn_desag_todos"):
                    try:
                        from models import cliente_isbn_forecast as _cif
                        with st.spinner("Desagregando proyección por ISBN..."):
                            desag_all = _cif.desagregar(clientes=proy_en_vista)
                        if desag_all is not None and len(desag_all) > 0:
                            st.session_state["_desag_all_csv"] = desag_all.to_csv(index=False).encode("utf-8-sig")
                            st.session_state["_desag_all_n"] = len(desag_all)
                            st.success(f"✓ {len(desag_all):,} filas cliente×ISBN×año generadas.")
                        else:
                            st.warning("No se pudo generar (falta el mix o las proyecciones).")
                    except Exception as e:
                        st.error(f"Error al desagregar: {e}")
                if st.session_state.get("_desag_all_csv"):
                    st.download_button(
                        f"⬇️ Descargar matriz ISBN×cliente×año ({st.session_state.get('_desag_all_n',0):,} filas)",
                        data=st.session_state["_desag_all_csv"],
                        file_name=f"proy_isbn_cliente_{pd.Timestamp.now():%Y%m%d}.csv",
                        mime="text/csv", use_container_width=True,
                    )
            else:
                st.caption("No hay clientes proyectables en la vista actual.")
            st.stop()

        # ============================================================
        # DESDE AQUÍ: vista de UN cliente específico
        # ============================================================
        # KPIs del cliente
        sub_cli = serie_cli_filt[serie_cli_filt["cliente"] == cliente_sel]
        # v3.15: categoría y último vendedor del cliente
        _cat_cli = cat_por_cliente.get(cliente_sel, "NUEVO")
        _vend_cli = vendedor_por_cliente.get(cliente_sel, "—")
        _CAT_LBL = {"PROYECTABLE": "🟢 Proyectable", "INACTIVO": "🔴 Inactivo (>12m)",
                    "NUEVO": "🟡 Nuevo (<12m)", "ESPORADICO": "🟠 Esporádico"}
        st.caption(
            f"Categoría: **{_CAT_LBL.get(_cat_cli, _cat_cli)}**  ·  "
            f"Último vendedor: **{_vend_cli or '—'}**"
        )
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Unidades totales", f"{int(sub_cli['unidades'].sum()):,}")
        kc2.metric("Valor total (COP)", f"${int(sub_cli['valor'].sum()):,}")
        kc3.metric("Meses con compra", f"{sub_cli['mes'].nunique()}")
        kc4.metric("ISBNs distintos", f"{int(sub_cli['n_isbns'].sum())}")

        # Descarga CSV de este cliente
        _csv_uno = _construir_csv_clientes([cliente_sel])
        st.download_button(
            "⬇️ Descargar CSV de este cliente",
            data=_csv_uno.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"cliente_{cliente_sel[:30].replace(' ','_')}_{pd.Timestamp.now():%Y%m%d}.csv",
            mime="text/csv",
        )

        # Serie mensual agregada (sumando todas las combinaciones del filtro)
        serie_cli_agg = sub_cli.groupby("mes").agg(
            unidades=("unidades", "sum"),
            valor=("valor", "sum"),
        ).reset_index().sort_values("mes")

        col_hist_y_cli = "valor" if metrica_cli == "Valor ($)" else "unidades"
        titulo_y_cli = "Valor mensual (COP)" if metrica_cli == "Valor ($)" else "Unidades mes"

        # Columnas de proyección según métrica seleccionada (v3.14)
        usar_valor_cli = (metrica_cli == "Valor ($)")
        col_pred = "prediccion_valor" if usar_valor_cli else "prediccion_unidades"
        col_p10  = "p10_valor" if usar_valor_cli else "p10_unidades"
        col_p90  = "p90_valor" if usar_valor_cli else "p90_unidades"
        # Retrocompat: si el parquet viejo no tiene columnas de unidades
        if proy_cli_all is not None and col_pred not in proy_cli_all.columns:
            col_pred, col_p10, col_p90 = "prediccion", "p10", "p90"

        # Override del cliente (v3.14): aplicar a la proyección mostrada
        from models import overrides_cliente_store as ovc
        _ov_cli_actual = ovc.get_override(cliente_sel)

        fig_cli = go.Figure()
        fig_cli.add_trace(go.Scatter(
            x=serie_cli_agg["mes"], y=serie_cli_agg[col_hist_y_cli],
            mode="lines+markers",
            line=dict(color="#1f4e79", width=2), marker=dict(size=6),
            name="Histórico real",
            fill="tozeroy", fillcolor="rgba(31,78,121,0.15)",
        ))

        # --- v3.14: proyección del cliente (valor O unidades) ---
        proy_de_este = None
        if proy_cli_all is not None:
            proy_de_este = proy_cli_all[proy_cli_all["cliente"] == cliente_sel].copy()
        if proy_de_este is not None and len(proy_de_este) > 0:
            proy_de_este["ds"] = pd.to_datetime(proy_de_este["ds"])
            # Proyección BASE (sin override) — línea gris punteada de referencia
            fig_cli.add_trace(go.Scatter(
                x=proy_de_este["ds"], y=proy_de_este[col_pred],
                mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dot"),
                name="Proyección base",
            ))
            # Proyección AJUSTADA con override
            proy_aj = ovc.aplicar_overrides(proy_de_este)
            # Banda de incertidumbre (sobre la ajustada)
            fig_cli.add_trace(go.Scatter(
                x=proy_aj["ds"], y=proy_aj[col_p90], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_cli.add_trace(go.Scatter(
                x=proy_aj["ds"], y=proy_aj[col_p10], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)", showlegend=False, hoverinfo="skip"))
            nombre_proy = "Proyección ajustada (presupuesto)" if _ov_cli_actual else "Proyección"
            color_proy = "#10b981" if _ov_cli_actual else "#3b82f6"
            fig_cli.add_trace(go.Scatter(
                x=proy_aj["ds"], y=proy_aj[col_pred], mode="lines",
                line=dict(color=color_proy, width=2.8), name=nombre_proy))
            if len(serie_cli_agg) > 0:
                fig_cli.add_vline(x=serie_cli_agg["mes"].max(), line_width=1,
                                  line_dash="dash", line_color="#94a3b8")
        else:
            if perfiles_cli is not None:
                perfil_de_este = perfiles_cli[perfiles_cli["cliente"] == cliente_sel]
                if len(perfil_de_este) > 0:
                    perfil = perfil_de_este["perfil"].iloc[0]
                    msgs = {
                        "INACTIVO": "Cliente sin compras en los últimos 12 meses. No se proyecta a futuro.",
                        "NUEVO_NO_PROYECTABLE": "Pocos meses de historia para una proyección confiable.",
                        "ESPORADICO_GRANDE": "Compras esporádicas de alto valor (no recurrentes). No proyectable como serie continua.",
                        "OTRO_NO_PROYECTABLE": "El cliente no cumple el umbral de historia + actividad para proyectar.",
                    }
                    st.info(f"💡 **{perfil}**: {msgs.get(perfil, 'No proyectable.')}")

        fig_cli.update_layout(
            height=380,
            yaxis_title=titulo_y_cli,
            yaxis=dict(tickformat=",.0f"),
            hovermode="x unified",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_cli, use_container_width=True)

        # =================================================================
        # OVERRIDE DE CRECIMIENTO DEL CLIENTE (v3.14) — para presupuesto
        # =================================================================
        if proy_de_este is not None and len(proy_de_este) > 0:
            st.markdown("##### 🎚️ Ajuste de presupuesto para este cliente")
            st.caption(
                "Ajusta la proyección para construir tu **presupuesto** 2027-2030. "
                "La **escala** sube/baja el nivel; el **crecimiento anual** lo hace "
                "crecer de forma compuesta cada año. Aplica a unidades e ingresos por igual."
            )
            colo1, colo2, colo3 = st.columns([1, 1, 2])
            with colo1:
                esc_in = st.number_input(
                    "Escala (×nivel)", min_value=0.0, max_value=5.0, step=0.05,
                    value=float(_ov_cli_actual["escala"]) if _ov_cli_actual else 1.0,
                    key=f"ovc_esc_{cliente_sel}",
                )
            with colo2:
                gro_in = st.number_input(
                    "Crecimiento anual %", min_value=-50.0, max_value=100.0, step=1.0,
                    value=float(_ov_cli_actual["crecimiento_anual_pct"]) if _ov_cli_actual else 0.0,
                    key=f"ovc_gro_{cliente_sel}",
                )
            with colo3:
                nota_in = st.text_input(
                    "Nota (por qué)", value=_ov_cli_actual.get("nota", "") if _ov_cli_actual else "",
                    placeholder="ej: meta comercial 2027, contrato nuevo...",
                    key=f"ovc_nota_{cliente_sel}",
                )
            cbtn1, cbtn2, _ = st.columns([1, 1, 2])
            with cbtn1:
                if st.button("💾 Guardar ajuste", type="primary",
                             use_container_width=True, key=f"ovc_save_{cliente_sel}"):
                    ovc.set_override(cliente_sel, esc_in, gro_in, nota_in)
                    st.cache_data.clear(); st.rerun()
            with cbtn2:
                if _ov_cli_actual and st.button("↩️ Quitar ajuste", use_container_width=True,
                                                 key=f"ovc_del_{cliente_sel}"):
                    ovc.eliminar(cliente_sel)
                    st.cache_data.clear(); st.rerun()

            # Tabla comparativa base vs ajustado (presupuesto anual)
            proy_de_este["anio"] = proy_de_este["ds"].dt.year
            base_anual = proy_de_este.groupby("anio").agg(
                v=("prediccion_valor" if "prediccion_valor" in proy_de_este else "prediccion", "sum"),
                u=("prediccion_unidades" if "prediccion_unidades" in proy_de_este else "prediccion", "sum"),
            )
            proy_aj2 = ovc.aplicar_overrides(proy_de_este)
            proy_aj2["anio"] = pd.to_datetime(proy_aj2["ds"]).dt.year
            aj_anual = proy_aj2.groupby("anio").agg(
                v=("prediccion_valor" if "prediccion_valor" in proy_aj2 else "prediccion", "sum"),
                u=("prediccion_unidades" if "prediccion_unidades" in proy_aj2 else "prediccion", "sum"),
            )
            tabla_pres = pd.DataFrame({
                "Año": base_anual.index,
                "Unid. base": base_anual["u"].astype(int).values,
                "Unid. presupuesto": aj_anual["u"].astype(int).values,
                "Ingreso base": base_anual["v"].values,
                "Ingreso presupuesto": aj_anual["v"].values,
            })
            tabla_pres["Ingreso base"] = tabla_pres["Ingreso base"].apply(lambda x: f"${x/1e6:,.0f}M")
            tabla_pres["Ingreso presupuesto"] = tabla_pres["Ingreso presupuesto"].apply(lambda x: f"${x/1e6:,.0f}M")
            tabla_pres["Unid. base"] = tabla_pres["Unid. base"].apply(lambda x: f"{x:,}")
            tabla_pres["Unid. presupuesto"] = tabla_pres["Unid. presupuesto"].apply(lambda x: f"{x:,}")
            st.dataframe(tabla_pres, hide_index=True, use_container_width=True)
            if _ov_cli_actual:
                st.success(
                    f"✓ Ajuste activo: escala ×{_ov_cli_actual['escala']:.2f}, "
                    f"crecimiento {_ov_cli_actual['crecimiento_anual_pct']:+.0f}%/año"
                    + (f" · _{_ov_cli_actual['nota']}_" if _ov_cli_actual.get("nota") else "")
                )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Histórico anual del cliente ({'valor COP' if metrica_cli == 'Valor ($)' else 'unidades'})**")
            s_y_cli = serie_cli_agg.copy()
            s_y_cli["año"] = pd.to_datetime(s_y_cli["mes"]).dt.year
            hist_anual_cli = s_y_cli.groupby("año")[col_hist_y_cli].sum().reset_index()
            hist_anual_cli[col_hist_y_cli] = hist_anual_cli[col_hist_y_cli].apply(lambda x: f"{int(x):,}")
            st.dataframe(
                hist_anual_cli.rename(columns={col_hist_y_cli: ("Valor (COP)" if metrica_cli == "Valor ($)" else "Unidades")}),
                hide_index=True, use_container_width=True,
            )
        with col_b:
            st.markdown(f"**Top clases compradas por este cliente**")
            top_clases = sub_cli.groupby("clase").agg(
                unidades=("unidades", "sum"),
                valor=("valor", "sum"),
            ).reset_index().sort_values(col_hist_y_cli, ascending=False).head(10)
            top_clases["unidades"] = top_clases["unidades"].apply(lambda x: f"{int(x):,}")
            top_clases["valor"] = top_clases["valor"].apply(lambda x: f"${int(x):,}")
            st.dataframe(top_clases, hide_index=True, use_container_width=True)

        # ─────────────────────────────────────────────────────────────
        # v3.12 — Desgloses adicionales del cliente: versión, categoría
        #         de precio, y top ISBNs (si la combinación existe)
        # ─────────────────────────────────────────────────────────────
        st.markdown("**Desglose de compras de este cliente**")
        cda, cdb, cdc = st.columns(3)

        with cda:
            st.markdown("_Por versión bíblica_")
            if "version" in sub_cli.columns:
                top_ver = sub_cli.groupby("version").agg(
                    unidades=("unidades", "sum"), valor=("valor", "sum"),
                ).reset_index().sort_values(col_hist_y_cli, ascending=False).head(10)
                top_ver = top_ver[top_ver["version"].astype(str).str.strip() != ""]
                top_ver["unidades"] = top_ver["unidades"].apply(lambda x: f"{int(x):,}")
                top_ver["valor"] = top_ver["valor"].apply(lambda x: f"${int(x):,}")
                top_ver.columns = ["Versión", "Unidades", "Valor"]
                st.dataframe(top_ver, hide_index=True, use_container_width=True)
            else:
                st.caption("Columna versión no disponible.")

        with cdb:
            st.markdown("_Por categoría de precio_")
            if "categoria_precio" in sub_cli.columns:
                _lbl_cat = {"economica": "Económica", "semi_economica": "Semi-econ.",
                            "media": "Media", "semi_fina": "Semi-fina", "fina": "Fina",
                            "no_aplica": "No aplica"}
                top_cat = sub_cli.groupby("categoria_precio").agg(
                    unidades=("unidades", "sum"), valor=("valor", "sum"),
                ).reset_index().sort_values(col_hist_y_cli, ascending=False)
                top_cat["categoria_precio"] = top_cat["categoria_precio"].map(
                    lambda c: _lbl_cat.get(str(c), str(c)))
                top_cat["unidades"] = top_cat["unidades"].apply(lambda x: f"{int(x):,}")
                top_cat["valor"] = top_cat["valor"].apply(lambda x: f"${int(x):,}")
                top_cat.columns = ["Categoría", "Unidades", "Valor"]
                st.dataframe(top_cat, hide_index=True, use_container_width=True)
            else:
                st.caption("Columna categoría de precio no disponible.")

        with cdc:
            st.markdown("_Estacionalidad (mes del año)_")
            s_mes = sub_cli.copy()
            s_mes["mes_num"] = pd.to_datetime(s_mes["mes"]).dt.month
            meses_nom = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
            est = s_mes.groupby("mes_num").agg(
                unidades=("unidades", "sum"), valor=("valor", "sum"),
            ).reindex(range(1, 13), fill_value=0).reset_index()
            est["Mes"] = est["mes_num"].map(lambda m: meses_nom[m-1])
            col_show = "valor" if metrica_cli == "Valor ($)" else "unidades"
            est["Valor"] = est[col_show].apply(
                lambda x: (f"${int(x):,}" if metrica_cli == "Valor ($)" else f"{int(x):,}"))
            st.dataframe(est[["Mes", "Valor"]], hide_index=True,
                         use_container_width=True, height=300)

        # ─────────────────────────────────────────────────────────────
        # v3.9 — TABLA DE ESTADÍSTICOS DESCRIPTIVOS MENSUALES (CLIENTE)
        # ─────────────────────────────────────────────────────────────
        # Misma lógica que en el Explorador por ISBN: sobre la serie
        # MENSUAL del cliente (ya agregada respetando los filtros activos).
        st.markdown("**📊 Estadísticos descriptivos — histórico mensual del cliente**")
        st.caption(
            "Cálculos sobre la serie mensual histórica del cliente (con los "
            "filtros aplicados). Útil para perfilar al cliente: CoV bajo = "
            "compra estable, skewness positivo = pocos meses con compras "
            "grandes (típico de iglesias/distribuidores con pedidos puntuales)."
        )
        if len(serie_cli_agg) > 0:
            tabla_stats_cli = calcular_estadisticos_mensuales(
                serie_cli_agg, col_unidades="unidades", col_valor="valor",
            )
            st.dataframe(tabla_stats_cli, hide_index=True, use_container_width=True)
        else:
            st.caption("Sin histórico mensual disponible para este cliente con los filtros actuales.")

        # =================================================================
        # PROYECCIÓN POR ISBN DE ESTE CLIENTE (Camino B, v3.16)
        # =================================================================
        if _cat_cli == "PROYECTABLE":
            st.markdown("##### 📦 Proyección por ISBN de este cliente (2027-2030)")
            st.caption(
                "El total proyectado del cliente (con sus ajustes) repartido entre "
                "sus ISBNs según el mix histórico de compra. La suma por ISBN "
                "reproduce el total del cliente."
            )
            try:
                from models import cliente_isbn_forecast as _cif
                desag_cli = _cif.desagregar_cliente(cliente_sel)
            except Exception as _e:
                desag_cli = None
                st.caption(f"_No disponible: {_e}_")
            if desag_cli is not None and len(desag_cli) > 0:
                anio_sel_isbn = st.radio(
                    "Año", options=[2027, 2028, 2029, 2030], horizontal=True,
                    key=f"anio_isbn_cli_{cliente_sel}",
                )
                vista = desag_cli[desag_cli["anio"] == anio_sel_isbn].copy()
                vista = vista.sort_values("valor_proy", ascending=False)
                vista_show = vista[["isbn", "descripcion", "unidades_proy", "valor_proy"]].head(50).copy()
                vista_show["unidades_proy"] = vista_show["unidades_proy"].apply(lambda x: f"{int(x):,}")
                vista_show["valor_proy"] = vista_show["valor_proy"].apply(lambda x: f"${int(x):,}")
                vista_show.columns = ["ISBN", "Descripción", "Unidades", "Valor"]
                st.dataframe(vista_show, hide_index=True, use_container_width=True,
                             height=min(38 + 35 * min(len(vista_show), 30), 450))
                if len(vista) > 50:
                    st.caption(f"Mostrando 50 de {len(vista):,} ISBNs. El CSV trae todos los años y todos los ISBNs.")
                st.download_button(
                    "⬇️ Descargar proyección ISBN×año de este cliente",
                    data=desag_cli.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"proy_isbn_{cliente_sel[:25].replace(' ','_')}_{pd.Timestamp.now():%Y%m%d}.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Sin mix histórico de ISBNs para desagregar este cliente.")
        else:
            st.info(
                f"💡 Este cliente es **{_CAT_LBL.get(_cat_cli, _cat_cli)}**, no se "
                f"proyecta, así que no hay desagregación por ISBN. Solo los "
                f"clientes proyectables tienen proyección por ISBN."
            )
