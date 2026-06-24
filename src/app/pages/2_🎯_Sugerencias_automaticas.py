"""
Página 3 — Sugerencias automáticas de novedades
================================================
Mejoras v3.4:
- Cache key correctamente invalidado por el botón "Regenerar"
- Sugerencias ya aprobadas se EXCLUYEN del listado
- Sección de gestión: ver aprobadas, eliminar individual o múltiple
- NO mezclar con novedades del simulador (cada una a su lugar)
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models import novedades_store, overrides_store
from models.sugeridos import (
    generar_todas_sugerencias,
    METAS_BIBLIAS,
    PISOS_BALANCE,
)

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"

st.set_page_config(page_title="Sugerencias automáticas", page_icon="🎯", layout="wide")
st.title("🎯 Sugerencias automáticas de novedades")
st.markdown(
    "Mix sugerido de novedades 2027-2030 que cierran el gap a las **metas conservadoras**, "
    "respetando capacidad operativa (15 SKUs/año) y aplicando criterios de diversificación "
    "basados en el estudio **PATMOS**."
)

st.info(
    "📌 **Apuntes PATMOS aplicados al sugerido**:\n\n"
    "- **Femenino** = oportunidad estratégica. Histórico SBC tiene 20% pero PATMOS muestra "
    "que las mujeres son 60% del segmento S1 (Activos) y mayoría del S5 (Influenciado-Inseguro). "
    "El sugerido empuja a mínimo 30% femenino/año.\n"
    "- **Juvenil (S6)** sub-atendido en catálogo. El sugerido garantiza al menos 1 SKU juvenil "
    "cada 2 años.\n"
    "- **Misioneras = universales evangelísticas**. Las compran iglesias para regalar a "
    "cualquier perfil sin distinción de género. Aunque el color default es azul (RVR), "
    "el sugerido las clasifica como `universal` y no las cuenta como cubierta masculina."
)

# =========================================================================
# CARGA
# =========================================================================
proy_path = DATA_STATE / "proyecciones_prophet.parquet"
modelo_path = DATA_STATE / "modelo_hedonico.joblib"
if not proy_path.exists() or not modelo_path.exists():
    st.warning(
        "⚠️ Para generar sugerencias necesitas correr primero:\n\n"
        "```bash\nuv run python src/models/run_all.py --with-forecasts\n```"
    )
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Generando sugerencias...", max_entries=1)
def cargar_sugerencias(permitir_incremento: bool, cache_key: int):
    """cache_key es parte del cache para invalidar al regenerar."""
    return generar_todas_sugerencias(permitir_incremento_capacidad=permitir_incremento)


if "sug_cache_key" not in st.session_state:
    st.session_state["sug_cache_key"] = 0
if "confirmar_reset_total" not in st.session_state:
    st.session_state["confirmar_reset_total"] = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    permitir_incremento = st.checkbox(
        "Incluir sugerencias fuera de capacidad",
        value=True,
        help="Marca para ver también las sugerencias que excederían los 15 SKUs/año "
             "(para mostrar a junta el déficit estructural)."
    )

    multiplicador = st.slider(
        "Multiplicador de predicciones",
        min_value=0.5, max_value=2.0, value=1.0, step=0.05,
        help="Ajusta todas las predicciones del sugerido. 1.0 = sin ajuste."
    )

    st.divider()
    st.markdown("### 🔄 Acciones de regeneración")

    # Botón 1: refrescar listado sin tocar aprobadas
    if st.button(
        "🔁 Refrescar listado",
        use_container_width=True,
        help="Re-genera el listado de propuestas. NO toca las sugerencias ya aprobadas."
    ):
        cargar_sugerencias.clear()
        st.session_state["sug_cache_key"] += 1
        st.success("✓ Listado refrescado.")
        st.rerun()

    # Botón 2: reset total con confirmación
    sug_apr_actuales = novedades_store.filtrar_por_origen("sugerencia_automatica")
    n_sug_apr = len(sug_apr_actuales)

    if st.button(
        f"🗑️ Resetear TODO desde cero",
        use_container_width=True,
        type="primary",
        help=f"Elimina las {n_sug_apr} sugerencias aprobadas + regenera el listado desde cero. "
             "NO toca las novedades manuales del simulador."
    ):
        st.session_state["confirmar_reset_total"] = True

    if st.session_state.get("confirmar_reset_total"):
        st.warning(
            f"⚠️ Vas a eliminar **{n_sug_apr} sugerencias aprobadas** "
            f"(con `origen=sugerencia_automatica`). Las novedades manuales del simulador "
            f"NO se tocan. ¿Confirmar?"
        )
        cy, cn = st.columns(2)
        if cy.button("✓ Sí, eliminar y regenerar", use_container_width=True, type="primary"):
            if sug_apr_actuales:
                ids = [n["id"] for n in sug_apr_actuales]
                novedades_store.eliminar_multiple(ids)
            cargar_sugerencias.clear()
            st.session_state["sug_cache_key"] += 1
            st.session_state["confirmar_reset_total"] = False
            st.success(f"✓ {n_sug_apr} aprobadas eliminadas. Listado regenerado desde cero.")
            st.rerun()
        if cn.button("✗ Cancelar", use_container_width=True):
            st.session_state["confirmar_reset_total"] = False
            st.rerun()

    st.divider()
    st.markdown("### 📐 Pisos de balance")
    st.caption(
        f"- Misioneras: mín {PISOS_BALANCE['min_misioneras']}/año\n"
        f"- Femenino: mín {PISOS_BALANCE['min_femenino_pct']*100:.0f}%\n"
        f"- Masculino: mín {PISOS_BALANCE['min_masculino_pct']*100:.0f}%\n"
        f"- Juvenil: mín {PISOS_BALANCE['min_juvenil_cada_2_anios']} cada 2 años"
    )

# Cargar sugerencias generadas
df_sug = cargar_sugerencias(permitir_incremento, st.session_state["sug_cache_key"])

# ─────────────────────────────────────────────────────────────────────────
# Mensaje de estado claro (apenas se entra a la página)
# ─────────────────────────────────────────────────────────────────────────
n_apr_total = len(novedades_store.filtrar_por_origen("sugerencia_automatica"))
if n_apr_total > 0:
    st.warning(
        f"📌 **Hay {n_apr_total} sugerencias automáticas YA APROBADAS** en el sistema "
        f"(persisten en `data/state/novedades_aprobadas.json` desde sesiones anteriores). "
        f"Estas son las que TÚ aprobaste antes — no se aprueban automáticamente. "
        f"Si quieres empezar desde cero, usa el botón **🗑️ Resetear TODO** en la barra lateral. "
        f"Para verlas/eliminarlas individualmente, ve a la sección _Sugerencias aprobadas_ más abajo."
    )
else:
    st.success(
        "✓ No hay sugerencias automáticas aprobadas. **El listado de abajo son PROPUESTAS** — "
        "solo se aprueban cuando hagas clic explícito en _Aprobar seleccionadas_ o _Aprobar año_."
    )

# =========================================================================
# FILTRAR SUGERENCIAS YA APROBADAS DEL LISTADO
# =========================================================================
# Para no proponer lo mismo dos veces, identificamos las sugerencias que ya
# fueron aprobadas comparando concepto_id + mes_lanzamiento + nombre.
sugerencias_aprobadas = novedades_store.filtrar_por_origen("sugerencia_automatica")
huellas_aprobadas = set()
for n in sugerencias_aprobadas:
    nombre = n.get("nombre", "")
    ml = n.get("mes_lanzamiento", "")
    huellas_aprobadas.add(f"{nombre}|{ml}")

if not df_sug.empty:
    df_sug = df_sug.copy()
    df_sug["_huella"] = df_sug["nombre_sugerido"].astype(str) + "|" + df_sug["mes_lanzamiento"].astype(str)
    df_sug_pendientes = df_sug[~df_sug["_huella"].isin(huellas_aprobadas)].copy()
    df_sug_pendientes = df_sug_pendientes.drop(columns=["_huella"])
else:
    df_sug_pendientes = pd.DataFrame()

# Aplicar multiplicador a TODOS los campos relevantes (incluyendo curva_mensual)
# Esto asegura que el efecto se vea en: tabla pendiente, gráficas, CSV exportado
# y novedad guardada al aprobar.
if not df_sug_pendientes.empty:
    df_sug_pendientes["demanda_anual_estimada"] = df_sug_pendientes["demanda_anual_estimada"] * multiplicador
    df_sug_pendientes["demanda_anual_p10"] = df_sug_pendientes["demanda_anual_p10"] * multiplicador
    df_sug_pendientes["demanda_anual_p90"] = df_sug_pendientes["demanda_anual_p90"] * multiplicador

    # Aplicar multiplicador a la curva_mensual (lista de dicts con ds, prediccion, p10, p90)
    def _aplicar_mult_curva(curva, mult):
        if not isinstance(curva, list):
            return curva
        return [
            {**e, "prediccion": e.get("prediccion", 0) * mult,
                  "p10": e.get("p10", 0) * mult,
                  "p90": e.get("p90", 0) * mult}
            for e in curva
        ]
    df_sug_pendientes["curva_mensual"] = df_sug_pendientes["curva_mensual"].apply(
        lambda c: _aplicar_mult_curva(c, multiplicador)
    )

# =========================================================================
# GAP A METAS CON SUGERENCIAS APLICADAS
# =========================================================================
st.divider()
st.subheader("Gap a metas — escenario actual vs simulación si apruebas pendientes")
st.caption(
    "**Aprobadas** = lo que YA decidiste lanzar (persistido en sistema). "
    "**Pendientes** = lo que el sugerido propone como simulación; **NO están aprobadas** "
    "hasta que hagas clic en _Aprobar_. Usa el toggle de abajo para incluirlas o no en la gráfica."
)

incluir_pendientes_en_grafico = st.checkbox(
    "Incluir pendientes en la gráfica (simulación 'what-if')",
    value=False,
    help="Por defecto muestra solo lo APROBADO. Marca esto para visualizar cómo quedaría "
         "el gap si aprobaras TODO el listado pendiente.",
    key="incluir_pendientes_gap",
)

proy = pd.read_parquet(proy_path)
feature_isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")

# Filtros globales (v3.12) — aparecen en el sidebar de esta página también.
# Aquí su efecto es informativo (las sugerencias siempre razonan sobre BIBLIAS),
# pero mantienen la consistencia del sidebar entre páginas.
try:
    _serie_fg = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    from app.filtros_globales import render_filtros_globales
    render_filtros_globales(feature_isbn, _serie_fg)
except Exception:
    pass

# Aplicar overrides de proyección (categoría + ISBN específico)
# Esto asegura que el gap a metas refleje los ajustes manuales del Explorador.
proy = overrides_store.aplicar_overrides_a_proyecciones(proy, feature_isbn, overrides_store.cargar())
proy["anio"] = pd.to_datetime(proy["ds"]).dt.year
biblias = feature_isbn[feature_isbn["clase"] == "BIBLIAS"]["isbn"].tolist()
aporte_catalogo = proy[proy["isbn"].isin(biblias)].groupby("anio")["yhat"].sum().to_dict()

# Desglose por origen + capacidad de TODAS las novedades aprobadas
desglose = novedades_store.obtener_aporte_anual_desglosado()

# Sumar el aporte de las pendientes (lo que el sugerido propone añadir)
aporte_pend_dentro = {}
aporte_pend_fuera = {}
if not df_sug_pendientes.empty:
    aporte_pend_dentro = (
        df_sug_pendientes[df_sug_pendientes["dentro_capacidad"]]
        .groupby("año")["demanda_anual_estimada"].sum().to_dict()
    )
    aporte_pend_fuera = (
        df_sug_pendientes[~df_sug_pendientes["dentro_capacidad"]]
        .groupby("año")["demanda_anual_estimada"].sum().to_dict()
    )

filas_resumen = []
for año in [2027, 2028, 2029, 2030]:
    meta = METAS_BIBLIAS[año]
    cat = aporte_catalogo.get(año, 0)
    d = desglose.get(año, {})
    apr_dentro = d.get("manual_dentro", 0) + d.get("sugerencia_dentro", 0)
    apr_fuera = d.get("manual_fuera", 0) + d.get("sugerencia_fuera", 0)
    sug_pend_d = aporte_pend_dentro.get(año, 0) if incluir_pendientes_en_grafico else 0
    sug_pend_f = aporte_pend_fuera.get(año, 0) if incluir_pendientes_en_grafico else 0
    total = cat + apr_dentro + apr_fuera + sug_pend_d + sug_pend_f
    gap_final = max(0, meta - total)
    filas_resumen.append({
        "año": año,
        "meta": int(meta),
        "catalogo": int(cat),
        "aprobadas_dentro": int(apr_dentro),
        "aprobadas_fuera": int(apr_fuera),
        "sugerencias_pendientes_dentro": int(sug_pend_d),
        "sugerencias_pendientes_fuera": int(sug_pend_f),
        "gap_final": int(gap_final),
    })
df_resumen = pd.DataFrame(filas_resumen)

fig = go.Figure()
fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["catalogo"],
                     name="Catálogo (Prophet+Decay)", marker_color="#1f4e79",
                     text=[f"{x:,}" for x in df_resumen["catalogo"]], textposition="inside"))
fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["aprobadas_dentro"],
                     name="✅ Aprobadas dentro cap.", marker_color="#10b981",
                     text=[f"{x:,}" if x > 0 else "" for x in df_resumen["aprobadas_dentro"]],
                     textposition="inside"))
fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["aprobadas_fuera"],
                     name="✅ Aprobadas FUERA cap.", marker_color="#8b5cf6",
                     text=[f"{x:,}" if x > 0 else "" for x in df_resumen["aprobadas_fuera"]],
                     textposition="inside"))
if incluir_pendientes_en_grafico:
    fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["sugerencias_pendientes_dentro"],
                         name="📋 Pendientes dentro (simulación)", marker_color="#86efac",
                         text=[f"{x:,}" if x > 0 else "" for x in df_resumen["sugerencias_pendientes_dentro"]],
                         textposition="inside"))
    fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["sugerencias_pendientes_fuera"],
                         name="📋 Pendientes FUERA (simulación)", marker_color="#c4b5fd",
                         text=[f"{x:,}" if x > 0 else "" for x in df_resumen["sugerencias_pendientes_fuera"]],
                         textposition="inside"))
fig.add_trace(go.Bar(x=df_resumen["año"], y=df_resumen["gap_final"],
                     name="❌ Gap aún no cubierto", marker_color="#f59e0b",
                     text=[f"{x:,}" for x in df_resumen["gap_final"]],
                     textposition="inside"))
fig.add_trace(go.Scatter(x=df_resumen["año"], y=df_resumen["meta"],
                          mode="lines+markers+text",
                          line=dict(color="#dc2626", width=3, dash="dash"),
                          marker=dict(size=12, symbol="diamond"),
                          name="Meta conservadora",
                          text=[f"{x:,}" for x in df_resumen["meta"]],
                          textposition="top center"))
fig.update_layout(barmode="stack", height=480,
                  yaxis_title="Unidades BIBLIAS", xaxis_title="Año",
                  legend=dict(orientation="h", y=1.14),
                  margin=dict(t=60, b=30))
st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# SUGERENCIAS APROBADAS — GESTIÓN
# =========================================================================
if sugerencias_aprobadas:
    st.divider()
    st.subheader(f"✅ Sugerencias automáticas aprobadas ({len(sugerencias_aprobadas)})")
    st.caption(
        "Marca con ✓ las sugerencias que quieras eliminar (devolverlas al pool de propuestas)."
    )

    apr_view = []
    for n in sugerencias_aprobadas:
        apr_view.append({
            "Eliminar": False,
            "ID": n.get("id", "")[:25],
            "Cap": "📈" if n.get("fuera_capacidad", False) else "✅",
            "Nombre": n.get("nombre", "")[:50],
            "TACO destino": n.get("taco_destino", "")[:30],
            "Lanzamiento": n.get("mes_lanzamiento", ""),
            "Demanda anual": f"{n.get('demanda_anual_estimada', 0):,.0f}",
            "Género": n.get("features", {}).get("familia_genero", ""),
        })
    df_apr = pd.DataFrame(apr_view)
    edited_apr = st.data_editor(
        df_apr,
        use_container_width=True,
        hide_index=True,
        height=280,
        column_config={
            "Eliminar": st.column_config.CheckboxColumn("✓", default=False, width="small"),
        },
        disabled=[c for c in df_apr.columns if c != "Eliminar"],
        key="editor_sug_aprobadas",
    )
    ids_eliminar_sug = edited_apr[edited_apr["Eliminar"]]["ID"].tolist()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(
            f"🗑️ Eliminar {len(ids_eliminar_sug)} marcada(s)",
            disabled=(len(ids_eliminar_sug) == 0),
            use_container_width=True,
            type="primary" if len(ids_eliminar_sug) > 0 else "secondary",
        ):
            ids_completos = set()
            for n in sugerencias_aprobadas:
                if n.get("id", "")[:25] in ids_eliminar_sug:
                    ids_completos.add(n.get("id"))
            n_elim = novedades_store.eliminar_multiple(list(ids_completos))
            st.success(f"✓ Eliminadas {n_elim} sugerencias. Volverán al pool de propuestas.")
            st.rerun()

# =========================================================================
# FILTROS DEL LISTADO PENDIENTE
# =========================================================================
if df_sug_pendientes.empty:
    st.divider()
    st.success("🎉 Sin sugerencias pendientes (ya aprobaste todo lo que se proponía o no hay gap).")
    st.stop()

st.divider()
st.subheader("Filtros del listado de sugerencias pendientes")
st.caption("Filtra el listado para enfocar tu revisión. Vacío = todos los valores.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    anios_filt = st.multiselect(
        "Año(s)", options=sorted(df_sug_pendientes["año"].unique()),
        default=sorted(df_sug_pendientes["año"].unique()),
    )
with c2:
    df_sug_pendientes["_mes_num"] = pd.to_datetime(df_sug_pendientes["mes_lanzamiento"]).dt.month
    meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    meses_disp = sorted(df_sug_pendientes["_mes_num"].unique())
    meses_filt_labels = st.multiselect(
        "Mes(es) lanzamiento",
        options=[f"{m:02d} - {meses_nombres[m-1]}" for m in meses_disp],
        default=[],
    )
    meses_filt = [int(m.split(" ")[0]) for m in meses_filt_labels] if meses_filt_labels else meses_disp
with c3:
    tipo_filt = st.multiselect(
        "Tipo TACO",
        options=["existente", "nuevo"],
        default=["existente", "nuevo"],
    )
with c4:
    capacidad_filt = st.multiselect(
        "Capacidad",
        options=["✅ Dentro de capacidad", "📈 Fuera de capacidad"],
        default=["✅ Dentro de capacidad", "📈 Fuera de capacidad"],
    )

c5, c6, c7, c8 = st.columns(4)
with c5:
    tacos_disp = sorted(df_sug_pendientes["taco_destino"].unique())
    taco_filt = st.multiselect(
        "TACO destino(s)",
        options=tacos_disp,
        default=[],
        placeholder="Vacío = todos",
    )
with c6:
    gen_filt = st.multiselect(
        "Género (efectivo)",
        options=sorted(df_sug_pendientes["familia_genero_efectivo"].unique()),
        default=sorted(df_sug_pendientes["familia_genero_efectivo"].unique()),
    )
with c7:
    mercado_filt = st.multiselect(
        "Mercado",
        options=sorted(df_sug_pendientes["mercado_principal"].unique()),
        default=sorted(df_sug_pendientes["mercado_principal"].unique()),
    )
with c8:
    cat_filt = st.multiselect(
        "Categoría precio",
        options=sorted(df_sug_pendientes["categoria_precio"].unique()),
        default=sorted(df_sug_pendientes["categoria_precio"].unique()),
    )

c9, c10 = st.columns(2)
with c9:
    prio_filt = st.multiselect(
        "Prioridad algoritmo",
        options=["alta", "media", "baja"],
        default=["alta", "media", "baja"],
    )
with c10:
    busqueda_nom = st.text_input(
        "🔎 Buscar nombre/TACO",
        placeholder="ej: misionera, mujer, juvenil...",
    )

# Aplicar filtros
df_filt = df_sug_pendientes.copy()
df_filt = df_filt[df_filt["año"].isin(anios_filt)]
df_filt = df_filt[df_filt["_mes_num"].isin(meses_filt)]
df_filt = df_filt[df_filt["tipo_taco"].isin(tipo_filt)]
if taco_filt:
    df_filt = df_filt[df_filt["taco_destino"].isin(taco_filt)]
df_filt = df_filt[df_filt["familia_genero_efectivo"].isin(gen_filt)]
df_filt = df_filt[df_filt["mercado_principal"].isin(mercado_filt)]
df_filt = df_filt[df_filt["categoria_precio"].isin(cat_filt)]
df_filt = df_filt[df_filt["prioridad"].isin(prio_filt)]
cap_values = []
if "✅ Dentro de capacidad" in capacidad_filt:
    cap_values.append(True)
if "📈 Fuera de capacidad" in capacidad_filt:
    cap_values.append(False)
df_filt = df_filt[df_filt["dentro_capacidad"].isin(cap_values)]
if busqueda_nom:
    mask = (
        df_filt["nombre_sugerido"].str.contains(busqueda_nom, case=False, na=False)
        | df_filt["taco_destino"].str.contains(busqueda_nom, case=False, na=False)
    )
    df_filt = df_filt[mask]

# =========================================================================
# LISTADO DE SUGERENCIAS CON SELECCIÓN
# =========================================================================
st.divider()
st.subheader(
    f"Sugerencias pendientes ({len(df_filt)} filtradas de {len(df_sug_pendientes)} "
    f"pendientes, de {len(df_sug) if not df_sug.empty else 0} totales generadas)"
)

if len(df_filt) == 0:
    st.warning("Sin sugerencias que coincidan con los filtros.")
else:
    df_view = df_filt.copy()
    df_view["estado_cap"] = df_view["dentro_capacidad"].map(
        {True: "✅", False: "📈"}
    )
    df_view["tipo_label"] = df_view["tipo_taco"].map(
        {"existente": "🔄 Var", "nuevo": "🆕 Nuevo"}
    )
    df_view["aprobar"] = False

    cols_visibles = [
        "aprobar", "estado_cap", "id_sugerencia", "año", "mes_lanzamiento",
        "tipo_label", "taco_destino", "nombre_sugerido",
        "familia_genero_efectivo", "mercado_principal", "categoria_precio",
        "precio_promedio", "descuento_promedio",
        "demanda_anual_estimada", "demanda_anual_p10", "demanda_anual_p90",
        "prioridad", "justificacion",
    ]
    df_view = df_view[cols_visibles]
    df_view.columns = [
        "Aprobar", "Cap", "ID", "Año", "Mes lanz", "Tipo", "TACO destino",
        "Nombre", "Género", "Mercado", "Categoría", "Precio", "Dto %",
        "Demanda anual", "p10", "p90", "Prioridad", "Justificación",
    ]

    edited = st.data_editor(
        df_view,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "Aprobar": st.column_config.CheckboxColumn(
                "Aprobar", default=False, width="small"),
            "Demanda anual": st.column_config.NumberColumn(format="%.0f"),
            "p10": st.column_config.NumberColumn(format="%.0f"),
            "p90": st.column_config.NumberColumn(format="%.0f"),
            "Precio": st.column_config.NumberColumn(format="$%.0f"),
            "Dto %": st.column_config.NumberColumn(format="%.1f%%"),
            "Justificación": st.column_config.TextColumn(width="large"),
        },
        disabled=[c for c in df_view.columns if c != "Aprobar"],
        key="editor_pendientes",
    )
    ids_seleccionados = edited[edited["Aprobar"]]["ID"].tolist()

    # =========================================================================
    # BOTONES DE APROBACIÓN
    # =========================================================================
    st.divider()
    st.subheader("Acciones")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button(
            f"✅ Aprobar seleccionadas ({len(ids_seleccionados)})",
            disabled=(len(ids_seleccionados) == 0),
            use_container_width=True,
            type="primary",
        ):
            n_ok = 0
            for id_sug in ids_seleccionados:
                fila = df_sug_pendientes[df_sug_pendientes["id_sugerencia"] == id_sug].iloc[0]
                novedad = {
                    "nombre": fila["nombre_sugerido"],
                    "concepto_id": fila["concepto_id"],
                    "tipo_taco": fila["tipo_taco"],
                    "taco_destino": fila["taco_destino"],
                    "mes_lanzamiento": fila["mes_lanzamiento"],
                    "features": {
                        "precio_promedio": float(fila["precio_promedio"]),
                        "descuento_promedio": float(fila["descuento_promedio"]),
                        "familia_genero": fila["familia_genero"],
                        "mercado_principal": fila["mercado_principal"],
                        "version": fila["version"],
                        "tamano_familia": fila["tamano_familia"],
                        "tipo_letra": fila["tipo_letra"],
                        "tiene_cierre": bool(fila["tiene_cierre"]),
                        "tiene_indice": bool(fila["tiene_indice"]),
                        "es_imitacion_cuero": bool(fila["es_imitacion_cuero"]),
                        "tiene_canto_dorado": bool(fila["tiene_canto_dorado"]),
                    },
                    "demanda_anual_estimada": float(fila["demanda_anual_estimada"]),
                    "demanda_anual_p10": float(fila["demanda_anual_p10"]),
                    "demanda_anual_p90": float(fila["demanda_anual_p90"]),
                    "ciclo_vida_meses": int(fila["ciclo_vida_meses"]),
                    "curva_mensual": list(fila["curva_mensual"]),
                    "origen": "sugerencia_automatica",
                    "justificacion": fila["justificacion"],
                    "fuera_capacidad": not bool(fila["dentro_capacidad"]),
                }
                novedades_store.agregar(novedad)
                n_ok += 1
            st.success(f"✅ Aprobadas {n_ok} sugerencias.")
            st.rerun()

    with c2:
        anio_aprobar = st.selectbox(
            "Aprobar todas DENTRO capacidad del año",
            options=sorted(df_filt["año"].unique()),
            key="anio_aprobar_dentro",
        )
        if st.button(
            f"✅ Aprobar año {anio_aprobar}",
            use_container_width=True,
        ):
            sub = df_sug_pendientes[
                (df_sug_pendientes["año"] == anio_aprobar) & (df_sug_pendientes["dentro_capacidad"])
            ]
            n_ok = 0
            for _, fila in sub.iterrows():
                novedad = {
                    "nombre": fila["nombre_sugerido"],
                    "concepto_id": fila["concepto_id"],
                    "tipo_taco": fila["tipo_taco"],
                    "taco_destino": fila["taco_destino"],
                    "mes_lanzamiento": fila["mes_lanzamiento"],
                    "features": {
                        "precio_promedio": float(fila["precio_promedio"]),
                        "descuento_promedio": float(fila["descuento_promedio"]),
                        "familia_genero": fila["familia_genero"],
                        "mercado_principal": fila["mercado_principal"],
                        "version": fila["version"],
                        "tamano_familia": fila["tamano_familia"],
                        "tipo_letra": fila["tipo_letra"],
                        "tiene_cierre": bool(fila["tiene_cierre"]),
                        "tiene_indice": bool(fila["tiene_indice"]),
                        "es_imitacion_cuero": bool(fila["es_imitacion_cuero"]),
                        "tiene_canto_dorado": bool(fila["tiene_canto_dorado"]),
                    },
                    "demanda_anual_estimada": float(fila["demanda_anual_estimada"]),
                    "demanda_anual_p10": float(fila["demanda_anual_p10"]),
                    "demanda_anual_p90": float(fila["demanda_anual_p90"]),
                    "ciclo_vida_meses": int(fila["ciclo_vida_meses"]),
                    "curva_mensual": list(fila["curva_mensual"]),
                    "origen": "sugerencia_automatica",
                    "justificacion": fila["justificacion"],
                    "fuera_capacidad": False,
                }
                novedades_store.agregar(novedad)
                n_ok += 1
            st.success(f"✅ Aprobadas {n_ok} sugerencias dentro de capacidad de {anio_aprobar}.")
            st.rerun()

    with c3:
        if st.button("📊 Exportar listado a CSV", use_container_width=True):
            csv = df_filt.drop(columns=["_mes_num", "curva_mensual"], errors="ignore").to_csv(index=False)
            st.download_button(
                "⬇️ Descargar",
                data=csv,
                file_name=f"sugerencias_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with c4:
        st.write("")
        st.caption(
            f"💡 Aporte total de filtradas: "
            f"**{df_filt['demanda_anual_estimada'].sum():,.0f} u/año estimado**"
        )

# =========================================================================
# RESUMEN POR CATEGORÍAS
# =========================================================================
st.divider()
st.subheader("Composición del pool original (todas las sugerencias generadas)")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Por género efectivo y año**")
    cross = df_sug.pivot_table(
        index="familia_genero_efectivo", columns="año",
        values="id_sugerencia", aggfunc="count", fill_value=0,
    )
    st.dataframe(cross, use_container_width=True)
with c2:
    st.markdown("**Por categoría y año**")
    cross2 = df_sug.pivot_table(
        index="categoria_precio", columns="año",
        values="id_sugerencia", aggfunc="count", fill_value=0,
    )
    st.dataframe(cross2, use_container_width=True)
