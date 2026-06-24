"""
Página 2 — Simulador de novedades (FUNCIONAL)
============================================
Para el equipo de Publicaciones: estima la demanda anual de una biblia nueva
basado en sus características, y la distribuye en 12 meses con perfil
estacional real (con picos cuatrimestrales).
"""
from __future__ import annotations
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.hedonic_model import predecir_novedad, buscar_comparables
from models.seasonality import perfil_para_producto, aplicar_perfil
from models import novedades_store
from features.text_extractors import construir_codigo_sbu_desde_inputs, parsear_sbu

# Catálogo MP (v3.11) — para mostrar descripción larga junto al código
try:
    from utils.catalogo_mp import label_taco_mp as _label_taco_mp_full
    def _label_taco_corto(codigo, max_len: int = 90):
        return _label_taco_mp_full(codigo, max_len=max_len)
except Exception:
    def _label_taco_corto(codigo, max_len: int = 90):
        return str(codigo)
from utils.dictionaries import (
    CAPACIDAD_NOVEDADES,
    DESCUENTO_DEFAULT_POR_CATEGORIA,
    DESCUENTO_MAX_POR_CATEGORIA,
    categoria_por_precio,
    SBU_VERSIONES,
    SBU_FAMILIA_PRODUCTO,
    SBU_FAMILIAS_MODELADAS,
    SBU_TAMANO,
    SBU_TAMANO_A_FAMILIA_TEXTUAL,
    SBU_PASTA,
    SBU_PASTA_LEGACY,
    SBU_SIMBOLOS,
    SBU_SIMBOLO_TO_ETIQUETA,
    SBU_ETIQUETA_TO_SIMBOLO,
    SBU_SIMBOLOS_FRECUENTES,
    SBU_SIMBOLOS_POCO_FRECUENTES,
    SBU_SIMBOLOS_LETRA,
    COLOR_TO_FAMILIA,
)

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"

st.set_page_config(page_title="Simulador de novedades", page_icon="🆕", layout="wide")


# =========================================================================
# CARGA
# =========================================================================
@st.cache_resource
def cargar_modelo():
    path = DATA_STATE / "modelo_hedonico.joblib"
    if not path.exists():
        return None, "Modelo hedónico no entrenado. Corre: uv run python src/models/hedonic_model.py"
    return joblib.load(path), None


@st.cache_resource
def cargar_perfiles():
    path = DATA_STATE / "perfiles_estacionales.pkl"
    if not path.exists():
        return None, "Perfiles estacionales no calculados. Corre: uv run python src/models/seasonality.py"
    with open(path, "rb") as f:
        return pickle.load(f), None


@st.cache_data(max_entries=1)
def cargar_isbn():
    return pd.read_parquet(DATA_PROC / "feature_isbn.parquet")


bundle, err1 = cargar_modelo()
perfiles, err2 = cargar_perfiles()
if err1 or err2:
    st.error(err1 or err2)
    st.stop()
isbn_df = cargar_isbn()
metricas = bundle["metricas"]

# Filtros globales (v3.12) — sidebar consistente entre páginas.
try:
    _serie_fg = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    from app.filtros_globales import render_filtros_globales
    render_filtros_globales(isbn_df, _serie_fg)
except Exception:
    pass

# =========================================================================
# HEADER
# =========================================================================
st.title("🆕 Simulador de demanda para novedades")
st.markdown("""
**Para el equipo de Publicaciones.** Estima cuánto venderá un producto nuevo
antes de aprobarlo, basado en el modelo hedónico entrenado con los ISBNs
históricos de BIBLIAS.
""")

st.warning(
    "⚠️ **Alerta estratégica de capacidad**\n\n"
    f"La capacidad actual de Publicaciones es **{CAPACIDAD_NOVEDADES['skus_por_ano']} SKUs nuevos/año** "
    f"({CAPACIDAD_NOVEDADES['conceptos_por_ano']} conceptos × "
    f"{CAPACIDAD_NOVEDADES['cubiertas_por_concepto']} cubiertas). "
    "Las metas conservadoras BIBLIAS 2029-2030 implican un gap a cubrir con "
    "novedades que **excede esta capacidad** (cada SKU nuevo necesitaría vender >100K u/año, "
    "lo cual es irreal incluso para blockbusters). \n\n"
    "**Es necesario fortalecer la capacidad del equipo de publicaciones** para alcanzar "
    "las metas 2029-2030. La página _Editor de pronóstico_ muestra el gap cuantitativo."
)

with st.expander("📖 Apuntes del estudio PATMOS aplicables a este simulador"):
    st.markdown("""
    El **estudio PATMOS** (Sociedad Bíblica Colombiana) identifica 7 segmentos de
    la población colombiana según afinidad con la fe bíblica. Insights relevantes
    para decidir novedades:

    - **Femenino = oportunidad #1**: las mujeres son **60% del segmento S1** (Activos)
      y mayoría del **S5** (Influenciado-Inseguro, ~18% de Colombia, 77% católicos).
      El catálogo SBC actual tiene solo 20% de demanda en femenino — hay espacio
      comercial para crecer al 30-35%. Programas SBC alineados: **Sí a la Familia**,
      **Sanar las Heridas**.

    - **Juvenil = segmento sub-atendido (S6)**: el catálogo apenas tiene 7% de demanda
      en juvenil. PATMOS sugiere que es un segmento crítico para el relevo generacional.
      Programa SBC alineado: **Caminata Bíblica**.

    - **Misioneras = universales evangelísticas**: las iglesias las compran por cajas
      para regalar sin distinguir género. Aunque la RVR azul domina ventas, el target
      real es **el Clúster 4 PATMOS completo** (segmentos S4-S5-S6).

    - **Segmentos S5 y S6 (Clúster 4 PATMOS)**: representan personas con afinidad
      pero sin discipulado activo. Las misioneras llegan a ellos via iglesias locales.
      Programa SBC alineado: **Un País en la Maleta**.
    """)

with st.sidebar:
    st.markdown("### 📊 Calidad del modelo")
    st.caption(f"Entrenado con **{metricas['n_train']}** BIBLIAS ACTIVAS/DECLINANDO")
    st.metric("R² (log)", f"{metricas['r2_log']:.3f}")
    st.metric("MAE (log)", f"{metricas['mae_log']:.3f}")
    st.metric("MAE (unidades)", f"{metricas['mae_real']:,.0f}")
    n_feats = metricas.get("n_features", 0)
    if n_feats >= 20:
        st.success(f"✓ Modelo v3.10 con codificación SBU ({n_feats} features)")
    else:
        st.warning(
            f"⚠️ Modelo legacy ({n_feats} features). Para activar la "
            f"codificación SBU completa, reentrena con:\n\n"
            f"`uv run python src/models/run_all.py`"
        )
    st.divider()
    st.markdown("### 🎯 Presets")
    preset = st.selectbox(
        "Cargar configuración",
        [
            "Personalizado",
            "Biblia comercial femenina (RVR065 imitación cuero + LG)",
            "Biblia comercial masculina (RVR065 imitación cuero + LG)",
            "Biblia juvenil económica (misionera tamaño 5)",
            "Biblia fina dorada para evento (RVR086 cuero + ZTI+LGi)",
            "Biblia específica para iglesia nacional (RVR066 PU)",
        ],
        index=1,
    )

# =========================================================================
# PRESETS — ahora se interpretan en clave SBU
# =========================================================================
PRESETS = {
    "Personalizado": {},
    "Biblia comercial femenina (RVR065 imitación cuero + LG)": {
        "sbu_version": "RVR", "sbu_familia": "0", "sbu_tamano": "6", "sbu_pasta": "5",
        "sbu_simbolos": ["concordancia_breve", "letra_grande",
                          "cierre", "indice", "palabras_jesus"],
        "color_seleccionado": "Rosa / fucsia (femenino)",
        "precio_promedio": 85_000, "descuento_promedio": 33.0,
        "mercado_objetivo": "ambos",
    },
    "Biblia comercial masculina (RVR065 imitación cuero + LG)": {
        "sbu_version": "RVR", "sbu_familia": "0", "sbu_tamano": "6", "sbu_pasta": "5",
        "sbu_simbolos": ["concordancia_breve", "letra_grande", "palabras_jesus"],
        "color_seleccionado": "Azul / negro / café (masculino)",
        "precio_promedio": 75_000, "descuento_promedio": 33.0,
        "mercado_objetivo": "ambos",
    },
    "Biblia juvenil económica (misionera tamaño 5)": {
        "sbu_version": "RVR", "sbu_familia": "0", "sbu_tamano": "5", "sbu_pasta": "0",
        "sbu_simbolos": ["letra_grande", "economica"],
        "color_seleccionado": "Naranja / verde (juvenil)",
        "precio_promedio": 25_000, "descuento_promedio": 22.0,
        "mercado_objetivo": "nacional",
    },
    "Biblia fina dorada para evento (RVR086 cuero + ZTI+LGi)": {
        "sbu_version": "RVR", "sbu_familia": "0", "sbu_tamano": "8", "sbu_pasta": "9",
        "sbu_simbolos": ["concordancia_amplia", "letra_gigante",
                          "cierre", "indice", "palabras_jesus"],
        "color_seleccionado": "Dorado / blanco (neutro)",
        "precio_promedio": 250_000, "descuento_promedio": 42.0,
        "mercado_objetivo": "ambos",
    },
    "Biblia específica para iglesia nacional (RVR066 PU)": {
        "sbu_version": "RVR", "sbu_familia": "0", "sbu_tamano": "6", "sbu_pasta": "6",
        "sbu_simbolos": ["letra_grande", "palabras_jesus"],
        "color_seleccionado": "Dorado / blanco (neutro)",
        "precio_promedio": 50_000, "descuento_promedio": 28.0,
        "mercado_objetivo": "nacional",
    },
}
defaults = PRESETS[preset]


# =========================================================================
# SECCIÓN 0 — PLANEACIÓN DEL LANZAMIENTO (se elige PRIMERO, v3.12)
# =========================================================================
st.subheader("1. Planeación del lanzamiento")
st.caption(
    "Define primero el destino productivo, el mes de lanzamiento y el ciclo "
    "de vida. Luego construirás las características SBU del producto."
)

# Selector de destino productivo
st.markdown("**Destino productivo del SKU nuevo**")
tipo_taco = st.radio(
    "¿Cuál es el destino productivo?",
    options=["existente", "nuevo"],
    format_func=lambda x: {
        "existente": "🔄 Variante a TACO MP existente (cubierta adicional)",
        "nuevo": "🆕 Concepto totalmente nuevo (TACO MP nuevo)",
    }[x],
    help=(
        "**Existente**: usa un TACO MP que ya está en el catálogo. Solo agregas una "
        "cubierta nueva (color/diseño). Consume 1 cubierta del techo de 15/año pero "
        "NO consume cupo de los 5 conceptos nuevos/año.\n\n"
        "**Nuevo**: TACO MP completamente nuevo (bloque interior nuevo). Consume 1 de "
        "los 5 conceptos/año + 1 cubierta del techo de 15/año. Idealmente lanzas 3 "
        "cubiertas del mismo concepto el mismo mes."
    ),
    horizontal=True,
)

col_taco, col_concepto = st.columns(2)
if tipo_taco == "existente":
    with col_taco:
        # Lista de TACOs existentes del catálogo, priorizando los de BIBLIAS activos
        biblias_isbn = isbn_df[isbn_df["clase"] == "BIBLIAS"]
        tacos_activos = (
            biblias_isbn[biblias_isbn["estado"] == "ACTIVO"]["taco_mp"]
            .dropna().unique().tolist()
        )
        tacos_otros = sorted(
            set(biblias_isbn["taco_mp"].dropna().unique()) - set(tacos_activos)
        )
        opciones_tacos = sorted(tacos_activos) + tacos_otros
        # Filtrar los POSIBLE IMPORTADO al final (son menos prioritarios)
        opciones_tacos = (
            [t for t in opciones_tacos if "POSIBLE IMPORTADO" not in str(t).upper()] +
            [t for t in opciones_tacos if "POSIBLE IMPORTADO" in str(t).upper()]
        )
        taco_existente = st.selectbox(
            "TACO MP existente",
            options=opciones_tacos,
            format_func=lambda c: _label_taco_corto(c),
            help="Selecciona el TACO MP del catálogo. Los TACOs de BIBLIAS ACTIVAS aparecen primero. La descripción larga viene del inventario MP.",
        )
        taco_destino_final = taco_existente
        # Info: cuántas cubiertas tiene actualmente
        n_isbns_taco = (
            biblias_isbn[biblias_isbn["taco_mp"] == taco_existente]["isbn"].nunique()
        )
        st.caption(f"📊 Este TACO tiene actualmente **{n_isbns_taco} cubiertas** en el catálogo.")
    with col_concepto:
        concepto_id = taco_existente or "CONCEPTO_EXISTENTE"
        st.info(
            f"💡 Como agregas cubierta a un TACO existente:\n\n"
            f"- **NO consumes cupo de los 5 conceptos/año** ✓\n"
            f"- Consumes **1 cubierta del techo de 15/año**\n"
            f"- El concepto_id queda como el TACO existente"
        )

else:  # nuevo
    with col_taco:
        nuevo_taco_nombre = st.text_input(
            "Nombre del TACO MP nuevo",
            value="",
            placeholder="ej: CLARIDAD 060 NUEVA TEMÁTICA",
            help="Nombre que tendrá el nuevo TACO MP. Si es CLARIDAD, debe coincidir "
                 "con el cronograma. Si no, usa una convención clara.",
        )
        taco_destino_final = nuevo_taco_nombre
    with col_concepto:
        concepto_id = st.text_input(
            "ID del concepto (agrupa cubiertas del mismo TACO nuevo)",
            value="",
            placeholder="ej: C001 - 2027 Nueva Mujer",
            help="Las 3 cubiertas del mismo concepto comparten TACO. Si vas a lanzar "
                 "las 3 cubiertas, asegúrate de usar el mismo concepto_id en las tres.",
        )
        st.warning(
            "⚠️ Como creas TACO MP nuevo:\n\n"
            "- **Consumes 1 de los 5 conceptos del año** ⚠️\n"
            "- Consumes 1 cubierta del techo de 15/año\n"
            "- Recuerda: regla operativa pide lanzar las 3 cubiertas el MISMO mes"
        )

col1, col2 = st.columns(2)
with col1:
    mes_lanzamiento = st.selectbox(
        "Mes de lanzamiento",
        options=[
            f"{año}-{mes:02d}"
            for año in [2027, 2028, 2029, 2030]
            for mes in range(1, 13)
        ],
        index=0,
        help="Mes en que la novedad estará disponible para venta.",
    )
with col2:
    ciclo_vida = st.number_input(
        "Ciclo de vida (meses)",
        min_value=12, max_value=60, value=30, step=6,
        help="Default 30. Por convención: 24-36 meses de vida útil del SKU.",
    )

# Checkbox: marcar como excedente de capacidad actual de publicaciones
excede_capacidad = st.checkbox(
    "📈 Marcar como novedad que **excede la capacidad actual** de Publicaciones",
    value=False,
    help=(
        "Si activas esto, la novedad se aprueba pero queda etiquetada como "
        "'fuera de capacidad' (en violeta en los tableros). Útil para simular el "
        "fortalecimiento de Publicaciones necesario para cerrar el gap a metas 2029-2030. "
        "**No consume cupo de los 15 SKUs/año actuales.**"
    ),
)


# =========================================================================
# CONSTRUCTOR DEL CÓDIGO SBU (v3.10)
# =========================================================================
st.subheader("2. Características del producto — Constructor SBU")
st.caption(
    "Codifica la novedad siguiendo la convención de las Sociedades Bíblicas "
    "Unidas (SBU). El **código se construye automáticamente** abajo a medida "
    "que seleccionas las características."
)

# --- Paso A: Versión bíblica -----------------------------------------------
versiones_opciones = [v[0] for v in SBU_VERSIONES]
versiones_label = {v[0]: f"{v[0]} — {v[1]}" for v in SBU_VERSIONES}
version_default = defaults.get("sbu_version", "RVR")
col_v, col_f = st.columns([1, 1])
with col_v:
    st.markdown("**A. Traducción**")
    sbu_version = st.selectbox(
        "Versión bíblica",
        options=versiones_opciones,
        format_func=lambda v: versiones_label[v],
        index=versiones_opciones.index(version_default) if version_default in versiones_opciones else 0,
        key="sbu_version_sel",
    )

# --- Paso B: Familia de producto -------------------------------------------
with col_f:
    st.markdown("**B. Familia de producto** (posición 2 del código)")
    familias_opciones = list(SBU_FAMILIA_PRODUCTO.keys())
    familia_default = defaults.get("sbu_familia", "0")
    sbu_familia = st.selectbox(
        "Familia",
        options=familias_opciones,
        format_func=lambda f: f"{f} — {SBU_FAMILIA_PRODUCTO[f]}",
        index=familias_opciones.index(familia_default) if familia_default in familias_opciones else 0,
        key="sbu_familia_sel",
    )

# Banner amarillo si la familia no es modelada
if sbu_familia not in SBU_FAMILIAS_MODELADAS:
    st.warning(
        f"⚠️ **Familia '{sbu_familia} — {SBU_FAMILIA_PRODUCTO[sbu_familia]}' "
        f"fuera del scope del modelo.** El modelo hedónico está entrenado SOLO "
        f"con biblias (familias 0 y '.'). Para Nuevos Testamentos, Porciones, "
        f"Selecciones y otros, la predicción tendrá incertidumbre alta y "
        f"posiblemente sesgada. Úsalo solo como referencia gruesa y valida "
        f"contra histórico equivalente."
    )

# --- Paso C: Tamaño del taco -----------------------------------------------
st.markdown("**C. Tamaño del taco** (posición 3)")
tam_opciones = list(SBU_TAMANO.keys())
tam_default = defaults.get("sbu_tamano", "6")
sbu_tamano = st.radio(
    "Tamaño",
    options=tam_opciones,
    format_func=lambda t: f"{t} — {SBU_TAMANO[t][0]} · {SBU_TAMANO[t][1]}",
    index=tam_opciones.index(tam_default) if tam_default in tam_opciones else 5,
    horizontal=True,
    key="sbu_tamano_sel",
    label_visibility="collapsed",
)

# --- Paso D: Tipo de pasta -------------------------------------------------
st.markdown("**D. Tipo de pasta o tapa** (posición 4)")
pasta_opciones = list(SBU_PASTA.keys())
pasta_default = defaults.get("sbu_pasta", "5")
sbu_pasta = st.selectbox(
    "Pasta",
    options=pasta_opciones,
    format_func=lambda p: f"{p} — {SBU_PASTA[p]}",
    index=pasta_opciones.index(pasta_default) if pasta_default in pasta_opciones else 4,
    key="sbu_pasta_sel",
    label_visibility="collapsed",
)

# --- Paso E: Símbolos de características específicas -----------------------
st.markdown("**E. Características específicas** (símbolos al final del código)")
st.caption(
    "Las **letras LM/LG/LGi/LSGi** son mutuamente excluyentes (sólo se puede "
    "una). Los símbolos con ⚠️ tienen poca data histórica — el modelo predice "
    "con incertidumbre más alta en esos casos."
)

defaults_sim = set(defaults.get("sbu_simbolos", []))

# Sub-paso E1: tipo de letra (radio, único)
st.markdown("**E.1 Tamaño de letra** (elige uno)")
letras_orden = ["letra_mediana", "letra_grande", "letra_gigante", "letra_super_gigante"]
letras_label = {
    "letra_mediana":       "LM — Letra Mediana (6–11 pt)",
    "letra_grande":        "LG — Letra Grande (12–14 pt)",
    "letra_gigante":       "LGi — Letra Gigante (15–17 pt)",
    "letra_super_gigante": "LSGi — Letra Súper Gigante (18+ pt)",
}
letra_default = next((s for s in letras_orden if s in defaults_sim), "letra_grande")
sbu_tipo_letra = st.radio(
    "Letra",
    options=letras_orden,
    format_func=lambda s: letras_label[s],
    index=letras_orden.index(letra_default),
    horizontal=True,
    key="sbu_letra_sel",
    label_visibility="collapsed",
)

# Sub-paso E2: símbolos no-letra (multi-select)
st.markdown("**E.2 Otras características** (selecciona las que apliquen)")
simbolos_no_letra = [
    (codigo, etiq, desc, pts, fam, frec)
    for codigo, etiq, desc, pts, fam, frec in SBU_SIMBOLOS
    if fam != "letra"
]
# Agrupar visualmente por familia
familias_simbolos = {}
for codigo, etiq, desc, pts, fam, frec in simbolos_no_letra:
    familias_simbolos.setdefault(fam, []).append((codigo, etiq, desc, frec))

NOMBRE_FAMILIA = {
    "encuad":   "Encuadernación",
    "notas":    "Notas / aparato crítico",
    "edicion":  "Acabados y formato",
}

simbolos_seleccionados: list[str] = []
cols_fam = st.columns(len(familias_simbolos))
for i, (fam, items) in enumerate(familias_simbolos.items()):
    with cols_fam[i]:
        st.markdown(f"_{NOMBRE_FAMILIA.get(fam, fam)}_")
        for codigo, etiq, desc, frec in items:
            marca = "" if frec else " ⚠️"
            checked_default = etiq in defaults_sim
            ck = st.checkbox(
                f"**{codigo}** — {desc}{marca}",
                value=checked_default,
                key=f"sbu_sym_{etiq}_ck",
                help=("Poco data histórica para este símbolo. La predicción "
                      "del modelo tendrá incertidumbre mayor." if not frec else None),
            )
            if ck:
                simbolos_seleccionados.append(etiq)
                if not frec:
                    st.caption(":orange[⚠️ Pocos comparables. Intervalo p10-p90 será amplio.]")

# Unir letra + símbolos seleccionados
todos_simbolos = simbolos_seleccionados + [sbu_tipo_letra]

# --- Código SBU generado ---------------------------------------------------
codigo_sbu = construir_codigo_sbu_desde_inputs(
    sbu_version, sbu_familia, sbu_tamano, sbu_pasta, todos_simbolos
)
# Texto descriptivo legible
def _describir_codigo(version, familia, tamano, pasta, sims):
    parts = [versiones_label[version]]
    parts.append(SBU_FAMILIA_PRODUCTO.get(familia, "?"))
    parts.append(f"Tamaño {tamano} · {SBU_TAMANO[tamano][0]}")
    parts.append(f"Pasta: {SBU_PASTA.get(pasta, SBU_PASTA_LEGACY.get(pasta, '?'))}")
    sim_descs = []
    for codigo, etiq, desc, *_ in SBU_SIMBOLOS:
        if etiq in sims:
            sim_descs.append(f"{codigo} ({desc})")
    if sim_descs:
        parts.append("Características: " + ", ".join(sim_descs))
    return " · ".join(parts)

descripcion_legible = _describir_codigo(
    sbu_version, sbu_familia, sbu_tamano, sbu_pasta, set(todos_simbolos)
)

st.divider()
st.markdown("#### 📦 Código SBU generado")
st.code(codigo_sbu, language=None)
st.caption(descripcion_legible)


# =========================================================================
# METADATA NO CODIFICADA EN SBU (sigue alimentando al modelo)
# =========================================================================
st.subheader("3. Metadata adicional (no codificada en SBU)")
st.caption(
    "Estos datos NO van en el código SBU pero sí entran al modelo (color → "
    "género, precio, descuento, mercado) o se guardan como referencia (marca)."
)

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("**Color dominante / género**")
    # Mapeo amigable: agrupamos colores por familia de género
    color_grupos = [
        ("Rosa / fucsia / lila (femenino)",         "femenino"),
        ("Azul / negro / café / gris (masculino)",  "masculino"),
        ("Naranja / verde / turquesa (juvenil)",    "juvenil"),
        ("Dorado / blanco / beige / rojo (neutro)", "neutro"),
    ]
    color_labels = [c[0] for c in color_grupos]
    color_default = defaults.get("color_seleccionado", color_labels[0])
    color_idx = color_labels.index(color_default) if color_default in color_labels else 0
    color_sel = st.selectbox(
        "Color",
        options=color_labels,
        index=color_idx,
        key="color_sel",
        label_visibility="collapsed",
        help=(
            "El género se infiere por el color dominante, NO por la "
            "nomenclatura del nombre. Esta clasificación calza con el "
            "histórico SBC."
        ),
    )
    familia_genero = dict(color_grupos)[color_sel]
    st.caption(f"Familia género: **{familia_genero}**")

with col_m2:
    st.markdown("**Precio y descuento**")
    precio = st.number_input(
        "Precio sugerido COP",
        min_value=5_000, max_value=500_000,
        value=int(defaults.get("precio_promedio", 85_000)),
        step=5_000,
    )
    cat_interna = categoria_por_precio(precio)
    cat_label = {
        "economica": "Económica (misionera)",
        "semi_economica": "Semi-económica",
        "media": "Media",
        "semi_fina": "Semi-fina",
        "fina": "Fina",
    }[cat_interna]
    st.caption(f"📌 Categoría auto: **{cat_label}**")

    desc_default = float(defaults.get(
        "descuento_promedio",
        DESCUENTO_DEFAULT_POR_CATEGORIA[cat_interna]
    ))
    desc_max = DESCUENTO_MAX_POR_CATEGORIA[cat_interna]
    descuento = st.slider(
        "Descuento estructural (%)",
        min_value=0.0,
        max_value=float(desc_max),
        value=min(desc_default, float(desc_max)),
        step=0.5,
    )
    if cat_interna == "economica" and descuento > 23:
        st.warning(f"⚠️ Económicas tope normal 23%. Estás en {descuento}% (excepcional).")

with col_m3:
    st.markdown("**Mercado y marca comercial**")
    mercados = ["ambos", "nacional", "internacional"]
    mercado_objetivo = st.selectbox(
        "Mercado objetivo",
        mercados,
        index=mercados.index(defaults.get("mercado_objetivo", "ambos")),
        help=(
            "**nacional**: cubierta con logo SBC (Colombia).\n\n"
            "**internacional**: cubierta con logos SBU (exportación).\n\n"
            "**ambos**: cubierta adaptada para los dos mercados (dispara demanda)."
        ),
    )
    marca = st.text_input(
        "Marca / temática libre",
        value=defaults.get("marca", ""),
        placeholder="ej: Mujer Virtuosa Coral, El Camino, Sí a la Familia",
        help=(
            "Nombre comercial NO codificado en SBU. Aparece en la descripción "
            "del SKU generado y en los informes."
        ),
    )

# Nombre de la novedad: combinación legible para humanos
nombre = (
    f"{codigo_sbu}"
    + (f" — {marca}" if marca else "")
    + f" ({familia_genero})"
)

# v3.12: descripción del SKU generado, ahora incluyendo la marca/temática.
# Se muestra aquí (después de capturar la marca) para que el usuario vea el
# nombre comercial completo del producto que está simulando.
descripcion_sku_completa = descripcion_legible + (
    f" · Marca/temática: {marca}" if marca else ""
)
st.markdown("#### 🏷️ SKU generado (con marca)")
st.code(nombre, language=None)
st.caption(descripcion_sku_completa)


# =========================================================================
# PREDICCIÓN
# =========================================================================
st.divider()

# Mapear las features SBU al diccionario que recibe el modelo. Incluye:
#   - Features legacy (familia_genero, tamano_familia, tipo_letra, etc.):
#     se derivan de la codificación SBU + color para compatibilidad.
#   - Features SBU nuevas (sbu_pasta, sbu_tamano, sbu_tipo_letra,
#     sbu_sym_*): se pasan directamente.
#   - Booleanas legacy (tiene_cierre, tiene_indice, es_imitacion_cuero,
#     tiene_canto_dorado): se derivan del set de símbolos SBU
#     correspondientes.
sims_set = set(todos_simbolos)
features = {
    # --- Precio / mercado / género (no codificados en SBU) ---
    "precio_promedio":     precio,
    "descuento_promedio":  descuento,
    "familia_genero":      familia_genero,
    "mercado_principal":   mercado_objetivo,
    "version":             sbu_version,

    # --- Features legacy derivadas (compatibilidad con modelo previo) ---
    "tamano_familia":      SBU_TAMANO_A_FAMILIA_TEXTUAL.get(sbu_tamano, "estandar"),
    "tipo_letra":          {
        "letra_mediana":       "mini",
        "letra_grande":        "letra_grande",
        "letra_gigante":       "letra_grande_compacta",
        "letra_super_gigante": "letra_super_gigante",
    }.get(sbu_tipo_letra, "letra_grande"),
    "tiene_cierre":        "cierre" in sims_set,
    "tiene_indice":        "indice" in sims_set,
    "es_imitacion_cuero":  (sbu_pasta == "5"),
    "tiene_canto_dorado":  False,  # no codificado en SBU; libre

    # --- Features SBU v3.10 ---
    "sbu_version":         sbu_version,
    "sbu_familia":         sbu_familia,
    "sbu_tamano":          sbu_tamano,
    "sbu_pasta":           sbu_pasta,
    "sbu_tipo_letra":      sbu_tipo_letra,
    "sbu_codigo_canonico": codigo_sbu,
}
# Dummies SBU (sin las de letra, ya están en sbu_tipo_letra)
for codigo, etiq, _desc, _pts, fam, _frec in SBU_SIMBOLOS:
    if fam == "letra":
        continue
    features[f"sbu_sym_{etiq}"] = int(etiq in sims_set)

st.subheader("4. Predicción de demanda")
pred = predecir_novedad(features, bundle)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Demanda anual estimada", f"{pred['demanda_anual_estimada']:,.0f} u")
col2.metric("⬇️ p10 (bajo)", f"{pred['intervalo_p10']:,.0f} u")
col3.metric("⬆️ p90 (alto)", f"{pred['intervalo_p90']:,.0f} u")
col4.metric("📅 Ciclo default", "30 meses")

# =========================================================================
# COMPARABLES
# =========================================================================
st.subheader("5. Validación contra ISBNs comparables")
st.caption(
    "Los 10 ISBNs históricos más similares — el modelo aplica los filtros SBU "
    "primero (versión, tamaño, pasta, letra) y relaja si no hay suficientes, "
    "luego ordena por cercanía de precio. **Si la predicción es muy diferente "
    "a la mediana de comparables, revisa los inputs.**"
)

comp = buscar_comparables(features, isbn_df, n_top=10)
if len(comp):
    mediana_c = comp["demanda_anual_madura"].median()
    media_c = comp["demanda_anual_madura"].mean()

    cA, cB, cC = st.columns(3)
    cA.metric("🎯 Predicción modelo", f"{pred['demanda_anual_estimada']:,.0f}")
    cB.metric("📊 Mediana comparables", f"{mediana_c:,.0f}")
    cC.metric("📊 Media comparables", f"{media_c:,.0f}")

    diff = pred["demanda_anual_estimada"] - mediana_c
    if abs(diff / max(mediana_c, 1)) > 0.5:
        if diff < 0:
            st.warning(
                f"⚠️ El modelo predice {abs(diff/mediana_c)*100:.0f}% MENOS que la "
                f"mediana de comparables. Posibles causas:\n\n"
                f"- Si elegiste `mercado_objetivo = nacional` y los comparables son "
                f"`ambos`, prueba con `ambos` (default comercial).\n"
                f"- El precio puede estar empujando a la baja vs comparables."
            )
        else:
            st.warning(
                f"⚠️ El modelo predice {(diff/mediana_c)*100:.0f}% MÁS que la "
                f"mediana de comparables. Revisa si la predicción es realista."
            )

    comp_view = comp.copy()
    comp_view["precio_promedio"] = comp_view["precio_promedio"].apply(
        lambda x: f"${x:,.0f}"
    )
    comp_view["demanda_anual_madura"] = comp_view["demanda_anual_madura"].apply(
        lambda x: f"{x:,.0f}"
    )
    # Construir nombres de columnas dinámicos según las columnas presentes
    column_label_map = {
        "isbn": "ISBN",
        "descripcion": "Descripción",
        "sbu_codigo_canonico": "Código SBU",
        "familia_genero": "Género",
        "version": "Versión",
        "tamano_familia": "Tamaño",
        "precio_promedio": "Precio",
        "demanda_anual_madura": "Demanda anual",
        "estado": "Estado",
    }
    comp_view.columns = [column_label_map.get(c, c) for c in comp_view.columns]
    st.dataframe(comp_view, use_container_width=True, hide_index=True, height=380)
else:
    st.info("No se encontraron comparables con esos filtros.")

# =========================================================================
# CURVA MENSUAL
# =========================================================================
st.subheader("6. Distribución mensual con picos cuatrimestrales reales")
st.caption(
    "Reparto usando perfil estacional histórico, segmentado por género × mercado. "
    "**No es distribución normal aplastada** — captura los picos reales: "
    "Marzo (Semana Santa), Junio-Julio (congresos), Octubre (Mes de la Biblia), "
    "Diciembre (pre-abasto enero)."
)

perfil = perfil_para_producto(perfiles, familia_genero=familia_genero, mercado=mercado_objetivo)
demanda_anual = pred["demanda_anual_estimada"]
curva = aplicar_perfil(demanda_anual, perfil)
curva_p10 = aplicar_perfil(pred["intervalo_p10"], perfil)
curva_p90 = aplicar_perfil(pred["intervalo_p90"], perfil)

meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=meses_nombres, y=curva_p90, fill=None, mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=meses_nombres, y=curva_p10, fill="tonexty", mode="lines",
    line=dict(width=0), fillcolor="rgba(99,102,241,0.15)",
    name="Intervalo p10-p90"
))
fig.add_trace(go.Scatter(
    x=meses_nombres, y=curva, mode="lines+markers+text",
    line=dict(color="#6366f1", width=3),
    marker=dict(size=10),
    text=[f"{int(v):,}" for v in curva],
    textposition="top center",
    name="Estimación central"
))
fig.update_layout(
    height=440,
    yaxis_title="Unidades mes",
    xaxis_title="Mes",
    hovermode="x unified",
    showlegend=True,
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=40, b=30),
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
pico_mes = meses_nombres[curva.argmax()]
valle_mes = meses_nombres[curva.argmin()]
col1.metric("🔝 Pico anual", f"{pico_mes} ({curva.max():,} u)")
col2.metric("⬇️ Valle anual", f"{valle_mes} ({curva.min():,} u)")
col3.metric("📊 Total año", f"{curva.sum():,} u")

# =========================================================================
# APROBACIÓN E INTEGRACIÓN AL PRONÓSTICO
# =========================================================================
st.divider()
st.subheader("7. Aprobar e integrar al pronóstico")
st.caption(
    "Al **aprobar** una novedad, queda persistida y suma su demanda al pronóstico agregado "
    "que verás en el Editor de pronóstico. La novedad se incluirá también en el CSV final 2027-2030 "
    "del Sprint 3b parte 2."
)

novedades_aprobadas = novedades_store.cargar()  # solo para conteo de capacidad

# Validación previa: capacidad disponible en el año seleccionado
año_lanzamiento = int(mes_lanzamiento[:4])
uso_pre = novedades_store.calcular_uso_capacidad(año_lanzamiento)
bloqueado_capacidad = False
mensaje_bloqueo = None
if not excede_capacidad:
    if tipo_taco == "nuevo" and uso_pre["conceptos_libres"] == 0:
        bloqueado_capacidad = True
        mensaje_bloqueo = (
            f"🚫 En **{año_lanzamiento}** ya tienes {uso_pre['conceptos_usados']}/5 conceptos "
            f"nuevos aprobados. No puedes aprobar otro concepto nuevo este año. "
            f"Puedes (a) cambiar a 'variante de TACO existente', (b) cambiar el mes de "
            f"lanzamiento a otro año, (c) eliminar otra novedad de concepto nuevo de {año_lanzamiento}, "
            f"o (d) marcar como **excede capacidad actual** (checkbox de arriba)."
        )
    elif uso_pre["cubiertas_libres"] == 0:
        bloqueado_capacidad = True
        mensaje_bloqueo = (
            f"🚫 En **{año_lanzamiento}** ya tienes {uso_pre['cubiertas_usadas']}/15 cubiertas "
            f"aprobadas. No puedes aprobar más SKUs dentro de capacidad. Cambia el mes a otro año, "
            f"elimina alguna novedad, o marca como **excede capacidad actual**."
        )
if bloqueado_capacidad:
    st.error(mensaje_bloqueo)

# v3.12: ID manual (ISBN/SKU). Si el usuario ya conoce el ISBN/SKU que tendrá
# esta novedad, puede escribirlo aquí. Si lo deja vacío, se autogenera.
col_id, col_btn = st.columns([2, 1])
with col_id:
    id_manual = st.text_input(
        "ID del producto (ISBN o SKU) — opcional",
        value="",
        placeholder="ej: 9789587461234 o SE1130100000 (vacío = autogenerar)",
        help="Si ya tienes el ISBN/SKU asignado a esta novedad, escríbelo. "
             "Si lo dejas vacío, el sistema genera un ID temporal único.",
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    boton_aprobar = st.button(
        "✅ Aprobar e integrar", type="primary",
        use_container_width=True, disabled=bloqueado_capacidad,
    )

if boton_aprobar:
    if not taco_destino_final:
        st.error("Debes especificar el TACO MP destino.")
    else:
        curva_mensual = novedades_store.generar_curva_mensual(
            mes_lanzamiento=mes_lanzamiento,
            demanda_anual=pred["demanda_anual_estimada"],
            perfil_estacional=perfil,
            ciclo_vida_meses=ciclo_vida,
            p10=pred["intervalo_p10"],
            p90=pred["intervalo_p90"],
        )
        novedad = {
            "nombre": nombre or f"Sim_{datetime.now():%Y%m%d_%H%M%S}",
            "concepto_id": concepto_id or f"CONCEPTO_{datetime.now():%Y%m%d_%H%M%S}",
            "tipo_taco": tipo_taco,  # "existente" o "nuevo"
            "taco_destino": taco_destino_final,
            "mes_lanzamiento": mes_lanzamiento,
            "features": features,
            # --- ID manual (v3.12): ISBN/SKU si el usuario lo proporcionó ---
            "id_producto_manual": id_manual.strip() if id_manual.strip() else None,
            # --- Bloque SBU (v3.10) ---
            "sbu_codigo": codigo_sbu,
            "sbu_version": sbu_version,
            "sbu_familia": sbu_familia,
            "sbu_tamano": sbu_tamano,
            "sbu_pasta": sbu_pasta,
            "sbu_tipo_letra": sbu_tipo_letra,
            "sbu_simbolos": sorted(sims_set),
            "color_seleccionado": color_sel,
            "marca": marca,
            # ---
            "demanda_anual_estimada": float(pred["demanda_anual_estimada"]),
            "demanda_anual_p10": float(pred["intervalo_p10"]),
            "demanda_anual_p90": float(pred["intervalo_p90"]),
            "comparables_mediana": float(comp["demanda_anual_madura"].median()) if len(comp) else None,
            "ciclo_vida_meses": int(ciclo_vida),
            "curva_mensual": [
                {
                    "ds": row["ds"].strftime("%Y-%m-%d"),
                    "prediccion": float(row["prediccion"]),
                    "p10": float(row.get("p10", 0)),
                    "p90": float(row.get("p90", 0)),
                }
                for _, row in curva_mensual.iterrows()
            ],
            "origen": "manual",
            "fuera_capacidad": bool(excede_capacidad),
        }
        # Si el usuario dio un ID manual, usarlo como id de la novedad
        if id_manual.strip():
            novedad["id"] = id_manual.strip()
        novedad_guardada = novedades_store.agregar(novedad)
        tipo_label = "Variante a TACO existente" if tipo_taco == "existente" else "Concepto NUEVO"
        cap_label = " 📈 [FUERA capacidad]" if excede_capacidad else ""
        st.success(
            f"✅ **Aprobada** ({tipo_label}{cap_label}) con ID `{novedad_guardada['id']}`. "
            f"Código SBU: `{codigo_sbu}`. Destino: `{taco_destino_final}`. "
            f"Suma anual proyectada: **{curva_mensual['prediccion'].sum():,.0f} u** "
            f"durante {ciclo_vida} meses desde {mes_lanzamiento}."
        )
        st.info("👉 Ve al **Explorador de pronóstico** para ver cómo esta novedad reduce el gap a metas.")
        st.rerun()

# =========================================================================
# NOVEDADES APROBADAS EN EL SIMULADOR (solo origen manual)
# =========================================================================
novedades_manuales = novedades_store.filtrar_por_origen("manual")
if novedades_manuales:
    st.divider()
    st.subheader(f"📋 Novedades aprobadas en este simulador ({len(novedades_manuales)})")
    st.caption(
        "Solo se muestran las novedades aprobadas desde este simulador. "
        "Las aprobadas en _Sugerencias automáticas_ se gestionan en esa otra página."
    )

    aprobadas_view = []
    for n in novedades_manuales:
        tipo = n.get("tipo_taco", "nuevo")
        fuera = n.get("fuera_capacidad", False)
        aprobadas_view.append({
            "Eliminar": False,
            "ID": n.get("id", "")[:25],
            "ID producto (ISBN/SKU)": n.get("id_producto_manual", "") or "",
            "Cap": "📈" if fuera else "✅",
            "Código SBU": n.get("sbu_codigo", ""),
            "Marca": n.get("marca", ""),
            "Tipo TACO": "🔄 Variante" if tipo == "existente" else "🆕 Nuevo",
            "Concepto / TACO": n.get("taco_destino", "")[:30],
            "Lanzamiento": n.get("mes_lanzamiento", ""),
            "Demanda anual": f"{n.get('demanda_anual_estimada', 0):,.0f}",
            "p10-p90": f"{n.get('demanda_anual_p10', 0):,.0f} - {n.get('demanda_anual_p90', 0):,.0f}",
            "Género": n.get("features", {}).get("familia_genero", ""),
            "Mercado": n.get("features", {}).get("mercado_principal", ""),
            "Ciclo (m)": n.get("ciclo_vida_meses", 30),
        })
    df_aprobadas = pd.DataFrame(aprobadas_view)
    edited = st.data_editor(
        df_aprobadas,
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "Eliminar": st.column_config.CheckboxColumn("✓", default=False, width="small"),
        },
        disabled=[c for c in df_aprobadas.columns if c != "Eliminar"],
        key="editor_aprobadas_manual",
    )
    ids_eliminar = edited[edited["Eliminar"]]["ID"].tolist()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(
            f"💡 {len(ids_eliminar)} marcadas para eliminar. "
            f"{sum(1 for n in novedades_manuales if n.get('fuera_capacidad'))} de las "
            f"{len(novedades_manuales)} están marcadas como FUERA de capacidad."
        )
    with col2:
        if st.button(
            f"🗑️ Eliminar {len(ids_eliminar)} marcada(s)",
            disabled=(len(ids_eliminar) == 0),
            use_container_width=True,
            type="primary" if len(ids_eliminar) > 0 else "secondary",
        ):
            # Construir set de IDs completos (no truncados)
            ids_completos = set()
            for n in novedades_manuales:
                if n.get("id", "")[:25] in ids_eliminar:
                    ids_completos.add(n.get("id"))
            n_elim = novedades_store.eliminar_multiple(list(ids_completos))
            st.success(f"✓ Eliminadas {n_elim} novedades.")
            st.rerun()

# =========================================================================
# CAPACIDAD POR AÑO
# =========================================================================
st.divider()
st.subheader("📦 Capacidad de lanzamiento por año")

st.info(
    f"💡 **Regla operativa SBC**:\n\n"
    f"- Máximo **{CAPACIDAD_NOVEDADES['conceptos_por_ano']} conceptos nuevos por año** "
    f"(TACOs MP nuevos). Cada concepto idealmente con {CAPACIDAD_NOVEDADES['cubiertas_por_concepto']} "
    f"cubiertas lanzadas el mismo mes.\n"
    f"- Máximo **{CAPACIDAD_NOVEDADES['skus_por_ano']} SKUs nuevos por año** en total. "
    f"Esto suma cubiertas nuevas a TACOs existentes + cubiertas de conceptos nuevos."
)

filas_capacidad = []
for año in [2027, 2028, 2029, 2030]:
    uso = novedades_store.calcular_uso_capacidad(año)
    filas_capacidad.append({
        "Año": año,
        "Conceptos nuevos": f"{uso['conceptos_usados']}/{uso['conceptos_max']}",
        "Cubiertas totales": f"{uso['cubiertas_usadas']}/{uso['cubiertas_max']}",
        "↳ De conceptos nuevos": uso.get("cubiertas_de_conceptos_nuevos", 0),
        "↳ Variantes a TACO existente": uso.get("variantes_existentes", 0),
        "Conceptos libres": uso["conceptos_libres"],
        "Cubiertas libres": uso["cubiertas_libres"],
        "% usado": f"{uso['pct_usado']}%",
    })
df_cap = pd.DataFrame(filas_capacidad)
st.dataframe(df_cap, use_container_width=True, hide_index=True)
