"""
Filtros globales compartidos entre TODAS las páginas (v3.12).

Por qué existe:
  Antes cada página tenía sus propios filtros (o ninguno). El usuario quería
  un set único de filtros — de dashboard y temporales — que viva en el sidebar
  de TODAS las páginas y afecte TODO de forma consistente.

Cómo funciona:
  - Streamlit multipage conserva st.session_state entre páginas.
  - Cada widget usa un `key` estable con prefijo `fg_` (filtro global).
  - La primera vez se inicializa con defaults sensatos; después, cualquier
    cambio que haga el usuario en una página se refleja en las demás porque
    todas leen del mismo session_state.

Uso típico en cada página:

    from app.filtros_globales import (
        render_filtros_globales, aplicar_filtros_isbn,
        aplicar_filtros_temporal_serie, label_periodo_actual,
    )

    filtros = render_filtros_globales(isbn, serie)
    isbn_f = aplicar_filtros_isbn(isbn, filtros)
    serie_f = aplicar_filtros_temporal_serie(serie, filtros)
    st.caption(f"Período: {label_periodo_actual(filtros)}")
"""
from __future__ import annotations
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

MESES_NOMBRES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Orden canónico para mostrar categorías de precio
ORDEN_CATEGORIA_PRECIO = [
    "economica", "semi_economica", "media", "semi_fina", "fina", "no_aplica"
]
LABEL_CATEGORIA_PRECIO = {
    "economica": "Económica",
    "semi_economica": "Semi-económica",
    "media": "Media",
    "semi_fina": "Semi-fina",
    "fina": "Fina",
    "no_aplica": "No aplica",
}
LABEL_GENERO = {
    "femenino": "Femenino",
    "masculino": "Masculino",
    "juvenil": "Juvenil",
    "neutro": "Neutro",
    "no_clasificado": "No clasificado",
}


def render_filtros_globales(
    isbn: pd.DataFrame,
    serie: pd.DataFrame,
    incluir_temporales: bool = True,
) -> Dict[str, Any]:
    """Dibuja los filtros globales en el sidebar y devuelve el dict de
    selecciones. Las claves de session_state (`fg_*`) garantizan que los
    valores persistan entre páginas.

    Args:
        isbn: feature_isbn (para extraer opciones disponibles)
        serie: serie mensual (para años disponibles)
        incluir_temporales: si False, omite los filtros de año/mes (útil
            en páginas que no manejan histórico).
    """
    # Asegurar columna de año en la serie
    if "anio" not in serie.columns and "mes" in serie.columns:
        serie = serie.copy()
        serie["anio"] = pd.to_datetime(serie["mes"]).dt.year

    with st.sidebar:
        st.markdown("### 🎚️ Filtros del dashboard")
        st.caption("Afectan TODAS las páginas de la app.")

        clase_opts = sorted([c for c in isbn["clase"].dropna().unique() if str(c).strip()])
        clase_default = ["BIBLIAS"] if "BIBLIAS" in clase_opts else []
        clase_sel = st.multiselect(
            "Clase", options=clase_opts,
            default=clase_default, key="fg_clase",
            help="Vacío = todas las clases.",
        )

        estado_opts = ["ACTIVO", "RECIENTE", "DECLINANDO", "DESCONTINUADO"]
        estado_sel = st.multiselect(
            "Estado", options=estado_opts,
            default=["ACTIVO", "RECIENTE"], key="fg_estado",
            help="Vacío = todos los estados.",
        )

        mercado_opts = ["nacional", "internacional", "ambos"]
        mercado_sel = st.multiselect(
            "Mercado", options=mercado_opts,
            default=mercado_opts, key="fg_mercado",
            help="Vacío = todos los mercados.",
        )

        # NUEVO v3.12: categoría de precio
        cat_opts = [c for c in ORDEN_CATEGORIA_PRECIO
                    if c in isbn["categoria_precio"].dropna().unique()]
        cat_sel = st.multiselect(
            "Categoría de precio", options=cat_opts,
            default=[], key="fg_cat_precio",
            format_func=lambda c: LABEL_CATEGORIA_PRECIO.get(c, c),
            help="Vacío = todas las categorías. Solo aplica a BIBLIAS.",
        )

        # NUEVO v3.12: familia por género
        gen_opts = [g for g in ["femenino", "masculino", "juvenil", "neutro", "no_clasificado"]
                    if g in isbn["familia_genero"].dropna().unique()]
        gen_sel = st.multiselect(
            "Familia por género", options=gen_opts,
            default=[], key="fg_genero",
            format_func=lambda g: LABEL_GENERO.get(g, g),
            help="Vacío = todos los géneros. El género se infiere por color.",
        )

        anios_sel: List[int] = []
        meses_sel: List[int] = []
        if incluir_temporales:
            st.divider()
            st.markdown("### 📅 Filtros temporales")
            st.caption("Aplican al histórico mostrado en gráficas y tablas.")
            anios_disponibles = sorted(serie["anio"].unique(), reverse=True)
            default_anios = (
                anios_disponibles[:2] if len(anios_disponibles) >= 2
                else anios_disponibles
            )
            anios_sel = st.multiselect(
                "Año(s)", options=anios_disponibles,
                default=default_anios, key="fg_anios",
                help="Vacío = todos los años.",
            )
            meses_labels = [f"{m:02d} - {MESES_NOMBRES[m-1]}" for m in range(1, 13)]
            meses_sel_labels = st.multiselect(
                "Mes(es)", options=meses_labels,
                default=[], key="fg_meses",
                help="Vacío = todos los meses.",
            )
            meses_sel = (
                [int(m.split(" ")[0]) for m in meses_sel_labels]
                if meses_sel_labels else list(range(1, 13))
            )

    return {
        "clase": clase_sel,
        "estado": estado_sel,
        "mercado": mercado_sel,
        "categoria_precio": cat_sel,
        "familia_genero": gen_sel,
        "anios": anios_sel,
        "meses": meses_sel,
    }


def aplicar_filtros_isbn(isbn: pd.DataFrame, filtros: Dict[str, Any]) -> pd.DataFrame:
    """Aplica los filtros de dashboard al universo de ISBNs (AND lógico).
    Filtros vacíos = no filtran (pasa todo)."""
    out = isbn
    if filtros.get("clase"):
        out = out[out["clase"].isin(filtros["clase"])]
    if filtros.get("estado"):
        out = out[out["estado"].isin(filtros["estado"])]
    if filtros.get("mercado"):
        out = out[out["mercado_principal"].isin(filtros["mercado"])]
    if filtros.get("categoria_precio"):
        out = out[out["categoria_precio"].isin(filtros["categoria_precio"])]
    if filtros.get("familia_genero"):
        out = out[out["familia_genero"].isin(filtros["familia_genero"])]
    return out


def aplicar_filtros_temporal_serie(
    serie: pd.DataFrame, filtros: Dict[str, Any]
) -> pd.DataFrame:
    """Filtra una serie mensual por los años/meses seleccionados.
    Requiere columna 'mes' (datetime)."""
    out = serie.copy()
    if "mes" not in out.columns:
        return out
    out["mes"] = pd.to_datetime(out["mes"])
    out["_anio"] = out["mes"].dt.year
    out["_mes_num"] = out["mes"].dt.month
    if filtros.get("anios"):
        out = out[out["_anio"].isin(filtros["anios"])]
    meses = filtros.get("meses", [])
    if meses and len(meses) < 12:
        out = out[out["_mes_num"].isin(meses)]
    return out.drop(columns=["_anio", "_mes_num"], errors="ignore")


def label_periodo_actual(filtros: Dict[str, Any]) -> str:
    """Texto humano del período temporal seleccionado."""
    anios = filtros.get("anios", [])
    meses = filtros.get("meses", [])
    tiene_meses = meses and len(meses) < 12
    if anios and tiene_meses:
        return f"{', '.join(map(str, anios))} · meses {','.join([MESES_NOMBRES[m-1] for m in meses])}"
    if anios:
        return f"años {', '.join(map(str, sorted(anios)))}"
    if tiene_meses:
        return f"meses {','.join([MESES_NOMBRES[m-1] for m in meses])} (todos los años)"
    return "todo el histórico"


def resumen_filtros_dashboard(filtros: Dict[str, Any]) -> str:
    """Texto humano de los filtros de dashboard activos (para banners)."""
    chips = []
    if filtros.get("clase"):
        chips.append("Clase: " + ", ".join(filtros["clase"]))
    if filtros.get("estado"):
        chips.append("Estado: " + ", ".join(filtros["estado"]))
    if filtros.get("mercado") and len(filtros["mercado"]) < 3:
        chips.append("Mercado: " + ", ".join(filtros["mercado"]))
    if filtros.get("categoria_precio"):
        chips.append("Categoría: " + ", ".join(
            LABEL_CATEGORIA_PRECIO.get(c, c) for c in filtros["categoria_precio"]))
    if filtros.get("familia_genero"):
        chips.append("Género: " + ", ".join(
            LABEL_GENERO.get(g, g) for g in filtros["familia_genero"]))
    return " · ".join(chips) if chips else "sin filtros de dashboard"
