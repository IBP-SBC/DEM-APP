"""
Página 5 — Guardar / Cargar estado del tablero
==============================================
Descarga y restaura el estado del tablero a un JSON.

El estado consiste en:
- Novedades aprobadas (manuales + sugeridos): data/state/novedades_aprobadas.json
- Correcciones TACO MP: data/state/correcciones_taco_mp.json

A diferencia de v3.3 que usaba st.session_state (que NO persiste entre recargas),
v3.4 lee y escribe directamente los archivos JSON de disco.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)
NOVEDADES_PATH = DATA_STATE / "novedades_aprobadas.json"
CORRECCIONES_PATH = DATA_STATE / "correcciones_taco_mp.json"
OVERRIDES_PATH = DATA_STATE / "overrides_proyeccion.json"

VERSION_FORMATO = "3.6"

st.set_page_config(page_title="Guardar / Cargar", page_icon="💾", layout="wide")
st.title("💾 Guardar / Cargar estado del tablero")
st.markdown("""
El estado del tablero incluye:
- **Novedades aprobadas** (manuales del simulador + aprobadas del sugerido automático)
- **Correcciones de TACO MP** aplicadas

Aquí puedes descargar TODO ese estado a un archivo JSON portable, y restaurarlo
después (útil para versionar tu trabajo, compartirlo con un colega, o tener un
respaldo antes de hacer cambios masivos).
""")


# =========================================================================
# HELPERS
# =========================================================================
def leer_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def construir_estado() -> dict:
    """Combina todos los archivos de estado en un solo dict."""
    return {
        "version_formato": VERSION_FORMATO,
        "fecha_guardado": datetime.now().isoformat(),
        "novedades_aprobadas": leer_json(NOVEDADES_PATH, []),
        "correcciones_taco_mp": leer_json(CORRECCIONES_PATH, {}),
        "overrides_proyeccion": leer_json(OVERRIDES_PATH, {"categorias": {}, "isbns": {}}),
    }


def aplicar_estado(estado: dict, modo: str = "reemplazar") -> tuple[int, int, int]:
    """
    Aplica el estado cargado a los archivos de disco.

    modo='reemplazar': sobrescribe completamente (default)
    modo='fusionar':   añade a lo existente sin borrar lo actual

    Returns: (n_novedades, n_correcciones, n_overrides) aplicadas.
    """
    novedades = estado.get("novedades_aprobadas", [])
    correcciones = estado.get("correcciones_taco_mp", {})
    overrides = estado.get("overrides_proyeccion", {"categorias": {}, "isbns": {}})

    if modo == "fusionar":
        # Fusionar novedades evitando duplicados por id
        novedades_actuales = leer_json(NOVEDADES_PATH, [])
        ids_existentes = {n.get("id") for n in novedades_actuales}
        for n in novedades:
            if n.get("id") not in ids_existentes:
                novedades_actuales.append(n)
        novedades = novedades_actuales

        # Fusionar correcciones (las del archivo importado prevalecen)
        correcciones_actuales = leer_json(CORRECCIONES_PATH, {})
        correcciones_actuales.update(correcciones)
        correcciones = correcciones_actuales

        # Fusionar overrides (los del archivo importado prevalecen)
        ov_actuales = leer_json(OVERRIDES_PATH, {"categorias": {}, "isbns": {}})
        ov_actuales.setdefault("categorias", {}).update(overrides.get("categorias", {}))
        ov_actuales.setdefault("isbns", {}).update(overrides.get("isbns", {}))
        overrides = ov_actuales

    with open(NOVEDADES_PATH, "w", encoding="utf-8") as f:
        json.dump(novedades, f, indent=2, ensure_ascii=False, default=str)
    with open(CORRECCIONES_PATH, "w", encoding="utf-8") as f:
        json.dump(correcciones, f, indent=2, ensure_ascii=False, default=str)
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False, default=str)

    n_ov = len(overrides.get("categorias", {})) + len(overrides.get("isbns", {}))
    return len(novedades), len(correcciones), n_ov


# =========================================================================
# ESTADO ACTUAL
# =========================================================================
estado_actual = construir_estado()
n_nov = len(estado_actual["novedades_aprobadas"])
n_corr = len(estado_actual["correcciones_taco_mp"])
n_ov_cat = len(estado_actual["overrides_proyeccion"].get("categorias", {}))
n_ov_isbn = len(estado_actual["overrides_proyeccion"].get("isbns", {}))

st.divider()
st.subheader("Estado actual del tablero")
c1, c2, c3 = st.columns(3)
c1.metric("Novedades aprobadas", f"{n_nov}")
c2.metric("Correcciones TACO MP", f"{n_corr}")
c3.metric("Overrides proyección", f"{n_ov_cat + n_ov_isbn}")

# Desglose
if n_nov > 0:
    nov_data = estado_actual["novedades_aprobadas"]
    n_manual = sum(1 for n in nov_data if n.get("origen", "manual") == "manual")
    n_sug = sum(1 for n in nov_data if n.get("origen") == "sugerencia_automatica")
    n_fuera = sum(1 for n in nov_data if n.get("fuera_capacidad", False))
    st.caption(
        f"📊 Desglose de novedades: **{n_manual}** del simulador · "
        f"**{n_sug}** sugerencias aprobadas · "
        f"**{n_fuera}** marcadas como FUERA de capacidad"
    )
if n_ov_cat + n_ov_isbn > 0:
    st.caption(
        f"🎚️ Overrides activos: **{n_ov_cat}** por categoría · **{n_ov_isbn}** por ISBN específico"
    )

st.divider()

# =========================================================================
# DESCARGAR
# =========================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("⬇️ Descargar estado actual")
    json_str = json.dumps(estado_actual, indent=2, default=str, ensure_ascii=False)
    st.json(estado_actual, expanded=False)
    st.download_button(
        label="📥 Descargar estado.json",
        data=json_str,
        file_name=f"estado_sbc_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        f"Incluye **{n_nov} novedades** + **{n_corr} correcciones TACO MP**. "
        f"Tamaño: ~{len(json_str)/1024:.1f} KB."
    )

# =========================================================================
# CARGAR
# =========================================================================
with col2:
    st.subheader("⬆️ Cargar estado guardado")
    archivo = st.file_uploader("Sube un estado.json previo", type=["json"], key="upload_estado")

    if archivo:
        try:
            contenido = archivo.read().decode("utf-8")
            estado_cargado = json.loads(contenido)

            # Validación básica
            if not isinstance(estado_cargado, dict):
                st.error("❌ El archivo no tiene formato válido (no es un objeto JSON).")
            elif "novedades_aprobadas" not in estado_cargado and "correcciones_taco_mp" not in estado_cargado:
                st.error("❌ El archivo no contiene las claves esperadas "
                          "(`novedades_aprobadas`, `correcciones_taco_mp`). "
                          "¿Es un estado antiguo de v3.3 o anterior?")
            else:
                n_nov_cargar = len(estado_cargado.get("novedades_aprobadas", []))
                n_corr_cargar = len(estado_cargado.get("correcciones_taco_mp", {}))
                ov_cargados = estado_cargado.get("overrides_proyeccion", {"categorias": {}, "isbns": {}})
                n_ov_cargar = len(ov_cargados.get("categorias", {})) + len(ov_cargados.get("isbns", {}))
                version_cargada = estado_cargado.get("version_formato", "desconocida")
                fecha_guardado = estado_cargado.get("fecha_guardado", "?")

                st.success(
                    f"✅ Archivo válido (v{version_cargada}, guardado {fecha_guardado[:19]}). "
                    f"Contiene **{n_nov_cargar} novedades**, **{n_corr_cargar} correcciones** "
                    f"y **{n_ov_cargar} overrides** de proyección."
                )

                st.json(estado_cargado, expanded=False)

                modo = st.radio(
                    "¿Cómo aplicar?",
                    options=["reemplazar", "fusionar"],
                    format_func=lambda x: {
                        "reemplazar": "🔄 Reemplazar todo el estado actual (los datos actuales se PIERDEN)",
                        "fusionar": "➕ Fusionar con el estado actual (mantener actuales + añadir cargados)",
                    }[x],
                    key="modo_carga",
                )

                if st.button("✅ Aplicar este estado al tablero", type="primary", use_container_width=True):
                    n_nov_final, n_corr_final, n_ov_final = aplicar_estado(estado_cargado, modo=modo)
                    st.success(
                        f"✅ Estado aplicado. Ahora el tablero tiene "
                        f"**{n_nov_final} novedades**, **{n_corr_final} correcciones** "
                        f"y **{n_ov_final} overrides** de proyección. "
                        f"Navega a las otras páginas para verificar."
                    )
                    st.cache_data.clear()
                    st.rerun()

        except json.JSONDecodeError as e:
            st.error(f"❌ El archivo no es un JSON válido: {e}")
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

st.divider()
st.caption(
    "💡 La descarga incluye TODAS las novedades aprobadas y correcciones. La carga restaura "
    "exactamente eso. Útil para versionar tu trabajo y compartir estados entre compañeros. "
    "Los archivos físicos viven en `data/state/novedades_aprobadas.json` y "
    "`data/state/correcciones_taco_mp.json`."
)
