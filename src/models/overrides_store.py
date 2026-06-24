"""
Overrides Store — Ajustes manuales de proyecciones
=====================================================
Sistema centralizado para ajustar las curvas proyectadas, ya sea por
CATEGORÍA (clase: BIBLIAS, LITERATURA, PORCIONES, etc.) o por ISBN específico.

Dos parámetros por override:
- **escala**: multiplicador constante en el tiempo (mover curva arriba/abajo).
  Rango sugerido 0.5 - 2.0. Default 1.0.
- **ciclo**: factor de duración del ciclo de vida. Aplicado como boost lineal
  creciente en el tiempo. Rango 0.5 - 2.0. Default 1.0.
  - ciclo > 1: el final del horizonte tiene más demanda (ciclo alargado)
  - ciclo < 1: el final del horizonte tiene menos demanda (ciclo acortado)

Reglas de prioridad:
- Si un ISBN tiene override específico, **anula** el de su categoría.
- Si NO tiene específico, hereda el de su categoría.
- Si la categoría tampoco tiene, no se modifica nada (escala=1.0, ciclo=1.0).

Persistencia: data/state/overrides_proyeccion.json

Estructura:
{
  "categorias": {
    "BIBLIAS":    {"escala": 1.20, "ciclo": 1.50},
    "LITERATURA": {"escala": 0.90, "ciclo": 1.00}
  },
  "isbns": {
    "9789588725079": {"escala": 0.50, "ciclo": 0.70}
  }
}
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)
OVERRIDES_PATH = DATA_STATE / "overrides_proyeccion.json"


# =========================================================================
# CRUD
# =========================================================================
def cargar() -> dict:
    """Carga el dict de overrides desde disco."""
    if not OVERRIDES_PATH.exists():
        return {"categorias": {}, "isbns": {}}
    try:
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "categorias" not in data:
            data["categorias"] = {}
        if "isbns" not in data:
            data["isbns"] = {}
        return data
    except (json.JSONDecodeError, IOError):
        return {"categorias": {}, "isbns": {}}


def guardar(overrides: dict) -> None:
    """Persiste el dict completo a disco (y a la nube si está activa)."""
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    _subir_nube(OVERRIDES_PATH, "overrides_proyeccion.json")


def _subir_nube(path, nombre: str) -> None:
    """Sube un archivo de estado a Supabase si la nube está activa (defensivo)."""
    try:
        from core import cloud_storage as _cloud
        if _cloud.nube_activa():
            _cloud.subir_archivo(path, nombre, subcarpeta="state")
    except Exception:
        pass


def set_categoria(clase: str, escala: float = 1.0, ciclo: float = 1.0) -> None:
    """Establece o actualiza el override de una categoría."""
    ov = cargar()
    if escala == 1.0 and ciclo == 1.0:
        # Si está en default, eliminar para mantener limpio
        ov["categorias"].pop(clase, None)
    else:
        ov["categorias"][clase] = {"escala": float(escala), "ciclo": float(ciclo)}
    guardar(ov)


def set_isbn(isbn: str, escala: float = 1.0, ciclo: float = 1.0) -> None:
    """Establece o actualiza el override de un ISBN específico."""
    ov = cargar()
    if escala == 1.0 and ciclo == 1.0:
        ov["isbns"].pop(isbn, None)
    else:
        ov["isbns"][isbn] = {"escala": float(escala), "ciclo": float(ciclo)}
    guardar(ov)


def eliminar_categoria(clase: str) -> bool:
    ov = cargar()
    existia = clase in ov["categorias"]
    ov["categorias"].pop(clase, None)
    guardar(ov)
    return existia


def eliminar_isbn(isbn: str) -> bool:
    ov = cargar()
    existia = isbn in ov["isbns"]
    ov["isbns"].pop(isbn, None)
    guardar(ov)
    return existia


def reset_todos() -> None:
    """Elimina TODOS los overrides (categorías E ISBNs)."""
    guardar({"categorias": {}, "isbns": {}})


def reset_categorias() -> int:
    """Elimina SOLO los overrides de categoría (los del slider de ajustes
    manuales). Preserva los overrides por ISBN del explorador.

    Returns: cuántas categorías se eliminaron.
    """
    ov = cargar()
    n = len(ov.get("categorias", {}))
    ov["categorias"] = {}
    guardar(ov)
    return n


def reset_isbns() -> int:
    """Elimina SOLO los overrides por ISBN. Preserva los de categoría.

    Returns: cuántos ISBNs se eliminaron.
    """
    ov = cargar()
    n = len(ov.get("isbns", {}))
    ov["isbns"] = {}
    guardar(ov)
    return n


def get_efectivo(isbn: str, clase: Optional[str] = None) -> dict:
    """
    Devuelve los overrides efectivos para un ISBN.
    Prioridad: ISBN específico > categoría > default (1.0, 1.0).
    """
    ov = cargar()
    if isbn in ov["isbns"]:
        return dict(ov["isbns"][isbn])
    if clase and clase in ov["categorias"]:
        return dict(ov["categorias"][clase])
    return {"escala": 1.0, "ciclo": 1.0}


# =========================================================================
# APLICACIÓN DE OVERRIDES A PROYECCIONES
# =========================================================================
def aplicar_overrides_a_proyecciones(
    proy_df: pd.DataFrame,
    feature_isbn_df: pd.DataFrame,
    overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Aplica los overrides a un dataframe de proyecciones.

    Parameters
    ----------
    proy_df : DataFrame con columnas ['isbn', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']
    feature_isbn_df : DataFrame del feature store (para obtener 'clase' por ISBN)
    overrides : dict (opcional). Si None, carga de disco.

    Returns
    -------
    DataFrame con yhat, yhat_lower, yhat_upper ajustados.
    """
    if overrides is None:
        overrides = cargar()

    if not overrides.get("categorias") and not overrides.get("isbns"):
        return proy_df  # nada que aplicar

    # Mapeo isbn → clase
    isbn_to_clase = feature_isbn_df.set_index("isbn")["clase"].to_dict()

    proy = proy_df.copy().sort_values(["isbn", "ds"]).reset_index(drop=True)
    proy["_idx"] = proy.index
    # Asegurar que las columnas a modificar son float (para evitar errores de dtype)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in proy.columns:
            proy[col] = proy[col].astype(float)

    cat_ov = overrides.get("categorias", {})
    isbn_ov = overrides.get("isbns", {})

    # Para cada ISBN, obtener su override efectivo
    isbns_unicos = proy["isbn"].unique()
    factor_escala_por_isbn = {}
    factor_ciclo_por_isbn = {}
    for isbn in isbns_unicos:
        if isbn in isbn_ov:
            factor_escala_por_isbn[isbn] = isbn_ov[isbn].get("escala", 1.0)
            factor_ciclo_por_isbn[isbn] = isbn_ov[isbn].get("ciclo", 1.0)
        else:
            clase = isbn_to_clase.get(isbn, "DESCONOCIDO")
            if clase in cat_ov:
                factor_escala_por_isbn[isbn] = cat_ov[clase].get("escala", 1.0)
                factor_ciclo_por_isbn[isbn] = cat_ov[clase].get("ciclo", 1.0)
            else:
                factor_escala_por_isbn[isbn] = 1.0
                factor_ciclo_por_isbn[isbn] = 1.0

    # Aplicar por ISBN: escala constante × boost lineal del ciclo
    for isbn in isbns_unicos:
        esc = factor_escala_por_isbn[isbn]
        ciclo = factor_ciclo_por_isbn[isbn]
        if esc == 1.0 and ciclo == 1.0:
            continue
        mask = proy["isbn"] == isbn
        n = mask.sum()
        if n == 0:
            continue
        # boost_ciclo: lineal de 1.0 a `ciclo` a lo largo de la serie
        boost = np.linspace(1.0, ciclo, n)
        factor_combinado = esc * boost
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            if col in proy.columns:
                vals = proy.loc[mask, col].values
                proy.loc[mask, col] = vals * factor_combinado

    return proy.drop(columns=["_idx"])


def aplicar_overrides_a_curva_isbn(
    serie_df: pd.DataFrame,
    isbn: str,
    clase: str,
    overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Aplica overrides a la serie de UN ISBN.
    serie_df: DataFrame con columnas ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """
    if overrides is None:
        overrides = cargar()

    eff = get_efectivo(isbn, clase)
    esc = eff["escala"]
    ciclo = eff["ciclo"]
    if esc == 1.0 and ciclo == 1.0:
        return serie_df

    serie = serie_df.copy().sort_values("ds").reset_index(drop=True)
    n = len(serie)
    if n == 0:
        return serie
    boost = np.linspace(1.0, ciclo, n)
    factor = esc * boost
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in serie.columns:
            serie[col] = serie[col].values * factor
    return serie


# =========================================================================
# RESUMEN / DIAGNÓSTICO
# =========================================================================
def resumen() -> dict:
    """Devuelve resumen de overrides activos para mostrar al usuario."""
    ov = cargar()
    return {
        "n_categorias": len(ov.get("categorias", {})),
        "n_isbns": len(ov.get("isbns", {})),
        "categorias": ov.get("categorias", {}),
        "isbns": ov.get("isbns", {}),
    }


# =========================================================================
# TEST CLI
# =========================================================================
if __name__ == "__main__":
    print("Test de overrides_store")
    reset_todos()
    set_categoria("BIBLIAS", escala=1.2, ciclo=1.5)
    set_isbn("9789588725079", escala=0.5, ciclo=0.7)
    r = resumen()
    print(f"Categorías activas: {r['n_categorias']}")
    print(f"ISBNs activos: {r['n_isbns']}")
    print(f"  BIBLIAS: {r['categorias'].get('BIBLIAS')}")
    print(f"  ISBN específico: {r['isbns'].get('9789588725079')}")

    # Test efectivo
    eff = get_efectivo("9789588725079", "BIBLIAS")
    print(f"Efectivo ISBN específico: {eff} (debe usar el del ISBN, no categoría)")
    assert eff["escala"] == 0.5

    eff2 = get_efectivo("OTRO_ISBN", "BIBLIAS")
    print(f"Efectivo ISBN sin override: {eff2} (debe heredar BIBLIAS)")
    assert eff2["escala"] == 1.2

    reset_todos()
    print("\n✓ Test pasa")
