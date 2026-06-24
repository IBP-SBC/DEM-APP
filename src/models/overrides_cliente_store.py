"""
Overrides de proyección por CLIENTE (v3.14).

Permite a Alberto ajustar la proyección de cada cliente para construir el
PRESUPUESTO de unidades e ingresos 2027-2030. Cada override tiene:

  - escala (float, default 1.0): multiplicador de nivel. Sube/baja todo el
    volumen proyectado del cliente (ej. 1.2 = 20% más que la proyección base).
  - crecimiento_anual_pct (float, default 0.0): crecimiento compuesto adicional
    año a año sobre la proyección base (ej. 8 = +8% compuesto cada año desde
    el primer año proyectado).
  - nota (str): texto libre para recordar por qué se ajustó.

Aplicación a un mes ds (con año Y, siendo Y0 el primer año proyectado):
    factor = escala * (1 + crecimiento_anual_pct/100) ** (Y - Y0)
    valor_ajustado    = valor_base    * factor
    unidades_ajustadas = unidades_base * factor

El mismo factor aplica a unidades e ingresos (el crecimiento del negocio con
ese cliente arrastra ambos). Persistencia: data/state/overrides_cliente.json
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
_PATH = _ROOT / "data" / "state" / "overrides_cliente.json"

ANIO_BASE_PROYECCION = 2027  # primer año proyectado (Y0)


# =========================================================================
# PERSISTENCIA
# =========================================================================
def cargar() -> Dict[str, Any]:
    if _PATH.exists():
        try:
            with open(_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def guardar(data: Dict[str, Any]) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        from core import cloud_storage as _cloud
        if _cloud.nube_activa():
            _cloud.subir_archivo(_PATH, "overrides_cliente.json", subcarpeta="state")
    except Exception:
        pass


# =========================================================================
# API
# =========================================================================
def set_override(cliente: str, escala: float = 1.0,
                 crecimiento_anual_pct: float = 0.0, nota: str = "") -> None:
    """Crea/actualiza el override de un cliente. Si escala=1 y crecimiento=0,
    lo elimina (vuelve a la proyección base)."""
    data = cargar()
    if abs(escala - 1.0) < 1e-9 and abs(crecimiento_anual_pct) < 1e-9 and not nota:
        data.pop(cliente, None)
    else:
        data[cliente] = {
            "escala": float(escala),
            "crecimiento_anual_pct": float(crecimiento_anual_pct),
            "nota": nota or "",
            "actualizado": datetime.now().isoformat(timespec="seconds"),
        }
    guardar(data)


def get_override(cliente: str) -> Optional[Dict[str, Any]]:
    return cargar().get(cliente)


def eliminar(cliente: str) -> bool:
    data = cargar()
    if cliente in data:
        data.pop(cliente)
        guardar(data)
        return True
    return False


def reset_todos() -> int:
    data = cargar()
    n = len(data)
    guardar({})
    return n


def factor_para(cliente: str, anio: int, overrides: Optional[Dict] = None) -> float:
    """Devuelve el factor multiplicador para un cliente en un año dado."""
    ov = (overrides or cargar()).get(cliente)
    if not ov:
        return 1.0
    escala = float(ov.get("escala", 1.0))
    g = float(ov.get("crecimiento_anual_pct", 0.0)) / 100.0
    return escala * ((1 + g) ** (anio - ANIO_BASE_PROYECCION))


# =========================================================================
# APLICACIÓN A LAS PROYECCIONES
# =========================================================================
def aplicar_overrides(proy: pd.DataFrame,
                      overrides: Optional[Dict] = None) -> pd.DataFrame:
    """Aplica los overrides a un DataFrame de proyecciones de cliente.

    Espera columnas: cliente, ds, y alguna de prediccion(_valor/_unidades),
    p10*, p90*. Devuelve una copia con las columnas escaladas y una columna
    booleana 'tiene_override'.
    """
    if proy is None or len(proy) == 0:
        return proy
    ov = overrides if overrides is not None else cargar()
    if not ov:
        out = proy.copy()
        out["tiene_override"] = False
        return out

    out = proy.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["_anio"] = out["ds"].dt.year
    out["tiene_override"] = out["cliente"].isin(ov.keys())

    # Calcular factor por fila
    def _factor(row):
        return factor_para(row["cliente"], int(row["_anio"]), ov)
    factores = out.apply(_factor, axis=1)

    cols_escalar = [c for c in [
        "prediccion", "p10", "p90",
        "prediccion_valor", "p10_valor", "p90_valor",
        "prediccion_unidades", "p10_unidades", "p90_unidades",
    ] if c in out.columns]
    for c in cols_escalar:
        out[c] = out[c] * factores

    return out.drop(columns=["_anio"], errors="ignore")


def resumen() -> Dict[str, int]:
    data = cargar()
    return {"n_clientes_con_override": len(data)}
