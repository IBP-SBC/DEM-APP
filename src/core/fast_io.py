"""
Lectura rápida de Excel con python-calamine (fallback a openpyxl).

En ventas, calamine bajó la lectura del Excel de ~36s a ~8s. Aquí aplica a la
ingesta del SIESA y al histórico (pasos de ESCRITORIO). Para los parquets del
feature store no hace falta (ya son rápidos).
"""
from __future__ import annotations
import pandas as pd


def leer_excel(ruta, **kwargs) -> pd.DataFrame:
    """Lee un Excel usando calamine si está disponible; si no, openpyxl.

    Acepta los mismos kwargs que pd.read_excel (sheet_name, dtype, etc.).
    """
    try:
        import python_calamine  # noqa: F401
        return pd.read_excel(ruta, engine="calamine", **kwargs)
    except Exception:
        # Fallback robusto: openpyxl (siempre disponible)
        kwargs.pop("engine", None)
        try:
            return pd.read_excel(ruta, engine="openpyxl", **kwargs)
        except Exception:
            return pd.read_excel(ruta, **kwargs)
