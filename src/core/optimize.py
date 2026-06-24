"""
Optimización de memoria de DataFrames (para Streamlit Community Cloud, 1 GB).

En ventas bajamos el dataset de ~136 MB a ~21 MB (85% menos) sin cambiar ningún
número. La técnica: convertir columnas de texto repetido (baja/media
cardinalidad) a `category` y hacer downcast de enteros — SIN tocar los floats
de montos (para no perder precisión).

CUIDADO documentado (nos pasó en ventas):
  `.apply()`/`.map()` sobre una columna `category` se ejecuta sobre las
  CATEGORÍAS (incluido el vacío), no solo los valores presentes. Por eso,
  cualquier `lista.index(v)` dentro de un `.apply` debe ser defensivo
  (`... if v in lista else <default>`). Esa revisión se hace en el código que
  consume estas columnas, no aquí.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# Umbral de cardinalidad: si una columna de texto tiene <= 50% de valores
# únicos respecto al total, se convierte a category.
_UMBRAL_CARDINALIDAD = 0.5


def optimizar_memoria(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con tipos optimizados.

    - object/string de baja-media cardinalidad → category
    - int64 → downcast al entero más pequeño que quepa
    - float: NO se tocan (precisión de montos). Solo float64→float32 si se
      pide explícitamente (no por defecto).

    No cambia ningún VALOR, solo el dtype de almacenamiento.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    n = len(out)

    for col in out.columns:
        s = out[col]
        # ¿Es columna de texto? Robusto a pandas 2.x (object) y 3.x (str/string)
        dtype_name = str(s.dtype).lower()
        es_texto = (
            s.dtype == object
            or dtype_name in ("string", "str")
            or dtype_name.startswith("string")
        )
        if es_texto:
            try:
                n_unicos = s.nunique(dropna=False)
                if n > 0 and (n_unicos / n) <= _UMBRAL_CARDINALIDAD:
                    out[col] = s.astype("category")
            except Exception:
                pass
        # Enteros → downcast
        elif pd.api.types.is_integer_dtype(s):
            try:
                out[col] = pd.to_numeric(s, downcast="integer")
            except Exception:
                pass
        # Floats: NO tocar (montos). Se preserva float64.

    if verbose:
        antes = df.memory_usage(deep=True).sum() / 1e6
        despues = out.memory_usage(deep=True).sum() / 1e6
        ahorro = (1 - despues / antes) * 100 if antes > 0 else 0
        print(f"   optimizar_memoria: {antes:.1f} MB → {despues:.1f} MB "
              f"({ahorro:.0f}% menos)")

    return out


def memoria_mb(df: pd.DataFrame) -> float:
    """Memoria del DataFrame en MB (deep)."""
    if df is None:
        return 0.0
    return df.memory_usage(deep=True).sum() / 1e6
