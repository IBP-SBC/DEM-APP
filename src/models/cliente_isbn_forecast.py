"""
Desagregación de la proyección por cliente entre sus ISBNs (Camino B, v3.16).

Idea (consensuada con Alberto):
  No hacemos un pronóstico independiente por cada par cliente×ISBN (sería
  ruido: la mayoría de pares tienen 1-3 compras). En cambio, tomamos la
  proyección TOTAL del cliente (la del modelo de cliente, con los overrides de
  presupuesto aplicados) y la REPARTIMOS entre los ISBNs que ese cliente compra,
  según su participación histórica (mix).

  Para cada cliente y año:
    unidades_isbn = proy_unidades_cliente_año × share_unidades_isbn
    valor_isbn    = proy_valor_cliente_año    × share_valor_isbn

  donde:
    share_unidades_isbn = unidades_hist_isbn / unidades_hist_total_cliente
    share_valor_isbn    = valor_hist_isbn    / valor_hist_total_cliente

  Se usan shares separados para unidades y valor porque un ISBN caro pesa más
  en valor que en unidades. Así la suma por ISBN reproduce exactamente el total
  proyectado del cliente, tanto en unidades como en valor.

Solo aplica a clientes PROYECTABLES (los que tienen proyección). Los inactivos,
nuevos y esporádicos no tienen proyección, así que no se desagregan.

Output: cliente, isbn, descripcion, anio, unidades_proy, valor_proy, categoria,
        ultimo_vendedor, tiene_override.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
_DATA_PROC = _ROOT / "data" / "processed"
_DATA_STATE = _ROOT / "data" / "state"

ANIOS = [2027, 2028, 2029, 2030]


def _cargar_mix() -> Optional[pd.DataFrame]:
    p = _DATA_PROC / "mix_cliente_isbn.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # Optimización de memoria (category) — seguro bajo pandas 3 (observed=True)
    try:
        from core.optimize import optimizar_memoria
        df = optimizar_memoria(df)
    except Exception:
        pass
    return df


def desagregar(
    clientes: Optional[List[str]] = None,
    aplicar_overrides_cliente: bool = True,
) -> Optional[pd.DataFrame]:
    """Desagrega la proyección de cliente entre sus ISBNs.

    Args:
        clientes: lista de clientes a incluir. None = todos los proyectables.
        aplicar_overrides_cliente: si True, usa la proyección con los overrides
            de presupuesto que haya configurado el usuario.

    Returns:
        DataFrame cliente×isbn×año o None si faltan insumos.
    """
    # Proyección por cliente (mensual → anual)
    proy_path = _DATA_STATE / "proyecciones_cliente.parquet"
    if not proy_path.exists():
        return None
    proy = pd.read_parquet(proy_path)
    if len(proy) == 0:
        return None

    # Aplicar overrides de presupuesto
    if aplicar_overrides_cliente:
        try:
            from models import overrides_cliente_store as ovc
            proy = ovc.aplicar_overrides(proy)
        except Exception:
            pass

    proy = proy.copy()
    proy["anio"] = pd.to_datetime(proy["ds"]).dt.year
    cv = "prediccion_valor" if "prediccion_valor" in proy.columns else "prediccion"
    cu = "prediccion_unidades" if "prediccion_unidades" in proy.columns else None

    # Total anual por cliente
    agg = {"valor": (cv, "sum")}
    if cu:
        agg["unidades"] = (cu, "sum")
    proy_anual = proy.groupby(["cliente", "anio"]).agg(**agg).reset_index()

    if clientes is not None:
        proy_anual = proy_anual[proy_anual["cliente"].isin(clientes)]
    if len(proy_anual) == 0:
        return pd.DataFrame()

    clientes_proy = proy_anual["cliente"].unique().tolist()

    # Mix histórico de esos clientes
    mix = _cargar_mix()
    if mix is None:
        return None
    mix = mix[mix["cliente"].isin(clientes_proy)].copy()
    if len(mix) == 0:
        return pd.DataFrame()

    # Shares por cliente
    tot_cli = mix.groupby("cliente").agg(
        u_tot=("unidades_hist", "sum"),
        v_tot=("valor_hist", "sum"),
    ).reset_index()
    mix = mix.merge(tot_cli, on="cliente", how="left")
    mix["share_u"] = mix["unidades_hist"] / mix["u_tot"].where(mix["u_tot"] > 0, 1)
    mix["share_v"] = mix["valor_hist"] / mix["v_tot"].where(mix["v_tot"] > 0, 1)

    # Cruce: para cada (cliente, isbn, share) × (cliente, anio, total)
    cruce = mix.merge(proy_anual, on="cliente", how="inner")
    cruce["unidades_proy"] = (
        cruce.get("unidades", 0) * cruce["share_u"]
    ).round(0) if cu else 0
    cruce["valor_proy"] = (cruce["valor"] * cruce["share_v"]).round(0)

    # Metadata: categoría, vendedor, override
    perf_path = _DATA_STATE / "perfiles_cliente.parquet"
    if perf_path.exists():
        perf = pd.read_parquet(perf_path, columns=["cliente", "categoria"])
        cat_map = dict(zip(perf["cliente"], perf["categoria"]))
    else:
        cat_map = {}
    fc_path = _DATA_PROC / "feature_cliente.parquet"
    if fc_path.exists():
        fc = pd.read_parquet(fc_path, columns=["cliente", "ultimo_vendedor"])
        vend_map = dict(zip(fc["cliente"], fc["ultimo_vendedor"]))
    else:
        vend_map = {}
    try:
        from models import overrides_cliente_store as ovc
        ov_keys = set(ovc.cargar().keys())
    except Exception:
        ov_keys = set()

    cruce["categoria"] = cruce["cliente"].map(cat_map).fillna("PROYECTABLE")
    cruce["ultimo_vendedor"] = cruce["cliente"].map(vend_map).fillna("")
    cruce["tiene_override"] = cruce["cliente"].isin(ov_keys)

    out_cols = ["cliente", "isbn", "descripcion", "anio",
                "unidades_proy", "valor_proy", "categoria",
                "ultimo_vendedor", "tiene_override"]
    out = cruce[[c for c in out_cols if c in cruce.columns]].copy()
    out = out.sort_values(["cliente", "anio", "valor_proy"],
                          ascending=[True, True, False])
    return out.reset_index(drop=True)


def desagregar_cliente(cliente: str) -> Optional[pd.DataFrame]:
    """Desagregación para un solo cliente (atajo)."""
    return desagregar(clientes=[cliente])
