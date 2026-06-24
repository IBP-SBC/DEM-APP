"""
Modelo de proyección de demanda por CLIENTE (v3.11).

Por qué este modelo y no Prophet por cliente:
  - Hay ~12.300 clientes únicos, pero la cola es larguísima: top 50 cubre 63%
    del valor histórico, top 500 cubre 92%. Prophet por cliente sería caro
    (varios minutos) y poco confiable en clientes con pocos meses de historia.
  - Las decisiones que requieren proyección por cliente son comerciales
    (presupuesto del vendedor, programación de visitas), no operativas
    (compras, producción). Toleran intervalos amplios.

Estrategia honesta:
  Para cada cliente clasificamos por PERFIL y aplicamos un método acorde:

  1) RECURRENTE_GRANDE: top 100 por valor + n_meses >= 18 + última compra <= 6m
     - Tendencia: regresión lineal sobre los últimos 24 meses (ventas mensuales)
     - Estacionalidad: perfil mensual propio del cliente (si tiene >= 24 meses)
       o el perfil global. Pesos suavizados con Laplace para evitar inestabilidad.
     - Decay opcional si la tendencia es < -10% anual.
     - p10/p90 a partir de la dispersión observada en los últimos 12 meses.

  2) RECURRENTE_MEDIO: posición 101-1000 + n_meses >= 12 + activo
     - Promedio robusto últimos 12 meses × perfil estacional global del segmento
     - Sin tendencia (los datos no la sostienen)

  3) ESPORADICO: valor alto pero n_meses < 12 (Iglesia El Lugar de su Presencia,
     Distribuidora Celestial, etc).
     - NO proyectar a futuro como serie continua. Mostrar histórico + alerta
       de "este cliente compra puntualmente, no proyectamos".

  4) INACTIVO: última compra > 12 meses atrás.
     - NO proyectar. Marcar como churned.

  5) NUEVO_NO_PROYECTABLE: muy pocos meses (< 6) y poco valor.
     - NO proyectar.

Output: data/state/proyecciones_cliente.parquet con columnas
  cliente, ds (mes), prediccion, p10, p90, perfil, metodo
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
DATA_PROC  = Path(__file__).parent.parent.parent / "data" / "processed"
DATA_STATE = Path(__file__).parent.parent.parent / "data" / "state"

# Configuración del modelo
ANIOS_PROYECCION = [2027, 2028, 2029, 2030]
MESES_PROYECCION = pd.date_range("2027-01-01", "2030-12-01", freq="MS")

# Umbrales para clasificación de perfiles
TOP_N_GRANDE       = 100
TOP_N_MEDIO        = 1000
MIN_MESES_GRANDE   = 18
MIN_MESES_MEDIO    = 12
MIN_MESES_PROY     = 6     # mínimo absoluto para considerar proyectable
MAX_MESES_INACT    = 12    # > 12m sin comprar → INACTIVO
LOOKBACK_TENDENCIA = 24    # meses para calcular tendencia
LOOKBACK_BASE      = 12    # meses para promedio base

# Decay: si la tendencia anual del cliente es muy negativa, atenuamos
DECAY_THRESHOLD_ANUAL = -0.10   # -10% anual
DECAY_FLOOR_RATIO     = 0.30    # piso 30% del último mes (no bajar más)


# =========================================================================
# CARGA Y AGREGACIÓN
# =========================================================================
def cargar_serie_mensual_cliente() -> pd.DataFrame:
    """Devuelve serie mensual (cliente, mes, valor, unidades, n_facturas)."""
    serie_raw = pd.read_parquet(
        DATA_PROC / "ventas_mensual_cliente_clase.parquet"
    )
    serie = (
        serie_raw.groupby(["cliente", "mes"], as_index=False)
        .agg(
            valor=("valor", "sum"),
            unidades=("unidades", "sum"),
            n_facturas=("n_facturas", "sum"),
        )
    )
    serie["mes"] = pd.to_datetime(serie["mes"])
    return serie


# =========================================================================
# CLASIFICACIÓN DE PERFIL
# =========================================================================
def clasificar_clientes(serie: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Devuelve DataFrame por cliente con perfil (motor) + categoría (negocio).

    Categorías de negocio (definidas con Alberto, v3.15):
      - NUEVO:       antigüedad < 12 meses (poco histórico para proyectar).
      - INACTIVO:    lleva >= 12 meses en la base pero sin comprar en los
                     últimos 12 meses.
      - ESPORADICO:  historia larga (>=12m de antigüedad) y compró en el último
                     año, pero en tan pocos meses distintos que la serie es
                     demasiado intermitente para proyectar con confianza.
      - PROYECTABLE: historia suficiente y compras regulares → el modelo proyecta.

    El `perfil` (RECURRENTE_GRANDE/MEDIO, etc.) se conserva para el motor de
    proyección; la `categoria` es la lente de negocio para filtrar y reportar.
    """
    resumen = (
        serie.groupby("cliente", as_index=False)
        .agg(
            valor_total=("valor", "sum"),
            unidades_total=("unidades", "sum"),
            n_meses=("mes", "nunique"),
            primera_compra=("mes", "min"),
            ultima_compra=("mes", "max"),
        )
        .sort_values("valor_total", ascending=False)
    )
    resumen["posicion"] = range(1, len(resumen) + 1)
    resumen["meses_desde_ultima"] = (
        (fecha_corte - resumen["ultima_compra"]).dt.days // 30
    )
    # Antigüedad = meses desde la primera compra hasta el corte
    resumen["antiguedad_meses"] = (
        (fecha_corte - resumen["primera_compra"]).dt.days // 30
    )
    # Densidad de compra = meses con compra / antigüedad (0-1)
    resumen["densidad_compra"] = (
        resumen["n_meses"] / resumen["antiguedad_meses"].clip(lower=1)
    ).clip(upper=1.0)

    def asignar_categoria(row):
        # 1) Nuevo: poco histórico desde su primera compra
        if row["antiguedad_meses"] < MAX_MESES_INACT:   # < 12 meses en la base
            return "NUEVO"
        # 2) Inactivo: lleva tiempo en la base pero no compra hace > 12 meses
        if row["meses_desde_ultima"] > MAX_MESES_INACT:
            return "INACTIVO"
        # 3) Tiene antigüedad y compró en el último año:
        #    proyectable si tiene suficientes meses de compra; si no, esporádico
        if row["n_meses"] >= MIN_MESES_MEDIO:
            return "PROYECTABLE"
        return "ESPORADICO"

    resumen["categoria"] = resumen.apply(asignar_categoria, axis=1)

    def asignar_perfil(row):
        cat = row["categoria"]
        if cat == "NUEVO":
            return "NUEVO_NO_PROYECTABLE"
        if cat == "INACTIVO":
            return "INACTIVO"
        if cat == "ESPORADICO":
            return ("ESPORADICO_GRANDE" if row["valor_total"] > 50_000_000
                    else "ESPORADICO")
        # PROYECTABLE → grande o medio según posición y robustez
        if row["posicion"] <= TOP_N_GRANDE and row["n_meses"] >= MIN_MESES_GRANDE:
            return "RECURRENTE_GRANDE"
        return "RECURRENTE_MEDIO"

    resumen["perfil"] = resumen.apply(asignar_perfil, axis=1)
    return resumen


# =========================================================================
# MÉTODOS DE PROYECCIÓN
# =========================================================================
def perfil_estacional_global(serie: pd.DataFrame) -> np.ndarray:
    """Devuelve perfil estacional global (12 valores que suman 1)
    calculado sobre los años completos del histórico."""
    s = serie.copy()
    s["anio"] = s["mes"].dt.year
    s["mes_num"] = s["mes"].dt.month
    # Solo años completos (12 meses)
    completos = s.groupby("anio")["mes_num"].nunique()
    anios_ok = completos[completos == 12].index.tolist()
    if not anios_ok:
        return np.ones(12) / 12
    s = s[s["anio"].isin(anios_ok)]
    por_mes = s.groupby("mes_num")["valor"].sum()
    total = por_mes.sum()
    if total <= 0:
        return np.ones(12) / 12
    perfil = np.zeros(12)
    for m in range(1, 13):
        perfil[m - 1] = (por_mes.get(m, 0) / total)
    return perfil


def perfil_estacional_cliente(
    serie_cliente: pd.DataFrame,
    perfil_global: np.ndarray,
    alpha: float = 12.0,
) -> np.ndarray:
    """Perfil propio del cliente suavizado con Laplace hacia el global.

    Si el cliente tiene >= 24 meses, su perfil pesa más. Si tiene 12,
    pesa 50/50 con el global. Si tiene 6, pesa 33/66.
    """
    s = serie_cliente.copy()
    s["mes_num"] = s["mes"].dt.month
    por_mes = s.groupby("mes_num")["valor"].sum()
    total_cliente = por_mes.sum()
    n_meses = len(s["mes"].unique())

    perfil_propio = np.zeros(12)
    for m in range(1, 13):
        perfil_propio[m - 1] = (
            por_mes.get(m, 0) / total_cliente if total_cliente > 0 else 1 / 12
        )

    # Mezcla: w_propio = n_meses / (n_meses + alpha)
    w_propio = n_meses / (n_meses + alpha)
    mezclado = w_propio * perfil_propio + (1 - w_propio) * perfil_global
    return mezclado / mezclado.sum()


def tendencia_anual(serie_cliente: pd.DataFrame, lookback_meses: int = 24,
                    col: str = "valor") -> float:
    """Devuelve crecimiento anual estimado del cliente (regresión sobre
    log(`col`+1) en los últimos `lookback_meses`). 0.0 si es inestable."""
    s = serie_cliente.sort_values("mes").tail(lookback_meses).copy()
    if len(s) < 6:
        return 0.0
    s["t"] = np.arange(len(s))
    y = np.log1p(s[col].values)
    try:
        coefs = np.polyfit(s["t"].values, y, 1)
        anual = np.exp(coefs[0] * 12) - 1
        return float(np.clip(anual, -0.5, 0.8))
    except Exception:
        return 0.0


def base_mensual_promedio(
    serie_cliente: pd.DataFrame,
    lookback_meses: int = 12,
    col: str = "valor",
) -> Tuple[float, float, float]:
    """Devuelve (mediana, media, std) de `col` mensual en los últimos
    `lookback_meses`, considerando solo meses con compras."""
    s = serie_cliente.sort_values("mes").tail(lookback_meses)
    vals = s[col].values
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    return float(np.median(vals)), float(np.mean(vals)), float(np.std(vals))


def _proyectar_serie_col(serie_cliente, perfil_global, meses_proy, col,
                          con_tendencia: bool):
    """Proyecta una columna ('valor' o 'unidades') mes a mes.
    Devuelve dict {ds: (pred, p10, p90)}."""
    _med, media12, std12 = base_mensual_promedio(serie_cliente, LOOKBACK_BASE, col)
    mediana12 = _med
    if con_tendencia:
        base = media12
    else:
        base = mediana12 if mediana12 > 0 else media12
    if base <= 0:
        return {ds: (0.0, 0.0, 0.0) for ds in meses_proy}

    perfil = perfil_estacional_cliente(serie_cliente, perfil_global)
    if perfil.std() < 0.01:
        perfil = perfil_global.copy()

    base_anual = base * 12
    ano_base = serie_cliente["mes"].max().year
    cv = (std12 / media12) if media12 > 0 else (0.5 if con_tendencia else 0.7)

    tend_anual = 0.0
    if con_tendencia:
        tend_anual = tendencia_anual(serie_cliente, LOOKBACK_TENDENCIA, col)
        if tend_anual < DECAY_THRESHOLD_ANUAL:
            tend_anual = max(tend_anual, -0.20)

    out = {}
    for ds in meses_proy:
        if con_tendencia:
            delta = (ds.year + ds.month / 12) - (ano_base + 1)
            valor_anual = base_anual * ((1 + tend_anual) ** delta)
            v = valor_anual * perfil[ds.month - 1]
            v = max(v, base * DECAY_FLOOR_RATIO * perfil[ds.month - 1])
        else:
            v = base_anual * perfil_global[ds.month - 1]
        spread = 1.28 * min(cv, 1.5)
        out[ds] = (float(v), float(max(0, v * np.exp(-spread))), float(v * np.exp(spread)))
    return out


def proyectar_cliente_recurrente_grande(
    serie_cliente: pd.DataFrame,
    perfil_global: np.ndarray,
    meses_proy: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Proyección con tendencia + estacionalidad propia, en valor Y unidades."""
    val = _proyectar_serie_col(serie_cliente, perfil_global, meses_proy, "valor", True)
    uni = _proyectar_serie_col(serie_cliente, perfil_global, meses_proy, "unidades", True)
    filas = []
    for ds in meses_proy:
        v = val.get(ds, (0, 0, 0))
        u = uni.get(ds, (0, 0, 0))
        filas.append({
            "ds": ds,
            "prediccion": v[0], "p10": v[1], "p90": v[2],       # valor (retrocompat)
            "prediccion_valor": v[0], "p10_valor": v[1], "p90_valor": v[2],
            "prediccion_unidades": u[0], "p10_unidades": u[1], "p90_unidades": u[2],
            "metodo": "tendencia_estacional",
        })
    return pd.DataFrame(filas)


def proyectar_cliente_recurrente_medio(
    serie_cliente: pd.DataFrame,
    perfil_global: np.ndarray,
    meses_proy: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Proyección sin tendencia (promedio robusto × estacionalidad global),
    en valor Y unidades."""
    val = _proyectar_serie_col(serie_cliente, perfil_global, meses_proy, "valor", False)
    uni = _proyectar_serie_col(serie_cliente, perfil_global, meses_proy, "unidades", False)
    filas = []
    for ds in meses_proy:
        v = val.get(ds, (0, 0, 0))
        u = uni.get(ds, (0, 0, 0))
        filas.append({
            "ds": ds,
            "prediccion": v[0], "p10": v[1], "p90": v[2],
            "prediccion_valor": v[0], "p10_valor": v[1], "p90_valor": v[2],
            "prediccion_unidades": u[0], "p10_unidades": u[1], "p90_unidades": u[2],
            "metodo": "promedio_estacional",
        })
    return pd.DataFrame(filas)


def _df_proy_cero(meses_proy) -> pd.DataFrame:
    return pd.DataFrame({
        "ds":         list(meses_proy),
        "prediccion": [0.0] * len(meses_proy),
        "p10":        [0.0] * len(meses_proy),
        "p90":        [0.0] * len(meses_proy),
        "metodo":     ["no_proyectable"] * len(meses_proy),
    })


# =========================================================================
# PIPELINE COMPLETO
# =========================================================================
def proyectar_todos_los_clientes() -> pd.DataFrame:
    """Genera proyecciones 2027-2030 para todos los clientes proyectables.

    Output columnas: cliente, ds, prediccion, p10, p90, perfil, metodo
    """
    print("📥 Cargando ventas mensuales por cliente...")
    serie = cargar_serie_mensual_cliente()
    print(f"   ✓ {len(serie):,} filas, {serie['cliente'].nunique():,} clientes")

    fecha_corte = serie["mes"].max()
    print(f"   Última fecha en histórico: {fecha_corte.date()}")

    print("\n🧮 Clasificando clientes por perfil...")
    perfiles = clasificar_clientes(serie, fecha_corte)
    distribucion = perfiles["perfil"].value_counts()
    print("   Distribución:")
    for p, n in distribucion.items():
        pct_valor = (perfiles[perfiles["perfil"] == p]["valor_total"].sum()
                     / perfiles["valor_total"].sum() * 100)
        print(f"     {p:<25} {n:>6} clientes  ({pct_valor:>5.1f}% del valor)")

    print("\n📊 Calculando perfil estacional global...")
    perfil_global = perfil_estacional_global(serie)
    print(f"   Picos: {', '.join([f'M{i+1}={perfil_global[i]*100:.1f}%' for i in np.argsort(-perfil_global)[:3]])}")

    print("\n🎯 Generando proyecciones por cliente...")
    proyectables_grandes = perfiles[perfiles["perfil"] == "RECURRENTE_GRANDE"]
    proyectables_medios  = perfiles[perfiles["perfil"] == "RECURRENTE_MEDIO"]
    no_proyectables_total = (
        len(perfiles) - len(proyectables_grandes) - len(proyectables_medios)
    )

    print(f"   RECURRENTE_GRANDE: {len(proyectables_grandes)} con tendencia")
    print(f"   RECURRENTE_MEDIO:  {len(proyectables_medios)} con promedio")
    print(f"   No proyectables:   {no_proyectables_total}")

    todas_proy = []
    n_total = len(proyectables_grandes) + len(proyectables_medios)
    n_done = 0

    for _, fila in proyectables_grandes.iterrows():
        cliente = fila["cliente"]
        serie_c = serie[serie["cliente"] == cliente].copy()
        proy = proyectar_cliente_recurrente_grande(serie_c, perfil_global, MESES_PROYECCION)
        proy["cliente"] = cliente
        proy["perfil"]  = fila["perfil"]
        todas_proy.append(proy)
        n_done += 1
        if n_done % 25 == 0:
            print(f"     {n_done}/{n_total} clientes procesados")

    for _, fila in proyectables_medios.iterrows():
        cliente = fila["cliente"]
        serie_c = serie[serie["cliente"] == cliente].copy()
        proy = proyectar_cliente_recurrente_medio(serie_c, perfil_global, MESES_PROYECCION)
        proy["cliente"] = cliente
        proy["perfil"]  = fila["perfil"]
        todas_proy.append(proy)
        n_done += 1
        if n_done % 100 == 0:
            print(f"     {n_done}/{n_total} clientes procesados")

    if not todas_proy:
        print("   ⚠️ No hay clientes proyectables.")
        return pd.DataFrame()

    out = pd.concat(todas_proy, ignore_index=True)
    cols_out = ["cliente", "ds", "prediccion", "p10", "p90",
                "prediccion_valor", "p10_valor", "p90_valor",
                "prediccion_unidades", "p10_unidades", "p90_unidades",
                "perfil", "metodo"]
    cols_out = [c for c in cols_out if c in out.columns]
    out = out[cols_out]
    print(f"   ✓ {len(out):,} filas de proyección ({n_done} clientes × 48 meses)")

    # Persistir
    DATA_STATE.mkdir(parents=True, exist_ok=True)
    out_path = DATA_STATE / "proyecciones_cliente.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n💾 Guardado: {out_path}")

    # También guardar el resumen por cliente con perfil
    resumen_path = DATA_STATE / "perfiles_cliente.parquet"
    perfiles.to_parquet(resumen_path, index=False)
    print(f"💾 Guardado: {resumen_path}")

    return out


# =========================================================================
# CARGADORES PARA LA APP
# =========================================================================
def cargar_proyecciones_cliente() -> Optional[pd.DataFrame]:
    """Devuelve las proyecciones persistidas. None si no existen."""
    path = DATA_STATE / "proyecciones_cliente.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def cargar_perfiles_cliente() -> Optional[pd.DataFrame]:
    """Devuelve la tabla de perfiles por cliente. None si no existe."""
    path = DATA_STATE / "perfiles_cliente.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


if __name__ == "__main__":
    print("=" * 70)
    print("PROYECCIÓN DE DEMANDA POR CLIENTE — v3.11")
    print("=" * 70)
    inicio = datetime.now()
    proyectar_todos_los_clientes()
    print(f"\n⏱  Tiempo total: {(datetime.now() - inicio).total_seconds():.0f}s")
    print("=" * 70)
