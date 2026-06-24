"""
Forecast Runner — paraleliza el entrenamiento Prophet sobre los ISBNs candidatos.

Entrada:
  - feature_isbn.parquet
  - ventas_mensual_isbn.parquet
  - eventos_eclesiasticos.parquet

Salida:
  - data/state/proyecciones_prophet.parquet
        Columnas: isbn, ds, yhat, yhat_lower, yhat_upper, fuente, nivel_madurez,
                  taco_mp_proyectado (con migración CLARIDAD aplicada)
  - data/state/forecast_log.csv
        Bitácora con éxito/falla y madurez por ISBN

Uso:
    uv run python src/models/forecast_runner.py [--limite N] [--cores K]

  --limite N : entrena solo los TOP N ISBNs por unidades (para validar rápido)
  --cores K  : número de procesos paralelos (default = todos los disponibles - 1)
"""
from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.forecast_prophet import (
    proyectar_isbn,
    proyectar_decay,
    ResultadoProphet,
    HORIZONTE_FIN,
    MIN_MESES_PROPHET,
)
from utils.dictionaries import CICLO_VIDA_DEFAULT_MESES

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)

OUT_PROYECCIONES = DATA_STATE / "proyecciones_prophet.parquet"
OUT_LOG = DATA_STATE / "forecast_log.csv"


# =========================================================================
# CLASIFICACIÓN DE ISBNs PARA PROCESAR
# =========================================================================
def clasificar_isbns(feature_isbn: pd.DataFrame, serie: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada ISBN en una de estas categorías para el forecasting:

    - 'prophet'   → BIBLIAS ACTIVO/RECIENTE con >=MIN_MESES_PROPHET meses con venta
    - 'decay'     → BIBLIAS DECLINANDO con ventas en últimos 6 meses
    - 'cero'      → DESCONTINUADO o sin ventas recientes
    - 'no_proyectar' → fuera de catálogo de proyección (no BIBLIAS)

    Returns:
        feature_isbn con columna nueva 'forecast_metodo'
    """
    fecha_corte = serie["mes"].max()
    meses_recientes = fecha_corte - pd.DateOffset(months=6)

    # Última venta efectiva (mes con unidades > 0)
    ultima_venta = serie.groupby("isbn")["mes"].max().reset_index()
    ultima_venta.columns = ["isbn", "ultima_venta_mes"]
    df = feature_isbn.merge(ultima_venta, on="isbn", how="left")

    def asignar(row):
        if row["clase"] != "BIBLIAS":
            return "no_proyectar"
        if row["estado"] in ("ACTIVO", "RECIENTE") and row["n_meses_con_venta"] >= MIN_MESES_PROPHET:
            return "prophet"
        if row["estado"] == "DECLINANDO" and pd.notna(row.get("ultima_venta_mes")):
            if row["ultima_venta_mes"] >= meses_recientes:
                return "decay"
        if row["estado"] in ("ACTIVO", "RECIENTE") and row["n_meses_con_venta"] < MIN_MESES_PROPHET:
            return "decay"  # poca historia, usar decay desde nivel actual
        return "cero"

    df["forecast_metodo"] = df.apply(asignar, axis=1)
    return df


# =========================================================================
# WORKERS
# =========================================================================
def _worker_prophet(args) -> ResultadoProphet:
    """Worker para Prophet (1 ISBN). Recibe tupla por joblib."""
    isbn, serie_isbn, eventos, fecha_corte, demanda_anual_madura, mes_corte_claridad = args
    try:
        return proyectar_isbn(
            isbn=isbn,
            serie_isbn=serie_isbn,
            eventos_eclesiasticos=eventos,
            fecha_corte=fecha_corte,
            demanda_anual_madura=demanda_anual_madura,
            mes_corte_claridad=mes_corte_claridad,
        )
    except Exception as e:
        return ResultadoProphet(
            isbn=isbn, exito=False, n_obs=len(serie_isbn),
            nivel_madurez="ERROR", proyeccion=None, cap_aplicado=None,
            mensaje=f"Exception: {str(e)[:200]}"
        )


def _worker_decay(args) -> ResultadoProphet:
    """Worker para decay (1 ISBN)."""
    isbn, serie_isbn, fecha_corte = args
    return proyectar_decay(
        isbn=isbn, serie_isbn=serie_isbn, fecha_corte=fecha_corte,
    )


# =========================================================================
# PROCESAMIENTO CLARIDAD
# =========================================================================
def calcular_mes_corte_claridad(row) -> Optional[pd.Timestamp]:
    """
    Devuelve el primer mes en que el ISBN deja de existir porque migra
    a CLARIDAD. None si no migra.
    """
    if not row.get("migra_a_claridad", False):
        return None
    mes_str = row.get("mes_inicio_claridad", "")
    if not mes_str:
        return None
    try:
        return pd.to_datetime(mes_str)
    except Exception:
        return None


def aplicar_taco_proyectado(proy_df: pd.DataFrame, feature_isbn: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columna taco_mp_proyectado: TACO MP del ISBN ORIGINAL hasta el mes
    de migración (exclusivo), y TACO DESTINO CLARIDAD desde ahí.

    Como cortamos las proyecciones del ISBN viejo en mes_corte_claridad - 1,
    en este momento taco_mp_proyectado = taco_mp original. Los TACOs CLARIDAD
    nuevos se ingresarán por separado en una etapa posterior (Sprint 3b)
    sumando los flujos de los ISBNs que migran a ellos.
    """
    mapeo = feature_isbn.set_index("isbn")[["taco_mp", "migra_a_claridad",
                                              "taco_destino_claridad",
                                              "mes_inicio_claridad"]]
    out = proy_df.merge(
        mapeo.reset_index(),
        on="isbn", how="left",
    )
    # Por defecto, taco_mp_proyectado = taco_mp original
    out["taco_mp_proyectado"] = out["taco_mp"]
    return out


# =========================================================================
# MAIN
# =========================================================================
def main(limite: Optional[int] = None, cores: int = -1):
    print("=" * 70)
    print("FORECAST RUNNER — PROPHET POR ISBN")
    print("=" * 70)
    print(f"⏱  Inicio: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

    feature_isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")
    serie = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    eventos = pd.read_parquet(DATA_PROC / "eventos_eclesiasticos.parquet")
    fecha_corte = serie["mes"].max()
    print(f"\n📅 Fecha corte histórico: {fecha_corte:%Y-%m-%d}")
    print(f"📅 Horizonte fin:         {HORIZONTE_FIN:%Y-%m-%d}")

    # Clasificación
    fc = clasificar_isbns(feature_isbn, serie)
    print(f"\n📋 Distribución de métodos:")
    print(fc["forecast_metodo"].value_counts().to_string())

    # Tareas Prophet
    fc_prophet = fc[fc["forecast_metodo"] == "prophet"].copy()
    if limite is not None:
        fc_prophet = fc_prophet.nlargest(limite, "unidades_total")
        print(f"\n⚠️  Modo --limite {limite}: solo top {limite} ISBNs por unidades")

    tareas_prophet = []
    for _, row in fc_prophet.iterrows():
        isbn = row["isbn"]
        s_isbn = serie[serie["isbn"] == isbn].copy()
        if len(s_isbn) < MIN_MESES_PROPHET:
            continue
        tareas_prophet.append((
            isbn, s_isbn, eventos, fecha_corte,
            row.get("demanda_anual_madura"),
            calcular_mes_corte_claridad(row),
        ))

    # Tareas decay
    fc_decay = fc[fc["forecast_metodo"] == "decay"].copy()
    if limite is not None:
        fc_decay = fc_decay.nlargest(min(limite, len(fc_decay)), "unidades_total")
    tareas_decay = []
    for _, row in fc_decay.iterrows():
        isbn = row["isbn"]
        s_isbn = serie[serie["isbn"] == isbn].copy()
        tareas_decay.append((isbn, s_isbn, fecha_corte))

    print(f"\n🔢 Tareas:")
    print(f"   Prophet: {len(tareas_prophet)} ISBNs")
    print(f"   Decay:   {len(tareas_decay)} ISBNs")

    # Ejecución paralela
    print(f"\n⚙️  Ejecutando con {cores if cores > 0 else 'todos los'} cores...")
    t0 = time.time()

    resultados_prophet = Parallel(n_jobs=cores, verbose=5)(
        delayed(_worker_prophet)(arg) for arg in tareas_prophet
    )
    print(f"\n   Prophet completado en {time.time()-t0:.1f}s")

    t1 = time.time()
    resultados_decay = Parallel(n_jobs=cores, verbose=2)(
        delayed(_worker_decay)(arg) for arg in tareas_decay
    )
    print(f"   Decay completado en {time.time()-t1:.1f}s")

    resultados = resultados_prophet + resultados_decay

    # Consolidar proyecciones
    proyecciones = []
    log_rows = []
    for r in resultados:
        log_rows.append({
            "isbn": r.isbn,
            "exito": r.exito,
            "n_obs": r.n_obs,
            "nivel_madurez": r.nivel_madurez,
            "cap_aplicado": r.cap_aplicado,
            "mensaje": r.mensaje,
        })
        if r.exito and r.proyeccion is not None:
            proyecciones.append(r.proyeccion)

    if proyecciones:
        proy_df = pd.concat(proyecciones, ignore_index=True)
        proy_df = aplicar_taco_proyectado(proy_df, feature_isbn)
        proy_df.to_parquet(OUT_PROYECCIONES, index=False)
        print(f"\n💾 Proyecciones guardadas: {OUT_PROYECCIONES}")
        print(f"   {len(proy_df):,} filas isbn-mes")
    else:
        print("\n⚠️  Sin proyecciones generadas")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUT_LOG, index=False)
    print(f"   {OUT_LOG} ({len(log_df)} entradas)")

    # Métricas finales
    n_exito = log_df["exito"].sum()
    n_fallo = (~log_df["exito"]).sum()
    print(f"\n📊 Resultado:")
    print(f"   ✓ Éxitos:  {n_exito}")
    print(f"   ✗ Fallos:  {n_fallo}")
    print(f"   Distribución madurez:")
    for m, n in log_df["nivel_madurez"].value_counts().items():
        print(f"      {m}: {n}")

    # Verificación: suma anual proyectada vs metas
    if proyecciones:
        proy_df["anio"] = pd.to_datetime(proy_df["ds"]).dt.year
        suma_anio = proy_df.groupby("anio")["yhat"].sum().astype(int)
        print(f"\n📈 Suma anual proyectada (suma de yhat por año):")
        for a, s in suma_anio.items():
            print(f"   {a}: {s:>10,} u")

    print(f"\n⏱  Total: {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=int, default=None,
                        help="Solo top N ISBNs por unidades (para validar rápido)")
    parser.add_argument("--cores", type=int, default=-1,
                        help="Número de cores para joblib (default: todos)")
    args = parser.parse_args()
    main(limite=args.limite, cores=args.cores)
