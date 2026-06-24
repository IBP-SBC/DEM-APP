"""
Forecasting Prophet por ISBN
============================
Entrena un modelo Prophet independiente para cada ISBN BIBLIA ACTIVO/RECIENTE,
usando los eventos eclesiásticos como regresores (holidays).

Decisiones econométricas clave:

1. growth='linear'
   No queremos saturación lógica (no es producto adoptivo, no hay techo natural).

2. seasonality_mode='multiplicative'
   Los picos cuatrimestrales crecen con la tendencia. Si un ISBN sube su nivel
   2x, sus picos de mes biblia también deberían crecer ~2x. La aditiva no
   capturaba esto bien en v3.2.

3. yearly_seasonality=True (datos mensuales >=24 meses lo permiten).
   weekly/daily=False (granularidad mensual).

4. holidays=eventos_eclesiasticos
   prior_scale específico por evento (mes_biblia=15 es el más fuerte).

5. Cap por madurez del ISBN (POST-predicción):
   - MADURO (>=24 meses): cap = min(p95_mensual × 1.5, demanda_anual × 1.8 / 12)
   - RECIENTE 12-24m: cap = nivel_reciente × 1.4
   - RECIENTE 6-12m:  cap = nivel_reciente × 1.6

6. Horizonte: desde el primer mes faltante hasta diciembre 2030.

7. Manejo de migración CLARIDAD:
   - Si el ISBN migra a CLARIDAD en mes M: se proyecta hasta M-1
   - Desde M, la demanda se redirige al TACO destino (en forecast_runner)
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Optional, NamedTuple
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Silenciar warnings de Prophet (informational, no errores)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


# =========================================================================
# CONSTANTES
# =========================================================================
HORIZONTE_FIN = pd.Timestamp("2030-12-01")
MIN_MESES_PROPHET = 6   # mínimo histórico para Prophet
MIN_MESES_ESTACIONAL = 24  # mínimo para yearly_seasonality real
CAP_MULTIPLICADOR_MADURO_P95 = 1.5
CAP_MULTIPLICADOR_MADURO_ANUAL = 1.8
CAP_MULTIPLICADOR_RECIENTE_LARGO = 1.4   # 12-24 meses
CAP_MULTIPLICADOR_RECIENTE_CORTO = 1.6   # 6-12 meses


class ResultadoProphet(NamedTuple):
    """Resultado del entrenamiento+proyección de un ISBN."""
    isbn: str
    exito: bool
    n_obs: int
    nivel_madurez: str  # 'MADURO' / 'RECIENTE_LARGO' / 'RECIENTE_CORTO' / 'INSUFICIENTE'
    proyeccion: Optional[pd.DataFrame]   # ds, yhat, yhat_lower, yhat_upper
    cap_aplicado: Optional[float]
    mensaje: Optional[str]


# =========================================================================
# CLASIFICACIÓN DE MADUREZ
# =========================================================================
def clasificar_madurez(meses_vida: float, n_obs: int) -> str:
    """
    Clasifica madurez del ISBN.
    Usa meses_vida (tiempo desde primera venta) NO n_meses_con_venta,
    porque hay ISBNs bursty (ej. exportación esporádica) que tienen
    pocos meses con venta pero llevan años en el catálogo.

    También requiere mínimo de observaciones (n_obs) para Prophet.
    """
    if n_obs < MIN_MESES_PROPHET:
        return "INSUFICIENTE"
    if meses_vida < 12:
        return "RECIENTE_CORTO"
    if meses_vida < 24:
        return "RECIENTE_LARGO"
    return "MADURO"


# =========================================================================
# CÁLCULO DEL CAP
# =========================================================================
def calcular_cap(serie_isbn: pd.DataFrame, demanda_anual_madura: Optional[float],
                  madurez: str) -> float:
    """
    Cap mensual por madurez del ISBN. Aplicado POST-Prophet para evitar
    proyecciones explosivas.

    Cambio v3.0a (vs v3.2): usar p99 en lugar de p95 + anclaje anual.
    El anclaje a demanda_anual_madura subestimaba ISBNs bursty (exportación
    esporádica concentrada en pocos meses con picos genuinos).
    p99×1.3 captura el techo histórico real con margen modesto de crecimiento.

    Args:
        serie_isbn: dataframe con columnas mes, unidades
        demanda_anual_madura: solo informativo (no se usa en el cap directo)
        madurez: nivel de madurez
    """
    if len(serie_isbn) == 0:
        return float('inf')

    if madurez == "MADURO":
        # p99 captura picos genuinos; 1.3 permite crecimiento modesto
        p99 = serie_isbn["unidades"].quantile(0.99)
        # piso: que el cap no caiga debajo del máximo histórico
        max_obs = serie_isbn["unidades"].max()
        return float(max(p99 * 1.3, max_obs))

    # Para RECIENTES: nivel = mediana de los últimos meses
    n_recientes = min(6, len(serie_isbn))
    nivel = serie_isbn["unidades"].tail(n_recientes).median()
    max_obs = serie_isbn["unidades"].max()

    if madurez == "RECIENTE_LARGO":
        # 1.4x mediana, pero al menos el máximo observado
        return float(max(nivel * CAP_MULTIPLICADOR_RECIENTE_LARGO, max_obs))
    elif madurez == "RECIENTE_CORTO":
        # 1.6x mediana, pero al menos el máximo observado
        return float(max(nivel * CAP_MULTIPLICADOR_RECIENTE_CORTO, max_obs))

    return float('inf')


def aplicar_ancla_anual(
    proy_df: pd.DataFrame,
    serie_isbn: pd.DataFrame,
    factor_max: float = 1.5,
) -> tuple[pd.DataFrame, Optional[dict]]:
    """
    Reescala proyecciones anuales si exceden el techo anual del ISBN.

    Razón: Prophet puede aprender tendencias crecientes razonables a corto
    plazo pero proyectadas a 5 años producen crecimientos compuestos
    irreales (ej. +75% acumulado). El catálogo SBC es maduro; el crecimiento
    se da por NUEVAS novedades, no por ISBNs existentes.

    Cálculo del techo:
        demanda_ref = media de los últimos 2 años completos (>=10 meses)
        techo_anual = demanda_ref × factor_max

    Si suma proyectada de un año > techo: reescalar TODO el año por
    techo/suma proyectada. Las bandas (lower/upper) se reescalan
    proporcionalmente.

    Args:
        proy_df: dataframe con ds, yhat, yhat_lower, yhat_upper
        serie_isbn: histórico mensual del ISBN (columnas mes, unidades)
        factor_max: 1.5 = permitir 50% sobre demanda reciente
    """
    if len(proy_df) == 0 or len(serie_isbn) == 0:
        return proy_df, None

    s = serie_isbn.copy()
    s["anio"] = pd.to_datetime(s["mes"]).dt.year
    venta_anual = s.groupby("anio").agg(
        unidades=("unidades", "sum"),
        n_meses=("mes", "count"),
    ).reset_index()
    venta_anual_compl = venta_anual[venta_anual["n_meses"] >= 10]

    if len(venta_anual_compl) == 0:
        return proy_df, None

    # Demanda referencia: máximo de últimos 2 años completos (uso max para
    # no penalizar el ISBN si su año más reciente tuvo crecimiento real)
    venta_recientes = venta_anual_compl.sort_values("anio").tail(2)
    demanda_ref = float(venta_recientes["unidades"].max())
    techo_anual = demanda_ref * factor_max

    out = proy_df.copy()
    out["anio_temp"] = pd.to_datetime(out["ds"]).dt.year
    factores = {}
    for anio in sorted(out["anio_temp"].unique()):
        mask = out["anio_temp"] == anio
        suma = out.loc[mask, "yhat"].sum()
        if suma > techo_anual:
            factor = techo_anual / suma
            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                out.loc[mask, col] *= factor
            factores[int(anio)] = round(factor, 3)
        else:
            factores[int(anio)] = 1.0

    return out.drop("anio_temp", axis=1), {
        "demanda_ref": demanda_ref,
        "techo_anual": techo_anual,
        "factores": factores,
    }


# =========================================================================
# ENTRENAR PROPHET PARA 1 ISBN
# =========================================================================
def proyectar_isbn(
    isbn: str,
    serie_isbn: pd.DataFrame,
    eventos_eclesiasticos: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    horizonte_fin: pd.Timestamp = HORIZONTE_FIN,
    demanda_anual_madura: Optional[float] = None,
    mes_corte_claridad: Optional[pd.Timestamp] = None,
) -> ResultadoProphet:
    """
    Entrena Prophet en serie_isbn y proyecta hasta horizonte_fin.

    Args:
        isbn: identificador del ISBN
        serie_isbn: filas isbn-mes con columnas (mes, unidades)
        eventos_eclesiasticos: holidays para Prophet
        fecha_corte: último mes con datos en el histórico general
        horizonte_fin: hasta cuándo proyectar (default dic 2030)
        demanda_anual_madura: para anclar cap si está disponible
        mes_corte_claridad: si el ISBN migra a CLARIDAD, se proyecta hasta este
                             mes (exclusivo). Más allá su demanda se redirige.
    """
    from prophet import Prophet

    # Validaciones
    serie = serie_isbn.copy().sort_values("mes")
    n_obs = len(serie)
    # Calcular meses_vida desde la propia serie
    if n_obs > 0:
        meses_vida = ((serie["mes"].max() - serie["mes"].min()).days / 30) + 1
    else:
        meses_vida = 0
    madurez = clasificar_madurez(meses_vida, n_obs)

    if madurez == "INSUFICIENTE":
        return ResultadoProphet(
            isbn=isbn, exito=False, n_obs=n_obs, nivel_madurez=madurez,
            proyeccion=None, cap_aplicado=None,
            mensaje=f"Solo {n_obs} meses de historia (mínimo {MIN_MESES_PROPHET})"
        )

    # Preparar dataframe Prophet
    df = pd.DataFrame({
        "ds": pd.to_datetime(serie["mes"]),
        "y": serie["unidades"].astype(float).values,
    })

    # Usar yearly_seasonality solo si tenemos >= 24 meses
    yearly = n_obs >= MIN_MESES_ESTACIONAL

    # Modelo
    try:
        m = Prophet(
            growth="linear",
            seasonality_mode="multiplicative",
            yearly_seasonality=yearly,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=eventos_eclesiasticos,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            interval_width=0.8,   # ±80% para p10-p90
            mcmc_samples=0,
            uncertainty_samples=200,  # reducimos para ir más rápido
        )
        m.fit(df)
    except Exception as e:
        return ResultadoProphet(
            isbn=isbn, exito=False, n_obs=n_obs, nivel_madurez=madurez,
            proyeccion=None, cap_aplicado=None,
            mensaje=f"Error en fit: {str(e)[:120]}"
        )

    # Construir future hasta horizonte_fin
    fechas_futuras = pd.date_range(
        start=df["ds"].max() + pd.DateOffset(months=1),
        end=horizonte_fin,
        freq="MS",
    )
    future = pd.concat([
        df[["ds"]],
        pd.DataFrame({"ds": fechas_futuras}),
    ], ignore_index=True)

    try:
        forecast = m.predict(future)
    except Exception as e:
        return ResultadoProphet(
            isbn=isbn, exito=False, n_obs=n_obs, nivel_madurez=madurez,
            proyeccion=None, cap_aplicado=None,
            mensaje=f"Error en predict: {str(e)[:120]}"
        )

    # Cap post-predicción
    cap = calcular_cap(serie, demanda_anual_madura, madurez)
    proy = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    proy["yhat"] = proy["yhat"].clip(lower=0, upper=cap)
    proy["yhat_lower"] = proy["yhat_lower"].clip(lower=0, upper=cap)
    proy["yhat_upper"] = proy["yhat_upper"].clip(lower=0, upper=cap)

    # Mantener solo los meses futuros (post fecha_corte)
    proy = proy[proy["ds"] > fecha_corte].copy()

    # Anclaje anual: evitar sobreestimación a 5 años por composición de tendencia
    proy, info_ancla = aplicar_ancla_anual(proy, serie)

    # Si el ISBN migra a CLARIDAD, cortar antes del mes de migración
    if mes_corte_claridad is not None:
        proy = proy[proy["ds"] < mes_corte_claridad].copy()

    proy["isbn"] = isbn
    proy["fuente"] = "prophet"
    proy["nivel_madurez"] = madurez

    return ResultadoProphet(
        isbn=isbn, exito=True, n_obs=n_obs, nivel_madurez=madurez,
        proyeccion=proy, cap_aplicado=cap,
        mensaje=f"ancla:{info_ancla['factores']}" if info_ancla else None,
    )


# =========================================================================
# PROYECCIÓN POR DECAY (ISBNs DECLINANDO)
# =========================================================================
def proyectar_decay(
    isbn: str,
    serie_isbn: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    horizonte_fin: pd.Timestamp = HORIZONTE_FIN,
    half_life_meses: float = 6.0,
) -> ResultadoProphet:
    """
    Proyección de decaimiento exponencial para ISBNs DECLINANDO/DESCONTINUADO
    que aún tienen ventas en los últimos meses.
    """
    serie = serie_isbn.copy().sort_values("mes")
    if len(serie) == 0:
        return ResultadoProphet(
            isbn=isbn, exito=False, n_obs=0, nivel_madurez="DECAY",
            proyeccion=None, cap_aplicado=None,
            mensaje="Sin observaciones para decay"
        )

    # Nivel base: mediana últimos 3 meses (o todo lo disponible)
    n_recientes = min(3, len(serie))
    nivel = serie["unidades"].tail(n_recientes).median()
    if nivel <= 0:
        nivel = serie["unidades"].mean()

    # Construir fechas futuras
    fechas = pd.date_range(
        start=fecha_corte + pd.DateOffset(months=1),
        end=horizonte_fin,
        freq="MS",
    )
    # Decay exponencial: y(t) = nivel * 0.5^(t/half_life)
    meses_t = np.arange(1, len(fechas) + 1, dtype=float)
    yhat = nivel * (0.5 ** (meses_t / half_life_meses))
    yhat = np.clip(yhat, 0, None)

    proy = pd.DataFrame({
        "ds": fechas,
        "yhat": yhat,
        "yhat_lower": yhat * 0.5,
        "yhat_upper": yhat * 1.5,
    })
    proy["isbn"] = isbn
    proy["fuente"] = "decay"
    proy["nivel_madurez"] = "DECAY"

    return ResultadoProphet(
        isbn=isbn, exito=True, n_obs=len(serie), nivel_madurez="DECAY",
        proyeccion=proy, cap_aplicado=None, mensaje=None,
    )


# =========================================================================
# TEST RÁPIDO
# =========================================================================
if __name__ == "__main__":
    DATA_PROC = ROOT / "data" / "processed"

    isbn_df = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")
    serie = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    eventos = pd.read_parquet(DATA_PROC / "eventos_eclesiasticos.parquet")
    fecha_corte = serie["mes"].max()

    # Tomar el top 1 ISBN BIBLIA ACTIVO por unidades
    biblias_activas = isbn_df[
        (isbn_df["clase"] == "BIBLIAS") & (isbn_df["estado"] == "ACTIVO")
    ].nlargest(3, "unidades_total")

    print(f"Fecha corte histórico: {fecha_corte:%Y-%m-%d}")
    print(f"Horizonte fin: {HORIZONTE_FIN:%Y-%m-%d}")
    print(f"\nTesteando con top 3 BIBLIAS ACTIVAS:")
    for _, row in biblias_activas.iterrows():
        isbn = row["isbn"]
        s = serie[serie["isbn"] == isbn]
        print(f"\n[{isbn}] {row['descripcion'][:50]}")
        print(f"  meses con venta: {len(s)} | demanda anual madura: {row.get('demanda_anual_madura', 'NA')}")
        r = proyectar_isbn(
            isbn=isbn,
            serie_isbn=s,
            eventos_eclesiasticos=eventos,
            fecha_corte=fecha_corte,
            demanda_anual_madura=row.get("demanda_anual_madura"),
        )
        if r.exito:
            print(f"  ✓ {r.nivel_madurez} | cap={r.cap_aplicado:,.0f} | proyección {len(r.proyeccion)} meses")
            print(f"    Primeros 6 meses proyectados:")
            for _, p in r.proyeccion.head(6).iterrows():
                print(f"      {p['ds']:%Y-%m}: yhat={p['yhat']:,.0f} [{p['yhat_lower']:,.0f}-{p['yhat_upper']:,.0f}]")
        else:
            print(f"  ✗ {r.mensaje}")
