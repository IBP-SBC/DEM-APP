"""
Perfil estacional realista para distribuir demanda anual en 12 meses.

Reemplaza la distribución normal aplastada del tablero v8 anterior con
patrones reales calculados del histórico, capturando los 4 picos
cuatrimestrales causados por eventos eclesiásticos:

- Marzo: Semana Santa (compra anticipada de evangelización)
- Julio: Congresos anuales (mes completo)
- Octubre: Mes de la Biblia (pico mayor del año)
- Diciembre últimos 15 días: Pre-abastecimiento eventos enero

Los perfiles son SEGMENTADOS por familia de género (color) y mercado,
porque mujeres compran más alrededor del Día de la Mujer (marzo) y
las ventas internacionales tienen otra distribución.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)


def calcular_perfiles_estacionales(
    serie_mensual: pd.DataFrame,
    feature_isbn: pd.DataFrame,
    anios_validos: tuple[int, ...] = (2022, 2023, 2024, 2025),
) -> dict:
    """
    Calcula perfiles estacionales (% mensual del año) segmentados.

    Returns:
        dict con perfiles por nivel de segmentación:
        - 'global'                  → vector de 12 valores que suman 1.0
        - 'por_genero'              → dict {familia_genero: vector}
        - 'por_mercado'             → dict {mercado: vector}
        - 'por_genero_mercado'      → dict {(genero,mercado): vector}
    """
    # Filtrar a BIBLIAS y años con datos completos
    isbn_b = feature_isbn[feature_isbn["clase"] == "BIBLIAS"]
    serie = serie_mensual[serie_mensual["isbn"].isin(isbn_b["isbn"])].copy()
    serie["anio"] = serie["mes"].dt.year
    serie["mes_num"] = serie["mes"].dt.month
    serie = serie[serie["anio"].isin(anios_validos)]

    # Pegar features de ISBN
    serie = serie.merge(
        isbn_b[["isbn", "familia_genero", "mercado_principal"]],
        on="isbn", how="left"
    )

    # ----- PERFIL GLOBAL -----
    perfil_global = (
        serie.groupby("mes_num")["unidades"].sum()
        / serie["unidades"].sum()
    ).reindex(range(1, 13), fill_value=0).values

    # ----- POR GÉNERO -----
    perfil_genero = {}
    for genero in serie["familia_genero"].unique():
        sub = serie[serie["familia_genero"] == genero]
        if sub["unidades"].sum() < 1000:
            continue  # muy pocos datos, no segmentar
        perfil = (
            sub.groupby("mes_num")["unidades"].sum()
            / sub["unidades"].sum()
        ).reindex(range(1, 13), fill_value=0).values
        perfil_genero[genero] = perfil

    # ----- POR MERCADO -----
    perfil_mercado = {}
    for mercado in serie["mercado_principal"].unique():
        sub = serie[serie["mercado_principal"] == mercado]
        if sub["unidades"].sum() < 1000:
            continue
        perfil = (
            sub.groupby("mes_num")["unidades"].sum()
            / sub["unidades"].sum()
        ).reindex(range(1, 13), fill_value=0).values
        perfil_mercado[mercado] = perfil

    # ----- POR GÉNERO × MERCADO -----
    perfil_gm = {}
    for genero in serie["familia_genero"].unique():
        for mercado in serie["mercado_principal"].unique():
            sub = serie[
                (serie["familia_genero"] == genero)
                & (serie["mercado_principal"] == mercado)
            ]
            if sub["unidades"].sum() < 500:
                continue
            perfil = (
                sub.groupby("mes_num")["unidades"].sum()
                / sub["unidades"].sum()
            ).reindex(range(1, 13), fill_value=0).values
            perfil_gm[(genero, mercado)] = perfil

    return {
        "global": perfil_global,
        "por_genero": perfil_genero,
        "por_mercado": perfil_mercado,
        "por_genero_mercado": perfil_gm,
        "anios_usados": list(anios_validos),
    }


def aplicar_perfil(
    demanda_anual: float,
    perfil: np.ndarray | list[float],
) -> np.ndarray:
    """
    Distribuye una demanda anual en 12 valores mensuales usando un perfil.

    Args:
        demanda_anual: total anual estimado
        perfil: vector de 12 fracciones que suma ~1.0

    Returns:
        array de 12 valores enteros (redondeados) que suman aprox. demanda_anual
    """
    perfil = np.array(perfil)
    perfil = perfil / perfil.sum()  # renormalizar por seguridad
    distribucion = perfil * demanda_anual
    # Redondear conservando el total
    enteros = np.floor(distribucion).astype(int)
    sobrante = int(round(demanda_anual - enteros.sum()))
    # Distribuir el sobrante en los meses con mayor fracción decimal
    decimales = distribucion - enteros
    orden = np.argsort(-decimales)
    for i in range(abs(sobrante)):
        enteros[orden[i % 12]] += np.sign(sobrante)
    return enteros


def perfil_para_producto(
    perfiles: dict,
    familia_genero: str | None = None,
    mercado: str | None = None,
) -> np.ndarray:
    """
    Elige el perfil más específico disponible para un producto.

    Prioridad:
    1. genero × mercado (más específico)
    2. genero solo
    3. mercado solo
    4. global
    """
    if familia_genero and mercado:
        key = (familia_genero, mercado)
        if key in perfiles["por_genero_mercado"]:
            return perfiles["por_genero_mercado"][key]
    if familia_genero and familia_genero in perfiles["por_genero"]:
        return perfiles["por_genero"][familia_genero]
    if mercado and mercado in perfiles["por_mercado"]:
        return perfiles["por_mercado"][mercado]
    return perfiles["global"]


def main():
    """Calcula perfiles y los guarda en data/state/."""
    print("=" * 70)
    print("CÁLCULO DE PERFILES ESTACIONALES")
    print("=" * 70)

    serie = pd.read_parquet(DATA_PROC / "ventas_mensual_isbn.parquet")
    isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")

    perfiles = calcular_perfiles_estacionales(serie, isbn)

    # Guardar (usamos pickle por simplicidad para los dict de arrays)
    import pickle
    with open(DATA_STATE / "perfiles_estacionales.pkl", "wb") as f:
        pickle.dump(perfiles, f)

    # Reporte
    meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    print(f"\n📅 PERFIL ESTACIONAL GLOBAL (años {perfiles['anios_usados']}):")
    for i, m in enumerate(meses_nombres):
        pct = perfiles["global"][i] * 100
        barra = "█" * int(pct * 2)
        print(f"   {m}: {pct:5.2f}%  {barra}")

    print(f"\n💼 Picos identificados:")
    perfil = perfiles["global"]
    picos = np.argsort(-perfil)[:4]
    for p in sorted(picos):
        print(f"   {meses_nombres[p]}: {perfil[p]*100:.2f}%")

    print(f"\n🎯 Perfiles por género (n={len(perfiles['por_genero'])}):")
    for g, p in perfiles["por_genero"].items():
        top = np.argsort(-p)[:3]
        print(f"   {g:18s}: top meses = "
              f"{', '.join([f'{meses_nombres[i]}({p[i]*100:.1f}%)' for i in sorted(top)])}")

    print(f"\n🌍 Perfiles por mercado (n={len(perfiles['por_mercado'])}):")
    for m, p in perfiles["por_mercado"].items():
        top = np.argsort(-p)[:3]
        print(f"   {m:18s}: top meses = "
              f"{', '.join([f'{meses_nombres[i]}({p[i]*100:.1f}%)' for i in sorted(top)])}")

    print(f"\n💾 Guardado en: {DATA_STATE/'perfiles_estacionales.pkl'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
