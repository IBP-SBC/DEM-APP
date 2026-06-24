"""
Anclaje agregado a metas conservadoras BIBLIAS 2027-2030.

Lógica:
1. Toma las proyecciones Prophet del catálogo actual (no se modifican).
2. Calcula la suma anual y la compara contra las metas inmutables:
        2027 → 1,673,298
        2028 → 2,139,351
        2029 → 2,666,471
        2030 → 3,243,508
3. Calcula el GAP por año = meta - suma_proyectada
4. El GAP debe cubrirse con NOVEDADES (Sprint 3b).
   La capacidad máxima de novedades es 15 SKUs/año.

Output:
    - data/state/anclaje_metas.csv
    - Reporta el gap por año al log
    - Útil para que el sugerido automático del Sprint 3b sepa cuánto cubrir.

Importante: este módulo NO reescala las proyecciones Prophet a la baja
para cumplir metas. Si Prophet dice que el catálogo da X, no podemos
forzar X+gap por arte de magia. Lo que sí podemos es:
- Aceptar X como piso (lo que va a aportar el catálogo actual)
- Identificar el gap (lo que necesitamos cubrir con novedades)
- En Sprint 3b: sugerir el mix de novedades que cierra el gap respetando
  el techo operativo de 15 SKUs/año.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"

# =========================================================================
# METAS CONSERVADORAS INMUTABLES
# =========================================================================
METAS_BIBLIAS = {
    2027: 1_673_298,
    2028: 2_139_351,
    2029: 2_666_471,
    2030: 3_243_508,
}


def calcular_aporte_catalogo(
    proyecciones: pd.DataFrame,
    feature_isbn: pd.DataFrame,
) -> Dict[int, int]:
    """
    Aporte del catálogo actual (Prophet + decay) por año futuro.
    Solo considera BIBLIAS.
    """
    # Filtrar a BIBLIAS
    isbn_biblias = feature_isbn[feature_isbn["clase"] == "BIBLIAS"]["isbn"].tolist()
    p = proyecciones[proyecciones["isbn"].isin(isbn_biblias)].copy()
    p["anio"] = pd.to_datetime(p["ds"]).dt.year
    suma_anual = p.groupby("anio")["yhat"].sum().astype(int).to_dict()
    return suma_anual


def calcular_gap_a_metas(aporte_catalogo: Dict[int, int]) -> pd.DataFrame:
    """
    Tabla de gap por año:
        anio | meta | aporte_catalogo | gap_a_cubrir_con_novedades | gap_pct
    """
    rows = []
    for anio, meta in METAS_BIBLIAS.items():
        aporte = aporte_catalogo.get(anio, 0)
        gap = meta - aporte
        rows.append({
            "anio": anio,
            "meta": meta,
            "aporte_catalogo": aporte,
            "gap_a_cubrir_con_novedades": max(gap, 0),
            "excedente_catalogo": max(-gap, 0),
            "gap_pct_de_meta": round(max(gap, 0) / meta * 100, 1),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("ANCLAJE A METAS CONSERVADORAS BIBLIAS 2027-2030")
    print("=" * 70)

    proy_path = DATA_STATE / "proyecciones_prophet.parquet"
    if not proy_path.exists():
        print(f"❌ No existe {proy_path}.")
        print("   Corre primero: uv run python src/models/forecast_runner.py")
        return

    proy = pd.read_parquet(proy_path)
    feature_isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")

    aporte = calcular_aporte_catalogo(proy, feature_isbn)
    print(f"\n📈 Aporte del catálogo actual por año:")
    for anio, suma in sorted(aporte.items()):
        print(f"   {anio}: {suma:>12,} u")

    gap_df = calcular_gap_a_metas(aporte)
    print(f"\n🎯 GAP a cubrir con novedades:")
    print()
    print(gap_df.to_string(index=False))
    print()

    # Guardar
    out_path = DATA_STATE / "anclaje_metas.csv"
    gap_df.to_csv(out_path, index=False)
    print(f"💾 Guardado: {out_path}")

    # Diagnóstico
    total_gap = gap_df["gap_a_cubrir_con_novedades"].sum()
    print(f"\n📋 Diagnóstico:")
    print(f"   Suma de gaps 2027-2030: {total_gap:>12,} u")
    print(f"   Capacidad max novedades: 15 SKUs/año × 4 años = 60 SKUs")
    print(f"   Demanda promedio por SKU nuevo: {total_gap / 60:>10,.0f} u")
    if total_gap / 60 > 30_000:
        print(f"   ⚠️  Cada SKU nuevo necesitaría vender >{total_gap/60:,.0f} u — "
              f"es ambicioso. Revisar realismo de metas conservadoras.")
    elif total_gap / 60 > 15_000:
        print(f"   📊 Cada SKU nuevo necesitaría vender ~{total_gap/60:,.0f} u — "
              f"alcanzable con mix balanceado.")
    else:
        print(f"   ✅ Demanda objetivo por SKU nuevo razonable.")

    print("=" * 70)


if __name__ == "__main__":
    main()
