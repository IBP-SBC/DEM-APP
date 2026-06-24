"""
Pipeline completo de modelos.
Corre en orden:
  1. build_features.py        (si no existe el feature store)
  2. seasonality.py           (calcula perfiles estacionales)
  3. hedonic_model.py         (entrena el modelo hedónico)

Uso:
  uv run python src/models/run_all.py
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PROC = ROOT / "data" / "processed"

steps = [
    ("Feature Store",          "src/features/build_features.py"),
    ("Perfiles estacionales",  "src/models/seasonality.py"),
    ("Modelo hedónico",        "src/models/hedonic_model.py"),
    ("Modelo de clientes",     "src/models/cliente_forecast.py"),
]


def main(skip_forecasts: bool = True):
    print("=" * 70)
    print("PIPELINE COMPLETO — SBC DEMANDA")
    print("=" * 70)

    # 1. Verificar feature store
    if not (DATA_PROC / "feature_isbn.parquet").exists():
        print("⚠️  Feature store no encontrado. Construyéndolo primero...")

    for nombre, script in steps:
        print(f"\n▶️  {nombre}: ejecutando {script}")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"❌ Falló: {nombre}")
            sys.exit(1)

    if not skip_forecasts:
        print(f"\n▶️  Forecasting Prophet por ISBN (Sprint 3a)")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(ROOT / "src/models/run_forecasts.py")],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print("❌ Falló: forecasting")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ Pipeline completado exitosamente")
    print("=" * 70)
    print("\nSiguiente paso:")
    print("   uv run streamlit run src/app/Home.py")
    if skip_forecasts:
        print("\nPara generar las proyecciones Prophet (Sprint 3a, toma ~8 min):")
        print("   uv run python src/models/run_forecasts.py")
    print("\nNavega a la página 'Simulador novedades' para probar el modelo hedónico.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-forecasts", action="store_true",
                        help="También correr forecasting Prophet (toma ~8 min adicionales)")
    args = parser.parse_args()
    main(skip_forecasts=not args.with_forecasts)
