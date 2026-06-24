"""
Orquestador del Sprint 3a — Forecasting completo + Anclaje a metas.

Corre en secuencia:
  1. forecast_runner.py    (Prophet por ISBN + decay)
  2. anchor_metas.py        (gap a metas conservadoras)

Uso:
  uv run python src/models/run_forecasts.py [--limite N] [--cores K]

  --limite N : top N ISBNs (para validar rápido). Sin esto procesa todos.
  --cores K  : procesos paralelos (default = todos disponibles)
"""
from __future__ import annotations
import argparse
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(limite: int | None = None, cores: int = -1):
    print("=" * 70)
    print("ORQUESTADOR SPRINT 3a — FORECASTING + ANCLAJE")
    print("=" * 70)

    t0 = time.time()

    # 1. forecast_runner
    cmd1 = [sys.executable, str(ROOT / "src/models/forecast_runner.py")]
    if limite is not None:
        cmd1.extend(["--limite", str(limite)])
    cmd1.extend(["--cores", str(cores)])
    print(f"\n▶️  Paso 1/2: forecast_runner")
    result = subprocess.run(cmd1, cwd=ROOT)
    if result.returncode != 0:
        print("❌ forecast_runner falló")
        sys.exit(1)

    # 2. anchor_metas
    cmd2 = [sys.executable, str(ROOT / "src/models/anchor_metas.py")]
    print(f"\n▶️  Paso 2/2: anchor_metas")
    result = subprocess.run(cmd2, cwd=ROOT)
    if result.returncode != 0:
        print("❌ anchor_metas falló")
        sys.exit(1)

    print(f"\n✅ Sprint 3a completado en {(time.time()-t0)/60:.1f} min")
    print(f"\nSiguiente paso (Sprint 3b):")
    print(f"   - Sugerido automático de novedades para cerrar el gap")
    print(f"   - Generación del CSV demanda 2027-2030 descargable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--cores", type=int, default=-1)
    args = parser.parse_args()
    main(limite=args.limite, cores=args.cores)
