"""
Subir artefactos calculados a Supabase (paso de ESCRITORIO, v3.17).

Tras reentrenar en el Mac (run_all + run_forecasts), este script sube a
Supabase los artefactos que la nube necesita: feature store, proyecciones
Prophet, modelo hedónico, perfiles y proyecciones de cliente.

Lee las credenciales de .streamlit/secrets.toml (el mismo que usa la app).

Uso:
    uv run python src/models/subir_a_nube.py
"""
from __future__ import annotations
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _cargar_secrets_a_entorno():
    """Lee .streamlit/secrets.toml y lo expone como un st.secrets simulado,
    para reutilizar core.cloud_storage sin Streamlit corriendo.

    Robusto a problemas típicos de Mac: comillas tipográficas (smart quotes)
    que mete TextEdit, BOM, y errores de formato (con mensaje claro en vez de
    un traceback)."""
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        print(f"❌ No se encontró {secrets_path}")
        print("   Crea .streamlit/secrets.toml con la sección [supabase].")
        sys.exit(1)

    # Leer como texto y sanear (BOM + comillas tipográficas → rectas)
    texto = secrets_path.read_text(encoding="utf-8-sig")  # -sig quita BOM
    reemplazos = {
        "\u201c": '"', "\u201d": '"',   # comillas dobles curvas
        "\u2018": "'", "\u2019": "'",   # comillas simples curvas
        "\u00a0": " ",                   # espacio no separable
    }
    for malo, bueno in reemplazos.items():
        texto = texto.replace(malo, bueno)

    try:
        secrets = tomllib.loads(texto)
    except tomllib.TOMLDecodeError as e:
        print("❌ El archivo .streamlit/secrets.toml tiene un error de formato.")
        print(f"   Detalle: {e}")
        print("")
        print("   Suele pasar al editarlo con TextEdit (mete comillas curvas).")
        print("   Recréalo limpio con este comando (una sola línea):")
        print('   printf \'[usuarios]\\nSBC001 = "TU_CLAVE"\\n\\n[supabase]\\n'
              'url = "https://TU_PROYECTO.supabase.co"\\nkey = "sb_secret_..."\\n'
              'bucket = "sbc-demanda"\\n\' > .streamlit/secrets.toml')
        sys.exit(1)

    if "supabase" not in secrets:
        print("❌ Falta la sección [supabase] en secrets.toml.")
        sys.exit(1)

    import types
    st_mock = types.ModuleType("streamlit")
    st_mock.secrets = secrets
    sys.modules["streamlit"] = st_mock


def main():
    _cargar_secrets_a_entorno()
    from core import cloud_storage as cloud

    if not cloud.nube_activa():
        print("❌ Supabase no está configurado en secrets.toml ([supabase] url/key).")
        sys.exit(1)

    ok, msg = cloud.probar_conexion()
    print(f"{'✓' if ok else '❌'} {msg}")
    if not ok:
        sys.exit(1)

    proc_dir = ROOT / "data" / "processed"
    state_dir = ROOT / "data" / "state"

    print("\n📤 Subiendo artefactos del feature store (processed/)...")
    n_ok = n_fail = 0
    for nombre in cloud.ARTEFACTOS_PROCESSED:
        local = proc_dir / nombre
        if not local.exists():
            print(f"   ⚠️  falta {nombre} (omitido)")
            continue
        if cloud.subir_archivo(local, nombre, subcarpeta="processed"):
            print(f"   ✓ {nombre}")
            n_ok += 1
        else:
            print(f"   ❌ {nombre}")
            n_fail += 1

    print("\n📤 Subiendo modelos y proyecciones (state/)...")
    for nombre in cloud.ARTEFACTOS_STATE_MODELO:
        local = state_dir / nombre
        if not local.exists():
            print(f"   ⚠️  falta {nombre} (omitido)")
            continue
        if cloud.subir_archivo(local, nombre, subcarpeta="state"):
            print(f"   ✓ {nombre}")
            n_ok += 1
        else:
            print(f"   ❌ {nombre}")
            n_fail += 1

    print(f"\n{'✓' if n_fail == 0 else '⚠️'} Subida completa: {n_ok} OK, {n_fail} fallos.")
    print("La app en la nube tomará estos archivos al sincronizar "
          "(panel ☁️ en el Home → 'Re-sincronizar').")


if __name__ == "__main__":
    main()
