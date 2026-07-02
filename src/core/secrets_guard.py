"""
Guardián de secrets.toml — evita que la app truene por un secrets mal formado.

Problema recurrente en Windows/Mac: si el usuario edita .streamlit/secrets.toml
con TextEdit/Notepad y el editor mete comillas tipográficas (" " ' ') o un BOM,
el parser TOML de Streamlit falla al arrancar con un error críptico.

Este guardián se llama AL INICIO del Home, ANTES de tocar st.secrets: lee el
archivo, y si detecta comillas curvas / BOM, reescribe una versión limpia en
disco (solo si hace falta). Es seguro: convertir comillas curvas a rectas es
siempre correcto en TOML.
"""
from __future__ import annotations
from pathlib import Path


# Comillas tipográficas → rectas, y espacios no separables → normales
_REEMPLAZOS = {
    "\u201c": '"', "\u201d": '"',   # " "
    "\u2018": "'", "\u2019": "'",   # ' '
    "\u00a0": " ",                   # espacio no separable
}


def sanear_secrets(root: Path) -> bool:
    """Si .streamlit/secrets.toml tiene comillas curvas o BOM, lo reescribe
    limpio. Devuelve True si hizo una corrección. Defensivo: nunca lanza."""
    try:
        path = root / ".streamlit" / "secrets.toml"
        if not path.exists():
            return False
        # utf-8-sig quita el BOM si está
        original = path.read_text(encoding="utf-8-sig")
        limpio = original
        for malo, bueno in _REEMPLAZOS.items():
            limpio = limpio.replace(malo, bueno)
        if limpio != original:
            path.write_text(limpio, encoding="utf-8")
            return True
        return False
    except Exception:
        return False
