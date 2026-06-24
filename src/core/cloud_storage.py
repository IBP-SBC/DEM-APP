"""
Persistencia en la nube con Supabase Storage (API REST) — sbc_demanda.

Por qué:
  El filesystem de Streamlit Community Cloud es EFÍMERO: se borra en cada
  reinicio. Para que el estado editable (overrides, novedades, correcciones)
  y los artefactos calculados en escritorio (proyecciones, modelos) sobrevivan,
  se guardan en Supabase Storage y se HIDRATAN al iniciar la sesión.

Diseño:
  - Totalmente DEFENSIVO: si no hay secrets de Supabase configurados, todas las
    funciones degradan a no-op y la app sigue funcionando 100% en local.
  - Usa la API REST de Storage con `requests` (sin SDK pesado).
  - Acepta las secret keys nuevas de Supabase (formato sb_secret_...) en los
    headers `apikey` y `Authorization: Bearer`.
  - Normaliza la URL: si se pega con /rest/v1/ al final, se reduce a la base
    https://PROYECTO.supabase.co (esto nos costó un bug en ventas).

Config (st.secrets['supabase']): url, key, bucket. Prefijo fijo: demanda/.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, List

import requests

# Bucket DEDICADO a esta app (sbc-demanda). Al ser dedicado, no usamos un
# prefijo de subcarpeta de app: guardamos directo en processed/ y state/.
PREFIJO_APP = ""
_TIMEOUT = 30


# =========================================================================
# CONFIG (defensiva — nunca lanza si falta algo)
# =========================================================================
def _get_secrets() -> Optional[dict]:
    """Lee st.secrets['supabase'] si existe. None si no hay nada configurado."""
    try:
        import streamlit as st
        if "supabase" in st.secrets:
            s = st.secrets["supabase"]
            return {
                "url": str(s.get("url", "")).strip(),
                "key": str(s.get("key", "")).strip(),
                "bucket": str(s.get("bucket", "sbc-demanda")).strip(),
            }
    except Exception:
        pass
    return None


def normalizar_url(url: str) -> str:
    """Reduce la URL a la base https://PROYECTO.supabase.co.

    Acepta que el usuario pegue la URL con /rest/v1/ al final (lo que nos
    costó un bug en ventas) y la limpia.
    """
    u = (url or "").strip().rstrip("/")
    # Quitar sufijos REST típicos
    for suf in ["/rest/v1", "/storage/v1", "/auth/v1"]:
        if u.endswith(suf):
            u = u[: -len(suf)]
    # Si quedó algo con /rest/v1/ en medio, cortar ahí
    if "/rest/v1" in u:
        u = u.split("/rest/v1")[0]
    return u.rstrip("/")


def nube_activa() -> bool:
    """True si hay credenciales de Supabase configuradas y completas."""
    s = _get_secrets()
    return bool(s and s["url"] and s["key"])


def _headers(content_type: Optional[str] = None) -> dict:
    s = _get_secrets() or {}
    key = s.get("key", "")
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if content_type:
        h["Content-Type"] = content_type
    return h


def _base_url() -> str:
    s = _get_secrets() or {}
    return normalizar_url(s.get("url", ""))


def _bucket() -> str:
    s = _get_secrets() or {}
    return s.get("bucket", "sbc-demanda")


def _ruta_remota(nombre: str, subcarpeta: str = "state") -> str:
    """Construye la ruta dentro del bucket. Con bucket dedicado (PREFIJO_APP
    vacío): {subcarpeta}/{nombre}. Si hubiera prefijo: {prefijo}/{subcarpeta}/{nombre}."""
    nombre = nombre.lstrip("/")
    partes = [p for p in [PREFIJO_APP, subcarpeta] if p]
    return "/".join(partes + [nombre]) if partes else nombre


# =========================================================================
# OPERACIONES
# =========================================================================
def subir_archivo(local_path, nombre: Optional[str] = None,
                  subcarpeta: str = "state") -> bool:
    """Sube un archivo local a Supabase Storage (upsert). False si falla o
    si la nube no está activa."""
    if not nube_activa():
        return False
    local_path = Path(local_path)
    if not local_path.exists():
        return False
    nombre = nombre or local_path.name
    ruta = _ruta_remota(nombre, subcarpeta)
    url = f"{_base_url()}/storage/v1/object/{_bucket()}/{ruta}"
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        h = _headers("application/octet-stream")
        h["x-upsert"] = "true"
        r = requests.post(url, headers=h, data=data, timeout=_TIMEOUT)
        if r.status_code in (200, 201):
            return True
        # Algunos despliegues requieren PUT para upsert
        r2 = requests.put(url, headers=h, data=data, timeout=_TIMEOUT)
        return r2.status_code in (200, 201)
    except Exception:
        return False


def subir_bytes(data: bytes, nombre: str, subcarpeta: str = "state") -> bool:
    """Sube bytes directamente (sin archivo en disco)."""
    if not nube_activa():
        return False
    ruta = _ruta_remota(nombre, subcarpeta)
    url = f"{_base_url()}/storage/v1/object/{_bucket()}/{ruta}"
    try:
        h = _headers("application/octet-stream")
        h["x-upsert"] = "true"
        r = requests.post(url, headers=h, data=data, timeout=_TIMEOUT)
        if r.status_code in (200, 201):
            return True
        r2 = requests.put(url, headers=h, data=data, timeout=_TIMEOUT)
        return r2.status_code in (200, 201)
    except Exception:
        return False


def bajar_archivo(nombre: str, local_path, subcarpeta: str = "state") -> bool:
    """Descarga un archivo de Supabase a una ruta local. False si no existe
    o falla."""
    if not nube_activa():
        return False
    ruta = _ruta_remota(nombre, subcarpeta)
    url = f"{_base_url()}/storage/v1/object/{_bucket()}/{ruta}"
    try:
        r = requests.get(url, headers=_headers(), timeout=_TIMEOUT)
        if r.status_code == 200 and r.content:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False


def existe_remoto(nombre: str, subcarpeta: str = "state") -> bool:
    """True si el archivo existe en la nube."""
    archivos = listar(subcarpeta)
    return nombre in archivos


def listar(subcarpeta: str = "state") -> List[str]:
    """Lista nombres de archivos en demanda/{subcarpeta}/. Lista vacía si falla."""
    if not nube_activa():
        return []
    prefijo = "/".join([p for p in [PREFIJO_APP, subcarpeta] if p])
    url = f"{_base_url()}/storage/v1/object/list/{_bucket()}"
    try:
        body = {"prefix": (prefijo + "/") if prefijo else "", "limit": 200,
                "offset": 0, "sortBy": {"column": "name", "order": "asc"}}
        r = requests.post(url, headers=_headers("application/json"),
                          json=body, timeout=_TIMEOUT)
        if r.status_code == 200:
            return [item["name"] for item in r.json()
                    if item.get("name") and not item["name"].endswith("/")]
        return []
    except Exception:
        return []


def probar_conexion() -> tuple[bool, str]:
    """Hace una llamada REAL a Supabase y devuelve (ok, mensaje con el error
    exacto si falla). Para el panel de diagnóstico del Home."""
    if not nube_activa():
        return False, "No hay credenciales de Supabase en st.secrets['supabase']."
    url = f"{_base_url()}/storage/v1/object/list/{_bucket()}"
    try:
        body = {"prefix": (f"{PREFIJO_APP}/" if PREFIJO_APP else ""), "limit": 1}
        r = requests.post(url, headers=_headers("application/json"),
                          json=body, timeout=_TIMEOUT)
        if r.status_code == 200:
            return True, f"Conexión OK · bucket '{_bucket()}' · base {_base_url()}"
        return False, (f"HTTP {r.status_code} al listar el bucket "
                       f"'{_bucket()}'. Respuesta: {r.text[:200]}")
    except requests.exceptions.Timeout:
        return False, f"Timeout ({_TIMEOUT}s) conectando a {_base_url()}."
    except Exception as e:
        return False, f"Error de conexión: {type(e).__name__}: {str(e)[:200]}"


# =========================================================================
# HIDRATACIÓN AL INICIAR
# =========================================================================
# Artefactos que la nube necesita (productos del cálculo de escritorio).
# subcarpeta remota → lista de (nombre_archivo, ruta_local_relativa)
ARTEFACTOS_PROCESSED = [
    "feature_isbn.parquet", "feature_cliente.parquet",
    "ventas_mensual_isbn.parquet", "ventas_mensual_cliente_clase.parquet",
    "mix_cliente_isbn.parquet", "feature_canal.parquet",
    "eventos_eclesiasticos.parquet",
]
ARTEFACTOS_STATE_MODELO = [
    "proyecciones_prophet.parquet", "modelo_hedonico.joblib",
    "metricas_hedonico.json", "importancias_hedonico.csv",
    "perfiles_estacionales.pkl", "proyecciones_cliente.parquet",
    "perfiles_cliente.parquet",
]
# Estado editable (se baja al iniciar y se sube al modificar)
ESTADO_EDITABLE = [
    "overrides_proyeccion.json", "novedades_aprobadas.json",
    "correcciones_taco_mp.json", "overrides_cliente.json",
    "ingesta_meta.json", "ventas_ejecutadas_siesa.parquet",
    "novedades_catalogo_anio.parquet",
]


def hidratar_desde_nube(root: Path, solo_si_falta: bool = True) -> dict:
    """Baja desde Supabase los archivos que falten en disco al iniciar sesión.

    Args:
        root: raíz del proyecto (donde está data/).
        solo_si_falta: si True, solo baja lo que no exista localmente
            (los artefactos pesados se bajan una vez por sesión). El estado
            editable se baja siempre para reflejar cambios de otra sesión.

    Returns:
        dict con conteos: {'processed': n, 'modelos': n, 'estado': n}.
    """
    if not nube_activa():
        return {"processed": 0, "modelos": 0, "estado": 0, "activa": False}

    proc_dir = root / "data" / "processed"
    state_dir = root / "data" / "state"
    res = {"processed": 0, "modelos": 0, "estado": 0, "activa": True}

    for nombre in ARTEFACTOS_PROCESSED:
        destino = proc_dir / nombre
        if solo_si_falta and destino.exists():
            continue
        if bajar_archivo(nombre, destino, subcarpeta="processed"):
            res["processed"] += 1

    for nombre in ARTEFACTOS_STATE_MODELO:
        destino = state_dir / nombre
        if solo_si_falta and destino.exists():
            continue
        if bajar_archivo(nombre, destino, subcarpeta="state"):
            res["modelos"] += 1

    # El estado editable se baja SIEMPRE (refleja cambios de otras sesiones)
    for nombre in ESTADO_EDITABLE:
        destino = state_dir / nombre
        if bajar_archivo(nombre, destino, subcarpeta="state"):
            res["estado"] += 1

    return res


def subir_estado(root: Path, nombre: str) -> bool:
    """Sube un archivo de estado editable a la nube (tras modificarlo).
    Detecta si está en processed o state por la lista de artefactos."""
    if not nube_activa():
        return False
    state_dir = root / "data" / "state"
    local = state_dir / nombre
    if not local.exists():
        return False
    return subir_archivo(local, nombre, subcarpeta="state")
