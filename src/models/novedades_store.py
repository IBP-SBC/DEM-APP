"""
Sistema de Novedades Aprobadas
================================
Gestiona las novedades simuladas que el equipo de Publicaciones aprueba
formalmente para incluir en el pronóstico operativo.

Persistencia: data/state/novedades_aprobadas.json

Estructura de una novedad aprobada:
{
    "id": "nov_20260511_134522",        # único
    "nombre": "RVR060 Nueva Mujer Coral",
    "concepto_id": "C001",              # agrupa cubiertas del mismo TACO
    "fecha_aprobacion": "2026-05-11T13:45:22",
    "mes_lanzamiento": "2027-01",       # YYYY-MM
    "taco_destino": "CLARIDAD 060 MUJER VIRTUOSA",  # TACO MP que producirá
    "features": {...},                  # inputs del simulador
    "demanda_anual_estimada": 2589,
    "demanda_anual_p10": 1050,
    "demanda_anual_p90": 6380,
    "ciclo_vida_meses": 30,
    "curva_mensual": [(YYYY-MM, prediccion, p10, p90), ...],
    "estado": "aprobado"
}

Capacidad operativa (de dictionaries.CAPACIDAD_NOVEDADES):
- 5 conceptos nuevos por año (un concepto = 1 TACO)
- 3 cubiertas por concepto (mismo TACO, distintos colores)
- 15 SKUs nuevos máximos por año
"""
from __future__ import annotations
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.dictionaries import CAPACIDAD_NOVEDADES

DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)
NOVEDADES_PATH = DATA_STATE / "novedades_aprobadas.json"


# =========================================================================
# CRUD
# =========================================================================
def cargar() -> list[dict]:
    """Carga la lista de novedades aprobadas desde disco."""
    if not NOVEDADES_PATH.exists():
        return []
    try:
        with open(NOVEDADES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def guardar(novedades: list[dict]) -> None:
    """Persiste la lista de novedades a disco (y a la nube si está activa)."""
    with open(NOVEDADES_PATH, "w", encoding="utf-8") as f:
        json.dump(novedades, f, indent=2, ensure_ascii=False, default=str)
    try:
        from core import cloud_storage as _cloud
        if _cloud.nube_activa():
            _cloud.subir_archivo(NOVEDADES_PATH, "novedades_aprobadas.json", subcarpeta="state")
    except Exception:
        pass


def agregar(novedad: dict) -> dict:
    """Agrega una novedad. Le asigna ID único si no tiene."""
    novedades = cargar()
    if "id" not in novedad:
        # ID con timestamp + sufijo aleatorio para garantizar unicidad
        # aún en ejecuciones en ráfaga (microsegundos pueden colisionar)
        sufijo = uuid.uuid4().hex[:6]
        novedad["id"] = f"nov_{datetime.now():%Y%m%d_%H%M%S}_{sufijo}"
    if "fecha_aprobacion" not in novedad:
        novedad["fecha_aprobacion"] = datetime.now().isoformat()
    if "estado" not in novedad:
        novedad["estado"] = "aprobado"
    # Asegurar campos de tracking
    if "origen" not in novedad:
        novedad["origen"] = "manual"  # "manual" | "sugerencia_automatica"
    if "fuera_capacidad" not in novedad:
        novedad["fuera_capacidad"] = False
    novedades.append(novedad)
    guardar(novedades)
    return novedad


def eliminar(novedad_id: str) -> bool:
    """Elimina una novedad por ID. Retorna True si la encontró."""
    novedades = cargar()
    n_orig = len(novedades)
    novedades = [n for n in novedades if n.get("id") != novedad_id]
    if len(novedades) < n_orig:
        guardar(novedades)
        return True
    return False


def eliminar_multiple(novedad_ids: list[str]) -> int:
    """Elimina varias novedades por ID. Retorna cuántas eliminó."""
    novedades = cargar()
    n_orig = len(novedades)
    ids_set = set(novedad_ids)
    novedades = [n for n in novedades if n.get("id") not in ids_set]
    eliminadas = n_orig - len(novedades)
    if eliminadas > 0:
        guardar(novedades)
    return eliminadas


def set_override_escala(novedad_id: str, escala: float) -> bool:
    """Establece un multiplicador de escala (override) sobre la curva de una
    novedad (v3.12). escala=1.0 lo elimina. Retorna True si encontró la novedad.

    El override se aplica al vuelo en las funciones de aporte (anual, mensual,
    desglosado), así que ajusta TODO lo que muestre la novedad sin reescribir
    su curva original."""
    novedades = cargar()
    encontrada = False
    for n in novedades:
        if n.get("id") == novedad_id:
            if abs(escala - 1.0) < 1e-9:
                n.pop("override_escala", None)
            else:
                n["override_escala"] = float(escala)
            encontrada = True
            break
    if encontrada:
        guardar(novedades)
    return encontrada


def _escala_nov(nov: dict) -> float:
    """Devuelve el multiplicador de override de una novedad (1.0 si no tiene)."""
    try:
        return float(nov.get("override_escala", 1.0))
    except (TypeError, ValueError):
        return 1.0


def filtrar_por_origen(origen: str) -> list[dict]:
    """Devuelve las novedades aprobadas del origen indicado.
    origen: 'manual' | 'sugerencia_automatica' | 'todos'"""
    novedades = cargar()
    if origen == "todos":
        return novedades
    return [n for n in novedades if n.get("origen", "manual") == origen]


def ids_aprobados_sugerencias() -> set[str]:
    """Devuelve los concepto_id + taco_destino + nombre de las sugerencias
    automáticas ya aprobadas. Útil para excluirlas del listado de propuestas."""
    aprobadas_sug = filtrar_por_origen("sugerencia_automatica")
    huellas = set()
    for n in aprobadas_sug:
        # Huella única: nombre + año del mes_lanzamiento
        nombre = n.get("nombre", "")
        ml = n.get("mes_lanzamiento", "")
        anio = ml[:4] if ml else ""
        huellas.add(f"{nombre}|{anio}")
    return huellas


def actualizar_estado(novedad_id: str, nuevo_estado: str) -> bool:
    """Actualiza el estado de una novedad (aprobado/rechazado/ejecutado)."""
    novedades = cargar()
    for n in novedades:
        if n.get("id") == novedad_id:
            n["estado"] = nuevo_estado
            n["fecha_ultima_actualizacion"] = datetime.now().isoformat()
            guardar(novedades)
            return True
    return False


# =========================================================================
# CAPACIDAD
# =========================================================================
def calcular_uso_capacidad(anio: int, solo_aprobadas: bool = True) -> dict:
    """
    Calcula el uso de capacidad de un año específico.

    Distingue:
    - Conceptos nuevos: novedades con tipo_taco='nuevo' (consumen cupo de
      los 5 conceptos/año porque requieren TACO MP nuevo)
    - Cubiertas totales: todas las novedades del año (incluye variantes
      a TACOs existentes), techo 15/año.

    Returns:
        dict con conceptos_usados, cubiertas_usadas, conceptos_max,
              cubiertas_max, pct_usado, conceptos_libres, cubiertas_libres,
              variantes_existentes (cubiertas a TACOs existentes)
    """
    novedades = cargar()
    if solo_aprobadas:
        novedades = [n for n in novedades if n.get("estado") == "aprobado"]
    novedades_anio = [
        n for n in novedades
        if n.get("mes_lanzamiento", "").startswith(str(anio))
    ]

    # Distinguir por tipo de TACO destino
    # Para retro-compatibilidad: si no tiene "tipo_taco" se asume "nuevo"
    nuevos = [n for n in novedades_anio
              if n.get("tipo_taco", "nuevo") == "nuevo"]
    existentes = [n for n in novedades_anio
                  if n.get("tipo_taco") == "existente"]

    # Agrupar conceptos nuevos por concepto_id (cubiertas del mismo TACO nuevo)
    conceptos_unicos = {
        n.get("concepto_id", n["id"])
        for n in nuevos
    }
    conceptos_usados = len(conceptos_unicos)
    cubiertas_usadas = len(novedades_anio)

    return {
        "anio": anio,
        "conceptos_usados": conceptos_usados,
        "conceptos_max": CAPACIDAD_NOVEDADES["conceptos_por_ano"],
        "conceptos_libres": max(0, CAPACIDAD_NOVEDADES["conceptos_por_ano"] - conceptos_usados),
        "cubiertas_usadas": cubiertas_usadas,
        "cubiertas_max": CAPACIDAD_NOVEDADES["skus_por_ano"],
        "cubiertas_libres": max(0, CAPACIDAD_NOVEDADES["skus_por_ano"] - cubiertas_usadas),
        "pct_usado": round(cubiertas_usadas / CAPACIDAD_NOVEDADES["skus_por_ano"] * 100, 1),
        "variantes_existentes": len(existentes),
        "cubiertas_de_conceptos_nuevos": len(nuevos),
    }


# =========================================================================
# GENERACIÓN DE CURVA MENSUAL
# =========================================================================
def generar_curva_mensual(
    mes_lanzamiento: str,
    demanda_anual: float,
    perfil_estacional: np.ndarray,
    ciclo_vida_meses: int = 30,
    p10: Optional[float] = None,
    p90: Optional[float] = None,
    decay_anual: float = 0.15,
) -> pd.DataFrame:
    """
    Genera la curva mensual proyectada para una novedad desde su mes
    de lanzamiento hasta el fin del ciclo de vida.

    Lógica:
    - Año 1: demanda_anual × perfil_estacional (12 meses)
    - Año 2: demanda_anual × perfil_estacional × (1 - decay_anual)
    - Año 3+: demanda_anual × perfil_estacional × (1 - 2*decay_anual)
    - Después del ciclo de vida: 0

    Args:
        mes_lanzamiento: "YYYY-MM"
        demanda_anual: estimación central anual
        perfil_estacional: array de 12 fracciones que suman 1.0
        ciclo_vida_meses: default 30
        p10, p90: bandas opcionales
        decay_anual: factor de decay año a año (default 15%)

    Returns:
        DataFrame con columnas: ds, prediccion, p10, p90, mes_relativo
    """
    mes_inicio = pd.Timestamp(f"{mes_lanzamiento}-01")
    fechas = pd.date_range(mes_inicio, periods=ciclo_vida_meses, freq="MS")

    perfil = np.asarray(perfil_estacional, dtype=float)
    perfil = perfil / perfil.sum()  # normalizar

    pred = []
    pred_p10 = []
    pred_p90 = []
    for i, fecha in enumerate(fechas):
        # Anio relativo desde lanzamiento (0=año 1)
        anio_rel = i // 12
        # Mes calendario (0-11)
        mes_cal = fecha.month - 1
        # Factor de decay por año
        factor = max(0, 1 - decay_anual * anio_rel)
        # Valor mensual
        valor = demanda_anual * perfil[mes_cal] * factor
        pred.append(valor)
        if p10 is not None:
            pred_p10.append(p10 * perfil[mes_cal] * factor)
        if p90 is not None:
            pred_p90.append(p90 * perfil[mes_cal] * factor)

    df = pd.DataFrame({
        "ds": fechas,
        "prediccion": pred,
        "mes_relativo": list(range(1, ciclo_vida_meses + 1)),
    })
    if p10 is not None:
        df["p10"] = pred_p10
    if p90 is not None:
        df["p90"] = pred_p90
    return df


# =========================================================================
# AGREGACIÓN PARA INCORPORAR AL PRONÓSTICO
# =========================================================================
def obtener_aporte_anual_novedades(
    novedades: Optional[list[dict]] = None,
    solo_aprobadas: bool = True,
) -> dict[int, float]:
    """
    Suma el aporte total proyectado de las novedades por año.
    Útil para integrar en el gap-vs-metas del Editor de pronóstico.
    """
    if novedades is None:
        novedades = cargar()
    if solo_aprobadas:
        novedades = [n for n in novedades if n.get("estado") == "aprobado"]

    aporte = {}
    for nov in novedades:
        esc = _escala_nov(nov)
        curva = nov.get("curva_mensual", [])
        for entrada in curva:
            if isinstance(entrada, dict):
                ds = entrada.get("ds")
                pred = entrada.get("prediccion", 0)
            elif isinstance(entrada, (list, tuple)) and len(entrada) >= 2:
                ds, pred = entrada[0], entrada[1]
            else:
                continue
            try:
                anio = pd.Timestamp(ds).year
                aporte[anio] = aporte.get(anio, 0) + float(pred) * esc
            except (ValueError, TypeError):
                continue
    return aporte


def obtener_aporte_anual_desglosado(
    solo_aprobadas: bool = True,
) -> dict:
    """
    Devuelve el aporte anual desglosado por origen y dentro/fuera capacidad:
    {
        2027: {
            "manual_dentro": 1000,
            "manual_fuera": 0,
            "sugerencia_dentro": 5000,
            "sugerencia_fuera": 3000,
        },
        ...
    }
    """
    novedades = cargar()
    if solo_aprobadas:
        novedades = [n for n in novedades if n.get("estado") == "aprobado"]

    aporte = {}
    for nov in novedades:
        origen = nov.get("origen", "manual")
        fuera = nov.get("fuera_capacidad", False)
        esc = _escala_nov(nov)
        # Determinar la subcategoría
        prefijo = "sugerencia" if origen == "sugerencia_automatica" else "manual"
        sufijo = "fuera" if fuera else "dentro"
        clave = f"{prefijo}_{sufijo}"

        for entrada in nov.get("curva_mensual", []):
            if isinstance(entrada, dict):
                ds = entrada.get("ds")
                pred = entrada.get("prediccion", 0)
            else:
                continue
            try:
                anio = pd.Timestamp(ds).year
                if anio not in aporte:
                    aporte[anio] = {
                        "manual_dentro": 0.0,
                        "manual_fuera": 0.0,
                        "sugerencia_dentro": 0.0,
                        "sugerencia_fuera": 0.0,
                    }
                aporte[anio][clave] += float(pred) * esc
            except (ValueError, TypeError):
                continue
    # Agregar campo 'total' por año (suma de los 4 buckets)
    for anio in aporte:
        aporte[anio]["total"] = sum(v for k, v in aporte[anio].items() if k != "total")
    return aporte


def obtener_aporte_mensual_novedades(
    novedades: Optional[list[dict]] = None,
    solo_aprobadas: bool = True,
) -> pd.DataFrame:
    """
    Devuelve la suma mensual proyectada de todas las novedades aprobadas.
    Output: DataFrame con columnas ds, prediccion, p10, p90.
    """
    if novedades is None:
        novedades = cargar()
    if solo_aprobadas:
        novedades = [n for n in novedades if n.get("estado") == "aprobado"]

    rows = []
    for nov in novedades:
        nov_id = nov.get("id")
        nombre = nov.get("nombre", "")
        esc = _escala_nov(nov)
        for entrada in nov.get("curva_mensual", []):
            if isinstance(entrada, dict):
                rows.append({
                    "novedad_id": nov_id,
                    "nombre": nombre,
                    "ds": pd.Timestamp(entrada.get("ds")),
                    "prediccion": float(entrada.get("prediccion", 0)) * esc,
                    "p10": float(entrada.get("p10", 0)) * esc,
                    "p90": float(entrada.get("p90", 0)) * esc,
                })

    if not rows:
        return pd.DataFrame(columns=["ds", "prediccion", "p10", "p90"])

    df = pd.DataFrame(rows)
    df_agg = df.groupby("ds").agg(
        prediccion=("prediccion", "sum"),
        p10=("p10", "sum"),
        p90=("p90", "sum"),
        n_novedades=("novedad_id", "nunique"),
    ).reset_index()
    return df_agg


# =========================================================================
# TEST EN CLI
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEST NOVEDADES STORE")
    print("=" * 70)

    # Limpiar store de prueba
    if NOVEDADES_PATH.exists():
        NOVEDADES_PATH.unlink()

    # Agregar una novedad
    nov_test = {
        "nombre": "RVR060 Test Femenino Coral",
        "concepto_id": "C001",
        "mes_lanzamiento": "2027-01",
        "taco_destino": "CLARIDAD 060 MUJER VIRTUOSA",
        "features": {"precio_promedio": 85000, "familia_genero": "femenino"},
        "demanda_anual_estimada": 2589,
        "demanda_anual_p10": 1050,
        "demanda_anual_p90": 6380,
        "ciclo_vida_meses": 30,
        "curva_mensual": [
            {"ds": "2027-01-01", "prediccion": 250, "p10": 100, "p90": 600},
            {"ds": "2027-02-01", "prediccion": 215, "p10": 86, "p90": 530},
        ],
    }
    nov = agregar(nov_test)
    print(f"\n✓ Agregada novedad ID: {nov['id']}")

    # Verificar capacidad
    cap = calcular_uso_capacidad(2027)
    print(f"\n✓ Capacidad 2027:")
    for k, v in cap.items():
        print(f"    {k}: {v}")

    # Aporte anual
    aporte = obtener_aporte_anual_novedades()
    print(f"\n✓ Aporte anual de novedades:")
    for anio, suma in sorted(aporte.items()):
        print(f"    {anio}: {suma:,.0f}")

    # Limpiar
    eliminar(nov["id"])
    print(f"\n✓ Test limpiado")
