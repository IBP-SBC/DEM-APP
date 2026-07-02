"""
Generador automático de sugerencias de novedades
==================================================
Algoritmo greedy con restricciones de portafolio para cerrar el gap a
las metas conservadoras 2027-2030, respetando capacidad operativa
(5 conceptos nuevos + 15 cubiertas/año) y aplicando criterios de
diversificación basados en el estudio PATMOS.

Decisiones de diseño (consensuadas con Alberto):

1. MISIONERAS son UNIVERSALES, no masculinas: aunque históricamente
   están dominadas por color azul, las iglesias las compran para
   evangelizar a TODOS los segmentos, no solo hombres. En el sugerido
   se contabilizan como "neutro/universal" para los pisos de balance.

2. Pisos de balance (sobre 15 cubiertas/año):
   - Mínimo 1 misionera (variante o concepto nuevo en categoría económica)
   - Mínimo 30% femenino (vs histórico 20%, empuje vía PATMOS)
   - Mínimo 40% masculino (mantener histórico 50%)
   - Mínimo 1 juvenil cada 2 años (PATMOS S6 sub-atendido)

3. Prioridad: variantes a TACOs activos > conceptos nuevos.
   Las variantes consumen solo cupo de cubiertas, no conceptos.

4. Conceptos nuevos: auto-completar 3 cubiertas (mismo TACO, colores
   variados según género).

5. Mes de lanzamiento óptimo por género:
   - Femenino → octubre (mes Biblia)
   - Masculino → marzo (Semana Santa)
   - Juvenil → agosto (escolar)
   - Neutro/universal → junio o enero

6. Dos sets de sugerencias:
   - DENTRO de capacidad actual (15 SKUs/año)
   - FUERA de capacidad (etiqueta "incremento_capacidad") para que
     Alberto pueda mostrar a junta cuánto fortalecimiento requiere.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import pickle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.hedonic_model import predecir_novedad
from models.seasonality import perfil_para_producto, aplicar_perfil
from models import novedades_store
from utils.dictionaries import (
    CAPACIDAD_NOVEDADES,
    DESCUENTO_DEFAULT_POR_CATEGORIA,
    categoria_por_precio,
)

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"


# =========================================================================
# CONSTANTES — METAS Y BALANCE
# =========================================================================
from core.config import METAS_BIBLIAS  # objetivos estrategicos (fuente unica)

# Pisos de balance por año (sobre el total de cubiertas del año)
PISOS_BALANCE = {
    "min_misioneras": 1,        # al menos 1 económica/misionera por año
    "min_femenino_pct": 0.30,   # 30% del total año
    "min_masculino_pct": 0.40,  # 40% del total año
    "min_juvenil_cada_2_anios": 1,  # al menos 1 juvenil cada 2 años
}

# Mes óptimo de lanzamiento por género (aprovecha picos estacionales)
MES_OPTIMO_POR_GENERO = {
    "femenino": 10,   # octubre - mes biblia
    "masculino": 3,   # marzo - semana santa
    "juvenil": 8,     # agosto - vuelta a clases
    "neutro": 6,      # junio - mid-year
    "universal": 6,   # universal/misioneras
}

# Color sugerido al generar cubiertas de un concepto nuevo
COLORES_POR_GENERO_CONCEPTO = {
    "femenino": ["ROSA", "LILA", "CORAL"],
    "masculino": ["AZUL", "NEGRO", "CAFE"],
    "juvenil": ["VERDE", "NARANJA", "TURQUESA"],
    "neutro": ["DORADO", "BLANCO", "BEIGE"],
    "universal": ["AZUL", "ROJO", "NEGRO"],  # misioneras universales
}


# =========================================================================
# CLASIFICACIÓN UNIVERSAL/MISIONERA
# =========================================================================
def es_taco_misionero(taco_mp: str) -> bool:
    """Identifica si un TACO es MISIONERA (universal evangelística)."""
    if not isinstance(taco_mp, str):
        return False
    t = taco_mp.upper()
    return "MISIONERA" in t


def genero_efectivo(familia_genero: str, taco_mp: str = "") -> str:
    """
    Devuelve el género EFECTIVO para fines del balance de portafolio.
    Las misioneras son siempre 'universal' aunque su color sea masculino,
    porque las iglesias las compran para evangelizar a todos los
    segmentos del Clúster 4 PATMOS.
    """
    if es_taco_misionero(taco_mp):
        return "universal"
    return familia_genero


# =========================================================================
# ESTRUCTURA DE SUGERENCIA
# =========================================================================
@dataclass
class Sugerencia:
    id_sugerencia: str
    año: int
    mes_lanzamiento: str  # "YYYY-MM"
    tipo_taco: str        # "existente" | "nuevo"
    taco_destino: str
    concepto_id: str
    nombre_sugerido: str
    # Features para predicción
    familia_genero: str
    familia_genero_efectivo: str  # universal si es misionera
    mercado_principal: str
    version: str
    tamano_familia: str
    tipo_letra: str
    categoria_precio: str
    precio_promedio: float
    descuento_promedio: float
    tiene_cierre: bool = False
    tiene_indice: bool = False
    es_imitacion_cuero: bool = True
    tiene_canto_dorado: bool = False
    # Predicciones
    demanda_anual_estimada: float = 0.0
    demanda_anual_p10: float = 0.0
    demanda_anual_p90: float = 0.0
    # Meta-info
    prioridad: str = "media"   # "alta" | "media" | "baja"
    dentro_capacidad: bool = True
    justificacion: str = ""
    ciclo_vida_meses: int = 30
    curva_mensual: list = field(default_factory=list)

    def features_dict(self) -> dict:
        """Retorna dict de features para predecir_novedad."""
        return {
            "precio_promedio": self.precio_promedio,
            "descuento_promedio": self.descuento_promedio,
            "familia_genero": self.familia_genero,
            "tamano_familia": self.tamano_familia,
            "tipo_letra": self.tipo_letra,
            "mercado_principal": self.mercado_principal,
            "version": self.version,
            "tiene_cierre": self.tiene_cierre,
            "tiene_indice": self.tiene_indice,
            "es_imitacion_cuero": self.es_imitacion_cuero,
            "tiene_canto_dorado": self.tiene_canto_dorado,
        }


# =========================================================================
# CONSTRUCCIÓN DE CANDIDATOS
# =========================================================================
def construir_pool_candidatos(
    feature_isbn: pd.DataFrame,
    bundle_hedonico: dict,
    año: int,
    n_variantes_por_taco: int = 2,
) -> list[Sugerencia]:
    """
    Construye pool grande de candidatos: variantes a TACOs top + conceptos
    nuevos por combinaciones rentables.
    """
    candidatos: list[Sugerencia] = []
    biblias = feature_isbn[feature_isbn["clase"] == "BIBLIAS"]
    activos = biblias[biblias["estado"] == "ACTIVO"]

    # ----- 1. VARIANTES A TOP TACOs ACTIVOS -----
    # Top 15 TACOs por demanda anual madura
    tacos_top = (
        activos.groupby("taco_mp").agg(
            n_isbns=("isbn", "count"),
            demanda_total=("demanda_anual_madura", "sum"),
            precio_mediano=("precio_promedio", "median"),
            descuento_promedio=("descuento_promedio", "mean"),
        ).reset_index()
    )
    # Excluir POSIBLE IMPORTADO
    tacos_top = tacos_top[
        ~tacos_top["taco_mp"].astype(str).str.upper().str.contains(
            "POSIBLE IMPORTADO", na=False
        )
    ]
    tacos_top = tacos_top.sort_values("demanda_total", ascending=False).head(15)

    for _, taco_row in tacos_top.iterrows():
        taco_mp = taco_row["taco_mp"]
        if pd.isna(taco_mp) or not taco_mp:
            continue
        # Características modales del TACO
        sub = activos[activos["taco_mp"] == taco_mp]
        if len(sub) == 0:
            continue
        # Género dominante del TACO
        gen_modo = sub["familia_genero"].mode()
        gen_principal = gen_modo.iloc[0] if len(gen_modo) else "neutro"
        # Si es misionero, marcarlo como universal
        gen_efectivo = genero_efectivo(gen_principal, taco_mp)
        # Mercado dominante
        merc_modo = sub["mercado_principal"].mode()
        merc_principal = merc_modo.iloc[0] if len(merc_modo) else "ambos"
        # Versión
        ver_modo = sub["version"].dropna().mode()
        version = ver_modo.iloc[0] if len(ver_modo) else "RVR"
        # Tamaño
        tam_modo = sub["tamano_familia"].dropna().mode()
        tamano = tam_modo.iloc[0] if len(tam_modo) else "letra_grande"
        # Precio y descuento
        precio = float(taco_row["precio_mediano"])
        if precio < 5000 or pd.isna(precio):
            precio = 50000.0
        cat = categoria_por_precio(precio)
        descuento = float(taco_row["descuento_promedio"]) if pd.notna(
            taco_row["descuento_promedio"]) else DESCUENTO_DEFAULT_POR_CATEGORIA[cat]

        # Generar N variantes con colores distintos
        colores = COLORES_POR_GENERO_CONCEPTO.get(
            gen_efectivo if gen_efectivo == "universal" else gen_principal,
            COLORES_POR_GENERO_CONCEPTO["neutro"]
        )[:n_variantes_por_taco]

        for i, color in enumerate(colores):
            sug = Sugerencia(
                id_sugerencia=f"SUG_VAR_{año}_{taco_mp[:20]}_{color}".replace(" ", "_"),
                año=año,
                mes_lanzamiento=f"{año}-{MES_OPTIMO_POR_GENERO.get(gen_efectivo, 6):02d}",
                tipo_taco="existente",
                taco_destino=taco_mp,
                concepto_id=taco_mp,  # las variantes comparten concepto_id=taco
                nombre_sugerido=f"{taco_mp} {color.title()} ({año})",
                familia_genero=gen_principal,
                familia_genero_efectivo=gen_efectivo,
                mercado_principal=merc_principal,
                version=version,
                tamano_familia=tamano,
                tipo_letra="letra_grande",
                categoria_precio=cat,
                precio_promedio=precio,
                descuento_promedio=descuento,
                tiene_cierre=False,
                tiene_indice=False,
                es_imitacion_cuero=True,
                tiene_canto_dorado=False,
                prioridad="alta" if es_taco_misionero(taco_mp) else "media",
                justificacion=(
                    f"Variante {color.lower()} a TACO existente exitoso "
                    f"({taco_row['demanda_total']:,.0f} u/año histórico, "
                    f"{taco_row['n_isbns']} ISBNs en catálogo). "
                    + ("Universal evangelística (misionera)." if es_taco_misionero(taco_mp)
                       else "Bajo riesgo, capacidad de planta probada.")
                ),
            )
            # Predecir demanda
            pred = predecir_novedad(sug.features_dict(), bundle_hedonico)
            sug.demanda_anual_estimada = pred["demanda_anual_estimada"]
            sug.demanda_anual_p10 = pred["intervalo_p10"]
            sug.demanda_anual_p90 = pred["intervalo_p90"]
            candidatos.append(sug)

    # ----- 2. CONCEPTOS NUEVOS POR COMBINACIÓN RENTABLE -----
    # Combinaciones (género, categoria, precio_default, descuento_default)
    # basadas en análisis histórico SBC
    combinaciones_rentables = [
        # (género, mercado, categoría, precio, descuento, descripción)
        ("femenino", "ambos", "semi_fina", 130_000, 38.0, "Mujer Virtuosa premium (Sí a la Familia)"),
        ("femenino", "ambos", "media", 75_000, 33.0, "Mujer comercial estándar"),
        ("femenino", "ambos", "semi_economica", 45_000, 28.0, "Mujer accesible (Sanar Heridas)"),
        ("masculino", "ambos", "economica", 28_000, 22.0, "Masculino misionera RVR (Buenas Nuevas)"),
        ("masculino", "ambos", "semi_fina", 130_000, 38.0, "Hombre estudio profundo"),
        ("masculino", "ambos", "media", 75_000, 33.0, "Hombre comercial estándar"),
        ("juvenil", "ambos", "economica", 28_000, 22.0, "Juvenil económica (Caminata Bíblica S6)"),
        ("juvenil", "ambos", "semi_economica", 45_000, 28.0, "Juvenil media"),
        ("neutro", "ambos", "economica", 25_000, 22.0, "Universal evangelística (Un País en la Maleta)"),
        ("neutro", "ambos", "media", 75_000, 33.0, "Familiar estándar"),
    ]
    for gen, mercado, cat, precio, dto, justif in combinaciones_rentables:
        for color_idx in range(3):  # 3 cubiertas por concepto nuevo
            colores = COLORES_POR_GENERO_CONCEPTO.get(gen, COLORES_POR_GENERO_CONCEPTO["neutro"])
            color = colores[color_idx % len(colores)]
            concepto_label = f"NUEVO_{gen}_{cat}_{año}".upper()
            taco_propuesto = f"CLARIDAD 060 NUEVO {gen.upper()} {cat.upper()}"
            sug = Sugerencia(
                id_sugerencia=f"SUG_NUEVO_{año}_{gen}_{cat}_{color}".upper(),
                año=año,
                mes_lanzamiento=f"{año}-{MES_OPTIMO_POR_GENERO.get(gen, 6):02d}",
                tipo_taco="nuevo",
                taco_destino=taco_propuesto,
                concepto_id=concepto_label,
                nombre_sugerido=f"{justif} {color.title()}",
                familia_genero=gen,
                familia_genero_efectivo=gen,
                mercado_principal=mercado,
                version="RVR",
                tamano_familia="letra_grande",
                tipo_letra="letra_grande",
                categoria_precio=cat,
                precio_promedio=float(precio),
                descuento_promedio=float(dto),
                tiene_cierre=(cat in ("semi_fina", "fina")),
                tiene_indice=(cat == "fina"),
                es_imitacion_cuero=True,
                tiene_canto_dorado=(cat == "fina"),
                prioridad="media" if cat == "economica" else "baja",
                justificacion=justif,
            )
            pred = predecir_novedad(sug.features_dict(), bundle_hedonico)
            sug.demanda_anual_estimada = pred["demanda_anual_estimada"]
            sug.demanda_anual_p10 = pred["intervalo_p10"]
            sug.demanda_anual_p90 = pred["intervalo_p90"]
            candidatos.append(sug)

    return candidatos


# =========================================================================
# ALGORITMO GREEDY CON RESTRICCIONES
# =========================================================================
def seleccionar_sugerencias_anio(
    candidatos: list[Sugerencia],
    año: int,
    gap_restante: float,
    cubiertas_libres: int,
    conceptos_libres: int,
    permitir_incremento_capacidad: bool = True,
    juvenil_cada_2: bool = True,  # añadir juvenil si toca este año
) -> list[Sugerencia]:
    """
    Selecciona greedy las sugerencias para un año respetando pisos de
    balance y capacidad. Si gap no se cierra con capacidad disponible y
    permitir_incremento_capacidad=True, agrega sugerencias adicionales
    marcadas como dentro_capacidad=False.
    """
    seleccionadas: list[Sugerencia] = []
    if gap_restante <= 0:
        return seleccionadas

    # Score = demanda × multiplicador por tipo de TACO
    def score(s: Sugerencia) -> float:
        mult = {"alta": 3.0, "media": 2.0, "baja": 1.0}[s.prioridad]
        risk_factor = 1.0 if s.tipo_taco == "existente" else 0.7
        return s.demanda_anual_estimada * mult * risk_factor

    pool = sorted(candidatos, key=score, reverse=True)
    conceptos_nuevos_seleccionados = set()
    aporte_acumulado = 0.0

    def cumple_pisos_actual() -> dict:
        """Calcula uso actual del piso de balance."""
        total = len(seleccionadas)
        n_misioneras = sum(1 for s in seleccionadas
                           if es_taco_misionero(s.taco_destino) or s.categoria_precio == "economica")
        n_fem = sum(1 for s in seleccionadas
                    if s.familia_genero_efectivo == "femenino")
        n_masc = sum(1 for s in seleccionadas
                     if s.familia_genero_efectivo == "masculino")
        n_juv = sum(1 for s in seleccionadas
                    if s.familia_genero_efectivo == "juvenil")
        return {
            "total": total,
            "misioneras": n_misioneras,
            "femenino_pct": n_fem / total if total else 0,
            "masculino_pct": n_masc / total if total else 0,
            "juvenil": n_juv,
        }

    def puede_aprobar(s: Sugerencia, dentro_cap: bool = True) -> bool:
        """Verifica si la sugerencia respeta restricciones de capacidad."""
        if not dentro_cap:
            return True  # no hay límite, etiqueta "incremento_capacidad"
        if len(seleccionadas) >= cubiertas_libres:
            return False
        if s.tipo_taco == "nuevo":
            n_conceptos_actuales = len(conceptos_nuevos_seleccionados)
            if s.concepto_id in conceptos_nuevos_seleccionados:
                return True  # ya hay otras cubiertas de este concepto, no consume cupo extra
            if n_conceptos_actuales >= conceptos_libres:
                return False
        return True

    def aplicar(s: Sugerencia, dentro_cap: bool = True):
        s.dentro_capacidad = dentro_cap
        seleccionadas.append(s)
        if s.tipo_taco == "nuevo":
            conceptos_nuevos_seleccionados.add(s.concepto_id)

    # --- PASO 1: cumplir piso de misioneras (mín 1) ---
    misioneras_candidatas = [s for s in pool
                              if es_taco_misionero(s.taco_destino) or s.categoria_precio == "economica"]
    for s in misioneras_candidatas:
        if cumple_pisos_actual()["misioneras"] >= PISOS_BALANCE["min_misioneras"]:
            break
        if puede_aprobar(s):
            aplicar(s)
            aporte_acumulado += s.demanda_anual_estimada
            pool.remove(s)

    # --- PASO 2: cumplir piso de femenino (30% sobre cubiertas_libres) PRIMERO ---
    # Lo hacemos ANTES del greedy general para garantizar balance PATMOS,
    # aunque sacrifique algo de aporte total.
    min_femenino = max(1, int(np.ceil(cubiertas_libres * PISOS_BALANCE["min_femenino_pct"])))
    femeninos_pool = [s for s in pool if s.familia_genero_efectivo == "femenino"]
    n_fem_actual = sum(1 for s in seleccionadas if s.familia_genero_efectivo == "femenino")
    for s in femeninos_pool:
        if n_fem_actual >= min_femenino:
            break
        if puede_aprobar(s):
            aplicar(s)
            aporte_acumulado += s.demanda_anual_estimada
            pool.remove(s)
            n_fem_actual += 1

    # --- PASO 3: piso juvenil (cada 2 años) ---
    if juvenil_cada_2:
        juveniles = [s for s in pool if s.familia_genero_efectivo == "juvenil"]
        n_juv_actual = sum(1 for s in seleccionadas if s.familia_genero_efectivo == "juvenil")
        for s in juveniles[:PISOS_BALANCE["min_juvenil_cada_2_anios"]]:
            if n_juv_actual >= PISOS_BALANCE["min_juvenil_cada_2_anios"]:
                break
            if puede_aprobar(s):
                aplicar(s)
                aporte_acumulado += s.demanda_anual_estimada
                pool.remove(s)
                n_juv_actual += 1

    # --- PASO 4: greedy maximizar aporte hasta cerrar gap o llenar capacidad ---
    for s in list(pool):
        if aporte_acumulado >= gap_restante:
            break
        if puede_aprobar(s, dentro_cap=True):
            aplicar(s, dentro_cap=True)
            aporte_acumulado += s.demanda_anual_estimada
            pool.remove(s)

    # --- PASO 4: si gap aún no cerrado y permitido, agregar "fuera de capacidad" ---
    if permitir_incremento_capacidad and aporte_acumulado < gap_restante:
        for s in list(pool):
            if aporte_acumulado >= gap_restante:
                break
            # Marcar como fuera de capacidad
            aplicar(s, dentro_cap=False)
            aporte_acumulado += s.demanda_anual_estimada
            pool.remove(s)
            # Limitar a 10 extra para no saturar
            extras = sum(1 for x in seleccionadas if not x.dentro_capacidad)
            if extras >= 10:
                break

    return seleccionadas


# =========================================================================
# GENERAR CURVA MENSUAL PARA CADA SUGERENCIA
# =========================================================================
def aplicar_curva_mensual(
    sugerencia: Sugerencia,
    perfiles_estacionales: dict,
) -> Sugerencia:
    """Genera la curva mensual de la sugerencia a partir del perfil."""
    perfil = perfil_para_producto(
        perfiles_estacionales,
        familia_genero=sugerencia.familia_genero,
        mercado=sugerencia.mercado_principal,
    )
    curva = novedades_store.generar_curva_mensual(
        mes_lanzamiento=sugerencia.mes_lanzamiento,
        demanda_anual=sugerencia.demanda_anual_estimada,
        perfil_estacional=perfil,
        ciclo_vida_meses=sugerencia.ciclo_vida_meses,
        p10=sugerencia.demanda_anual_p10,
        p90=sugerencia.demanda_anual_p90,
    )
    sugerencia.curva_mensual = [
        {
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "prediccion": float(row["prediccion"]),
            "p10": float(row.get("p10", 0)),
            "p90": float(row.get("p90", 0)),
        }
        for _, row in curva.iterrows()
    ]
    return sugerencia


# =========================================================================
# GENERACIÓN COMPLETA 2027-2030
# =========================================================================
def generar_todas_sugerencias(
    permitir_incremento_capacidad: bool = True,
) -> pd.DataFrame:
    """
    Genera el set completo de sugerencias 2027-2030.
    Considera novedades ya aprobadas (para descontar de capacidad libre).
    """
    feature_isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")
    bundle = joblib.load(DATA_STATE / "modelo_hedonico.joblib")
    with open(DATA_STATE / "perfiles_estacionales.pkl", "rb") as f:
        perfiles = pickle.load(f)

    # Aporte catálogo (Prophet)
    proy_path = DATA_STATE / "proyecciones_prophet.parquet"
    if proy_path.exists():
        proy = pd.read_parquet(proy_path)
        proy["anio"] = pd.to_datetime(proy["ds"]).dt.year
        isbn_biblias = feature_isbn[feature_isbn["clase"] == "BIBLIAS"]["isbn"].tolist()
        aporte_catalogo = (
            proy[proy["isbn"].isin(isbn_biblias)]
            .groupby("anio")["yhat"].sum().to_dict()
        )
    else:
        aporte_catalogo = {}

    # Aporte de novedades ya aprobadas
    aporte_aprobadas = novedades_store.obtener_aporte_anual_novedades()

    todas: list[Sugerencia] = []
    for año in [2027, 2028, 2029, 2030]:
        gap = METAS_BIBLIAS[año] - aporte_catalogo.get(año, 0) - aporte_aprobadas.get(año, 0)
        if gap <= 0:
            continue

        uso = novedades_store.calcular_uso_capacidad(año)
        cubiertas_libres = uso["cubiertas_libres"]
        conceptos_libres = uso["conceptos_libres"]

        candidatos = construir_pool_candidatos(feature_isbn, bundle, año)
        seleccionadas = seleccionar_sugerencias_anio(
            candidatos=candidatos,
            año=año,
            gap_restante=gap,
            cubiertas_libres=cubiertas_libres,
            conceptos_libres=conceptos_libres,
            permitir_incremento_capacidad=permitir_incremento_capacidad,
            juvenil_cada_2=(año % 2 == 1),  # años impares
        )

        # Aplicar curva mensual
        for s in seleccionadas:
            aplicar_curva_mensual(s, perfiles)
        todas.extend(seleccionadas)

    # Convertir a DataFrame
    if not todas:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(s) for s in todas])
    return df


# =========================================================================
# TEST EN CLI
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GENERADOR DE SUGERENCIAS AUTOMÁTICAS 2027-2030")
    print("=" * 70)

    df = generar_todas_sugerencias()
    if len(df) == 0:
        print("⚠️  No se generaron sugerencias (¿ya se cerró el gap?)")
    else:
        print(f"\n✓ Generadas {len(df)} sugerencias")
        print("\n📊 Por año y tipo:")
        print(df.groupby(["año", "tipo_taco", "dentro_capacidad"]).size().to_string())

        print("\n📊 Por género efectivo:")
        print(df.groupby(["año", "familia_genero_efectivo"]).size().to_string())

        print("\n📊 Aporte total estimado por año:")
        agg = df.groupby(["año", "dentro_capacidad"])["demanda_anual_estimada"].sum().astype(int)
        print(agg.to_string())

        # Top 5 sugerencias 2027
        print("\n🏆 TOP 10 sugerencias 2027 (por demanda estimada):")
        top = df[df["año"] == 2027].nlargest(10, "demanda_anual_estimada")
        for _, r in top.iterrows():
            cap_label = "✅" if r["dentro_capacidad"] else "📈 +cap"
            print(f"   {cap_label} {r['nombre_sugerido'][:50]:50s} | "
                  f"{r['demanda_anual_estimada']:>6,.0f} u/año | "
                  f"{r['familia_genero_efectivo']:10s} | "
                  f"{r['categoria_precio']:15s} | {r['tipo_taco']}")
