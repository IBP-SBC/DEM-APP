"""
Configuración central de sbc_demanda — ÚNICA FUENTE DE VERDAD.

Por qué existe:
  Las metas estratégicas de Biblias 2027-2030 se usan en tres lugares (el gap
  a metas del Home, el motor de sugerencias y el anclaje de Prophet). Antes
  estaban escritas a mano en los tres archivos; si una cambiaba y otra no, la
  app mostraba un objetivo y el motor trabajaba contra otro, sin avisar.

  Ahora viven SOLO aquí. Cambiar una meta a futuro es editar un único número
  en este archivo, y es imposible que se desincronicen.
"""
from __future__ import annotations

# =========================================================================
# OBJETIVOS ESTRATÉGICOS — Metas conservadoras de Biblias 2027-2030
# =========================================================================
# Si los objetivos cambian, se edita SOLO este diccionario.
METAS_BIBLIAS = {
    2027: 1_673_298,
    2028: 2_139_351,
    2029: 2_666_471,
    2030: 3_243_508,
}

# Total del cuatrienio (derivado; no editar a mano).
TOTAL_METAS_BIBLIAS = sum(METAS_BIBLIAS.values())  # 9_722_628
