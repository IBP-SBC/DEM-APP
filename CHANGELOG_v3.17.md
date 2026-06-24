# Sprint v3.17 — Migración a la nube (GitHub + Streamlit Cloud + Supabase)

**Fecha:** 19 junio 2026

## POR QUÉ
Llevar la app de demanda a la nube como ventas, sin perder funcionalidad. El
cálculo pesado (Prophet + hedónico, 8-11 min) NO es viable en Community Cloud
(1 GB RAM, corta procesos largos), así que se separa en dos modos sobre el
mismo código: escritorio produce artefactos, nube consume y edita.

## QUÉ CAMBIÓ

### Plomería de nube
- NUEVO `core/cloud_storage.py`: API REST de Supabase (requests), defensiva.
  Normaliza la URL (quita /rest/v1/), acepta claves sb_secret_ en apikey +
  Bearer, nube_activa(), subir/bajar/listar, hidratar_desde_nube(),
  probar_conexion() con error exacto. Si no hay secrets, todo es no-op.
- NUEVO `core/auth.py`: login por st.secrets['usuarios'] (libre en local).
- Home: login + hidratación al iniciar + panel de estado de nube. El cargador
  SIESA queda INFORMATIVO en la nube (la ingesta + reentrenamiento es escritorio).
- Hooks de subida automática a Supabase en los 4 stores de estado (overrides
  de proyección, novedades aprobadas, correcciones TACO MP, overrides de cliente).
- NUEVO `models/subir_a_nube.py`: sube los artefactos calculados a Supabase
  desde escritorio (tras reentrenar).

### Optimización de memoria (Community Cloud 1 GB)
- NUEVO `core/optimize.py`: optimizar_memoria(df) convierte texto de baja/media
  cardinalidad a category y hace downcast de enteros, SIN tocar floats de
  montos. Validado: NO cambia ningún número. Ahorro 53->17.5 MB (67%).
  Seguro bajo pandas 3.0 (groupby category usa observed=True).
- @st.cache_data con max_entries=1 en las 6 páginas (no acumula versiones).

### Lectura rápida
- NUEVO `core/fast_io.py`: leer_excel con python-calamine + fallback openpyxl.
  Aplicado a la ingesta SIESA y al histórico (escritorio).

### Configuración del repo
- requirements.txt con versiones fijas (SIN prophet: la nube no entrena).
- .gitignore (excluye datos, secrets, entorno), .streamlit/config.toml
  (solo [browser]), .streamlit/secrets.toml.example, README.md.

## VALIDACIÓN
- optimizar_memoria no cambia números (desagregación 0.000% diff).
- Las 7 páginas pasan AppTest en modo local completo.
- Modo nube simulado SIN artefactos pesados: todas las páginas degradan con
  gracia (pág 4 blindada).
- Degradación sin credenciales: todo no-op, app corre local.

## PENDIENTE DE CONFIRMAR EN DESPLIEGUE
- La conexión real a Supabase no se pudo probar desde el entorno de desarrollo
  (allowlist de red). El panel probar_conexion() la confirma en la nube.
- Versiones de requirements.txt: alinear con las del Mac de Alberto si difieren.
