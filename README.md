# SBC Demanda — Proyección de demanda 2027-2030

App Streamlit de proyección de demanda de la Sociedad Bíblica Colombiana.
Funciona en **dos modos sobre el mismo código**:

- **Escritorio (productor):** corre el cálculo pesado (Prophet por ISBN +
  modelo hedónico LightGBM, 8-11 min) y sube los artefactos a Supabase.
- **Nube (consumidor + editor):** Streamlit Community Cloud. No calcula nada
  pesado; hidrata los artefactos desde Supabase y deja explorar el pronóstico,
  simular/aprobar novedades, generar sugerencias, aplicar overrides y
  correcciones TACO MP, construir el presupuesto por cliente y descargar el
  CSV final 2027-2030. El estado editable se persiste en Supabase.

## Qué hay en este repo

Solo código y configuración. **Los datos NO van al repo** (viven en Supabase y
en el escritorio).

## Despliegue en la nube (Streamlit Community Cloud)

1. Repo privado en GitHub (IBP-SBC/DEM-APP). No subir secrets.toml ni datos.
2. Streamlit Community Cloud -> New app -> este repo, rama y src/app/Home.py.
3. Settings -> Secrets: pegar el contenido de secrets.toml real (usuarios +
   Supabase). Ver .streamlit/secrets.toml.example.
4. La app instala requirements.txt, pide login y al entrar hidrata desde Supabase.

## Flujo de escritorio (cuando llegan ventas nuevas)

    uv run python src/models/run_all.py          # feature store + modelos
    uv run python src/models/run_forecasts.py    # Prophet por ISBN (8-11 min)
    uv run python src/models/subir_a_nube.py     # sube artefactos a Supabase

Luego en la nube: panel de nube -> "Re-sincronizar desde la nube".

## Persistencia (Supabase Storage)

Bucket sbc-demanda: processed/ (feature store) y
state/ (modelos, proyecciones y estado editable). El estado editable
se sube solo al modificarlo y se hidrata al iniciar sesión.
