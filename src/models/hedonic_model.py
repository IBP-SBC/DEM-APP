"""
Modelo Hedónico de Demanda Anual de BIBLIAS.

Predice la demanda anual estabilizada de una biblia (nueva o existente)
a partir de sus características de producto, precio y mercado.

Especificación econométrica:

    log(demanda_anual_madura) = β₀
        + β₁·log(precio)
        + β₂·descuento_pct
        + β₃·categoria_precio (5 niveles, ref: media)
        + β₄·familia_genero (4+1 niveles)
        + β₅·tamano_familia (7 niveles)
        + β₆·tipo_letra
        + β₇·mercado_principal (3 niveles)
        + β₈·canal_principal (top niveles)
        + β₉·lista_precios_principal (top niveles)
        + β₁₀·tiene_cierre (dummy)
        + β₁₁·tiene_indice (dummy)
        + β₁₂·es_imitacion_cuero (dummy)
        + β₁₃·share_internacional
        + ε

Estimador: LightGBM gradient-boosted trees con 5-fold CV.
Razón: captura interacciones no lineales (ej. el efecto de "letra grande"
puede ser distinto en biblias económicas vs finas) sin que tengamos que
especificarlas a mano. Para interpretabilidad usamos importancia +
permutation importance.

Output:
    - modelo entrenado (.joblib)
    - métricas (MAPE, MAE, R²) en data/state/metricas_hedonico.json
    - feature importance plot data
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
DATA_STATE = ROOT / "data" / "state"
DATA_STATE.mkdir(parents=True, exist_ok=True)

MODELO_PATH = DATA_STATE / "modelo_hedonico.joblib"
METRICAS_PATH = DATA_STATE / "metricas_hedonico.json"
IMPORTANCIAS_PATH = DATA_STATE / "importancias_hedonico.csv"


# =========================================================================
# DEFINICIÓN DE FEATURES
# =========================================================================
# CRITERIO: solo features EXÓGENAS, decidibles al diseñar el producto.
# El descuento SÍ es exógeno porque es decisión estructural por categoría:
#   - Económicas/misioneras: tope 23% (excepcional 25%)
#   - Otras categorías: 28-45% según fineza
# Quitamos canal_principal y lista_precios_principal porque sí son outputs.
#
# v3.10: añadimos features de la codificación SBU canónica.
#   - sbu_pasta:  distingue rústica/vinilo/dura/jean/imitación/PU/cuero
#   - sbu_tamano: 1..8 (más granular que tamano_familia textual)
#   - sbu_tipo_letra: LM/LG/LGi/LSGi (más fino que tipo_letra legacy)
#   - dummies por símbolo (PJR, C, c, DK, D, P, T, EE, a, e, ue)
# El A/B test sobre el histórico muestra que el R² real prácticamente
# se duplica (0.20 → 0.38) y el MAE baja 15,7%.

# Símbolos que entran como dummies. Importamos la lista canónica de
# dictionaries.py para mantener una sola fuente de verdad.
from utils.dictionaries import SBU_SIMBOLOS

# Solo símbolos que NO son tipo de letra (los de letra ya están en
# sbu_tipo_letra como categórica, evitamos duplicar señal).
_SBU_SYM_DUMMIES = [
    f"sbu_sym_{etiq}"
    for codigo, etiq, _desc, _pts, fam, _frec in SBU_SIMBOLOS
    if fam != "letra"
]

FEATURES_NUMERICAS = [
    "log_precio",
    "descuento_promedio",   # decisión estructural por categoría al lanzar
]

FEATURES_CATEGORICAS = [
    "familia_genero",
    "tamano_familia",       # legacy textual (mini/bolsillo/...)
    "tipo_letra",           # legacy de extractor de tokens
    "mercado_principal",    # decisión: nacional XOR internacional (por logos SBU/SBC)
    "version",
    # --- SBU v3.10 ---
    "sbu_pasta",            # '0','2','3','4','5','6','9' o legacy '1','7'
    "sbu_tamano",           # '1'..'6','8'
    "sbu_tipo_letra",       # 'letra_mediana' | 'letra_grande' | 'letra_gigante' | 'letra_super_gigante'
]

FEATURES_BOOL = [
    "tiene_cierre",
    "tiene_indice",
    "es_imitacion_cuero",
    "tiene_canto_dorado",
    # --- Dummies SBU por símbolo (sin los de letra) ---
    *_SBU_SYM_DUMMIES,
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BOOL
TARGET = "log_demanda_anual"


# =========================================================================
# PREPARACIÓN DE DATOS
# =========================================================================
def preparar_dataset_training(
    feature_isbn: pd.DataFrame,
    min_meses_vida: int = 12,
    min_demanda: float = 30.0,
    cap_p99: bool = True,
) -> pd.DataFrame:
    """
    Filtra y prepara el dataset para entrenamiento del hedónico.

    Restricciones:
    - Solo BIBLIAS
    - Solo ACTIVOS o DECLINANDO (no DESCONTINUADOS, su madurez es del pasado)
    - meses_vida >= 12 (necesitamos al menos un año observado)
    - demanda anualizada >= 30 (excluir microventas)
    - cap a p99 (outliers extremos distorsionan)
    """
    df = feature_isbn[feature_isbn["clase"] == "BIBLIAS"].copy()
    df = df[df["demanda_anual_madura"].notna()]
    df = df[df["demanda_anual_madura"] >= min_demanda]
    df = df[df["meses_vida"] >= min_meses_vida]
    df = df[df["precio_promedio"] > 1000]
    df = df[df["estado"].isin(["ACTIVO", "DECLINANDO"])]

    # Cap outliers extremos
    if cap_p99 and len(df):
        p99 = df["demanda_anual_madura"].quantile(0.99)
        df.loc[df["demanda_anual_madura"] > p99, "demanda_anual_madura"] = p99

    # Target: log1p(demanda) — robusto a ceros y a la cola larga
    df["log_demanda_anual"] = np.log1p(df["demanda_anual_madura"])

    # Feature: log del precio
    df["log_precio"] = np.log(df["precio_promedio"])

    # Imputar nulls en categóricas como "desconocido"
    for c in FEATURES_CATEGORICAS:
        if c in df.columns:
            df[c] = df[c].fillna("desconocido").astype(str)
            df.loc[df[c] == "", c] = "desconocido"
            df.loc[df[c] == "nan", c] = "desconocido"
        else:
            df[c] = "desconocido"

    # Bools a int (LightGBM las quiere así).
    # En v3.10 algunas columnas (dummies SBU) pueden venir como float si el
    # parquet las dedujo así; fillna(0) primero para evitar errores en astype.
    for c in FEATURES_BOOL:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = 0

    return df[["isbn", "descripcion", "demanda_anual_madura"] + ALL_FEATURES + [TARGET]]


# =========================================================================
# ENTRENAMIENTO
# =========================================================================
def entrenar_modelo(
    df_train: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Entrena LightGBM con K-fold CV y retorna modelo + métricas.
    """
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

    X = df_train[ALL_FEATURES].copy()
    y = df_train[TARGET].copy()
    y_real = df_train["demanda_anual_madura"].copy()

    # Marcar categóricas para LightGBM
    for c in FEATURES_CATEGORICAS:
        X[c] = X[c].astype("category")

    # K-fold CV con predicciones out-of-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred_log = np.zeros(len(X))
    fold_scores = []

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.04,
        "num_leaves": 63,          # más capacidad para capturar interacciones
        "min_child_samples": 5,    # menos restrictivo, captura efectos en subgrupos
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.05,         # menos regularización
        "lambda_l2": 0.05,
        "max_depth": -1,
        "verbose": -1,
    }

    for fold, (idx_tr, idx_val) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[idx_tr], X.iloc[idx_val]
        y_tr, y_val = y.iloc[idx_tr], y.iloc[idx_val]

        train_set = lgb.Dataset(
            X_tr, label=y_tr, categorical_feature=FEATURES_CATEGORICAS
        )
        val_set = lgb.Dataset(
            X_val, label=y_val, categorical_feature=FEATURES_CATEGORICAS,
            reference=train_set
        )

        model = lgb.train(
            params, train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_pred_log[idx_val] = model.predict(X_val, num_iteration=model.best_iteration)

        fold_mae = mean_absolute_error(y_val, oof_pred_log[idx_val])
        fold_scores.append(fold_mae)

    # Entrenar modelo final en todo el dataset
    train_set_full = lgb.Dataset(X, label=y, categorical_feature=FEATURES_CATEGORICAS)
    model_final = lgb.train(
        params, train_set_full,
        num_boost_round=int(np.mean([m.best_iteration for m in [model]])) or 300,
    )

    # Métricas en espacio LOG
    mae_log = mean_absolute_error(y, oof_pred_log)
    r2_log = r2_score(y, oof_pred_log)

    # Métricas en espacio REAL (expm1 para des-transformar)
    y_pred_real = np.expm1(oof_pred_log)
    mae_real = mean_absolute_error(y_real, y_pred_real)
    mape_real = mean_absolute_percentage_error(y_real, y_pred_real)
    r2_real = r2_score(y_real, y_pred_real)

    # Importancia
    importancias = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_gain": model_final.feature_importance(importance_type="gain"),
        "importance_split": model_final.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    # Predicciones del modelo final para diagnóstico
    oof_df = df_train[["isbn", "descripcion", "demanda_anual_madura"]].copy()
    oof_df["prediccion"] = y_pred_real
    oof_df["error_pct"] = (
        (oof_df["prediccion"] - oof_df["demanda_anual_madura"])
        / oof_df["demanda_anual_madura"]
    ) * 100
    oof_df["error_abs"] = (oof_df["prediccion"] - oof_df["demanda_anual_madura"]).abs()

    return {
        "model": model_final,
        "metricas": {
            "n_train": len(X),
            "mae_log": float(mae_log),
            "r2_log": float(r2_log),
            "mae_real": float(mae_real),
            "mape_real": float(mape_real),
            "r2_real": float(r2_real),
            "fold_scores": [float(s) for s in fold_scores],
            "features_usadas": ALL_FEATURES,
            "n_features": len(ALL_FEATURES),
        },
        "importancias": importancias,
        "oof_predictions": oof_df,
        "params": params,
    }


# =========================================================================
# PREDICCIÓN PARA NOVEDADES
# =========================================================================
def predecir_novedad(
    features: dict,
    model_bundle: Optional[dict] = None,
) -> dict:
    """
    Predice la demanda anual de un producto nuevo basado en sus características.

    Args:
        features: dict con las características del producto. Claves esperadas:
            precio_promedio, descuento_promedio, share_internacional,
            categoria_precio, familia_genero, tamano_familia, tipo_letra,
            mercado_principal, canal_principal, lista_precios_principal,
            version, tiene_cierre, tiene_indice, es_imitacion_cuero,
            tiene_canto_dorado
        model_bundle: dict con model + metadata. Si None se carga del disco.

    Returns:
        dict con prediccion (unidades anuales), prediccion_log,
              p10 e p90 (intervalo aproximado por MAE histórico)
    """
    if model_bundle is None:
        model_bundle = joblib.load(MODELO_PATH)
    model = model_bundle["model"]
    mae_log = model_bundle["metricas"]["mae_log"]

    # Construir fila de input
    fila = {}
    fila["log_precio"] = np.log(max(features.get("precio_promedio", 50000), 1000))
    fila["descuento_promedio"] = float(features.get("descuento_promedio", 30.0))
    for c in FEATURES_CATEGORICAS:
        v = features.get(c, "desconocido")
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            v = "desconocido"
        fila[c] = str(v)
    for c in FEATURES_BOOL:
        fila[c] = int(bool(features.get(c, False)))

    X = pd.DataFrame([fila])[ALL_FEATURES]
    for c in FEATURES_CATEGORICAS:
        X[c] = X[c].astype("category")

    pred_log = float(model.predict(X)[0])
    pred = float(np.expm1(pred_log))

    # Intervalo aproximado: ±MAE_log en el espacio log
    p10 = float(np.expm1(pred_log - 1.28 * mae_log))  # ~10% inferior
    p90 = float(np.expm1(pred_log + 1.28 * mae_log))

    return {
        "demanda_anual_estimada": pred,
        "log_prediction": pred_log,
        "intervalo_p10": p10,
        "intervalo_p90": p90,
        "mae_log_modelo": mae_log,
    }


# =========================================================================
# MAIN - ENTRENAR Y GUARDAR
# =========================================================================
def buscar_comparables(
    features: dict,
    feature_isbn: pd.DataFrame,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Busca ISBNs históricos similares al producto descrito, como validación
    cualitativa de la predicción del modelo. La similitud se calcula sobre:
    familia_genero (igual), version (igual), tamaño (igual — preferimos
    sbu_tamano si existe), y distancia en log(precio). Cuando hay features
    SBU, se afina con sbu_pasta y sbu_tipo_letra.
    """
    df = feature_isbn[
        (feature_isbn["clase"] == "BIBLIAS")
        & (feature_isbn["demanda_anual_madura"].notna())
    ].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Filtrar por dimensiones categóricas iguales (relajación progresiva).
    # Probamos primero las features SBU (más precisas) y luego las legacy.
    candidatos = df.copy()
    cols_filtro_orden = [
        "sbu_version", "sbu_tamano", "sbu_pasta", "sbu_tipo_letra",
        "familia_genero", "version", "tamano_familia",
    ]
    for col in cols_filtro_orden:
        v = features.get(col)
        if v and v != "desconocido" and col in candidatos.columns:
            filtro = candidatos[col].astype(str) == str(v)
            if filtro.sum() >= 3:  # mantener filtro si hay suficientes
                candidatos = candidatos[filtro]

    if len(candidatos) == 0:
        candidatos = df

    # Distancia en log(precio)
    log_precio_target = np.log(max(features.get("precio_promedio", 50000), 1000))
    candidatos = candidatos.copy()
    candidatos["dist_precio"] = (np.log(candidatos["precio_promedio"]) - log_precio_target).abs()
    candidatos = candidatos.sort_values("dist_precio").head(n_top)

    # Columnas a mostrar: incluimos sbu_codigo_canonico si está disponible
    cols_show = [
        "isbn", "descripcion", "familia_genero", "version", "tamano_familia",
        "precio_promedio", "demanda_anual_madura", "estado"
    ]
    if "sbu_codigo_canonico" in candidatos.columns:
        cols_show.insert(2, "sbu_codigo_canonico")
    return candidatos[cols_show].reset_index(drop=True)


# =========================================================================
# MAIN - ENTRENAR Y GUARDAR
# =========================================================================
def main():
    print("=" * 70)
    print("ENTRENAMIENTO MODELO HEDÓNICO — DEMANDA ANUAL BIBLIAS")
    print("=" * 70)

    feature_isbn = pd.read_parquet(DATA_PROC / "feature_isbn.parquet")
    df = preparar_dataset_training(feature_isbn)
    print(f"\n📊 Dataset de entrenamiento: {len(df)} ISBNs")
    print(f"   Target: log1p(demanda_anual_madura)")
    print(f"   Rango target: [{df['log_demanda_anual'].min():.2f}, {df['log_demanda_anual'].max():.2f}]")
    print(f"   Rango real:   [{df['demanda_anual_madura'].min():.0f}, {df['demanda_anual_madura'].max():.0f}]")

    print(f"\n🧮 Features ({len(ALL_FEATURES)}):")
    print(f"   Numéricas:    {FEATURES_NUMERICAS}")
    print(f"   Categóricas:  {FEATURES_CATEGORICAS}")
    print(f"   Booleanas:    {FEATURES_BOOL}")

    print("\n⚙️  Entrenando LightGBM con 5-fold CV...")
    bundle = entrenar_modelo(df)

    print("\n📈 MÉTRICAS DE PERFORMANCE (out-of-fold):")
    m = bundle["metricas"]
    print(f"   MAE (log):           {m['mae_log']:.4f}")
    print(f"   R²  (log):           {m['r2_log']:.4f}")
    print(f"   MAE (unidades):      {m['mae_real']:,.0f}")
    print(f"   MAPE (unidades):     {m['mape_real']*100:.1f}%")
    print(f"   R²  (unidades):      {m['r2_real']:.4f}")
    print(f"   Folds MAE log:       {[f'{s:.3f}' for s in m['fold_scores']]}")

    print("\n🏆 IMPORTANCIA DE FEATURES (top 10 por gain):")
    imp = bundle["importancias"].head(10)
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance_gain"] / imp["importance_gain"].max() * 30)
        print(f"   {row['feature']:25s} {row['importance_gain']:>10,.0f}  {bar}")

    # Guardar
    joblib.dump(bundle, MODELO_PATH)
    with open(METRICAS_PATH, "w") as f:
        json.dump(bundle["metricas"], f, indent=2)
    bundle["importancias"].to_csv(IMPORTANCIAS_PATH, index=False)
    bundle["oof_predictions"].to_csv(DATA_STATE / "predicciones_oof.csv", index=False)

    print(f"\n💾 Guardado:")
    print(f"   {MODELO_PATH}")
    print(f"   {METRICAS_PATH}")
    print(f"   {IMPORTANCIAS_PATH}")
    print(f"   {DATA_STATE/'predicciones_oof.csv'}")

    # Test rápido: predecir una novedad ejemplo
    # En v3.10 incluimos también las features SBU para que la predicción
    # use el modelo completo. Usamos mercado_principal="ambos" porque es
    # la configuración de los SKUs exitosos del catálogo (las que son
    # 'solo nacional' tienen mediana 113 u/año, mientras 'ambos' tiene 2.300).
    # Eso refleja una realidad estructural: el mercado de exportación más
    # el nacional > solo nacional.
    print("\n🧪 TEST: predecir RVR065 imitación cuero femenina $85k, mercado=ambos, dto 33%")
    test_features = {
        "precio_promedio": 85000,
        "descuento_promedio": 33.0,
        "familia_genero": "femenino",
        "mercado_principal": "ambos",
        "version": "RVR",
        "tamano_familia": "letra_grande",
        "tipo_letra": "letra_grande",
        "tiene_cierre": False,
        "tiene_indice": False,
        "es_imitacion_cuero": True,
        "tiene_canto_dorado": False,
        # --- SBU (v3.10): código RVR065cLGPJR ---
        "sbu_version": "RVR",
        "sbu_familia": "0",
        "sbu_tamano": "6",
        "sbu_pasta": "5",
        "sbu_tipo_letra": "letra_grande",
        "sbu_sym_cierre": 0,
        "sbu_sym_indice": 0,
        "sbu_sym_palabras_jesus": 1,
        "sbu_sym_concordancia_amplia": 0,
        "sbu_sym_concordancia_breve": 1,
        "sbu_sym_deuteros_alej": 0,
        "sbu_sym_deuteros_sep": 0,
        "sbu_sym_estudio_economica": 0,
        "sbu_sym_ilustrado": 0,
        "sbu_sym_tematica": 0,
        "sbu_sym_acolchado": 0,
        "sbu_sym_economica": 0,
        "sbu_sym_ultraeconomica": 0,
    }
    pred = predecir_novedad(test_features, bundle)
    print(f"   Estimación anual: {pred['demanda_anual_estimada']:,.0f} unidades")
    print(f"   Intervalo 80%:    [{pred['intervalo_p10']:,.0f} - {pred['intervalo_p90']:,.0f}]")

    print("\n   🔍 ISBNs comparables reales (validación):")
    comp = buscar_comparables(test_features, feature_isbn, n_top=10)
    if len(comp):
        comp_print = comp.copy()
        comp_print["precio"] = comp_print["precio_promedio"].apply(lambda x: f"${x:,.0f}")
        comp_print["demanda"] = comp_print["demanda_anual_madura"].apply(lambda x: f"{x:,.0f}")
        for _, r in comp_print.iterrows():
            print(f"      {r['isbn'][:25]:25s} | {str(r['descripcion'])[:40]:40s} | "
                  f"{r['precio']:>10s} | {r['demanda']:>7s} u/año | {r['estado']}")
        mediana_real = comp["demanda_anual_madura"].median()
        media_real = comp["demanda_anual_madura"].mean()
        print(f"\n   📊 Comparables: mediana={mediana_real:,.0f} | media={media_real:,.0f}")
        print(f"      Predicción del modelo: {pred['demanda_anual_estimada']:,.0f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
