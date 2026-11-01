#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
FASE 3 — Modelado predictivo, evaluación temporal y explainability (SHAP)
=========================================================================

Objetivo : predecir `riesgo_abstencion` municipal con información previa al
           período evaluado; evaluar fuera de muestra en 2024 y explicar con SHAP.
Alcance  : MODELO PREDICTIVO + EXPLICABILIDAD. NO demuestra causalidad.

Uso
---
    python 03_modelo_explicabilidad.py --input "data/Salida/fase2_<timestamp>/features_municipales.parquet"    todo (train -> congela -> evalúa)
    
    python scripts/03_modelo_explicabilidad.py --stage train    # solo TRAIN/VALIDATION y congelado
    python scripts/03_modelo_explicabilidad.py --stage evaluate # solo evaluación 2024 + SHAP (carga modelo congelado)
    python scripts/03_modelo_explicabilidad.py --input ruta.parquet --out scripts/salida # usar rutas personalizadas

    # El mejor de estas opciones es la primera, que ejecuta todo el pipeline de forma reproducible. 

Diseño anti-contaminación
-------------------------
* La etapa `train` NUNCA lee filas de 2024 para ninguna decisión: se descartan
  del DataFrame inmediatamente después de la partición.
* El modelo se guarda en disco junto con su SHA-256 antes de evaluar 2024.
* La etapa `evaluate` carga el modelo desde disco, verifica el hash y solo
  entonces toca 2024. No hay re-ajuste posible tras ver las métricas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
RANDOM_STATE = 42
YEAR_TRAIN_MAX = 2018
YEAR_VALID = 2021
YEAR_TEST = 2024

KEYS = ["AELEC", "EDOCVE", "MPIOCVE"]
TARGET = "riesgo_abstencion"
FEATURES = [
    "tasa_jovenes_ln",
    "lag_riesgo_abstencion_1",
    "lag_riesgo_abstencion_2",
    "lag_voto_jovenes_1",
    "lag_voto_jovenes_2",
    "lag_tasa_jovenes_ln_1",
    "hist_mean_riesgo",
    "hist_std_riesgo",
    "hist_delta_riesgo_1",
    "state_lag_mean",
    "gap_riesgo_estado",
    "log_ln_total",
    "log_ln_jovenes",
    "presidencial",
]
LAG_COL = "lag_riesgo_abstencion_1"
FORBIDDEN_FEATURES = {
    TARGET,
    "tasa_voto_jovenes",
    "tasa_voto_jovenes_obs",
    "votos_totales",
    "votos_jovenes",
    "lista_nominal_total",
    "lista_nominal_jovenes",
}
FEATURE_AVAILABILITY = {
    "tasa_jovenes_ln": "composicion demografica del ejercicio, conocida antes de la eleccion",
    "lag_riesgo_abstencion_1": "riesgo de abstencion del registro previo del municipio",
    "lag_riesgo_abstencion_2": "riesgo de abstencion de dos registros previos del municipio",
    "lag_voto_jovenes_1": "participacion juvenil observada en el registro previo",
    "lag_voto_jovenes_2": "participacion juvenil observada dos registros previos",
    "lag_tasa_jovenes_ln_1": "peso juvenil en la lista nominal del registro previo",
    "hist_mean_riesgo": "media historica de riesgo con ejercicios previos",
    "hist_std_riesgo": "desviacion historica de riesgo con ejercicios previos",
    "hist_delta_riesgo_1": "cambio de riesgo entre los dos registros previos",
    "state_lag_mean": "promedio estatal del riesgo rezagado en el mismo ejercicio",
    "gap_riesgo_estado": "diferencia del municipio respecto al promedio estatal rezagado",
    "log_ln_total": "tamano de la lista nominal total del ejercicio",
    "log_ln_jovenes": "tamano de la lista nominal juvenil del ejercicio",
    "presidencial": "indicador de eleccion presidencial (2012, 2018, 2024)",
}
REQUIRED = KEYS + [TARGET] + FEATURES

# Rejilla pequeña y fija (el volumen no justifica búsqueda exhaustiva).
# Se evalúa SOLO con VALIDATION (2021); 2024 no participa.
XGB_GRID = [
    dict(max_depth=3, learning_rate=0.05, min_child_weight=5, subsample=0.8,
         colsample_bytree=1.0, reg_lambda=5.0, reg_alpha=0.0),
    dict(max_depth=4, learning_rate=0.05, min_child_weight=10, subsample=0.8,
         colsample_bytree=1.0, reg_lambda=10.0, reg_alpha=0.1),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=20, subsample=0.8,
         colsample_bytree=1.0, reg_lambda=20.0, reg_alpha=0.1),
]
RF_GRID = [
    dict(n_estimators=300, max_depth=8, min_samples_leaf=20, max_features=1.0),
    dict(n_estimators=300, max_depth=12, min_samples_leaf=50, max_features=1.0),
]
XGB_N_ESTIMATORS = 3000
XGB_EARLY_STOP = 100

log = logging.getLogger("fase3")


class PipelineError(RuntimeError):
    """Condición crítica: el pipeline debe detenerse."""


def check(cond: bool, msg: str) -> None:
    """Validación fail-fast con mensaje claro (no depende de assert / -O)."""
    if not cond:
        log.critical("VALIDACIÓN FALLIDA: %s", msg)
        raise PipelineError(msg)
    log.info("OK  | %s", msg)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def setup_logging(out_dir: Path) -> None:
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(out_dir / "logs" / "modelo_explicabilidad.log",
                                  mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def jdump(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def lib_versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "numpy": np.__version__, "pandas": pd.__version__}
    for name in ("pyarrow", "sklearn", "xgboost", "shap", "matplotlib", "joblib"):
        try:
            mod = __import__(name)
            v[name] = getattr(mod, "__version__", "desconocida")
        except Exception:
            v[name] = None
    return v


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 median_absolute_error, r2_score)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        # R² complementario: es relativo a la media del propio conjunto y puede
        # ser negativo fuera de muestra si hay cambio de nivel entre períodos.
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


# --------------------------------------------------------------------------
# 1. Carga y diagnóstico
# --------------------------------------------------------------------------
def validate_feature_contract() -> None:
    bad = sorted(set(FEATURES) & FORBIDDEN_FEATURES)
    check(not bad, f"Ninguna feature viola el contrato anti-fuga: {bad}")
    missing = sorted(set(FEATURES) - set(FEATURE_AVAILABILITY))
    check(not missing, f"Todas las features tienen declarada su disponibilidad temporal: {missing}")
    check(LAG_COL in FEATURES, f"El baseline ingenuo usa {LAG_COL}, que debe estar en FEATURES")
    log.info("Contrato de features: %d features, todas disponibles antes del ejercicio", len(FEATURES))


def load_and_diagnose(path: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    log.info("=== 1. CARGA Y DIAGNÓSTICO ===")
    check(path.exists(), f"El archivo de entrada existe: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"No se pudo leer el parquet: {e}") from e
    log.info("Filas=%d, columnas=%d", len(df), df.shape[1])
    log.info("Esquema:\n%s", df.dtypes.to_string())

    validate_feature_contract()
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    check(not missing_cols, f"Columnas críticas presentes (faltan: {missing_cols})")

    for c in ["AELEC", "EDOCVE", "MPIOCVE"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c in [TARGET] + FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    diag: dict = {"filas": int(len(df)), "columnas": list(map(str, df.columns)),
                  "target": TARGET, "features": FEATURES,
                  "forbidden_features": sorted(FORBIDDEN_FEATURES),
                  "feature_availability": FEATURE_AVAILABILITY}

    # Ejercicios
    por_anio = (df.groupby("AELEC")
                  .agg(observaciones=("MPIOCVE", "size"),
                       municipios=("MPIOCVE", lambda s: 0))  # se recalcula abajo
                  .reset_index())
    por_anio["municipios"] = (df.groupby("AELEC")[["EDOCVE", "MPIOCVE"]]
                                .apply(lambda g: g.drop_duplicates().shape[0])
                                .values)
    log.info("Ejercicios disponibles:\n%s", por_anio.to_string(index=False))
    diag["ejercicios"] = por_anio.to_dict(orient="records")

    esperados = {2009, 2012, 2015, 2018, 2021, 2024}
    presentes = set(df["AELEC"].dropna().astype(int).unique())
    extra = sorted(presentes - esperados)
    ausentes = sorted(esperados - presentes)
    diag["anios_inesperados"] = extra
    diag["anios_esperados_ausentes"] = ausentes
    if extra:
        log.warning("Ejercicios adicionales a los esperados: %s. Los <=2018 entran "
                    "a TRAIN; los demás distintos de 2021/2024 y >2018 quedan "
                    "EXCLUIDOS de la partición.", extra)
    if ausentes:
        log.warning("Ejercicios esperados ausentes: %s", ausentes)

    # Nulos / infinitos
    nulos = df[[TARGET] + FEATURES].isna().sum()
    log.info("Nulos:\n%s", nulos.to_string())
    diag["nulos"] = {k: int(v) for k, v in nulos.items()}
    inf_counts = {c: int(np.isinf(df[c].to_numpy(dtype=float)).sum())
                  for c in [TARGET] + FEATURES}
    diag["infinitos"] = inf_counts
    check(sum(inf_counts.values()) == 0, f"Sin infinitos en target/features {inf_counts}")
    log.info("Nulos del target (total panel): %d", int(df[TARGET].isna().sum()))
    # Política target: NO se imputa. Nulos en target de años de entrenamiento/
    # validación/test se excluyen explícitamente y se reportan; si afectan a
    # 2024 de forma masiva se emite WARNING.
    diag["nulos_target_total"] = int(df[TARGET].isna().sum())

    # Duplicidades
    dup = int(df.duplicated(subset=KEYS).sum())
    diag["duplicados_municipio_anio"] = dup
    check(dup == 0, f"Sin duplicidades municipio/año (duplicados={dup})")

    jdump(diag, out_dir / "modelo_abstencion" / "diagnostico_dataset.json")
    return df, diag


def audit_lag(df: pd.DataFrame) -> dict:
    """Comprueba que el lag sea información ANTERIOR al ejercicio objetivo."""
    log.info("=== 2. AUDITORÍA DEL LAG ===")
    d = df.sort_values(["EDOCVE", "MPIOCVE", "AELEC"]).copy()
    g = d.groupby(["EDOCVE", "MPIOCVE"], sort=False)
    d["_target_prev"] = g[TARGET].shift(1)
    d["_year_prev"] = g["AELEC"].shift(1)

    # (a) Primer ejercicio de cada municipio: el lag debería ser NaN.
    first = d["_year_prev"].isna()
    first_lag_notnull = int(d.loc[first, LAG_COL].notna().sum())

    # (b) Coincidencia del lag con el target del ejercicio ANTERIOR observado.
    comp = d.loc[d[LAG_COL].notna() & d["_target_prev"].notna()]
    match_prev = float(np.isclose(comp[LAG_COL], comp["_target_prev"], atol=1e-9).mean()) if len(comp) else np.nan

    # (c) Coincidencia con el target del MISMO ejercicio (señal de fuga).
    comp2 = d.loc[d[LAG_COL].notna() & d[TARGET].notna()]
    match_same = float(np.isclose(comp2[LAG_COL], comp2[TARGET], atol=1e-12).mean()) if len(comp2) else np.nan
    corr_same = float(comp2[LAG_COL].corr(comp2[TARGET])) if len(comp2) > 2 else np.nan

    # (d) Específico de 2024
    t = d[d["AELEC"] == YEAR_TEST]
    t = t.loc[t[LAG_COL].notna() & t[TARGET].notna()]
    match_same_2024 = float(np.isclose(t[LAG_COL], t[TARGET], atol=1e-12).mean()) if len(t) else np.nan
    t2 = d[(d["AELEC"] == YEAR_TEST) & d[LAG_COL].notna() & d["_target_prev"].notna()]
    match_prev_2024 = float(np.isclose(t2[LAG_COL], t2["_target_prev"], atol=1e-9).mean()) if len(t2) else np.nan

    res = {
        "primer_ejercicio_con_lag_no_nulo": first_lag_notnull,
        "coincidencia_lag_vs_target_anterior": match_prev,
        "coincidencia_lag_vs_target_mismo_ejercicio": match_same,
        "correlacion_lag_vs_target_mismo_ejercicio": corr_same,
        "coincidencia_lag_vs_target_anterior_2024": match_prev_2024,
        "coincidencia_lag_vs_target_mismo_ejercicio_2024": match_same_2024,
    }
    log.info("Auditoría lag: %s", json.dumps(res, ensure_ascii=False))

    # Reglas de detención
    check(np.isnan(match_same) or match_same < 0.05,
          f"Lag NO coincide con el target del mismo ejercicio (coincidencia={match_same:.4f})")
    check(np.isnan(match_same_2024) or match_same_2024 < 0.05,
          f"Lag de 2024 NO coincide con el target 2024 (coincidencia={match_same_2024:.4f})")
    if not np.isnan(match_prev):
        if match_prev < 0.95:
            # No es fuga, pero el lag no es exactamente el ejercicio previo
            # observado (p. ej. huecos de panel o lag de otra definición).
            log.warning("El lag coincide con el target previo solo en %.2f%% de los "
                        "casos. Revisar definición en Fase 2 (no es evidencia de fuga).",
                        100 * match_prev)
            res["warning_definicion_lag"] = True
    if first_lag_notnull > 0:
        log.warning("%d filas de primer ejercicio del municipio tienen lag no nulo; "
                    "revisar construcción del lag en Fase 2.", first_lag_notnull)
    return res


# --------------------------------------------------------------------------
# 3. Partición temporal
# --------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, diag: dict):
    log.info("=== 3. PARTICIÓN TEMPORAL ===")
    train = df[df["AELEC"] <= YEAR_TRAIN_MAX].copy()
    valid = df[df["AELEC"] == YEAR_VALID].copy()
    test = df[df["AELEC"] == YEAR_TEST].copy()

    excluidas = df[(df["AELEC"] > YEAR_TRAIN_MAX) &
                   (~df["AELEC"].isin([YEAR_VALID, YEAR_TEST]))]
    diag["filas_excluidas_por_anio_fuera_de_particion"] = int(len(excluidas))
    diag["anios_excluidos"] = sorted(excluidas["AELEC"].unique().astype(int).tolist())
    if len(excluidas):
        log.warning("Ejercicios fuera de la partición (excluidos): %s",
                    diag["anios_excluidos"])

    check(len(train) > 0 and len(valid) > 0 and len(test) > 0,
          f"Los tres conjuntos son no vacíos (train={len(train)}, valid={len(valid)}, test={len(test)})")
    check(test["AELEC"].nunique() == 1 and int(test["AELEC"].iloc[0]) == YEAR_TEST,
          "TEST contiene únicamente 2024")
    check(int(train["AELEC"].max()) <= YEAR_TRAIN_MAX, "TRAIN: max(AELEC) <= 2018")
    check(valid["AELEC"].nunique() == 1 and int(valid["AELEC"].iloc[0]) == YEAR_VALID,
          "VALIDATION contiene únicamente 2021")
    check(int(valid["AELEC"].min()) == YEAR_VALID and int(valid["AELEC"].max()) == YEAR_VALID,
          "min/max VALIDATION == 2021")
    check(int(test["AELEC"].min()) == YEAR_TEST and int(test["AELEC"].max()) == YEAR_TEST,
          "min/max TEST == 2024")
    check(not (train["AELEC"] == YEAR_VALID).any(), "TRAIN no contiene 2021")
    check(not (train["AELEC"] == YEAR_TEST).any(), "TRAIN no contiene 2024")
    check(not (valid["AELEC"] == YEAR_TEST).any(), "VALIDATION no contiene 2024")
    check(not (test["AELEC"] < YEAR_TEST).any(), "TEST no contiene años anteriores")
    check(int(train["AELEC"].max()) < int(valid["AELEC"].min()) < int(test["AELEC"].min()),
          "Orden temporal estricto train < validation < test")
    return train, valid, test


def prepare_xy(part: pd.DataFrame, name: str, diag: dict):
    """Excluye filas sin features completas o sin target. No imputa nada."""
    n0 = len(part)
    nan_target = int(part[TARGET].isna().sum())
    nan_feat = int(part[FEATURES].isna().any(axis=1).sum())
    mask = part[TARGET].notna() & part[FEATURES].notna().all(axis=1)
    kept = part.loc[mask].copy()
    dropped = n0 - len(kept)
    info = {"filas_originales": n0, "filas_usadas": int(len(kept)),
            "excluidas": int(dropped),
            "pct_excluidas": float(100 * dropped / n0) if n0 else None,
            "nulos_target": nan_target, "filas_con_feature_nula": nan_feat}
    diag.setdefault("politica_nulos", {})[name] = info
    log.info("%s: usadas=%d / %d (excluidas=%d, %.2f%%; target nulo=%d; feature nula=%d)",
             name, len(kept), n0, dropped, info["pct_excluidas"] or 0, nan_target, nan_feat)
    check(len(kept) > 0, f"{name}: quedan filas utilizables tras excluir incompletas")
    X = kept[FEATURES].astype(float)
    y = kept[TARGET].astype(float)
    check(TARGET not in X.columns, f"{name}: el target no está dentro de X")
    check(not any(c.startswith(TARGET) for c in X.columns), f"{name}: ninguna columna derivada del target en X (lag permitido solo si es 'lag_')")
    check(not (set(X.columns) & FORBIDDEN_FEATURES), f"{name}: ninguna feature del contrato anti-fuga en X")
    check(len(X) == len(y), f"{name}: dimensiones X/y coinciden ({len(X)})")
    check(not np.isinf(X.to_numpy()).any() and not np.isinf(y.to_numpy()).any(),
          f"{name}: sin infinitos en X ni y")
    return kept, X, y


# --------------------------------------------------------------------------
# 4. Entrenamiento / selección (solo TRAIN + VALIDATION)
# --------------------------------------------------------------------------
def gpu_decision() -> dict:
    # Dataset pequeño y features tabulares: la CPU es suficiente y más reproducible.
    return {"backend": "cpu", "motivo": f"dataset pequeño ({len(FEATURES)} features); GPU no aporta ventaja"}


def train_and_select(Xtr, ytr, Xva, yva) -> tuple[object, dict]:
    log.info("=== 4. ENTRENAMIENTO Y SELECCIÓN (solo TRAIN/VALIDATION) ===")
    set_seeds(RANDOM_STATE)
    from sklearn.ensemble import RandomForestRegressor
    candidates: list[dict] = []
    best = None

    try:
        import xgboost as xgb
        have_xgb = True
    except Exception as e:  # noqa: BLE001
        log.warning("xgboost no disponible (%s). Se usará solo Random Forest.", e)
        have_xgb = False

    if have_xgb:
        for i, hp in enumerate(XGB_GRID):
            m = xgb.XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS, objective="reg:squarederror",
                tree_method="hist", device="cpu", random_state=RANDOM_STATE,
                n_jobs=-1, eval_metric="rmse", early_stopping_rounds=XGB_EARLY_STOP,
                **hp)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pv = m.predict(Xva)
            mv = regression_metrics(yva, pv)
            rec = {"algoritmo": "XGBRegressor", "id": f"xgb_{i}", "params": hp,
                   "best_iteration": int(m.best_iteration),
                   "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]}
            candidates.append(rec)
            log.info("Candidato %s: val_RMSE=%.5f val_MAE=%.5f best_iter=%d",
                     rec["id"], mv["RMSE"], mv["MAE"], rec["best_iteration"])
            if best is None or mv["RMSE"] < best[0]["val_RMSE"]:
                best = (rec, m)

    for i, hp in enumerate(RF_GRID):
        m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **hp)
        m.fit(Xtr, ytr)
        pv = m.predict(Xva)
        mv = regression_metrics(yva, pv)
        rec = {"algoritmo": "RandomForestRegressor", "id": f"rf_{i}", "params": hp,
               "val_RMSE": mv["RMSE"], "val_MAE": mv["MAE"]}
        candidates.append(rec)
        log.info("Candidato %s: val_RMSE=%.5f val_MAE=%.5f", rec["id"], mv["RMSE"], mv["MAE"])
        if best is None or mv["RMSE"] < best[0]["val_RMSE"]:
            best = (rec, m)

    rec, model = best
    log.info("MODELO SELECCIONADO (criterio: menor RMSE en VALIDATION 2021): %s %s",
             rec["id"], rec["params"])
    sel = {"criterio": "RMSE en VALIDATION (2021)", "candidatos": candidates,
           "seleccionado": rec}
    return model, sel


def save_model(model, sel: dict, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    algo = sel["seleccionado"]["algoritmo"]
    for old in model_dir.glob("modelo.*"):
        old.unlink()
    if algo == "XGBRegressor":
        p = model_dir / "modelo.json"
        model.save_model(p)
    else:
        import joblib
        p = model_dir / "modelo.joblib"
        joblib.dump(model, p)
    return p


def load_model(model_path: Path, algo: str):
    if algo == "XGBRegressor":
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(model_path)
        return m
    import joblib
    return joblib.load(model_path)


# --------------------------------------------------------------------------
# 5. Evaluación 2024 + SHAP
# --------------------------------------------------------------------------
def compute_shap(model, algo: str, X: pd.DataFrame):
    """TreeExplainer; fallback a pred_contribs de XGBoost (SHAP exacto de árboles)."""
    method = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        base = explainer.expected_value
        base = float(np.ravel(base)[0])
        method = "shap.TreeExplainer"
    except Exception as e:  # noqa: BLE001
        log.warning("shap.TreeExplainer falló (%s).", e)
        if algo != "XGBRegressor":
            raise PipelineError(f"No se pudo calcular SHAP: {e}") from e
        import xgboost as xgb
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        sv, base = contrib[:, :-1], float(contrib[0, -1])
        method = "xgboost.pred_contribs (fallback, SHAP exacto de árboles)"
        log.warning("Usando fallback: %s", method)
    sv = np.asarray(sv, dtype=float)
    return sv, base, method


def evaluate_and_explain(df_test_raw: pd.DataFrame, model, sel: dict, out_dir: Path,
                         model_dir: Path, diag: dict) -> tuple[dict, dict]:
    log.info("=== 5. EVALUACIÓN HOLD-OUT 2024 + SHAP ===")
    algo = sel["seleccionado"]["algoritmo"]
    kept, Xte, yte = prepare_xy(df_test_raw, "TEST", diag)
    kept = kept.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)
    yte = yte.reset_index(drop=True)

    pred = model.predict(Xte)
    m_test = regression_metrics(yte, pred)
    log.info("TEST 2024: MAE=%.5f RMSE=%.5f MedAE=%.5f R2=%s",
             m_test["MAE"], m_test["RMSE"], m_test["MedAE"], m_test["R2"])

    # Baseline ingenuo (diagnóstico, NO se usa para elegir nada)
    base_naive = regression_metrics(yte, Xte[LAG_COL].to_numpy())

    # Resultados por observación
    res = kept[KEYS].copy()
    res["riesgo_abstencion_real"] = yte.to_numpy()
    res["prediccion_modelo"] = pred
    res["error"] = res["prediccion_modelo"] - res["riesgo_abstencion_real"]
    res["error_abs"] = res["error"].abs()

    # Diagnóstico de errores (no es un ranking político)
    q = res["error_abs"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    por_estado = (res.groupby("EDOCVE")
                     .agg(n=("error", "size"), error_medio=("error", "mean"),
                          MAE=("error_abs", "mean"),
                          RMSE=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))))
                     .reset_index())
    n_mun = int(res[["EDOCVE", "MPIOCVE"]].drop_duplicates().shape[0])
    diag_err = {
        "municipios_evaluados": n_mun,
        "error_medio_sesgo": float(res["error"].mean()),
        "error_abs_cuantiles": {str(k): float(v) for k, v in q.items()},
        "error_abs_max": float(res["error_abs"].max()),
        "baseline_naive_lag": base_naive,
    }

    shap_dir = out_dir / "modelo_abstencion" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # ---- SHAP sobre TEST 2024 únicamente
    sv, base_value, shap_method = compute_shap(model, algo, Xte)
    check(sv.shape == Xte.shape, f"SHAP: forma {sv.shape} == X {Xte.shape}")
    check(np.isfinite(sv).all(), "SHAP: valores finitos")
    recon = base_value + sv.sum(axis=1)
    add_err = float(np.max(np.abs(recon - pred)))
    log.info("SHAP aditividad: max|base+sum(shap)-pred| = %.3e", add_err)
    if add_err > 1e-3:
        log.warning("Aditividad SHAP con error %.3e (> 1e-3). Revisar compatibilidad "
                    "shap/xgboost.", add_err)

    shap_df = pd.DataFrame(sv, columns=[f"shap_{c}" for c in FEATURES])
    shap_out = pd.concat([res[KEYS].reset_index(drop=True), Xte, shap_df], axis=1)
    shap_out["shap_base_value"] = base_value
    shap_out["prediccion_modelo"] = pred
    shap_out.to_parquet(shap_dir / "shap_values_test_2024.parquet", index=False)

    global_imp = (pd.DataFrame({"feature": FEATURES,
                                "mean_abs_shap": np.abs(sv).mean(axis=0)})
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True))
    global_imp.to_csv(shap_dir / "shap_global_mean_abs.csv", index=False)
    log.info("SHAP global (importancia predictiva del modelo, NO causalidad):\n%s",
             global_imp.to_string(index=False))

    # ---- 3 explicaciones locales deterministas: primeras 3 por (EDOCVE, MPIOCVE)
    order = res.sort_values(["EDOCVE", "MPIOCVE"], kind="mergesort").head(3).index
    locales = []
    for idx in order:
        contribs = []
        for j, f in enumerate(FEATURES):
            contribs.append({"feature": f, "valor_observado": float(Xte.loc[idx, f]),
                             "shap": float(sv[idx, j]),
                             "direccion": ("empuja la predicción hacia un valor MAYOR"
                                           if sv[idx, j] > 0 else
                                           "empuja la predicción hacia un valor MENOR"
                                           if sv[idx, j] < 0 else "sin contribución")})
        contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)
        for c in contribs:
            c["interpretacion"] = (
                f"Para este municipio, {c['feature']}={c['valor_observado']:.4f} "
                f"contribuyó {c['shap']:+.5f} a la predicción del modelo respecto al "
                f"valor base ({base_value:.4f}); {c['direccion']}. Es una contribución "
                f"a la predicción, no una relación causal.")
        locales.append({
            "EDOCVE": int(res.loc[idx, "EDOCVE"]), "MPIOCVE": int(res.loc[idx, "MPIOCVE"]),
            "AELEC": int(res.loc[idx, "AELEC"]),
            "riesgo_abstencion_real": float(res.loc[idx, "riesgo_abstencion_real"]),
            "prediccion_modelo": float(res.loc[idx, "prediccion_modelo"]),
            "error": float(res.loc[idx, "error"]),
            "valor_base_shap": float(base_value),
            "shap_por_feature": {f: float(sv[idx, j]) for j, f in enumerate(FEATURES)},
            "top3_contribuciones_abs": contribs[:3],
        })
    jdump({"criterio_seleccion": "primeros 3 municipios de TEST 2024 ordenados por (EDOCVE, MPIOCVE)",
           "explicaciones": locales}, shap_dir / "explicaciones_locales_3_municipios.json")

    # ---- Persistencia de resultados
    res.to_parquet(out_dir / "modelo_abstencion" / "predicciones_test_2024.parquet", index=False)
    por_estado.to_csv(out_dir / "modelo_abstencion" / "errores_por_estado_2024.csv", index=False)

    # ---- Gráficos (cada uno con propósito analítico)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        gi = global_imp.iloc[::-1]
        ax.barh(gi["feature"], gi["mean_abs_shap"])
        ax.set_xlabel("mean(|SHAP|) — impacto predictivo (no causal)")
        ax.set_title("Importancia global SHAP — TEST 2024")
        fig.tight_layout(); fig.savefig(shap_dir / "shap_global_bar.png", dpi=150); plt.close(fig)

        try:
            import shap
            plt.figure()
            shap.summary_plot(sv, Xte, show=False)
            plt.tight_layout(); plt.savefig(shap_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        except Exception as e:  # noqa: BLE001
            log.warning("No se generó el beeswarm: %s", e)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(res["riesgo_abstencion_real"], res["prediccion_modelo"], s=4, alpha=0.4)
        lo = float(min(res["riesgo_abstencion_real"].min(), res["prediccion_modelo"].min()))
        hi = float(max(res["riesgo_abstencion_real"].max(), res["prediccion_modelo"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Real 2024"); ax.set_ylabel("Predicción")
        ax.set_title("Predicción vs real — TEST 2024 (calibración)")
        fig.tight_layout(); fig.savefig(out_dir / "modelo_abstencion" / "pred_vs_real_2024.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(res["error"], bins=60)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("Error (pred − real)"); ax.set_ylabel("Municipios")
        ax.set_title("Distribución del error — TEST 2024 (sesgo y colas)")
        fig.tight_layout(); fig.savefig(out_dir / "modelo_abstencion" / "distribucion_errores_2024.png", dpi=150); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("Gráficos no generados: %s", e)

    shap_summary = {"metodo": shap_method, "valor_base": base_value,
                    "max_error_aditividad": add_err,
                    "global": global_imp.to_dict(orient="records"),
                    "ruta": str(shap_dir)}
    result = {"metricas_test_2024": m_test, "diagnostico_error": diag_err,
              "shap": shap_summary, "locales": locales}
    return result, diag_err


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def stage_train(df, out_dir: Path, diag: dict, input_path: Path) -> dict:
    train, valid, test = temporal_split(df, diag)
    n_test_raw = int(len(test))
    # Desde aquí 2024 NO se usa en ninguna decisión: se descarta de memoria.
    del test

    tr_k, Xtr, ytr = prepare_xy(train, "TRAIN", diag)
    va_k, Xva, yva = prepare_xy(valid, "VALIDATION", diag)

    n_tr, n_va = len(Xtr), len(Xva)
    n_tot = n_tr + n_va + n_test_raw
    props = {"n_train_filas_originales": int(len(train)), "n_validation_filas_originales": int(len(valid)),
             "n_test_filas_originales": n_test_raw,
             "n_train_usadas": n_tr, "n_validation_usadas": n_va,
             "proporcion_train": n_tr / n_tot, "proporcion_validation": n_va / n_tot,
             "proporcion_test": n_test_raw / n_tot}
    log.info("Tamaños/proporciones: %s", json.dumps(props))

    model, sel = train_and_select(Xtr, ytr, Xva, yva)
    m_tr = regression_metrics(ytr, model.predict(Xtr))
    m_va = regression_metrics(yva, model.predict(Xva))
    log.info("Diagnóstico TRAIN: %s", m_tr)
    log.info("Diagnóstico VALIDATION: %s", m_va)
    base_va = regression_metrics(yva, Xva[LAG_COL].to_numpy())

    model_dir = out_dir / "modelo_abstencion"
    model_path = save_model(model, sel, model_dir)
    digest = sha256(model_path)
    log.info("MODELO CONGELADO: %s  sha256=%s", model_path, digest)

    frozen = {
        "fecha_hora_congelado": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(input_path), "dataset_sha256": sha256(input_path),
        "target": TARGET, "features": FEATURES,
        "particion_temporal": {"train": f"AELEC <= {YEAR_TRAIN_MAX}",
                               "validation": f"AELEC == {YEAR_VALID}",
                               "test": f"AELEC == {YEAR_TEST}"},
        "random_state": RANDOM_STATE, "hardware": gpu_decision(),
        "algoritmo": sel["seleccionado"]["algoritmo"],
        "hiperparametros": sel["seleccionado"]["params"],
        "best_iteration": sel["seleccionado"].get("best_iteration"),
        "seleccion": sel, "tamanos": props,
        "politica_nulos": diag.get("politica_nulos"),
        "metricas_diagnostico": {"train": m_tr, "validation": m_va,
                                 "baseline_naive_lag_validation": base_va},
        "ruta_modelo": str(model_path), "modelo_sha256": digest,
        "versiones": lib_versions(),
        "nota": "Modelo congelado ANTES de evaluar 2024. No se re-ajusta tras ver el test.",
    }
    jdump(frozen, model_dir / "modelo_congelado.json")
    return frozen


def stage_evaluate(df, out_dir: Path, diag: dict) -> dict:
    model_dir = out_dir / "modelo_abstencion"
    fz_path = model_dir / "modelo_congelado.json"
    check(fz_path.exists(), f"Existe el registro del modelo congelado {fz_path} (ejecutar --stage train primero)")
    frozen = json.loads(fz_path.read_text(encoding="utf-8"))
    mp = Path(frozen["ruta_modelo"])
    check(mp.exists(), f"Existe el archivo del modelo {mp}")
    check(sha256(mp) == frozen["modelo_sha256"],
          "El hash del modelo coincide con el congelado (no fue modificado tras ver 2024)")
    check(frozen["features"] == FEATURES and frozen["target"] == TARGET,
          "Features/target del modelo congelado coinciden con la configuración actual")

    sel = frozen["seleccion"]
    model = load_model(mp, sel["seleccionado"]["algoritmo"])
    test = df[df["AELEC"] == YEAR_TEST].copy()
    check(test["AELEC"].nunique() == 1 and int(test["AELEC"].iloc[0]) == YEAR_TEST, "TEST es 2024")
    diag.setdefault("politica_nulos", {})

    result, _ = evaluate_and_explain(test, model, sel, out_dir, model_dir, diag)
    return {"frozen": frozen, "result": result}


def default_input_path(project: Path) -> Path:
    salida = project / "data" / "Salida"
    candidatos = sorted(salida.glob("fase2_*/features_municipales.parquet"))
    if candidatos:
        return candidatos[-1]
    return salida / "fase2_<timestamp>" / "features_municipales.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 3 — Modelo predictivo + SHAP")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None, help="Directorio 'salida'")
    ap.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else base / "salida"
    input_path = Path(args.input) if args.input else default_input_path(base.parent)
    setup_logging(out_dir)
    set_seeds(RANDOM_STATE)
    started = datetime.now()
    log.info("##### FASE 3 | stage=%s | input=%s #####", args.stage, input_path)

    estado, error_msg = "PASSED", None
    registro: dict = {}
    try:
        df, diag = load_and_diagnose(input_path, out_dir)
        lag_audit = audit_lag(df)
        diag["auditoria_lag"] = lag_audit
        if diag["anios_inesperados"] or lag_audit.get("warning_definicion_lag"):
            estado = "WARNING"

        frozen = result = None
        if args.stage in ("all", "train"):
            frozen = stage_train(df, out_dir, diag, input_path)
        if args.stage in ("all", "evaluate"):
            ev = stage_evaluate(df, out_dir, diag)
            frozen, result = ev["frozen"], ev["result"]

        mdir = out_dir / "modelo_abstencion"
        if result is not None:
            m = result["metricas_test_2024"]
            if m["R2"] is not None and m["R2"] < 0:
                estado = "WARNING"
                log.warning("R² de TEST 2024 negativo: el modelo es peor que predecir la "
                            "media de 2024 (posible cambio de nivel entre períodos).")
            if result["shap"]["max_error_aditividad"] > 1e-3:
                estado = "WARNING"
            metricas = {
                "test_2024": m,
                "diagnostico_error_2024": result["diagnostico_error"],
                "validation_2021": frozen["metricas_diagnostico"]["validation"],
                "train": frozen["metricas_diagnostico"]["train"],
                "baseline_naive_lag_validation": frozen["metricas_diagnostico"]["baseline_naive_lag_validation"],
                "nota": "Métrica principal: TEST 2024 (hold-out). TRAIN/VALIDATION solo diagnóstico.",
            }
            jdump(metricas, mdir / "metricas.json")

        if frozen is not None:
            meta = {k: frozen[k] for k in
                    ("dataset", "dataset_sha256", "target", "features", "particion_temporal",
                     "random_state", "hardware", "algoritmo", "hiperparametros",
                     "best_iteration", "tamanos", "politica_nulos", "ruta_modelo",
                     "modelo_sha256", "versiones", "fecha_hora_congelado")}
            meta["seleccion_modelo"] = frozen["seleccion"]
            meta["auditoria_lag"] = lag_audit
            meta["diagnostico_dataset"] = {k: diag[k] for k in diag if k != "politica_nulos"}
            meta["ruta_shap"] = str(mdir / "shap") if result is not None else None
            meta["advertencia_metodologica"] = (
                "Modelo predictivo + explicabilidad. SHAP describe contribuciones a la "
                "predicción; NO demuestra causalidad.")
            jdump(meta, mdir / "metadata.json")

        registro = {
            "fecha_hora": started.isoformat(timespec="seconds"),
            "dataset": str(input_path),
            "filas_train": frozen["tamanos"]["n_train_usadas"] if frozen else None,
            "filas_validation": frozen["tamanos"]["n_validation_usadas"] if frozen else None,
            "filas_test": result["metricas_test_2024"]["n"] if result else None,
            "features": FEATURES, "target": TARGET,
            "algoritmo": frozen["algoritmo"] if frozen else None,
            "hiperparametros": frozen["hiperparametros"] if frozen else None,
            "random_state": RANDOM_STATE,
            "MAE": result["metricas_test_2024"]["MAE"] if result else None,
            "RMSE": result["metricas_test_2024"]["RMSE"] if result else None,
            "versiones": lib_versions(),
            "ruta_modelo": frozen["ruta_modelo"] if frozen else None,
            "ruta_shap": str(mdir / "shap") if result else None,
            "estado": estado,
        }
    except PipelineError as e:
        estado, error_msg = "FAILED", str(e)
        log.critical("PIPELINE DETENIDO: %s", e)
    except Exception as e:  # noqa: BLE001
        estado, error_msg = "CRITICAL", f"{type(e).__name__}: {e}"
        log.exception("ERROR INESPERADO")

    registro["estado"] = estado
    if error_msg:
        registro["error"] = error_msg
    registro.setdefault("fecha_hora", started.isoformat(timespec="seconds"))
    reg_path = out_dir / "modelo_abstencion" / "registro_experimentos.jsonl"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")

    log.info("##### ESTADO FINAL: %s #####", estado)
    return 0 if estado in ("PASSED", "WARNING") else 1


if __name__ == "__main__":
    sys.exit(main())