#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 4 — Laboratorio de escenarios y simulación contrafactual
=============================================================

Pregunta: ¿cuánto podría bajar la abstención municipal (2024) si la participación
juvenil aumentara 10 % relativo (con tope 1.0), manteniendo constante la
participación adulta?

ALCANCE: ejercicio CONDICIONAL, no causal. Se reporta una banda:
* escenario contable: identidad LN = SV + NV + NS, con voto adulto constante;
* escenario asociativo: asociación histórica (OLS ponderado por LN, corte 2024);
* punto medio: solo referencia de comunicación, no estimación identificada.

El modelo predictivo de Fase 3 se usa para el riesgo base del mapa; el impacto
del escenario es un ajuste contable/asociativo, no una feature contemporánea
del modelo (evita la fuga corregida en Fase 3).

Ejecutar desde la raíz de Geocivica_Predictiva:

    python scripts/04_laboratorio_escenarios.py
    python scripts/04_laboratorio_escenarios.py --dataset data/Salida/fase2_<timestamp>/features_municipales.parquet

Rutas (relativas a la raíz del proyecto):
    entrada dataset : data/Salida/fase2_<timestamp>/features_municipales.parquet
    entrada modelo  : scripts/salida/modelo_abstencion/    (artefacto de Fase 3)
    salidas         : scripts/salida/escenarios_finales_2024.csv
                      scripts/salida/escenario_2024.json
                      scripts/salida/logs/laboratorio_escenarios.log
    NOTA: `data/Salida/` (Fases 1-2) y `scripts/salida/` (Fases 3-5) son directorios distintos.

Reglas de seguridad del experimento
-----------------------------------
* NO reentrena ni reconstruye el modelo. Si no hay artefacto -> CRITICAL.
* El escenario contable solo modifica votos juveniles; el voto adulto queda fijo.
* El escenario asociativo se estima en el corte 2024 y se documenta como asociación.
* Sin aleatoriedad (municipios ordenados por EDOCVE, MPIOCVE).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Rutas (derivadas de la ubicación del script = Geocivica_Predictiva/scripts/)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "salida"                      # scripts/salida
MODEL_DIR = OUT_DIR / "modelo_abstencion"
LOG_DIR = OUT_DIR / "logs"
HIST_DIR = OUT_DIR / "escenarios_historial"


def latest_dataset(project: Path) -> Path:
    salida = project / "data" / "Salida"
    candidatos = sorted(salida.glob("fase2_*/features_municipales.parquet"))
    if candidatos:
        return candidatos[-1]
    return salida / "fase2_<timestamp>" / "features_municipales.parquet"


DEFAULT_DATASET = latest_dataset(PROJECT_DIR)
CSV_PATH = OUT_DIR / "escenarios_finales_2024.csv"
JSON_PATH = OUT_DIR / "escenario_2024.json"
INDEX_PATH = OUT_DIR / "runs_index_escenarios.jsonl"
LOG_PATH = LOG_DIR / "laboratorio_escenarios.log"


def rel(p: Path) -> str:
    """Ruta relativa a scripts/ (nunca absoluta) para logs y reportes."""
    try:
        return Path(os.path.relpath(Path(p).resolve(), SCRIPT_DIR)).as_posix()
    except ValueError:
        return Path(p).name


# --------------------------------------------------------------------------
# Configuración del experimento
# --------------------------------------------------------------------------
YEAR = 2024
INTERVENED = "tasa_voto_jovenes"
MULTIPLIER = 1.10
CAP = 1.0
KEYS = ["AELEC", "EDOCVE", "MPIOCVE"]
TARGET = "riesgo_abstencion"
ACCOUNTING_COLS = [TARGET, INTERVENED, "lista_nominal_total", "lista_nominal_jovenes",
                   "votos_totales", "votos_jovenes"]
REQUIRED = KEYS + ACCOUNTING_COLS
EPS = 1e-9                      # tolerancia numérica para "impacto = 0"
SCENARIO_LABEL = ("Aumento de 10 % relativo en la participacion juvenil con voto adulto "
                  "constante; banda contable-asociativa, NO causal.")

log = logging.getLogger("fase4")
VALIDATIONS: dict[str, str] = {}


class ScenarioError(RuntimeError):
    """Condición crítica. `status` ∈ {CRITICAL, FAILED}."""

    def __init__(self, msg: str, status: str = "CRITICAL"):
        super().__init__(msg)
        self.status = status


class _CountHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.warnings = 0

    def emit(self, record):
        if record.levelno == logging.WARNING:
            self.warnings += 1


COUNTER = _CountHandler()


def check(cond: bool, msg: str, key: str | None = None, status: str = "CRITICAL") -> None:
    """Validación fail-fast (no depende de `assert`, que se desactiva con -O)."""
    if key:
        VALIDATIONS[key] = "OK" if cond else "FALLÓ"
    if not cond:
        log.critical("VALIDACIÓN FALLIDA: %s", msg)
        raise ScenarioError(msg, status)
    log.info("OK  | %s", msg)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    COUNTER.warnings = 0
    log.addHandler(COUNTER)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("No se pudo leer %s: %s", rel(path), e)
        return None


def last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception as e:  # noqa: BLE001
        log.warning("No se pudo leer %s: %s", rel(path), e)
        return None


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.isoformat()
    return str(o)


def write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault),
                   encoding="utf-8")
    tmp.replace(path)


def archive_previous(path: Path) -> None:
    """No sobrescribe silenciosamente: mueve la versión previa a escenarios_historial/."""
    if path.exists():
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        dest = HIST_DIR / f"{path.stem}_{stamp}{path.suffix}"
        n = 1
        while dest.exists():
            dest = HIST_DIR / f"{path.stem}_{stamp}_{n}{path.suffix}"
            n += 1
        shutil.move(str(path), str(dest))
        log.info("Resultado previo archivado en %s", rel(dest))


def lib_versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "pandas": pd.__version__}
    for name in ("pyarrow", "xgboost", "sklearn", "joblib"):
        try:
            mod = __import__(name)
            v[name] = getattr(mod, "__version__", "desconocida")
        except Exception:  # noqa: BLE001
            v[name] = None
    return v


def desc(s: pd.Series) -> dict:
    return {"promedio": float(s.mean()), "mediana": float(s.median()),
            "minimo": float(s.min()), "maximo": float(s.max()),
            "desviacion_estandar": float(s.std(ddof=1)) if len(s) > 1 else 0.0}


def same_cols(a: pd.Series, b: pd.Series) -> bool:
    return bool(((a == b) | (a.isna() & b.isna())).all())


# --------------------------------------------------------------------------
# 1. Modelo de Fase 3
# --------------------------------------------------------------------------
def locate_and_load_model() -> tuple[object, dict]:
    log.info("=== 1. LOCALIZACIÓN DEL MODELO DE FASE 3 ===")
    check(MODEL_DIR.exists() and MODEL_DIR.is_dir(),
          f"Existe el directorio de Fase 3 {rel(MODEL_DIR)}", "modelo_correcto")
    contenido = sorted(p.name for p in MODEL_DIR.iterdir())
    log.info("Contenido de %s: %s", rel(MODEL_DIR), contenido)

    frozen = read_json(MODEL_DIR / "modelo_congelado.json")
    meta = read_json(MODEL_DIR / "metadata.json")
    reg = last_jsonl(MODEL_DIR / "registro_experimentos.jsonl")
    info = frozen or meta or {}
    algo = info.get("algoritmo") or (reg or {}).get("algoritmo")
    feats = info.get("features") or (reg or {}).get("features")
    log.info("Registro Fase 3 -> algoritmo=%s | features=%s | estado=%s",
             algo, feats, (reg or {}).get("estado"))

    if reg is not None and reg.get("estado") in ("FAILED", "CRITICAL"):
        VALIDATIONS["modelo_correcto"] = "FALLÓ"
        raise ScenarioError(
            f"El último experimento de Fase 3 terminó en {reg.get('estado')}: "
            "no hay un modelo final válido.")

    candidatos = [p for p in MODEL_DIR.glob("modelo.*")
                  if p.suffix in (".json", ".ubj", ".joblib", ".pkl")]
    if not candidatos:
        VALIDATIONS["modelo_correcto"] = "FALLÓ"
        raise ScenarioError(
            "Fase 3 NO dejó un artefacto de modelo (modelo.json / modelo.joblib) en "
            f"{rel(MODEL_DIR)}. Fase 4 requiere el modelo de Fase 3; NO se reentrena. "
            "Reejecuta `03_modelo_explicabilidad.py` y vuelve a lanzar esta fase.")

    elegido = None
    ruta_reg = (info.get("ruta_modelo") or (reg or {}).get("ruta_modelo"))
    if ruta_reg:  # solo el nombre de archivo: la ruta registrada puede ser absoluta/de otro equipo
        nombre = Path(str(ruta_reg).replace("\\", "/")).name
        elegido = next((p for p in candidatos if p.name == nombre), None)
    if elegido is None and algo:
        suf = ".json" if "XGB" in str(algo).upper() else ".joblib"
        elegido = next((p for p in candidatos if p.suffix == suf), None)
    if elegido is None and len(candidatos) == 1:
        elegido = candidatos[0]
    check(elegido is not None,
          f"Modelo final identificado sin ambigüedad (candidatos={[p.name for p in candidatos]})",
          "modelo_correcto")
    log.info("Modelo detectado: %s", rel(elegido))

    sha_ok = None
    esperado = (frozen or {}).get("modelo_sha256") or (meta or {}).get("modelo_sha256")
    sha = sha256(elegido)
    if esperado:
        sha_ok = (sha == esperado)
        check(sha_ok, "El SHA-256 del modelo coincide con el congelado en Fase 3",
              "modelo_correcto")
    else:
        log.warning("Fase 3 no registró SHA-256 del modelo; no se pudo verificar integridad.")

    if elegido.suffix in (".json", ".ubj"):
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(elegido))          # restaura el modelo sklearn-API de Fase 3
        algoritmo = "XGBRegressor"
        names = model.get_booster().feature_names
    else:
        import joblib
        model = joblib.load(elegido)
        algoritmo = type(model).__name__
        names = list(getattr(model, "feature_names_in_", []) or [])
    log.info("Algoritmo cargado: %s | features del modelo: %s", algoritmo, names)

    if algo and algoritmo.replace("Regressor", "").lower() not in str(algo).lower() \
            and str(algo).replace("Regressor", "").lower() not in algoritmo.lower():
        VALIDATIONS["modelo_correcto"] = "FALLÓ"
        raise ScenarioError(f"Algoritmo cargado ({algoritmo}) ≠ registrado en Fase 3 ({algo}).")

    # Features que espera el modelo (se respeta SU orden)
    model_feats = list(names) if names else list(feats or [])
    if not names:
        log.warning("El modelo no expone nombres de features; se usa el orden registrado en Fase 3.")
    check(bool(model_feats), "El modelo declara sus features", "features_compatibles")
    if feats:
        check(list(feats) == model_feats or set(feats) == set(model_feats),
              "Features del modelo coinciden con las registradas en Fase 3", "features_compatibles")
    check(TARGET not in model_feats and not any(k in model_feats for k in KEYS),
          "Ni el target ni las llaves (AELEC/EDOCVE/MPIOCVE) son predictores",
          "features_compatibles")
    check(INTERVENED not in model_feats,
          f"{INTERVENED} (variable contemporánea) NO es feature del modelo; el escenario se calcula aparte",
          "features_compatibles")
    VALIDATIONS.setdefault("modelo_correcto", "OK")

    return model, {"archivo": rel(elegido), "sha256": sha, "algoritmo": algoritmo,
                   "features": model_feats, "registro_fase3_estado": (reg or {}).get("estado"),
                   "random_state_fase3": info.get("random_state")}


# --------------------------------------------------------------------------
# 2. Datos 2024
# --------------------------------------------------------------------------
def load_2024(path: Path, model_feats: list[str]) -> tuple[pd.DataFrame, dict]:
    log.info("=== 2. DATASET ===")
    log.info("Ruta del dataset: %s", rel(path))
    check(path.exists() and path.is_file(), f"Existe el dataset {rel(path)}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        raise ScenarioError(f"No se pudo leer el parquet: {e}") from e
    log.info("Filas=%d | columnas=%d | ejercicios=%s", len(df), df.shape[1],
             sorted(pd.Series(df["AELEC"]).dropna().unique().tolist()) if "AELEC" in df else "N/D")
    cols = list(dict.fromkeys(REQUIRED + list(model_feats)))
    faltan = [c for c in cols if c not in df.columns]
    check(not faltan, f"Columnas requeridas presentes (faltan: {faltan})")

    for c in KEYS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c in [TARGET] + model_feats + [INTERVENED, "lista_nominal_total",
                                       "lista_nominal_jovenes", "votos_totales", "votos_jovenes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    test_df = df[df["AELEC"] == YEAR].copy()
    check(len(test_df) > 0, f"Existen registros de {YEAR}", "solo_2024")
    check(test_df["AELEC"].nunique() == 1 and int(test_df["AELEC"].iloc[0]) == YEAR,
          f"El filtro contiene únicamente {YEAR}", "solo_2024")
    check(not test_df.duplicated(subset=KEYS).any(), "Sin duplicidades municipio/año en 2024")

    n_total = len(test_df)
    num_cols = list(dict.fromkeys([TARGET] + model_feats + [INTERVENED, "lista_nominal_total",
                                                            "lista_nominal_jovenes", "votos_totales", "votos_jovenes"]))
    inf_cols = {c: int(np.isinf(test_df[c]).sum()) for c in num_cols}
    check(sum(inf_cols.values()) == 0, f"Sin infinitos en 2024 {inf_cols}", "sin_nan_inf")

    nulos_feat = test_df[model_feats].isna().any(axis=1)
    nulos_cont = test_df[[INTERVENED, "lista_nominal_total", "lista_nominal_jovenes",
                          "votos_totales", "votos_jovenes"]].isna().any(axis=1)
    nulos = nulos_feat | nulos_cont
    n_excl = int(nulos.sum())
    if n_excl:
        log.warning("%d de %d municipios 2024 (%.2f%%) tienen features nulas y se excluyen "
                    "(misma política que Fase 3; no se imputa).", n_excl, n_total,
                    100 * n_excl / n_total)
    test_df = test_df.loc[~nulos].copy()
    check(len(test_df) > 0, "Quedan municipios 2024 con features completas")
    n_tgt_null = int(test_df[TARGET].isna().sum())
    if n_tgt_null:
        log.warning("%d municipios 2024 tienen riesgo_abstencion nulo (se conservan, "
                    "pero sin valor observado).", n_tgt_null)

    test_df = (test_df.sort_values(["EDOCVE", "MPIOCVE"], kind="mergesort")
                      .reset_index(drop=True))
    n_mun = int(test_df[["EDOCVE", "MPIOCVE"]].drop_duplicates().shape[0])
    log.info("Municipios 2024 en el escenario: %d (de %d; excluidos=%d)", n_mun, n_total, n_excl)
    return test_df, {"archivo": rel(path), "sha256": sha256(path), "anio": YEAR,
                     "municipios_2024_totales": n_total, "municipios_excluidos_features_nulas": n_excl,
                     "municipios": n_mun, "riesgo_abstencion_nulos": n_tgt_null,
                     "features_modelo": list(model_feats)}


# --------------------------------------------------------------------------
# 3. Escenarios
# --------------------------------------------------------------------------
def fit_association(df: pd.DataFrame) -> dict:
    """OLS ponderado por LN en el corte 2024: riesgo ~ tasa_voto_jovenes.

    Es una ASOCIACIÓN descriptiva, no un efecto causal. Se usa solo para
    construir el extremo superior de la banda de sensibilidad.
    """
    x = df[INTERVENED].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    w = df["lista_nominal_total"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    sw = w / w.sum()
    XtWX = X.T @ (X * sw[:, None])
    XtWy = X.T @ (sw * y)
    beta = np.linalg.solve(XtWX, XtWy)
    pred = X @ beta
    ybar = float(np.average(y, weights=w))
    ss_res = float(np.sum(sw * (y - pred) ** 2))
    ss_tot = float(np.sum(sw * (y - ybar) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    log.info("Asociación 2024 (OLS ponderado por LN): riesgo = %.5f + (%.5f) × tasa_voto_jovenes | R2=%.4f",
             beta[0], beta[1], r2 if r2 is not None else float("nan"))
    check(np.isfinite(beta).all(), "Coeficientes de asociación finitos", "asociacion_valida")
    return {"metodo": "OLS ponderado por lista nominal, corte 2024",
            "intercepto": float(beta[0]), "beta_tasa_voto_jovenes": float(beta[1]),
            "r2_ponderado": None if r2 is None else float(r2), "n": int(len(df)),
            "advertencia": "Asociación descriptiva; no identifica un efecto causal."}


def build_scenarios(df24: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    log.info("=== 3. ESCENARIOS BASE, CONTABLE Y ASOCIATIVO ===")
    ln = df24["lista_nominal_total"].astype(float)
    lnj = df24["lista_nominal_jovenes"].astype(float)
    sv = df24["votos_totales"].astype(float)
    svj = df24["votos_jovenes"].astype(float)
    base_risk = df24[TARGET].astype(float)
    base_x = df24[INTERVENED].astype(float)

    check(bool((ln > 0).all()), "Lista nominal total positiva", "ln_positiva")
    check(bool((lnj <= ln).all()), "Lista nominal juvenil <= total", "lnj_le_ln")
    check(bool((svj <= lnj).all()) and bool(((sv - svj) >= -1e-9).all()),
          "Votos juveniles <= LN juvenil y voto adulto no negativo", "votos_coherentes")
    check(base_x.between(0, 1).all(),
          f"La tasa base {INTERVENED} está en [0, 1] (min={base_x.min():.4f}, max={base_x.max():.4f})")

    sim_svj = np.minimum(svj.to_numpy() * MULTIPLIER, lnj.to_numpy())
    adult_v = sv.to_numpy() - svj.to_numpy()
    sim_risk = (ln.to_numpy() - (adult_v + sim_svj)) / ln.to_numpy()
    impact_c = base_risk.to_numpy() - sim_risk
    sim_tasa = np.divide(sim_svj, lnj.to_numpy(), out=np.zeros_like(sim_svj),
                         where=lnj.to_numpy() > 0)
    limitados = (svj.to_numpy() * MULTIPLIER) > (lnj.to_numpy() + 1e-12)

    assoc = fit_association(df24)
    x_sim = np.minimum(base_x.to_numpy() * MULTIPLIER, CAP)
    impact_a = assoc["beta_tasa_voto_jovenes"] * (base_x.to_numpy() - x_sim)
    impact_mid = 0.5 * (impact_c + impact_a)

    check(np.isfinite(impact_c).all() and np.isfinite(impact_a).all() and np.isfinite(impact_mid).all(),
          "Impactos contable, asociativo y central finitos", "impactos_finitos")
    check(bool((sim_risk >= 0).all()) and bool((sim_risk <= 1).all()),
          "Riesgo contable simulado en [0,1]", "riesgo_simulado_valido")
    check(bool((base_x.to_numpy() <= x_sim + 1e-12).all()),
          "La tasa juvenil simulada no disminuye", "tasa_simulada_no_disminuye")

    out = df24.copy(deep=True)
    out["tasa_voto_jovenes_simulada"] = sim_tasa
    out["riesgo_contable_simulado"] = sim_risk
    out["impacto_contable"] = impact_c
    out["impacto_asociativo"] = impact_a
    out["impacto_central"] = impact_mid
    out["limitado_por_clip"] = limitados

    sat = {"municipios_modificados": int((sim_svj != svj.to_numpy()).sum()),
           "municipios_con_aumento_completo_10pct": int((~limitados & (svj.to_numpy() > 0)).sum()),
           "municipios_limitados_por_clip": int(limitados.sum()),
           "municipios_base_ya_en_1.0": int((base_x.to_numpy() >= CAP).sum()),
           "municipios_base_en_0_sin_cambio": int((base_x.to_numpy() == 0).sum())}
    stats = {"base": {**desc(base_x)}, "simulada": {**desc(pd.Series(sim_tasa))},
             "riesgo_base_observado": {**desc(base_risk)},
             "riesgo_contable_simulado": {**desc(pd.Series(sim_risk))},
             "saturacion": sat, "raw_max": float((svj.to_numpy() * MULTIPLIER).max()),
             "identidad": "riesgo = (LN_total - SV_total) / LN_total; voto adulto constante"}
    log.info("Tasa base      : media=%.5f min=%.5f max=%.5f", base_x.mean(), base_x.min(), base_x.max())
    log.info("Tasa simulada  : media=%.5f min=%.5f max=%.5f", sim_tasa.mean(), sim_tasa.min(), sim_tasa.max())
    log.info("Impacto contable (pp): ponderado=%.4f simple=%.4f | asociativo (pp): ponderado=%.4f simple=%.4f",
             100 * np.average(impact_c, weights=ln), 100 * impact_c.mean(),
             100 * np.average(impact_a, weights=ln), 100 * impact_a.mean())
    return out, stats, assoc


# --------------------------------------------------------------------------
# 4. Predicción e impacto
# --------------------------------------------------------------------------
def predict(model, X: pd.DataFrame, feats: list[str], name: str) -> np.ndarray:
    pred = np.asarray(model.predict(X[feats].astype("float64")), dtype=float)
    check(pred.shape[0] == len(X), f"{name}: dimensiones de la predicción ({pred.shape[0]}) == filas ({len(X)})")
    check(not np.isnan(pred).any() and not np.isinf(pred).any(), f"{name}: sin NaN/inf")
    return pred


def cross_check_fase3(res: pd.DataFrame) -> dict:
    """La predicción base debe reproducir la de Fase 3 sobre 2024 (mismo modelo, mismos datos)."""
    p = MODEL_DIR / "predicciones_test_2024.parquet"
    if not p.exists():
        log.warning("No existe %s; no se pudo contrastar la predicción base con Fase 3.", rel(p))
        return {"verificado": False}
    f3 = pd.read_parquet(p)
    need = KEYS + ["prediccion_modelo"]
    if any(c not in f3.columns for c in need):
        log.warning("%s no tiene las columnas esperadas; contraste omitido.", rel(p))
        return {"verificado": False}
    m = res[KEYS + ["prediccion_riesgo_base"]].merge(f3[need], on=KEYS, how="inner")
    check(len(m) > 0, "La predicción base se pudo cruzar con las predicciones de Fase 3",
          "modelo_correcto")
    diff = float((m["prediccion_riesgo_base"] - m["prediccion_modelo"]).abs().max())
    check(diff < 1e-5, f"Predicción base reproduce Fase 3 (max|delta|={diff:.2e}, n={len(m)})",
          "modelo_correcto")
    return {"verificado": True, "filas_cruzadas": int(len(m)), "max_abs_diff": diff}


def scenario_summary(res: pd.DataFrame, assoc: dict) -> dict:
    w = res["lista_nominal_total"].to_numpy(dtype=float)
    out: dict = {"metodo_asociacion": assoc, "escenario": SCENARIO_LABEL}
    for col in ("impacto_contable", "impacto_asociativo", "impacto_central", "impacto_intervencion"):
        s = res[col].to_numpy(dtype=float)
        out[col] = {"promedio_simple": float(s.mean()),
                    "promedio_ponderado_LN": float(np.average(s, weights=w)),
                    "mediana": float(np.median(s)),
                    "p10": float(np.quantile(s, 0.10)), "p90": float(np.quantile(s, 0.90))}
    c = out["impacto_contable"]["promedio_ponderado_LN"]
    a = out["impacto_asociativo"]["promedio_ponderado_LN"]
    out["banda_pp"] = {"limite_inferior": 100 * c, "limite_superior": 100 * a,
                       "punto_medio": 100 * (c + a) / 2}
    out["lectura"] = ("Banda condicional: el límite inferior es la identidad contable; el superior "
                      "es una asociación histórica. No es un efecto causal identificado.")
    return out


def run_prediction(model, minfo: dict, df_scen: pd.DataFrame):
    log.info("=== 4. PREDICCIÓN BASE + AJUSTE DEL ESCENARIO ===")
    feats = minfo["features"]
    pb = predict(model, df_scen, feats, "BASE")
    central = df_scen["impacto_central"].to_numpy(dtype=float)
    ps = np.clip(pb - central, 0.0, 1.0)

    keep = list(dict.fromkeys(KEYS + [TARGET, "lista_nominal_total", "tasa_jovenes_ln", INTERVENED,
                                      "lag_riesgo_abstencion_1", "tasa_voto_jovenes_simulada",
                                      "limitado_por_clip", "impacto_contable", "impacto_asociativo",
                                      "impacto_central"]))
    res = df_scen[keep].copy()
    res["prediccion_riesgo_base"] = pb
    res["prediccion_riesgo_simulado"] = ps
    res["impacto_intervencion"] = res["prediccion_riesgo_base"] - res["prediccion_riesgo_simulado"]
    check(np.isfinite(res["impacto_intervencion"]).all(), "impacto_intervencion finito",
          "sin_nan_inf")
    check(res.shape[0] == len(df_scen), "Mismo número de municipios en base y simulado",
          "predicciones_validas")
    VALIDATIONS["predicciones_validas"] = "OK"
    VALIDATIONS.setdefault("sin_nan_inf", "OK")
    res["error_base_vs_observado"] = res["prediccion_riesgo_base"] - res[TARGET]
    cross = cross_check_fase3(res)
    return res, cross


def analyze(res: pd.DataFrame) -> dict:
    imp = res["impacto_intervencion"]
    w = res["lista_nominal_total"].to_numpy(dtype=float)
    out = {"impacto": desc(imp),
           "reduccion_promedio_riesgo": float(imp.mean()),
           "reduccion_promedio_ponderada_LN": float(np.average(imp.to_numpy(dtype=float), weights=w)),
           "reduccion_mediana_riesgo": float(imp.median()),
           "prediccion_base": desc(res["prediccion_riesgo_base"]),
           "prediccion_simulada": desc(res["prediccion_riesgo_simulado"]),
           "riesgo_observado_promedio": float(res[TARGET].mean()) if res[TARGET].notna().any() else None,
           "riesgo_observado_ponderado_LN": float(np.average(res[TARGET].to_numpy(dtype=float), weights=w))}
    # Impacto según saturación (para interpretar el tope 1.0)
    grp = np.where(res["limitado_por_clip"], "limitado_por_clip", "aumento_completo_o_sin_cambio")
    out["impacto_por_saturacion"] = {
        k: {"n": int(len(g)), "impacto_medio": float(g["impacto_intervencion"].mean())}
        for k, g in res.groupby(grp)}
    # Extremos deterministas (empates resueltos por llaves) — diagnóstico, no ranking político
    ordered = res.sort_values(["impacto_intervencion", "EDOCVE", "MPIOCVE"], kind="mergesort")
    cols = ["EDOCVE", "MPIOCVE", "tasa_voto_jovenes", "impacto_contable", "impacto_asociativo",
            "impacto_central", "impacto_intervencion"]
    out["menor_impacto_top5"] = ordered.head(5)[cols].to_dict(orient="records")
    out["mayor_impacto_top5"] = ordered.tail(5).iloc[::-1][cols].to_dict(orient="records")
    return out


# --------------------------------------------------------------------------
# 5. Persistencia
# --------------------------------------------------------------------------
def save_outputs(res: pd.DataFrame, report: dict) -> None:
    log.info("=== 5. GUARDADO ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_previous(CSV_PATH)
    archive_previous(JSON_PATH)

    cols = ["EDOCVE", "MPIOCVE", "AELEC", TARGET, "lista_nominal_total", "tasa_jovenes_ln",
            INTERVENED, "tasa_voto_jovenes_simulada", "lag_riesgo_abstencion_1",
            "prediccion_riesgo_base", "prediccion_riesgo_simulado", "impacto_intervencion",
            "impacto_contable", "impacto_asociativo", "impacto_central",
            "limitado_por_clip", "error_base_vs_observado"]
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    res[cols].to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(CSV_PATH)
    VALIDATIONS["csv_generado"] = "OK"

    report["validaciones"] = dict(VALIDATIONS)
    report["archivos"] = {"csv": rel(CSV_PATH), "reporte": rel(JSON_PATH), "log": rel(LOG_PATH)}
    write_json_atomic(report, JSON_PATH)
    VALIDATIONS["reporte_generado"] = "OK"

    # Relectura de verificación
    back = pd.read_csv(CSV_PATH)
    check(len(back) == len(res) and set(cols) <= set(back.columns),
          f"CSV releído correctamente ({len(back)} filas, {back.shape[1]} columnas)")
    check(np.allclose(back["impacto_intervencion"].to_numpy(),
                      res["impacto_intervencion"].to_numpy(), atol=1e-12, equal_nan=True),
          "El impacto del CSV coincide con el calculado")
    rj = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    check(rj.get("estado") == report["estado"], "JSON releído correctamente")
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": report["run_id"], "fecha_hora": report["fecha_hora"],
                            "estado": report["estado"], "impacto_medio": report["impacto_medio"],
                            "csv": rel(CSV_PATH), "reporte": rel(JSON_PATH)},
                           ensure_ascii=False, default=jdefault) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 4 — Laboratorio de escenarios (contrafactual del modelo)")
    ap.add_argument("--dataset", default=None,
                    help="Ruta relativa a la raíz del proyecto del features_municipales.parquet (por defecto la de Fase 2)")
    args = ap.parse_args()

    dataset = (PROJECT_DIR / args.dataset) if args.dataset else DEFAULT_DATASET
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = datetime.now().isoformat(timespec="seconds")
    setup_logging()
    log.info("##### FASE 4 | INICIO | run_id=%s #####", run_id)
    log.info("Regla: %s × %.2f, clip(upper=%.1f), año %d", INTERVENED, MULTIPLIER, CAP, YEAR)

    estado, error = "PASSED", None
    report: dict = {}
    try:
        model, minfo = locate_and_load_model()
        df24, dinfo = load_2024(dataset, minfo["features"])
        df_scen, sstats, assoc = build_scenarios(df24)
        res, cross = run_prediction(model, minfo, df_scen)
        an = analyze(res)
        summary = scenario_summary(res, assoc)
        summary["estado"] = "BANDA CONDICIONAL"

        i, b, s = an["impacto"], an["prediccion_base"], an["prediccion_simulada"]
        log.info("Base      : media=%.6f mediana=%.6f min=%.6f max=%.6f", b["promedio"], b["mediana"], b["minimo"], b["maximo"])
        log.info("Simulado  : media=%.6f mediana=%.6f min=%.6f max=%.6f", s["promedio"], s["mediana"], s["minimo"], s["maximo"])
        log.info("Impacto   : media=%.6f mediana=%.6f min=%.6f max=%.6f sd=%.6f", i["promedio"], i["mediana"], i["minimo"], i["maximo"], i["desviacion_estandar"])
        log.info("Banda nacional (ponderada por LN): contable=%.4f pp | asociativo=%.4f pp | medio=%.4f pp",
                 100 * summary["impacto_contable"]["promedio_ponderado_LN"],
                 100 * summary["impacto_asociativo"]["promedio_ponderado_LN"],
                 summary["banda_pp"]["punto_medio"])
        log.info("Promedio simple del ajuste central entre %d municipios 2024: %.4f pp",
                 len(res), 100 * an["reduccion_promedio_riesgo"])
        estado = "WARNING" if COUNTER.warnings else "PASSED"

        report = {
            "run_id": run_id, "fecha_hora": started,
            "modelo_utilizado": minfo, "dataset_utilizado": dinfo, "anio": YEAR,
            "feature_intervenida": INTERVENED, "multiplicador": MULTIPLIER, "limite_superior": CAP,
            "regla": f"{INTERVENED}_simulada = min({INTERVENED} × {MULTIPLIER}, {CAP}); voto adulto constante",
            "numero_municipios": int(len(res)),
            "media_base": b["promedio"], "media_simulada": s["promedio"],
            "impacto_medio": summary["impacto_central"]["promedio_ponderado_LN"],
            "impacto_medio_contable": summary["impacto_contable"]["promedio_ponderado_LN"],
            "impacto_medio_asociativo": summary["impacto_asociativo"]["promedio_ponderado_LN"],
            "banda_pp": summary["banda_pp"],
            "impacto_mediano": i["mediana"],
            "impacto_minimo": i["minimo"], "impacto_maximo": i["maximo"],
            "municipios_limitados_por_clip": sstats["saturacion"]["municipios_limitados_por_clip"],
            "tasa_voto_jovenes": {"base": sstats["base"], "simulada": sstats["simulada"],
                                  "saturacion": sstats["saturacion"]},
            "riesgo_base_observado": sstats["riesgo_base_observado"],
            "riesgo_contable_simulado": sstats["riesgo_contable_simulado"],
            "identidad_contable": sstats["identidad"],
            "analisis_impacto": an, "contraste_con_fase3": cross,
            "resumen_escenario": summary,
            "resultado_hipotesis": summary,
            "promedio_nacional_definicion": (
                "Promedio ponderado por lista nominal de los impactos condicionales entre los "
                "municipios 2024 incluidos. El límite inferior es la identidad contable; el superior "
                "es una asociación histórica. No es un efecto causal ni equivale a la reducción "
                "observada de la abstención nacional."),
            "advertencia": ("Ejercicio condicional: identidad contable + asociación histórica. "
                            "NO es una estimación causal de una campaña o política real."),
            "determinismo": "Sin aleatoriedad; municipios ordenados por (EDOCVE, MPIOCVE).",
            "versiones": lib_versions(), "estado": estado,
        }
        VALIDATIONS.setdefault("csv_generado", "PENDIENTE")
        save_outputs(res, report)

        print("\n" + "=" * 78)
        print("BANDA NACIONAL DE REDUCCIÓN DEL RIESGO (ponderada por lista nominal)")
        print(f"  contable  : {summary['banda_pp']['limite_inferior']:.4f} pp")
        print(f"  asociativo: {summary['banda_pp']['limite_superior']:.4f} pp")
        print(f"  punto medio: {summary['banda_pp']['punto_medio']:.4f} pp")
        print(f"  Municipios: {len(res)} | Estado: {estado}")
        print("  (condicional; no es una estimación causal real)")
        print("=" * 78)
    except ScenarioError as e:
        estado, error = e.status, str(e)
        log.critical("FASE 4 DETENIDA [%s]: %s", e.status, e)
    except Exception as e:  # noqa: BLE001
        estado, error = "CRITICAL", f"{type(e).__name__}: {e}"
        log.exception("ERROR INESPERADO")

    if error:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fail = OUT_DIR / f"escenario_2024_{estado}_{run_id}.json"   # no pisa un resultado válido previo
        write_json_atomic({"run_id": run_id, "fecha_hora": started, "estado": estado, "error": error,
                           "validaciones": dict(VALIDATIONS), "versiones": lib_versions()}, fail)
        log.info("Reporte de fallo: %s", rel(fail))

    log.info("##### FASE 4 | ESTADO FINAL: %s #####", estado)
    return 0 if estado in ("PASSED", "PASSED + HIPÓTESIS NO CONFIRMADA", "WARNING") else 1


if __name__ == "__main__":
    sys.exit(main())