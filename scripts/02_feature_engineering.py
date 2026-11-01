#!/usr/bin/env python3
"""
CCPC - Fase 2: Feature engineering y panel municipal (AELEC, EDOCVE, MPIOCVE).

USO
    python 02_feature_engineering.py --phase1-out-root Prueba_2/salida
    python 02_feature_engineering.py --input <ruta>/ine_dataset_limpio.parquet --out-root Prueba_2/salida
    (requiere: pandas, numpy, pyarrow)

ARQUITECTURA Y JUSTIFICACION
- Lectura en streaming por lotes (pyarrow iter_batches, solo 6 columnas) + agregacion parcial por lote
  (groupby-sum) consolidada periodicamente. Memoria acotada por el numero de municipios x ejercicios
  (~1e4 filas), NO por el tamano del dataset de entrada. No se justifica DuckDB/Polars: la unica
  operacion pesada es una suma agrupada y la tabla resultante es diminuta. GPU: no aplica.
- Sumas en int64 (exactas). Las tasas se calculan y almacenan en float64: el panel es pequeno
  (decenas de miles de filas) y float32 no ahorra nada relevante; la exactitud facilita las validaciones.
- Toda validacion es explicita (conteo, porcentaje, ejemplos, causa, decision), no `assert`.
- Conservacion: se comparan los totales del panel contra totales acumulados de forma independiente
  durante el streaming (sum LN, sum SV, sum LN/SV de jovenes).
- Lag: registro anterior DISPONIBLE del mismo municipio; se anaden `lag_aelec_1` y `lag_saltos_1`
  para auditar si el rezago es el ejercicio inmediato o salta ejercicios.
- La logica especifica de Parquet esta aislada en `open_source` / `write_table_file` / `read_table_file`.
  Con entrada .csv (solo pruebas) el estado maximo es WARNING y `parquet_verificado=false`.

DECISIONES SOBRE EL PROMPT (ver reporte): salida en carpeta por ejecucion (no sobrescribe);
`jovenes_ln_cero` como bandera de la imputacion 0; columnas auxiliares de trazabilidad.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"
KEY = ["AELEC", "EDOCVE", "MPIOCVE"]
REQUIRED = KEY + ["LN", "SV", "EDAD"]
NULLABLE_PANEL = (
    "tasa_voto_jovenes_obs",
    "lag_riesgo_abstencion_1", "lag_riesgo_abstencion_2",
    "lag_aelec_1", "lag_aelec_2", "lag_saltos_1",
    "lag_voto_jovenes_1", "lag_voto_jovenes_2", "lag_tasa_jovenes_ln_1",
    "hist_mean_riesgo", "hist_std_riesgo", "hist_delta_riesgo_1",
    "state_lag_mean", "gap_riesgo_estado",
)
JOVEN_MIN, JOVEN_MAX = 18, 29
PHASE1_PROVENANCE = ["SEXO_ORIGINAL", "DEL_IMPUTADO", "ARCHIVO_ORIGEN", "ANIO_ORIGEN", "FILA_ORIGEN"]
BATCH_ROWS = 1_000_000
REJECTED_LN_WARN_PCT = 0.5   # umbral informativo (no bloqueante)
OK_PHASE1 = ("COMPLETADA", "COMPLETADA CON OBSERVACIONES")
log = logging.getLogger("fase2")


class PipelineError(Exception):
    def __init__(self, code, msg):
        self.code = code
        super().__init__(msg)


# =============================== E/S (aislada) ===============================
def open_source(path: Path):
    """Devuelve (columnas[(nombre,tipo)], n_filas|None, iterador(columnas)->DataFrames)."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        cols = [(f.name, str(f.type)) for f in pf.schema_arrow]

        def it(columns):
            for b in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
                yield b.to_pandas()
        return cols, pf.metadata.num_rows, it
    cols = [(c, "csv") for c in pd.read_csv(path, nrows=0).columns]

    def it(columns):
        yield from pd.read_csv(path, usecols=columns, chunksize=BATCH_ROWS)
    return cols, None, it


def write_table_file(df: pd.DataFrame, path: Path):
    if path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path),
                       compression="zstd", compression_level=3)
    else:
        df.to_csv(path, index=False)


def read_table_file(path: Path):
    """Devuelve (DataFrame, [(col,tipo)] | None, n_filas_metadata | None)."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        return pf.read().to_pandas(), [(f.name, str(f.type)) for f in pf.schema_arrow], pf.metadata.num_rows
    return pd.read_csv(path, float_precision="round_trip"), None, None


# =============================== UTILIDADES ===============================
def to_int(s: pd.Series, name: str) -> np.ndarray:
    if s.isna().any():
        raise PipelineError("NULOS_EN_COLUMNA_CRITICA", f"{name}: {int(s.isna().sum())} nulos en un lote. "
                            f"Fase 1 deberia haberlos aislado; revise la entrada.")
    if pd.api.types.is_integer_dtype(s):
        return s.to_numpy(dtype="int64")
    num = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    bad = num.isna() | (num != np.floor(num.fillna(0)))
    if bad.any():
        raise PipelineError("VALORES_NO_ENTEROS", f"{name}: {int(bad.sum())} valores no enteros; "
                            f"ejemplos={s[bad].head(5).tolist()}")
    return num.to_numpy(dtype="int64")


def smallest_int(lo, hi) -> str:
    for n in ("int8", "int16", "int32", "int64"):
        i = np.iinfo(n)
        if i.min <= lo and hi <= i.max:
            return n
    raise PipelineError("RANGO_ENTERO", f"fuera de int64: [{lo},{hi}]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, Path)):
        return str(o)
    return str(o)


# =============================== PIPELINE ===============================
class Phase2:
    def __init__(self, a):
        self.a = a
        self.t0 = time.perf_counter()
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.out_root = Path(a.out_root).resolve()
        self.run_dir = self.out_root / f"fase2_{self.stamp}"
        self.issues, self.validations = [], []
        self.R = {"pipeline_version": VERSION, "timestamp": self.stamp}
        self.debug_input = False

    # ---- infraestructura ----
    LV = {"CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING, "INFO": logging.INFO}

    def setup_logging(self):
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=False)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        log.setLevel(logging.INFO)
        for h in (logging.FileHandler(self.run_dir / "logs" / "feature_engineering.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            log.addHandler(h)

    def issue(self, sev, code, msg):
        self.issues.append({"severidad": sev, "codigo": code, "mensaje": msg})
        log.log(self.LV[sev], "[%s] %s", code, msg)

    def validation(self, name, result, evidence):
        self.validations.append({"validacion": name, "resultado": result, "evidencia": evidence})
        lvl = {"PASSED": logging.INFO, "WARNING": logging.WARNING, "NOT_RUN": logging.WARNING}.get(result, logging.CRITICAL)
        log.log(lvl, "VALIDACION %-28s %-8s %s", name, result, evidence)

    # ---- 1. entrada ----
    def step_input(self):
        a = self.a
        if a.input:
            path = Path(a.input).resolve()
            p1_dir = path.parent
        else:
            idx = Path(a.phase1_out_root).resolve() / "runs_index.csv"
            if not idx.exists():
                raise PipelineError("SIN_INDICE_FASE1", f"No existe {idx}. Ejecute la Fase 1 o use --input.")
            rows = list(csv.DictReader(open(idx, encoding="utf-8")))
            valid = [r for r in rows if r["status"] in OK_PHASE1]
            if not valid:
                raise PipelineError("FASE1_NO_VALIDA", f"Ninguna ejecucion de Fase 1 con estado {OK_PHASE1}. "
                                    f"Ultima: {rows[-1] if rows else 'ninguna'}")
            p1_dir = Path(valid[-1]["run_dir"])
            cands = [p1_dir / "ine_dataset_limpio.parquet", p1_dir / "ine_dataset_limpio.debug.csv"]
            path = next((c for c in cands if c.exists()), None)
            if path is None:
                raise PipelineError("DATASET_FASE1_AUSENTE", f"No hay dataset limpio en {p1_dir}")
        if not path.is_file():
            raise PipelineError("ENTRADA_INEXISTENTE", f"No existe {path}")
        self.path, self.p1_dir = path, p1_dir
        self.debug_input = path.suffix.lower() != ".parquet"
        if self.debug_input:
            self.issue("WARNING", "ENTRADA_NO_PARQUET", "Entrada CSV de prueba: estado maximo WARNING.")
        self.R["entrada"] = {"ruta": str(path), "bytes": path.stat().st_size}
        log.info("Entrada: %s (%.1f MB)", path, path.stat().st_size / 1e6)
        try:
            self.cols, self.n_meta, self.iterate = open_source(path)
        except Exception as e:
            raise PipelineError("ENTRADA_ILEGIBLE", f"No se pudo abrir {path}: {e!r}")
        names = [c for c, _ in self.cols]
        log.info("Esquema (%d columnas): %s", len(names), self.cols)
        miss = [c for c in REQUIRED if c not in names]
        if miss:
            raise PipelineError("COLUMNAS_CRITICAS_AUSENTES", f"Faltan {miss} en {path}. Columnas: {names}")
        self.R["entrada"].update(columnas=self.cols, filas_metadata=self.n_meta)
        if a.hash_input:
            self.R["entrada"]["sha256"] = sha256_file(path)
        rep = p1_dir / "reporte_calidad.json"
        self.p1_report = json.load(open(rep, encoding="utf-8")) if rep.exists() else None
        if self.p1_report is None:
            self.issue("WARNING", "SIN_REPORTE_FASE1", f"No hay reporte_calidad.json en {p1_dir}; no se puede "
                       "reconciliar filas ni leer el estado de Fase 1.")
        else:
            log.info("Fase 1: estado=%s calidad=%s", self.p1_report.get("status"), self.p1_report.get("calidad"))

    # ---- 2. streaming: diagnostico + agregacion ----
    def step_stream(self):
        cols = REQUIRED
        parts, tot = [], Counter()
        edad_vals, aelec_vals = set(), set()
        rows = 0
        for df in self.iterate(cols):
            n = len(df)
            rows += n
            a, e, m = (to_int(df[c], c) for c in KEY)
            ln, sv, ed = (to_int(df[c], c) for c in ("LN", "SV", "EDAD"))
            tot["neg_ln"] += int((ln < 0).sum())
            tot["neg_sv"] += int((sv < 0).sum())
            tot["sv_gt_ln"] += int((sv > ln).sum())
            tot["ln_cero_filas"] += int((ln == 0).sum())
            edad_vals.update(np.unique(ed).tolist())
            aelec_vals.update(np.unique(a).tolist())
            j = (ed >= JOVEN_MIN) & (ed <= JOVEN_MAX)
            tot["ln"] += int(ln.sum()); tot["sv"] += int(sv.sum())
            tot["ln_j"] += int(ln[j].sum()); tot["sv_j"] += int(sv[j].sum())
            tot["filas_j"] += int(j.sum())
            g = pd.DataFrame({"AELEC": a, "EDOCVE": e, "MPIOCVE": m, "LN": ln, "SV": sv,
                              "LN_J": np.where(j, ln, 0), "SV_J": np.where(j, sv, 0),
                              "N": 1, "N_J": j.astype("int64")})
            parts.append(g.groupby(KEY, sort=False).sum())
            if len(parts) >= 16:
                parts = [pd.concat(parts).groupby(level=[0, 1, 2], sort=False).sum()]
        if rows == 0:
            raise PipelineError("ENTRADA_VACIA", "El dataset de entrada no tiene filas.")
        self.agg = pd.concat(parts).groupby(level=[0, 1, 2], sort=False).sum().reset_index()
        self.tot, self.rows_in = tot, rows
        self.edad_vals = sorted(edad_vals)
        self.aelec_vals = sorted(aelec_vals)
        log.info("Filas leidas=%d | AELEC=%s | EDAD distintos=%d [%s..%s]", rows, self.aelec_vals,
                 len(self.edad_vals), self.edad_vals[0], self.edad_vals[-1])
        # reconciliacion con metadata y con Fase 1
        if self.n_meta is not None and self.n_meta != rows:
            raise PipelineError("FILAS_INCONSISTENTES", f"metadata={self.n_meta} vs leidas={rows}")
        if self.p1_report and self.p1_report.get("filas", {}).get("filas_finales") not in (None, rows):
            raise PipelineError("FILAS_NO_COINCIDEN_FASE1", f"Fase 1 reporta {self.p1_report['filas']['filas_finales']} "
                                f"filas finales; Fase 2 leyo {rows}.")
        diag = {"filas": rows, "columnas": len(self.cols), "aelec_distintos": self.aelec_vals,
                "n_estados": int(self.agg["EDOCVE"].nunique()),
                "n_municipios": int(len(self.agg[["EDOCVE", "MPIOCVE"]].drop_duplicates())),
                "cardinalidad": {"EDOCVE": int(self.agg["EDOCVE"].nunique()),
                                 "MPIOCVE_valores_distintos": int(self.agg["MPIOCVE"].nunique()),
                                 "AELEC": len(self.aelec_vals)},
                "edad": {"n_distintos": len(self.edad_vals), "min": self.edad_vals[0], "max": self.edad_vals[-1],
                         "valores": self.edad_vals if len(self.edad_vals) <= 200 else "omitido (>200)"},
                "negativos_LN": tot["neg_ln"], "negativos_SV": tot["neg_sv"], "filas_SV_mayor_LN": tot["sv_gt_ln"],
                "filas_LN_cero(informativo)": tot["ln_cero_filas"]}
        self.R["diagnostico_entrada"] = diag
        log.info("Diagnostico: %s", {k: v for k, v in diag.items() if k != "edad"})
        missing_young_ages = [x for x in range(JOVEN_MIN, JOVEN_MAX + 1) if x not in set(self.edad_vals)]
        if missing_young_ages:
            self.issue("WARNING", "EDADES_JOVENES_AUSENTES", f"Edades {missing_young_ages} no aparecen en los datos.")
        if tot["neg_ln"] or tot["neg_sv"] or tot["sv_gt_ln"]:
            self.issue("CRITICAL", "VALORES_IMPOSIBLES", f"LN<0={tot['neg_ln']} SV<0={tot['neg_sv']} SV>LN="
                       f"{tot['sv_gt_ln']} filas: contradice las garantias de la Fase 1.")
            raise PipelineError("VALORES_IMPOSIBLES", "Valores imposibles en LN/SV en la entrada.")

    # ---- 3. diagnosticos opcionales ----
    def step_duplicates_and_provenance(self):
        names = [c for c, _ in self.cols]
        info = {}
        if self.a.dup_check:
            data_cols = [c for c in names if c not in PHASE1_PROVENANCE]
            h1, h2 = [], []
            has_file = "ARCHIVO_ORIGEN" in names
            for df in self.iterate(data_cols + (["ARCHIVO_ORIGEN"] if has_file else [])):
                h1.append(pd.util.hash_pandas_object(df[data_cols], index=False).to_numpy())
                if has_file:
                    h2.append(pd.util.hash_pandas_object(df[data_cols + ["ARCHIVO_ORIGEN"]], index=False).to_numpy())
            n = sum(len(x) for x in h1)
            d_all = n - len(np.unique(np.concatenate(h1)))
            info["filas_duplicadas_exactas_sin_procedencia"] = int(d_all)
            if has_file:
                d_in = n - len(np.unique(np.concatenate(h2)))
                info["dup_dentro_del_mismo_archivo"] = int(d_in)
                info["dup_entre_archivos_distintos"] = int(d_all - d_in)
            if d_all:
                self.issue("WARNING", "FILAS_DUPLICADAS", f"{info} (hash 64-bit de columnas originales). No se eliminan: "
                           "se suman tal cual. Si el dataset carece de una columna que distinga filas legitimas "
                           "(p.ej. seccion), pueden ser filas validas; si son cargas repetidas inflan LN/SV.")
        if "ANIO_ORIGEN" in names:
            c = Counter()
            for df in self.iterate(["AELEC", "ANIO_ORIGEN"]):
                c.update(zip(to_int(df["AELEC"], "AELEC").tolist(), to_int(df["ANIO_ORIGEN"], "ANIO_ORIGEN").tolist()))
            mism = {f"{k[0]}|{k[1]}": v for k, v in c.items() if k[0] != k[1]}
            info["aelec_vs_anio_origen_distintos"] = mism
            if mism:
                self.issue("WARNING", "AELEC_NO_IGUAL_ANIO_ORIGEN", f"Pares (AELEC|ANIO_ORIGEN) distintos: {mism}. "
                           "ANIO_ORIGEN se infiere de la ruta del archivo; AELEC es la variable de panel.")
        self.R["diagnostico_calidad_adicional"] = info

    def step_rejected_ln(self):
        """Cuantifica LN que Fase 1 excluyo (sesgo de lista_nominal_total)."""
        out = {}
        ext = ".parquet" if not self.debug_input else ".debug.csv"
        kept = self.tot["ln"]
        rej_total = 0
        for kind in ("contabilidad", "edad"):
            p = self.p1_dir / f"registros_invalidos_{kind}{ext}"
            if not p.exists():
                continue
            try:
                _, nrows, it = open_source(p)
                rows = ln_sum = ln_bad = 0
                by_a = Counter()
                for df in it(["AELEC", "LN"]):
                    ln = pd.to_numeric(df["LN"].astype(str).str.strip(), errors="coerce")
                    ok = ln.notna() & (ln >= 0)
                    a = pd.to_numeric(df["AELEC"].astype(str).str.strip(), errors="coerce")
                    rows += len(df); ln_bad += int((~ok).sum()); ln_sum += int(ln[ok].sum())
                    for k, v in ln[ok].groupby(a[ok]).sum().items():
                        by_a[int(k)] += int(v)
                out[kind] = {"filas": rows, "LN_numerico_excluido": ln_sum, "filas_LN_no_numerico": ln_bad,
                             "LN_excluido_por_AELEC": dict(by_a)}
                rej_total += ln_sum
            except Exception as e:
                self.issue("WARNING", "RECHAZADOS_ILEGIBLES", f"{p}: {e!r}")
        pct = 100 * rej_total / (rej_total + kept) if (rej_total + kept) else 0.0
        out["pct_LN_excluido_sobre_total"] = round(pct, 4)
        self.R["ln_excluido_por_fase1"] = out
        if rej_total:
            lvl = "WARNING" if pct >= REJECTED_LN_WARN_PCT else "INFO"
            self.issue(lvl, "LN_EXCLUIDO_FASE1", f"Fase 1 excluyo LN={rej_total} ({pct:.3f}% del LN total conocido). "
                       f"lista_nominal_total subestima el total real en esa proporcion: {out}")
        elif not any(k in out for k in ("contabilidad", "edad")):
            log.info("Sin archivos de registros invalidos de Fase 1 (nada excluido o no disponibles).")

    # ---- 4. features ----
    def step_features(self):
        p = self.agg.rename(columns={"LN": "lista_nominal_total", "SV": "votos_totales",
                                     "LN_J": "lista_nominal_jovenes", "SV_J": "votos_jovenes",
                                     "N": "n_filas_origen", "N_J": "n_filas_origen_jovenes"})
        p = p.sort_values(["EDOCVE", "MPIOCVE", "AELEC"], kind="mergesort").reset_index(drop=True)
        ln, sv = p["lista_nominal_total"].to_numpy("float64"), p["votos_totales"].to_numpy("float64")
        lnj, svj = p["lista_nominal_jovenes"].to_numpy("float64"), p["votos_jovenes"].to_numpy("float64")
        bad_den = ln <= 0
        if bad_den.any():
            ex = p.loc[bad_den, KEY + ["lista_nominal_total"]].head(10).to_dict("records")
            self.R["denominador_lista_nominal_invalido"] = {"n": int(bad_den.sum()), "ejemplos": ex}
            self.issue("CRITICAL", "LN_TOTAL_NO_POSITIVA", f"{int(bad_den.sum())} municipio/ejercicio con "
                       f"lista_nominal_total<=0; ejemplos={ex}. Politica: detener (no se fabrica la tasa).")
            raise PipelineError("LN_TOTAL_NO_POSITIVA", "Denominador invalido en riesgo_abstencion.")
        p["riesgo_abstencion"] = (ln - sv) / ln
        p["tasa_jovenes_ln"] = lnj / ln
        zero = lnj == 0
        p["tasa_voto_jovenes"] = np.divide(svj, lnj, out=np.zeros_like(svj), where=~zero)
        p["jovenes_ln_cero"] = zero
        self.zero_cases = {"n": int(zero.sum()), "pct": round(100 * zero.mean(), 4),
                           "por_AELEC": {int(k): int(v) for k, v in p.loc[zero].groupby("AELEC").size().items()},
                           "significado": "denominador joven = 0 (sin poblacion joven registrada): tasa_voto_jovenes=0 "
                                          "NO es una tasa observada igual a cero; ver columna jovenes_ln_cero"}
        if zero.any():
            self.issue("WARNING", "DENOMINADOR_JOVEN_CERO", f"{self.zero_cases['n']} municipio/ejercicio "
                       f"({self.zero_cases['pct']}%) con lista_nominal_jovenes=0 -> tasa_voto_jovenes=0 "
                       f"(bandera jovenes_ln_cero=True). Para modelado se recomienda tratarlo como faltante.")
        # lag: registro anterior disponible del mismo municipio
        g = p.groupby(["EDOCVE", "MPIOCVE"], sort=False)
        p["lag_riesgo_abstencion_1"] = g["riesgo_abstencion"].shift(1)
        p["lag_riesgo_abstencion_2"] = g["riesgo_abstencion"].shift(2)
        lag_a = g["AELEC"].shift(1)
        lag_a2 = g["AELEC"].shift(2)
        p["lag_aelec_1"] = lag_a.astype("Int16")
        p["lag_aelec_2"] = lag_a2.astype("Int16")
        rank = {e: i for i, e in enumerate(self.aelec_vals)}
        cur_r = p["AELEC"].map(rank)
        lag_r = lag_a.map(rank)
        p["lag_saltos_1"] = (cur_r - lag_r - 1).astype("Int8")
        p["tasa_voto_jovenes_obs"] = p["tasa_voto_jovenes"].mask(zero)
        p["lag_voto_jovenes_1"] = g["tasa_voto_jovenes_obs"].shift(1)
        p["lag_voto_jovenes_2"] = g["tasa_voto_jovenes_obs"].shift(2)
        p["lag_tasa_jovenes_ln_1"] = g["tasa_jovenes_ln"].shift(1)
        p["hist_mean_riesgo"] = g["riesgo_abstencion"].transform(lambda s: s.shift(1).expanding().mean())
        p["hist_std_riesgo"] = g["riesgo_abstencion"].transform(lambda s: s.shift(1).expanding().std())
        p["hist_delta_riesgo_1"] = p["lag_riesgo_abstencion_1"] - p["lag_riesgo_abstencion_2"]
        p["state_lag_mean"] = p.groupby(["EDOCVE", "AELEC"])["lag_riesgo_abstencion_1"].transform("mean")
        p["gap_riesgo_estado"] = p["lag_riesgo_abstencion_1"] - p["state_lag_mean"]
        p["log_ln_total"] = np.log1p(p["lista_nominal_total"].astype("float64"))
        p["log_ln_jovenes"] = np.log1p(p["lista_nominal_jovenes"].astype("float64"))
        p["presidencial"] = p["AELEC"].isin([2012, 2018, 2024]).astype("int8")
        # tipos
        t = {}
        for c in ("AELEC", "EDOCVE", "MPIOCVE", "lista_nominal_total", "votos_totales", "lista_nominal_jovenes",
                  "votos_jovenes", "n_filas_origen", "n_filas_origen_jovenes", "presidencial"):
            t[c] = smallest_int(int(p[c].min()), int(p[c].max()))
            p[c] = p[c].astype(t[c])
        self.dtypes = t
        order = KEY + ["lista_nominal_total", "votos_totales", "riesgo_abstencion", "lista_nominal_jovenes",
                       "votos_jovenes", "tasa_jovenes_ln", "tasa_voto_jovenes", "tasa_voto_jovenes_obs",
                       "lag_riesgo_abstencion_1", "lag_riesgo_abstencion_2",
                       "lag_aelec_1", "lag_aelec_2", "lag_saltos_1",
                       "lag_voto_jovenes_1", "lag_voto_jovenes_2", "lag_tasa_jovenes_ln_1",
                       "hist_mean_riesgo", "hist_std_riesgo", "hist_delta_riesgo_1",
                       "state_lag_mean", "gap_riesgo_estado",
                       "log_ln_total", "log_ln_jovenes", "presidencial",
                       "jovenes_ln_cero", "n_filas_origen", "n_filas_origen_jovenes"]
        self.panel = p[order]
        log.info("Panel: %d filas x %d columnas | tipos enteros=%s", len(p), len(order), t)

    # ---- 5. validaciones (reutilizable antes y despues de exportar) ----
    def check_panel(self, df: pd.DataFrame, stage: str, strict_conservation=True):
        crit = []

        def rec(name, ok, ev, warn=False):
            res = "PASSED" if ok else ("WARNING" if warn else "CRITICAL")
            self.validation(f"{name} [{stage}]", res, ev)
            if res == "CRITICAL":
                crit.append(name)

        d = df
        dup = int(d.duplicated(KEY).sum())
        rec("Unicidad (AELEC,EDOCVE,MPIOCVE)", dup == 0, f"filas={len(d)} duplicadas={dup} unicas={len(d.drop_duplicates(KEY))}")
        for c in ("riesgo_abstencion", "tasa_jovenes_ln", "tasa_voto_jovenes"):
            v = d[c].astype("float64")
            out = (v < 0) | (v > 1) | ~np.isfinite(v)
            ex = d.loc[out, KEY + [c]].head(5).to_dict("records")
            rec(f"Rango [0,1] {c}", int(out.sum()) == 0,
                f"fuera_de_rango_o_no_finito={int(out.sum())} ({100*out.mean():.4f}%) min={v.min():.6f} max={v.max():.6f} ej={ex}")
        obs = d["tasa_voto_jovenes_obs"].astype("float64")
        obs_out = obs.notna() & ((obs < 0) | (obs > 1) | ~np.isfinite(obs))
        rec("Rango [0,1] tasa_voto_jovenes_obs", int(obs_out.sum()) == 0,
            f"fuera_de_rango_o_no_finito={int(obs_out.sum())} nulos={int(obs.isna().sum())}")
        for c in ("lag_voto_jovenes_1", "lag_voto_jovenes_2", "lag_tasa_jovenes_ln_1",
                  "hist_mean_riesgo", "hist_std_riesgo", "state_lag_mean"):
            v = d[c].astype("float64")
            out = v.notna() & ((v < 0) | (v > 1) | ~np.isfinite(v))
            rec(f"Rango [0,1] {c}", int(out.sum()) == 0,
                f"fuera_de_rango_o_no_finito={int(out.sum())} nulos={int(v.isna().sum())}")
        for c in ("hist_delta_riesgo_1", "gap_riesgo_estado"):
            v = d[c].astype("float64")
            out = v.notna() & ((v < -1) | (v > 1) | ~np.isfinite(v))
            rec(f"Rango [-1,1] {c}", int(out.sum()) == 0,
                f"fuera_de_rango_o_no_finito={int(out.sum())} nulos={int(v.isna().sum())}")
        nn = [c for c in d.columns if c not in NULLABLE_PANEL]
        nulls = {c: int(d[c].isna().sum()) for c in nn if d[c].isna().any()}
        rec("Nulos inesperados", not nulls, f"columnas con nulos fuera de NULLABLE_PANEL: {nulls or 'ninguna'}")
        fl = [c for c in d.columns if d[c].dtype.kind == "f"]
        infs = {c: int((~np.isfinite(d[c].dropna().to_numpy("float64"))).sum()) for c in fl}
        rec("Valores finitos", all(v == 0 for v in infs.values()), f"no finitos por columna={infs}")
        # formulas recomputadas de forma independiente
        ln, sv = d["lista_nominal_total"].astype("int64"), d["votos_totales"].astype("int64")
        lnj, svj = d["lista_nominal_jovenes"].astype("int64"), d["votos_jovenes"].astype("int64")
        r_ok = np.allclose(d["riesgo_abstencion"], (ln - sv) / ln, rtol=0, atol=1e-12)
        j_ok = np.allclose(d["tasa_jovenes_ln"], lnj / ln, rtol=0, atol=1e-12)
        den = lnj.to_numpy() > 0
        v_ok = np.allclose(d.loc[den, "tasa_voto_jovenes"], (svj / lnj)[den], rtol=0, atol=1e-12) and bool(
            (d.loc[~den, "tasa_voto_jovenes"] == 0).all())
        rec("Formulas recomputadas", r_ok and j_ok and v_ok, f"riesgo={r_ok} tasa_jovenes_ln={j_ok} tasa_voto_jovenes={v_ok}")
        rec("Jovenes <= total", bool((lnj <= ln).all() and (svj <= lnj).all() and (sv <= ln).all()),
            "LN_j<=LN, SV_j<=LN_j, SV<=LN")
        # conservacion contra totales acumulados en el streaming
        if strict_conservation:
            t = self.tot
            cons = {"LN": (int(ln.sum()), t["ln"]), "SV": (int(sv.sum()), t["sv"]),
                    "LN_J": (int(lnj.sum()), t["ln_j"]), "SV_J": (int(svj.sum()), t["sv_j"]),
                    "filas": (int(d["n_filas_origen"].astype("int64").sum()), self.rows_in)}
            rec("Conservacion de totales", all(a == b for a, b in cons.values()), f"(panel,entrada)={cons}")
        # lag: recomputo independiente por diccionario + fuga temporal
        dd = d.sort_values(["EDOCVE", "MPIOCVE", "AELEC"])
        hist, exp_vals, exp_aelec = {}, [], []
        for e, m, a, r in zip(dd["EDOCVE"], dd["MPIOCVE"], dd["AELEC"], dd["riesgo_abstencion"].astype("float64")):
            prev = hist.get((e, m))
            exp_vals.append(prev[1] if prev else np.nan)
            exp_aelec.append(prev[0] if prev else np.nan)
            hist[(e, m)] = (a, r)
        got = dd["lag_riesgo_abstencion_1"].astype("float64").to_numpy()
        got_a = dd["lag_aelec_1"].astype("float64").to_numpy()
        same = np.array_equal(np.nan_to_num(got, nan=-9), np.nan_to_num(np.array(exp_vals), nan=-9)) and \
            np.array_equal(np.nan_to_num(got_a, nan=-9), np.nan_to_num(np.array(exp_aelec), nan=-9))
        rec("Lag = registro previo del mismo municipio", same, "recomputo independiente (diccionario) coincide con shift por grupo")
        nonnull = dd["lag_aelec_1"].notna()
        leak = int((dd.loc[nonnull, "lag_aelec_1"].astype("int64") >= dd.loc[nonnull, "AELEC"].astype("int64")).sum())
        rec("Sin fuga temporal (lag_aelec < AELEC)", leak == 0, f"lags con ejercicio >= actual={leak}")
        # el valor del lag debe ser el riesgo de ese municipio en lag_aelec
        look = dd.loc[nonnull, ["EDOCVE", "MPIOCVE", "lag_aelec_1", "lag_riesgo_abstencion_1"]].astype(
            {"lag_aelec_1": "int64"}).merge(
            d[KEY + ["riesgo_abstencion"]].rename(columns={"AELEC": "lag_aelec_1"}),
            on=["EDOCVE", "MPIOCVE", "lag_aelec_1"], how="left")
        mism = int((~np.isclose(look["lag_riesgo_abstencion_1"].astype("float64"), look["riesgo_abstencion"], atol=1e-12)).sum())
        rec("Lag proviene del mismo municipio (lookup)", mism == 0 and len(look) == int(nonnull.sum()), f"inconsistencias={mism} de {len(look)}")
        n_mun = len(d[["EDOCVE", "MPIOCVE"]].drop_duplicates())
        n_null = int(d["lag_riesgo_abstencion_1"].isna().sum())
        rec("Nulos del lag = primer registro por municipio", n_null == n_mun,
            f"nulos={n_null} ({100*n_null/len(d):.2f}%) municipios={n_mun}")
        if stage == "pre-export":
            multi = int((d.groupby(["EDOCVE", "MPIOCVE"]).size() >= 2).sum())
            sal = d["lag_saltos_1"].dropna().astype("int64")
            self.R["lag"] = {"nulos": n_null, "pct_nulos": round(100 * n_null / len(d), 3), "municipios": n_mun,
                             "municipios_con_historial_suficiente(>=2 ejercicios)": multi,
                             "distribucion_saltos": {int(k): int(v) for k, v in sal.value_counts().sort_index().items()}}
            if (sal > 0).any():
                self.issue("WARNING", "LAG_CON_SALTOS", f"{int((sal > 0).sum())} registros cuyo lag salta ejercicios "
                           f"(municipio sin registro en algun ejercicio intermedio): {self.R['lag']['distribucion_saltos']}. "
                           "Por el requisito de 'anterior disponible' se usa ese registro; ver lag_saltos_1.")
        return crit

    # ---- 6. exportacion y verificacion posterior ----
    def step_export(self):
        crit = self.check_panel(self.panel, "pre-export")
        if crit:
            raise PipelineError("VALIDACION_CRITICA_FALLIDA", f"Validaciones criticas fallidas antes de exportar: {crit}")
        final = self.run_dir / ("features_municipales" + (".debug.csv" if self.debug_input else ".parquet"))
        # temporal con la extension real (el lector decide el formato por extension); se promueve con os.replace
        tmp = self.run_dir / ("tmp_features" + (".csv" if self.debug_input else ".parquet"))
        write_table_file(self.panel, tmp)
        back, schema, nrows = read_table_file(tmp)
        exp_cols = list(self.panel.columns)
        ok_cols = list(back.columns) == exp_cols
        self.validation("Parquet: columnas", "PASSED" if ok_cols else "CRITICAL", f"{len(back.columns)} columnas; esperadas {len(exp_cols)}")
        ok_rows = len(back) == len(self.panel) and (nrows is None or nrows == len(self.panel))
        self.validation("Parquet: filas", "PASSED" if ok_rows else "CRITICAL", f"leidas={len(back)} metadata={nrows} esperadas={len(self.panel)}")
        same = ok_cols and all(np.allclose(back[c].astype("float64"), self.panel[c].astype("float64"), rtol=0, atol=0, equal_nan=True)
                               for c in exp_cols)
        self.validation("Parquet: contenido identico", "PASSED" if same else "CRITICAL", "comparacion columna a columna (exacta)")
        crit2 = self.check_panel(back, "post-export")
        if not (ok_cols and ok_rows and same) or crit2:
            raise PipelineError("VERIFICACION_POST_EXPORT_FALLIDA", f"Fallo tras reabrir {tmp}; se conserva para inspeccion.")
        os.replace(tmp, final)
        self.final = final
        self.parquet_verified = not self.debug_input
        self.R["schema_parquet"] = schema
        if self.debug_input:
            self.validation("Parquet legible", "NOT_RUN", "entrada/sink de prueba CSV: NO se genero ni verifico Parquet")
        else:
            self.validation("Parquet legible", "PASSED", f"{final.name} reabierto: {len(back)} filas, esquema={schema}")

    # ---- 7. evidencia ----
    def step_evidence(self):
        d = self.panel
        d24 = d[d["AELEC"] == 2024]
        if len(d24) == 0:
            self.issue("WARNING", "SIN_2024", "No hay filas AELEC==2024; no se puede describir su distribucion.")
            desc = None
        else:
            desc = d24[["riesgo_abstencion", "tasa_voto_jovenes"]].describe().to_dict()
        self.desc_2024 = desc
        by = d.groupby("AELEC").agg(municipios=("MPIOCVE", "size"), LN=("lista_nominal_total", lambda s: int(s.astype("int64").sum())),
                                    SV=("votos_totales", lambda s: int(s.astype("int64").sum())),
                                    riesgo_medio_municipal=("riesgo_abstencion", "mean"),
                                    jovenes_ln_cero=("jovenes_ln_cero", "sum"))
        by["riesgo_nacional_ponderado"] = (by["LN"] - by["SV"]) / by["LN"]
        self.por_aelec = by.reset_index().to_dict("records")
        log.info("Resumen por AELEC:\n%s", by.to_string())
        if desc:
            log.info("Distribucion 2024:\n%s", d24[["riesgo_abstencion", "tasa_voto_jovenes"]].describe().to_string())
        h = int(pd.util.hash_pandas_object(self.panel, index=False).astype("uint64").sum() % (2**63))
        self.R["hash_contenido_panel"] = f"{h:016x}"

    def finalize(self, exc):
        dur = time.perf_counter() - self.t0
        sev = [i["severidad"] for i in self.issues]
        res = [v["resultado"] for v in self.validations]
        if isinstance(exc, PipelineError):
            status = "CRITICAL"
        elif exc is not None:
            status = "FAILED"
        elif "CRITICAL" in res or "CRITICAL" in sev:
            status = "CRITICAL"
        elif "WARNING" in sev or "WARNING" in res or "NOT_RUN" in res or self.debug_input:
            status = "WARNING"
        else:
            status = "PASSED"
        self.R.update(status=status, duracion_s=round(dur, 2), issues=self.issues, validaciones=self.validations,
                      argumentos=vars(self.a), parquet_verificado=bool(getattr(self, "parquet_verified", False)),
                      definicion_jovenes=f"{JOVEN_MIN} <= EDAD <= {JOVEN_MAX}",
                      entorno={"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                               "so": platform.platform(), "cpus": os.cpu_count()})
        if hasattr(self, "panel"):
            self.R["panel"] = {"filas_entrada": self.rows_in, "filas_finales": len(self.panel),
                               "municipios": int(len(self.panel[["EDOCVE", "MPIOCVE"]].drop_duplicates())),
                               "estados": int(self.panel["EDOCVE"].nunique()),
                               "ejercicios": self.aelec_vals, "columnas": list(self.panel.columns),
                               "tipos": {c: str(t) for c, t in self.panel.dtypes.items()}}
            self.R["denominador_joven_cero"] = self.zero_cases
            self.R["distribucion_2024"] = self.desc_2024 if hasattr(self, "desc_2024") else None
            self.R["resumen_por_aelec"] = getattr(self, "por_aelec", None)
        self.R["linaje_features"] = {
            "lista_nominal_total": "SUM(LN) por (AELEC,EDOCVE,MPIOCVE)",
            "votos_totales": "SUM(SV)",
            "riesgo_abstencion": "(lista_nominal_total - votos_totales) / lista_nominal_total  [incluye NS en el numerador: LN=SV+NV+NS]",
            "lista_nominal_jovenes": f"SUM(LN) donde {JOVEN_MIN}<=EDAD<={JOVEN_MAX}",
            "votos_jovenes": f"SUM(SV) donde {JOVEN_MIN}<=EDAD<={JOVEN_MAX}",
            "tasa_jovenes_ln": "lista_nominal_jovenes / lista_nominal_total",
            "tasa_voto_jovenes": "votos_jovenes / lista_nominal_jovenes; 0 si el denominador es 0 (ver jovenes_ln_cero)",
            "tasa_voto_jovenes_obs": "votos_jovenes / lista_nominal_jovenes; NaN si el denominador es 0 (version observada para modelado)",
            "lag_riesgo_abstencion_1": "riesgo_abstencion del registro anterior disponible del mismo (EDOCVE,MPIOCVE), orden AELEC asc",
            "lag_riesgo_abstencion_2": "riesgo_abstencion de dos registros previos disponibles del mismo municipio",
            "lag_aelec_1": "AELEC del registro usado como lag 1 (auditoria)",
            "lag_aelec_2": "AELEC del registro usado como lag 2 (auditoria)",
            "lag_saltos_1": "ejercicios globales omitidos entre el lag y el registro actual (0 = ejercicio inmediato)",
            "lag_voto_jovenes_1 / lag_voto_jovenes_2": "tasa_voto_jovenes_obs en el registro previo 1 / 2 del mismo municipio",
            "lag_tasa_jovenes_ln_1": "tasa_jovenes_ln en el registro previo 1 del mismo municipio",
            "hist_mean_riesgo / hist_std_riesgo": "media y desviacion de riesgo_abstencion usando solo ejercicios previos del municipio",
            "hist_delta_riesgo_1": "lag_riesgo_abstencion_1 - lag_riesgo_abstencion_2",
            "state_lag_mean": "media de lag_riesgo_abstencion_1 entre municipios del mismo estado y ejercicio",
            "gap_riesgo_estado": "lag_riesgo_abstencion_1 - state_lag_mean",
            "log_ln_total / log_ln_jovenes": "log1p(lista_nominal_total) / log1p(lista_nominal_jovenes)",
            "presidencial": "1 si AELEC es 2012, 2018 o 2024; 0 en el resto",
            "jovenes_ln_cero": "True si lista_nominal_jovenes == 0 (tasa_voto_jovenes imputada a 0, no observada)",
            "n_filas_origen / n_filas_origen_jovenes": "filas del dataset limpio agregadas (trazabilidad/conservacion)"}
        try:
            with open(self.run_dir / "reporte_feature_engineering.json", "x", encoding="utf-8") as f:
                json.dump(self.R, f, ensure_ascii=False, indent=2, default=jsonable)
            idx = self.out_root / "runs_index_fase2.csv"
            new = not idx.exists()
            with open(idx, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "run_dir", "status"])
                w.writerow([self.stamp, str(self.run_dir), status])
        except Exception as e:
            log.error("No se pudo escribir el reporte: %r", e)
        log.info("ESTADO FINAL=%s | duracion=%.1fs | salida=%s", status, dur, self.run_dir)
        return {"PASSED": 0, "WARNING": 1}.get(status, 2)

    def run(self):
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        log.info("Fase 2 v%s | inicio", VERSION)
        exc = None
        try:
            self.step_input()
            self.step_stream()
            self.step_duplicates_and_provenance()
            self.step_rejected_ln()
            self.step_features()
            self.step_evidence()
            self.step_export()
        except PipelineError as e:
            exc = e
            self.issue("CRITICAL", e.code, str(e))
        except Exception as e:
            exc = e
            log.exception("Error inesperado")
            self.issue("CRITICAL", "ERROR_INESPERADO", repr(e))
        return self.finalize(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help="ruta a ine_dataset_limpio.parquet (si no, se resuelve via runs_index.csv de Fase 1)")
    ap.add_argument("--phase1-out-root", default="Prueba_2/salida")
    ap.add_argument("--out-root", default="Prueba_2/salida")
    ap.add_argument("--no-dup-check", dest="dup_check", action="store_false")
    ap.add_argument("--no-hash-input", dest="hash_input", action="store_false")
    return Phase2(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())