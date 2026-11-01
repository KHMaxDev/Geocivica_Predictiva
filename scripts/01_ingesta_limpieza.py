#!/usr/bin/env python3
"""
CCPC - Fase 1: ingesta, limpieza, validacion y consolidacion -> Parquet.

USO
    python scripts/01_ingesta_limpieza.py --input-root data/CCPC --out-root data/Salida
    (requiere: pandas, numpy, pyarrow)

ARQUITECTURA Y JUSTIFICACION
- pandas + pyarrow, procesamiento por chunks (500k filas) en DOS pasadas:
    Pasada 1 (paralela por archivo, ProcessPool): valida estructura, cuenta campos por fila con el
        modulo csv, perfila SEXO/DEL/contables/EDAD y mide min/max para elegir el entero minimo REAL.
    Pasada 2 (secuencial, 1 escritor Parquet): transforma, aplica reglas, rechaza, escribe.
  Dos pasadas porque el tipo entero optimo no se conoce hasta ver todos los datos y un ParquetWriter
  exige esquema fijo desde el primer chunk. Secuencial en pasada 2 porque hay un unico escritor y el
  orden de salida debe ser determinista.
- Todo se lee como TEXTO (dtype=str, sin NA implicitos): ninguna conversion ocurre sin ser medida.
- Duckdb/Polars no estan justificados aun: el trabajo es por-fila con reglas propias y chunking basta.
  La GPU no aplica a parsing/validacion de CSV.
- Parquet: ZSTD nivel 3 (mejor ratio que Snappy, lectura rapida, soportado por pyarrow/DuckDB/Polars).
- La logica especifica de Parquet esta aislada en ParquetSink; el resto es independiente del formato.
  `--sink csv` existe SOLO para pruebas sin pyarrow y fuerza estado "REQUIERE VALIDACION".

DESVIACIONES / DECISIONES SOBRE EL PROMPT (ver reporte): exclusion evaluada sobre la ruta RELATIVA a
--input-root y normalizada (sin acentos/separadores); resultados en carpeta por ejecucion (no se
sobrescribe nada); columnas de procedencia anadidas; columnas no especificadas se conservan como texto.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_VERSION = "1.0.0"

# =============================== CONFIGURACION CENTRALIZADA ===============================
CRITICAL_COLS = ["SEXO", "DEL", "LN", "SV", "NV", "NS", "EDAD"]
CONTABLES = ["LN", "SV", "NV", "NS"]
SEXO_MAP = {0: "H", 1: "M", 2: "No_Binario"}
SEXO_FINAL_ALLOWED = ["H", "M", "No_Binario"]
SEXO_UNMAPPED = "No_Mapeado"                             # valores desconocidos: visibles, no ocultos
DEL_SENTINEL = -1                                        # imputacion temporal (con bandera DEL_IMPUTADO)
EDAD_MIN, EDAD_MAX = 18, 120
EXCLUDE_TOKENS = ["votoanticipado", "prisionpreventiva"]  # comparados tras normalizar
CHUNK_ROWS = 250_000
MAX_SCAN_WORKERS = 4
MAX_ISOLATED_FILE_FRACTION = 0.05
DEFAULT_MAX_REJECT_PCT = 5.0
COMPRESSION, COMPRESSION_LEVEL = "zstd", 3
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
DELIMS = [",", ";", "|", "\t"]
PROVENANCE = ["SEXO_ORIGINAL", "DEL_IMPUTADO", "ARCHIVO_ORIGEN", "ANIO_ORIGEN", "FILA_ORIGEN"]
MAX_EXAMPLES = 10

log = logging.getLogger("fase1")


class PipelineError(Exception):
    """Condicion CRITICA que bloquea la generacion del dataset final."""

    def __init__(self, code, msg):
        self.code = code
        super().__init__(msg)


# =============================== UTILIDADES ===============================
def norm_token(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def detect_encoding(path: Path, probe=4 * 1024 * 1024):
    with open(path, "rb") as f:
        raw = f.read(probe)
    full = len(raw) < probe
    for enc in ENCODINGS:
        try:
            (raw if full else raw[:-4]).decode(enc)
            return enc, ("latin-1 como ultimo recurso (encoding incierto)" if enc == "latin-1" else "ok")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"encoding no reconocido entre {ENCODINGS}")


def read_header(path: Path, enc: str):
    with open(path, "r", encoding=enc, newline="") as f:
        line = f.readline()
    counts = {d: line.count(d) for d in DELIMS}
    best = max(counts, key=counts.get)
    ties = [d for d, c in counts.items() if c == counts[best]]
    if counts[best] == 0 or len(ties) > 1:
        raise ValueError(f"delimitador no determinable en cabecera (conteos={counts})")
    hdr = next(csv.reader([line.lstrip("\ufeff")], delimiter=best))
    return [h.strip() for h in hdr], best


def parse_int_col(s: pd.Series):
    """Devuelve (valores float64 con NaN si no valido, motivo por fila). Sin conversion silenciosa."""
    t = s.str.strip()
    empty = (t == "").to_numpy()
    num = pd.to_numeric(t.where(~empty), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    inf = np.isinf(num)
    num = np.where(inf, np.nan, num)
    nonnum = ~empty & np.isnan(num)
    frac = ~np.isnan(num) & (num != np.floor(np.nan_to_num(num)))
    neg = ~np.isnan(num) & ~frac & (num < 0)
    reason = np.select([empty, nonnum, frac, neg], ["VACIO", "NO_NUMERICO", "DECIMAL", "NEGATIVO"], "OK")
    return num, reason.astype(object)


def smallest_int(lo, hi):
    for name in ("int8", "int16", "int32", "int64"):
        info = np.iinfo(name)
        if info.min <= lo and hi <= info.max:
            return name
    raise PipelineError("RANGO_ENTERO", f"valores fuera de int64: [{lo}, {hi}]")


def year_from_rel(rel: str):
    m = re.findall(r"(?<!\d)(20\d{2})(?!\d)", rel)
    return int(m[-1]) if m else None


def sha256_file(path: str):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_chunks(path, enc, delim):
    return pd.read_csv(path, sep=delim, encoding=enc, dtype=str, keep_default_na=False, na_filter=False,
                       chunksize=CHUNK_ROWS, index_col=False, on_bad_lines="error")


# =============================== PASADA 1 (worker) ===============================
def scan_file(path_str: str, rel: str) -> dict:
    """Valida y perfila un archivo. Devuelve solo tipos simples (picklable)."""
    p = Path(path_str)
    res = {"rel": rel, "path": path_str, "fatal": None, "missing_critical": []}
    try:
        enc, enc_note = detect_encoding(p)
        header, delim = read_header(p, enc)
        res.update(encoding=enc, enc_note=enc_note, delimiter=delim, header=header)
        dups = [h for h, c in Counter(header).items() if c > 1]
        if dups:
            raise ValueError(f"columnas duplicadas: {dups}")
        cm, rename = {}, {}
        for h in header:
            u = h.upper()
            if u in CRITICAL_COLS:
                if u in cm:
                    raise ValueError(f"columna critica ambigua: {u}")
                cm[u] = h
                if h != u:
                    rename[h] = u
        res["crit_map"], res["case_variants"] = cm, rename
        res["missing_critical"] = [c for c in CRITICAL_COLS if c not in cm]
        if res["missing_critical"]:
            return res

        # conteo de campos por fila con csv (pandas no detecta filas cortas/largas de forma fiable)
        nh, ragged, rows_csv, ragged_examples = len(header), 0, 0, []
        with open(p, "r", encoding=enc, newline="") as f:
            rd = csv.reader(f, delimiter=delim)
            next(rd)
            for r in rd:
                if not r:
                    continue
                rows_csv += 1
                if len(r) != nh:
                    ragged += 1
                    if len(ragged_examples) < 3:
                        ragged_examples.append((rows_csv, len(r)))
        res.update(rows_csv=rows_csv, ragged_rows=ragged, ragged_examples=ragged_examples)

        numeric = CONTABLES + ["DEL", "EDAD"]
        reasons = {c: Counter() for c in numeric}
        tokens = {c: Counter() for c in numeric}
        lo = {c: None for c in numeric}
        hi = {c: None for c in numeric}
        sexo, rows = Counter(), 0
        for ch in read_csv_chunks(p, enc, delim):
            rows += len(ch)
            for c in numeric:
                s = ch[cm[c]]
                num, rs = parse_int_col(s)
                for k, v in zip(*np.unique(rs.astype(str), return_counts=True)):
                    reasons[c][k] += int(v)
                bad = (rs != "OK") & (rs != "VACIO")
                if bad.any():
                    tokens[c].update(s[bad].str.strip().value_counts().head(10).to_dict())
                    tokens[c] = Counter(dict(tokens[c].most_common(20)))
                ok = rs == "OK"
                if c == "EDAD":
                    ok = ok & (num >= EDAD_MIN) & (num <= EDAD_MAX)
                if ok.any():
                    mn, mx = float(num[ok].min()), float(num[ok].max())
                    lo[c] = mn if lo[c] is None else min(lo[c], mn)
                    hi[c] = mx if hi[c] is None else max(hi[c], mx)
            sexo.update(ch[cm["SEXO"]].str.strip().value_counts().to_dict())
        res.update(rows=rows, reasons={c: dict(v) for c, v in reasons.items()},
                   tokens={c: dict(v) for c, v in tokens.items()}, lo=lo, hi=hi, sexo=dict(sexo))
    except Exception as e:  # el main decide la severidad
        res["fatal"] = f"{type(e).__name__}: {e}"
    return res


# =============================== SINKS ===============================
class ParquetSink:
    TYPES = {"string": "string", "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
             "bool": "bool_"}

    def __init__(self, path: Path, fields):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq, self.fields = pa, pq, fields
        self.schema = pa.schema([pa.field(n, getattr(pa, self.TYPES[t])()) for n, t in fields])
        self.writer = pq.ParquetWriter(str(path), self.schema, compression=COMPRESSION,
                                       compression_level=COMPRESSION_LEVEL)

    def write(self, df):
        self.writer.write_table(self.pa.Table.from_pandas(df, schema=self.schema, preserve_index=False))

    def close(self):
        self.writer.close()

    @staticmethod
    def read_batches(path, fields):
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(str(path)).iter_batches(batch_size=CHUNK_ROWS):
            yield b.to_pandas()

    @staticmethod
    def describe(path):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        return [(f.name, str(f.type)) for f in pf.schema_arrow], pf.metadata.num_rows


class CsvDebugSink:
    """SOLO PRUEBAS (sin pyarrow). No produce Parquet."""

    def __init__(self, path: Path, fields):
        self.path, self.fields, self.first = path, fields, True

    def write(self, df):
        df.to_csv(self.path, mode="w" if self.first else "a", header=self.first, index=False)
        self.first = False

    def close(self):
        if self.first:
            pd.DataFrame(columns=[n for n, _ in self.fields]).to_csv(self.path, index=False)

    @staticmethod
    def read_batches(path, fields):
        dt = {n: (str if t == "string" else bool if t == "bool" else t) for n, t in fields}
        yield from pd.read_csv(path, dtype=dt, keep_default_na=False, na_filter=False, chunksize=CHUNK_ROWS)

    @staticmethod
    def describe(path):
        n = sum(len(b) for b in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS))
        return None, n


# =============================== PIPELINE ===============================
class Pipeline:
    def __init__(self, a):
        self.a = a
        self.t0 = time.perf_counter()
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms: evita colision entre ejecuciones
        self.out_root = Path(a.out_root).resolve()
        self.run_dir = self.out_root / f"run_{self.stamp}"
        self.issues, self.inv, self.validations = [], [], []
        self.R = {"pipeline_version": PIPELINE_VERSION, "timestamp": self.stamp, "status": "FALLIDA"}
        self.sink_cls = ParquetSink if a.sink == "parquet" else CsvDebugSink
        self.ext = ".parquet" if a.sink == "parquet" else ".debug.csv"

    # ---- infraestructura ----
    def setup_logging(self):
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=False)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        log.setLevel(logging.INFO)
        for h in (logging.FileHandler(self.run_dir / "logs" / "ingesta_limpieza.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            log.addHandler(h)

    LEVELS = {"CRITICO": logging.CRITICAL, "ERROR": logging.ERROR, "ADVERTENCIA": logging.WARNING,
              "INFORMATIVO": logging.INFO}

    def issue(self, sev, code, msg, file=None):
        self.issues.append({"severidad": sev, "codigo": code, "mensaje": msg, "archivo": file})
        log.log(self.LEVELS[sev], "[%s] %s%s", code, msg, f" (archivo: {file})" if file else "")

    def validation(self, name, result, evidence):
        self.validations.append({"validacion": name, "resultado": result, "evidencia": evidence})
        log.info("VALIDACION %-12s %-5s %s", name, result, evidence)

    def write_csv(self, name, rows, fields):
        with open(self.run_dir / name, "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # ---- 1. descubrimiento ----
    def step_discover(self):
        root = Path(self.a.input_root).resolve()
        if not root.is_dir():
            raise PipelineError("RUTA_ENTRADA", f"No existe el directorio de entrada: {root}. "
                                f"Verifique --input-root.")
        self.root = root
        if any(t in norm_token(str(root)) for t in EXCLUDE_TOKENS):
            self.issue("INFORMATIVO", "RAIZ_CON_TOKEN", "La ruta raiz contiene un token de exclusion; la "
                       "exclusion se evalua solo sobre la ruta relativa a la raiz.")
        extra = re.compile(self.a.exclude_extra, re.I) if self.a.exclude_extra else None
        csvs = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"),
                      key=lambda p: str(p.relative_to(root)).lower())
        for p in csvs:
            rel = str(p.relative_to(root))
            hits = sorted({t for part in p.relative_to(root).parts for t in EXCLUDE_TOKENS
                           if t in norm_token(part)})
            size = p.stat().st_size
            rec = {"rel": rel, "path": str(p), "size_bytes": size, "estado": "INCLUIDO", "motivo": ""}
            if hits:
                rec.update(estado="EXCLUIDO", motivo="ruta contiene: " + ",".join(hits))
            elif extra and extra.search(rel):
                rec.update(estado="EXCLUIDO", motivo=f"--exclude-extra /{self.a.exclude_extra}/")
            elif size == 0:
                rec.update(estado="VACIO", motivo="tamano 0 bytes")
            else:
                try:
                    with open(p, "rb") as f:
                        f.read(1)
                except OSError as e:
                    rec.update(estado="ILEGIBLE", motivo=str(e))
            self.inv.append(rec)
        cnt = Counter(r["estado"] for r in self.inv)
        self.R["inventario"] = {"total_csv": len(csvs), **dict(cnt)}
        log.info("Inventario: %s", self.R["inventario"])
        for r in self.inv:
            if r["estado"] in ("VACIO", "ILEGIBLE"):
                self.issue("ERROR", r["estado"], r["motivo"], r["rel"])
        self.included = [r for r in self.inv if r["estado"] == "INCLUIDO"]
        if not self.included:
            raise PipelineError("SIN_ARCHIVOS", f"No hay CSV validos bajo {root} tras exclusiones.")
        # verificacion independiente de la exclusion
        for r in self.included:
            if any(t in norm_token(part) for part in Path(r["rel"]).parts for t in EXCLUDE_TOKENS):
                raise PipelineError("EXCLUSION_FALLIDA", f"Archivo prohibido incluido: {r['rel']}")

    # ---- 2. hashes y duplicados ----
    def step_hash(self):
        with ThreadPoolExecutor(4) as ex:
            for r, h in zip(self.included, ex.map(lambda r: sha256_file(r["path"]), self.included)):
                r["sha256"] = h
        groups = {}
        for r in self.included:
            groups.setdefault(r["sha256"], []).append(r["rel"])
        dups = [v for v in groups.values() if len(v) > 1]
        for v in dups:
            self.issue("ADVERTENCIA", "ARCHIVOS_DUPLICADOS", f"Contenido identico (SHA-256): {v}. "
                       "No se eliminan; decidir en revision.")
        self.R["duplicados_exactos"] = dups

    # ---- 3. pasada 1 ----
    def step_scan(self):
        args = [(r["path"], r["rel"]) for r in self.included]
        workers = max(1, min(MAX_SCAN_WORKERS, (os.cpu_count() or 2) - 1, len(args)))
        log.info("Pasada 1: %d archivos, %d workers", len(args), workers)
        if workers == 1:
            scans = [scan_file(*x) for x in args]
        else:
            with ProcessPoolExecutor(workers) as ex:
                scans = list(ex.map(scan_file, *zip(*args)))
        by_rel = {r["rel"]: r for r in self.included}
        crit_missing = []
        self.scans = []
        for s in scans:
            rec = by_rel[s["rel"]]
            if s["fatal"]:
                rec.update(estado="AISLADO", motivo=s["fatal"])
                self.issue("ERROR", "ARCHIVO_ILEGIBLE_O_CORRUPTO", s["fatal"], s["rel"])
            elif s["missing_critical"]:
                crit_missing.append((s["rel"], s["missing_critical"]))
            elif s["ragged_rows"] > 0:
                rec.update(estado="AISLADO", motivo=f"{s['ragged_rows']} filas con numero de campos distinto")
                self.issue("ERROR", "FILAS_IRREGULARES", f"{s['ragged_rows']} filas con distinto numero de "
                           f"campos (fila, n_campos)={s['ragged_examples']}", s["rel"])
            elif s["rows_csv"] != s["rows"]:
                rec.update(estado="AISLADO", motivo=f"filas csv={s['rows_csv']} vs pandas={s['rows']}")
                self.issue("ERROR", "CONTEO_FILAS_INCONSISTENTE", rec["motivo"], s["rel"])
            elif s["rows"] == 0:
                rec.update(estado="AISLADO", motivo="sin filas de datos")
                self.issue("ERROR", "SIN_FILAS", "Archivo con solo cabecera", s["rel"])
            else:
                if s["enc_note"] != "ok":
                    self.issue("ADVERTENCIA", "ENCODING_INCIERTO", s["enc_note"], s["rel"])
                if s["case_variants"]:
                    self.issue("ADVERTENCIA", "COLUMNAS_CASE", f"Cabeceras criticas con otra capitalizacion: "
                               f"{s['case_variants']}", s["rel"])
                self.scans.append(s)
        if crit_missing:
            for rel, m in crit_missing:
                self.issue("CRITICO", "COLUMNAS_CRITICAS_AUSENTES", f"Faltan {m}", rel)
            raise PipelineError("COLUMNAS_CRITICAS_AUSENTES",
                                f"{len(crit_missing)} archivo(s) sin columnas fundamentales. Revise el inventario o "
                                f"use --exclude-extra para excluir archivos que no son parte del alcance.")

    # ---- 4. esquema canonico y tipos ----
    def step_schema(self):
        by_rel = {r["rel"]: r for r in self.included}
        hdrs = Counter(tuple(s["header"]) for s in self.scans)
        if not hdrs:
            raise PipelineError("SIN_ARCHIVOS", "Ningun archivo supero la validacion de la pasada 1.")
        top = max(hdrs.values())
        ref = next(tuple(s["header"]) for s in self.scans if hdrs[tuple(s["header"])] == top)
        self.ref_header = list(ref)
        self.ref_cm = next(s["crit_map"] for s in self.scans if tuple(s["header"]) == ref)
        keep = []
        for s in self.scans:
            h = s["header"]
            if tuple(h) == ref:
                keep.append(s)
            elif set(h) == set(ref):
                self.issue("ADVERTENCIA", "ORDEN_COLUMNAS", "Mismo conjunto de columnas, distinto orden; "
                           "se reordena por nombre (regla conocida).", s["rel"])
                keep.append(s)
            else:
                miss, extra = [c for c in ref if c not in h], [c for c in h if c not in ref]
                by_rel[s["rel"]].update(estado="AISLADO", motivo=f"esquema distinto: faltan={miss} sobran={extra}")
                self.issue("ERROR", "ESQUEMA_DISTINTO", by_rel[s["rel"]]["motivo"], s["rel"])
        self.scans = keep
        isolated = sum(r["estado"] == "AISLADO" for r in self.included)
        frac = isolated / len(self.included)
        self.R["archivos_aislados"] = isolated
        if frac > MAX_ISOLATED_FILE_FRACTION:
            raise PipelineError("DEMASIADOS_AISLADOS", f"{isolated}/{len(self.included)} archivos aislados "
                                f"({frac:.1%} > {MAX_ISOLATED_FILE_FRACTION:.0%}). Esquema irreconciliable o "
                                f"alcance mal definido; revise inventario y log.")
        clash = [c for c in PROVENANCE if c in self.ref_header]
        if clash:
            raise PipelineError("COLISION_COLUMNAS", f"Las columnas {clash} ya existen en los datos.")

        # --- chequeos agregados de pasada 1 ---
        agg = {c: Counter() for c in CONTABLES + ["DEL", "EDAD"]}
        lo, hi = {}, {}
        sexo = Counter()
        for s in self.scans:
            sexo.update(s["sexo"])
            for c in agg:
                agg[c].update(s["reasons"][c])
                if s["lo"][c] is not None:
                    lo[c] = s["lo"][c] if c not in lo else min(lo[c], s["lo"][c])
                    hi[c] = s["hi"][c] if c not in hi else max(hi[c], s["hi"][c])
        self.p1 = {"reasons": {c: dict(v) for c, v in agg.items()}, "sexo_original": dict(sexo)}
        bad_del = {str(k): int(v) for k, v in agg["DEL"].items() if k not in ("OK", "VACIO")}
        if bad_del:
            toks = Counter()
            for s in self.scans:
                toks.update(s["tokens"]["DEL"])
            raise PipelineError("DEL_VALORES_NO_ENTEROS", f"DEL contiene valores que no son enteros no "
                                f"negativos: {bad_del}; ejemplos={dict(toks.most_common(10))}. No se imputa "
                                f"sobre una variable cuyo tipo no esta resuelto; decida el tratamiento.")
        for c in CONTABLES + ["EDAD"]:
            if c not in lo:
                raise PipelineError("SIN_VALORES_VALIDOS", f"{c} no tiene ningun valor numerico valido.")
        self.dtypes = {c: smallest_int(lo[c], hi[c]) for c in CONTABLES + ["EDAD"]}
        self.dtypes["DEL"] = smallest_int(min(lo.get("DEL", 0), DEL_SENTINEL), max(hi.get("DEL", 0), DEL_SENTINEL))
        self.ranges = {c: [lo.get(c), hi.get(c)] for c in lo}
        log.info("Tipos elegidos (min/max reales): %s | rangos: %s", self.dtypes, self.ranges)
        unexpected = {}
        for k, v in sexo.items():
            try:
                code = int(float(k)) if k != "" and float(k) == int(float(k)) else None
            except ValueError:
                code = None
            if not ((code in SEXO_MAP) or (k in SEXO_FINAL_ALLOWED)):
                unexpected[k] = v
        log.info("SEXO valores originales (pasada 1): %s", dict(sexo))
        if unexpected:
            self.issue("ERROR", "SEXO_VALORES_INESPERADOS", f"Valores fuera de {list(SEXO_MAP)}: {unexpected}. "
                       f"Se etiquetaran '{SEXO_UNMAPPED}' conservando el original en SEXO_ORIGINAL.")
        # esquema final
        f = []
        for h in self.ref_header:
            canon = next((k for k, v in self.ref_cm.items() if v == h), None)
            f.append((h, "string" if canon in (None, "SEXO") else self.dtypes[canon]))
        f += [("SEXO_ORIGINAL", "string"), ("DEL_IMPUTADO", "bool"), ("ARCHIVO_ORIGEN", "string"),
              ("ANIO_ORIGEN", "int16"), ("FILA_ORIGEN", "int64")]
        self.fields = f
        self.rej_fields = [(h, "string") for h in self.ref_header] + [("ARCHIVO_ORIGEN", "string"),
                                                                       ("FILA_ORIGEN", "int64"),
                                                                       ("MOTIVO", "string")]

    # ---- 5. transformacion de un chunk ----
    def transform_chunk(self, df, rel, anio, offset, st):
        cm = self.ref_cm
        n = len(df)
        fila = np.arange(offset + 1, offset + n + 1, dtype=np.int64)
        P = {c: parse_int_col(df[cm[c]]) for c in CONTABLES + ["DEL", "EDAD"]}

        # SEXO
        raw = df[cm["SEXO"]].str.strip()
        num = pd.to_numeric(raw.where(raw != ""), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        isint = ~np.isnan(num) & ~np.isinf(num) & (num == np.floor(np.nan_to_num(num)))
        mapped = np.full(n, SEXO_UNMAPPED, dtype=object)
        for code, label in SEXO_MAP.items():
            mapped[isint & (num == code)] = label
        lab = raw.isin(SEXO_FINAL_ALLOWED).to_numpy()
        mapped[lab] = raw.to_numpy()[lab]
        st["sexo_original"].update(raw.value_counts().to_dict())

        # motivos de rechazo (prioridad: contable -> EDAD -> cuadratura)
        reason = np.full(n, "", dtype=object)
        for c in CONTABLES:
            r = P[c][1]
            m = (reason == "") & (r != "OK")
            reason[m] = ["CONTABLE_%s_%s" % (c, x) for x in r[m]]
        er, en = P["EDAD"][1].copy(), P["EDAD"][0]
        er[(er == "OK") & ((en < EDAD_MIN) | (en > EDAD_MAX))] = "FUERA_DE_RANGO"
        m = (reason == "") & (er != "OK")
        reason[m] = ["EDAD_" + x for x in er[m]]
        ct_ok = np.all([P[c][1] == "OK" for c in CONTABLES], axis=0)
        ln, sv, nv, ns = (np.nan_to_num(P[c][0]) for c in CONTABLES)
        cuad_fail = ct_ok & (ln != sv + nv + ns)
        st["cuadratura_violaciones_detectadas"] += int(cuad_fail.sum())
        reason[(reason == "") & cuad_fail] = "CUADRATURA_LN_NE_SV_NV_NS"
        if cuad_fail.any() and len(st["ejemplos_cuadratura"]) < MAX_EXAMPLES:
            for i in np.flatnonzero(cuad_fail)[: MAX_EXAMPLES - len(st["ejemplos_cuadratura"])]:
                st["ejemplos_cuadratura"].append({"archivo": rel, "fila": int(fila[i]), "LN": ln[i], "SV": sv[i],
                                                  "NV": nv[i], "NS": ns[i]})
        dr = P["DEL"][1]
        if (~np.isin(dr, ["OK", "VACIO"])).any():
            raise PipelineError("DEL_CAMBIO_ENTRE_PASADAS", f"DEL con valores no validos en {rel}.")
        del_empty = dr == "VACIO"

        accept = reason == ""
        rej = ~accept
        st["rows_in"] += n
        st["rows_out"] += int(accept.sum())
        st["del_vacios_originales"] += int(del_empty.sum())
        st["del_imputados"] += int((del_empty & accept).sum())
        st["vacios_exactos"].update({h: int((df[h] == "").sum()) for h in self.ref_header})
        for k, v in zip(*np.unique(reason[rej].astype(str), return_counts=True)):
            st["motivos"][str(k)] += int(v)
            st["motivos_por_archivo"][rel][str(k)] += int(v)

        out = {}
        inv = {v: k for k, v in cm.items()}
        for h in self.ref_header:
            c = inv.get(h)
            if c == "SEXO":
                out[h] = mapped[accept]
            elif c == "DEL":
                v = np.where(del_empty, DEL_SENTINEL, np.nan_to_num(P["DEL"][0]))[accept]
                out[h] = v.astype(self.dtypes["DEL"])
            elif c:
                out[h] = np.nan_to_num(P[c][0])[accept].astype(self.dtypes[c])
            else:
                out[h] = df[h].to_numpy()[accept]
        out["SEXO_ORIGINAL"] = raw.to_numpy()[accept]
        out["DEL_IMPUTADO"] = del_empty[accept]
        out["ARCHIVO_ORIGEN"] = np.full(int(accept.sum()), rel, dtype=object)
        out["ANIO_ORIGEN"] = np.full(int(accept.sum()), -1 if anio is None else anio, dtype="int16")
        out["FILA_ORIGEN"] = fila[accept]
        clean = pd.DataFrame(out)
        st["sexo_final"].update(pd.Series(out[cm["SEXO"]]).value_counts().to_dict())
        rejected = {}
        if rej.any():
            r = df[rej].reset_index(drop=True)
            r["ARCHIVO_ORIGEN"], r["FILA_ORIGEN"], r["MOTIVO"] = rel, fila[rej], reason[rej]
            for kind, mask in (("contabilidad", np.array([x.startswith(("CONTABLE_", "CUADRATURA")) for x in reason[rej]])),
                               ("edad", np.array([x.startswith("EDAD_") for x in reason[rej]]))):
                if mask.any():
                    rejected[kind] = r[mask]
        # estadisticas de rangos finales
        for c in ("EDAD", "DEL"):
            v = out[cm[c]]
            if len(v):
                st["final_min"][c] = min(st["final_min"].get(c, v.min()), v.min())
                st["final_max"][c] = max(st["final_max"].get(c, v.max()), v.max())
        return clean, rejected

    # ---- 6. pasada 2 ----
    def step_transform(self):
        self.tmp_main = self.run_dir / ("ine_dataset_limpio" + self.ext + ".partial")
        self.sink = self.sink_cls(self.tmp_main, self.fields)
        self.rej_sinks, self.rej_paths = {}, {}
        st = {"rows_in": 0, "rows_out": 0, "del_vacios_originales": 0, "del_imputados": 0,
              "cuadratura_violaciones_detectadas": 0, "ejemplos_cuadratura": [], "sexo_original": Counter(),
              "sexo_final": Counter(), "vacios_exactos": Counter(), "motivos": Counter(),
              "motivos_por_archivo": {}, "final_min": {}, "final_max": {}}
        from collections import defaultdict
        st["motivos_por_archivo"] = defaultdict(Counter)
        by_rel = {r["rel"]: r for r in self.included}
        per_file = []
        for s in sorted(self.scans, key=lambda s: s["rel"].lower()):
            rec = by_rel[s["rel"]]
            if os.stat(rec["path"]).st_size != rec["size_bytes"]:
                raise PipelineError("ARCHIVO_MODIFICADO", f"{s['rel']} cambio durante la ejecucion.")
            anio = year_from_rel(s["rel"])
            if anio is None:
                self.issue("ADVERTENCIA", "SIN_ANIO", "No se pudo inferir el anio desde la ruta (ANIO_ORIGEN=-1).", s["rel"])
            before_in, before_out = st["rows_in"], st["rows_out"]
            offset = 0
            for ch in read_csv_chunks(rec["path"], s["encoding"], s["delimiter"]):
                ch.columns = [c.strip() for c in ch.columns]
                ch = ch[self.ref_header]
                clean, rejected = self.transform_chunk(ch, s["rel"], anio, offset, st)
                offset += len(ch)
                if len(clean):
                    self.sink.write(clean)
                for kind, rdf in rejected.items():
                    if kind not in self.rej_sinks:
                        pth = self.run_dir / f"registros_invalidos_{kind}{self.ext}.partial"
                        self.rej_paths[kind] = pth
                        self.rej_sinks[kind] = self.sink_cls(pth, self.rej_fields)
                    self.rej_sinks[kind].write(rdf[[n for n, _ in self.rej_fields]])
            fin, fout = st["rows_in"] - before_in, st["rows_out"] - before_out
            per_file.append({"archivo": s["rel"], "anio": anio, "filas_in": fin, "filas_out": fout,
                             "filas_rechazadas": fin - fout, "sha256": rec["sha256"], "encoding": s["encoding"],
                             "delimitador": s["delimiter"]})
            log.info("  %-60s in=%d out=%d rechazadas=%d", s["rel"][-60:], fin, fout, fin - fout)
        self.sink.close()
        for k in self.rej_sinks.values():
            k.close()
        self.st, self.per_file = st, per_file
        pct = 100 * (st["rows_in"] - st["rows_out"]) / max(st["rows_in"], 1)
        log.info("Filas: in=%d out=%d rechazadas=%d (%.3f%%)", st["rows_in"], st["rows_out"],
                 st["rows_in"] - st["rows_out"], pct)
        if st["rows_out"] == 0:
            raise PipelineError("DATASET_VACIO", "Ninguna fila supero las reglas de calidad.")
        if pct > self.a.max_reject_pct:
            raise PipelineError("RECHAZO_EXCESIVO", f"{pct:.2f}% de filas rechazadas supera --max-reject-pct="
                                f"{self.a.max_reject_pct}. Revise las reglas (p.ej. EDAD_MAX) con la evidencia "
                                f"en registros_invalidos_*; el dataset NO se promueve.")
        if st["rows_in"] - st["rows_out"]:
            self.issue("ERROR", "FILAS_RECHAZADAS", f"{st['rows_in'] - st['rows_out']} filas rechazadas "
                       f"({pct:.3f}%): {dict(st['motivos'])}")

    # ---- 7. validacion final sobre el artefacto escrito ----
    def step_validate(self):
        st, cm = self.st, self.ref_cm
        exp_names = [n for n, _ in self.fields]
        desc, nrows = self.sink_cls.describe(self.tmp_main)
        v_ok = True
        # esquema
        if desc is not None:
            exp = [(n, ("bool" if t == "bool" else t)) for n, t in self.fields]
            schema_ok = [(n, t) for n, t in desc] == exp
            self.validation("Esquema", "PASS" if schema_ok else "FAIL", f"{len(desc)} columnas; tipos criticos="
                            f"{ {c: self.dtypes[c] for c in self.dtypes} }")
            v_ok &= schema_ok
        else:
            self.validation("Esquema", "OBS", "sink de prueba: tipos no verificables en CSV")
        # recorrido de lectura
        tot = viol = null_crit = del_null = sexo_bad = edad_bad = del_imp = 0
        sexo_counts = Counter()
        allowed = set(SEXO_FINAL_ALLOWED) | {SEXO_UNMAPPED}
        emin = emax = None
        for b in self.sink_cls.read_batches(self.tmp_main, self.fields):
            tot += len(b)
            ln, sv, nv, ns = (b[cm[c]].astype("int64") for c in CONTABLES)
            viol += int((ln != sv + nv + ns).sum())
            null_crit += int(b[[cm[c] for c in CONTABLES]].isna().sum().sum())
            del_null += int(b[cm["DEL"]].isna().sum())
            del_imp += int(b["DEL_IMPUTADO"].sum())
            sexo_counts.update(b[cm["SEXO"]].value_counts(dropna=False).to_dict())
            sexo_bad += int((~b[cm["SEXO"]].isin(allowed)).sum())
            e = b[cm["EDAD"]]
            edad_bad += int(e.isna().sum() + ((e < EDAD_MIN) | (e > EDAD_MAX)).sum())
            emin = e.min() if emin is None else min(emin, e.min())
            emax = e.max() if emax is None else max(emax, e.max())
        self.final_sexo = dict(sexo_counts)
        self.validation("Filas", "PASS" if tot == st["rows_out"] == nrows else "FAIL",
                        f"leidas={tot} metadata={nrows} esperadas={st['rows_out']}")
        v_ok &= tot == st["rows_out"] == nrows
        self.validation("Cuadratura", "PASS" if viol == 0 and null_crit == 0 else "FAIL",
                        f"violaciones LN!=SV+NV+NS en salida={viol}; nulos contables={null_crit}; "
                        f"detectadas y eliminadas en proceso={st['cuadratura_violaciones_detectadas']}")
        v_ok &= viol == 0 and null_crit == 0
        unm = sexo_counts.get(SEXO_UNMAPPED, 0)
        self.validation("SEXO", "PASS" if sexo_bad == 0 and unm == 0 else ("OBS" if sexo_bad == 0 else "FAIL"),
                        f"distribucion final={self.final_sexo}; fuera de permitidos={sexo_bad}; {SEXO_UNMAPPED}={unm}")
        v_ok &= sexo_bad == 0
        self.validation("DEL", "PASS" if del_null == 0 and del_imp == st["del_imputados"] else "FAIL",
                        f"vacios originales={st['del_vacios_originales']} imputados en salida={del_imp} "
                        f"nulos finales={del_null}")
        v_ok &= del_null == 0 and del_imp == st["del_imputados"]
        self.validation("EDAD", "PASS" if edad_bad == 0 else "FAIL",
                        f"invalidos en salida={edad_bad}; rango final=[{emin},{emax}] regla=[{EDAD_MIN},{EDAD_MAX}]")
        v_ok &= edad_bad == 0
        if not v_ok:
            raise PipelineError("VALIDACION_FINAL_FALLIDA", "Una o mas validaciones criticas fallaron; el dataset "
                                "NO se promueve (se mantiene como .partial para inspeccion).")
        # promover (renombrado atomico)
        final = self.run_dir / ("ine_dataset_limpio" + self.ext)
        os.replace(self.tmp_main, final)
        self.R["salidas"] = {"dataset": str(final)}
        for k, pth in self.rej_paths.items():
            dest = Path(str(pth)[: -len(".partial")])
            os.replace(pth, dest)
            self.R["salidas"][f"registros_invalidos_{k}"] = str(dest)
        self.final_path = final
        self.validation("Parquet",
                        "PASS" if self.a.sink == "parquet" else "NO_EJECUTADO",
                        f"{final.name} existe, abre y contiene {tot} filas" if self.a.sink == "parquet"
                        else "sink de prueba CSV: NO se genero ni valido Parquet")

    # ---- 8. reporte ----
    def finalize(self, status_exc=None):
        dur = time.perf_counter() - self.t0
        st = getattr(self, "st", None)
        crit = [i for i in self.issues if i["severidad"] == "CRITICO"]
        if status_exc is not None or crit:
            status, label = "FALLIDA", "NO VALIDADA"
        elif self.a.sink != "parquet":
            status, label = "REQUIERE VALIDACION", "REQUIERE VALIDACION"
        else:
            obs = any(i["severidad"] == "ERROR" for i in self.issues) or any(
                v["resultado"] == "OBS" for v in self.validations)
            status, label = (("COMPLETADA CON OBSERVACIONES", "VALIDADA CON OBSERVACIONES") if obs
                             else ("COMPLETADA", "VALIDADA"))
        self.R.update(status=status, calidad=label, duracion_s=round(dur, 2), issues=self.issues,
                      validaciones=self.validations, argumentos=vars(self.a),
                      entorno={"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                               "so": platform.platform(), "cpus": os.cpu_count()},
                      config={"SEXO_MAP": SEXO_MAP, "DEL_SENTINEL": DEL_SENTINEL, "EDAD_RANGO": [EDAD_MIN, EDAD_MAX],
                              "EXCLUDE_TOKENS": EXCLUDE_TOKENS, "CHUNK_ROWS": CHUNK_ROWS,
                              "COMPRESION": f"{COMPRESSION}:{COMPRESSION_LEVEL}"})
        if hasattr(self, "dtypes"):
            self.R["tipos_finales"] = self.dtypes
            self.R["rangos_observados"] = self.ranges
            self.R["pasada1"] = self.p1
            self.R["esquema_canonico"] = [
                {"columna": n, "tipo_original": "string(CSV)", "tipo_final": t,
                 "transformacion": ("mapeo SEXO_MAP" if n == self.ref_cm["SEXO"] else
                                    f"imputacion {DEL_SENTINEL}+bandera" if n == self.ref_cm["DEL"] else
                                    "entero minimo real" if n in [self.ref_cm[c] for c in CONTABLES + ['EDAD']] else
                                    "procedencia/derivada" if n in PROVENANCE else "ninguna (texto)"),
                 "vacios_exactos_entrada": (st["vacios_exactos"].get(n) if st else None)}
                for n, t in self.fields]
        if st:
            fi, fo = st["rows_in"], st["rows_out"]
            self.R["filas"] = {"filas_originales": fi, "filas_invalidas": fi - fo, "filas_eliminadas": fi - fo,
                               "filas_finales": fo, "porcentaje_eliminado": round(100 * (fi - fo) / max(fi, 1), 4)}
            self.R["rechazos_por_motivo"] = dict(st["motivos"])
            self.R["cuadratura"] = {"violaciones_detectadas": st["cuadratura_violaciones_detectadas"],
                                    "ejemplos": st["ejemplos_cuadratura"]}
            self.R["SEXO"] = {"distribucion_original": dict(st["sexo_original"]),
                              "distribucion_final": getattr(self, "final_sexo", dict(st["sexo_final"]))}
            self.R["DEL"] = {"vacios_originales": st["del_vacios_originales"],
                             "pct_vacios_originales": round(100 * st["del_vacios_originales"] / max(fi, 1), 4),
                             "valor_imputacion": DEL_SENTINEL, "imputados_en_salida": st["del_imputados"]}
            self.R["por_archivo"] = self.per_file
        self.R["archivos"] = [{k: r.get(k) for k in ("rel", "estado", "motivo", "size_bytes", "sha256")} for r in self.inv]
        try:
            with open(self.run_dir / "reporte_calidad.json", "x", encoding="utf-8") as f:
                json.dump(self.R, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            self.write_csv("inventario_archivos.csv", self.inv, ["rel", "estado", "motivo", "size_bytes", "sha256"])
            self.write_csv("archivos_excluidos.csv", [r for r in self.inv if r["estado"] != "INCLUIDO"],
                           ["rel", "estado", "motivo", "size_bytes"])
            idx = self.out_root / "runs_index.csv"
            new = not idx.exists()
            with open(idx, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "run_dir", "status", "calidad"])
                w.writerow([self.stamp, str(self.run_dir), status, label])
        except Exception as e:
            log.error("No se pudo escribir el reporte: %r", e)
        log.info("ESTADO=%s | CALIDAD=%s | duracion=%.1fs | salida=%s", status, label, dur, self.run_dir)
        return 0 if status in ("COMPLETADA", "COMPLETADA CON OBSERVACIONES") else (1 if status != "FALLIDA" else 2)

    def run(self):
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        log.info("Fase 1 v%s | input=%s | sink=%s", PIPELINE_VERSION, self.a.input_root, self.a.sink)
        exc = None
        try:
            if self.a.sink == "parquet":
                try:
                    import pyarrow  # noqa: F401
                except ImportError:
                    raise PipelineError("SIN_PYARROW", "pyarrow no esta instalado: pip install pyarrow")
            self.step_discover()
            self.step_hash()
            self.step_scan()
            self.step_schema()
            self.step_transform()
            self.step_validate()
        except PipelineError as e:
            exc = e
            self.issue("CRITICO", e.code, str(e))
        except Exception as e:
            exc = e
            log.exception("Error inesperado")
            self.issue("CRITICO", "ERROR_INESPERADO", repr(e))
        return self.finalize(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", default="data/CCPC")
    ap.add_argument("--out-root", default="data/Salida")
    ap.add_argument("--max-reject-pct", type=float, default=DEFAULT_MAX_REJECT_PCT)
    ap.add_argument("--exclude-extra", default=None, help="regex adicional (case-insensitive) sobre la ruta relativa")
    ap.add_argument("--sink", choices=["parquet", "csv"], default="parquet", help="'csv' = SOLO pruebas")
    return Pipeline(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())