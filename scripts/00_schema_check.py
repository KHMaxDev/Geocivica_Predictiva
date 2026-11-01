#!/usr/bin/env python3
"""
CCPC - Fase 0: diagnostico no destructivo de esquemas (2021 vs 2024 vs Voto Anticipado 2024).

Decisiones tecnicas (justificacion):
- Solo libreria estandar (csv, codecs, hashlib, logging, json). Para esta fase solo se leen
  cabeceras y unas pocas filas: pandas/polars/DuckDB no aportan ventaja y anaden dependencias.
  Recomendados para Fase 1+ (comparacion de valores/tipos): DuckDB o Polars (lazy/streaming).
- Sin paralelismo ni GPU: leer ~32 cabeceras + N filas es I/O trivial; la GPU no aplica a CSV.
- Lectura en modo binario + decodificacion explicita: permite detectar BOM/encoding y fallar
  con claridad en lugar de que una libreria "adivine" silenciosamente.
- Se comparan las cabeceras de TODOS los CSV de cada carpeta contra el representante
  (coste ~0, evita que un solo archivo enmascare diferencias internas de la carpeta).
- Solo lectura sobre los datos; toda salida va a <root>/outputs/ con timestamp (sin sobrescribir).

Uso:
    python phase0_schema_check.py --root Prueba_2 [--sample-rows 200] [--hash]
"""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import logging
import re
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]  # latin-1 decodifica todo: ultimo recurso
DELIMS = [",", ";", "|", "\t"]
NULL_TOKENS = {"", "NA", "N/A", "NULL", "NONE", "NAN", "-", "S/D"}
log = logging.getLogger("phase0")


class Phase0Error(Exception):
    """Error critico con contexto estructurado (fail-fast)."""

    def __init__(self, what, where, file, expected, fix):
        self.what, self.where, self.file, self.expected, self.fix = what, where, file, expected, fix
        super().__init__(
            f"\n  Que ocurrio : {what}\n  Donde       : {where}\n  Archivo     : {file}\n"
            f"  Se esperaba : {expected}\n  Solucion    : {fix}"
        )


@dataclass
class FileProfile:
    path: str
    size_bytes: int
    encoding: str
    encoding_note: str
    bom: bool
    delimiter: str
    n_columns: int
    columns: list
    first_rows: list
    sample_rows_read: int
    field_count_mismatches: int
    duplicate_columns: list
    empty_column_names: list
    nonascii_columns: list
    whitespace_columns: list
    apparent_types: dict
    null_pct_sample: dict
    decimal_comma_suspects: list
    sha256: str | None = None
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------- descubrimiento
def _csvs(folder: Path):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"),
                  key=lambda p: p.name.lower())


def find_folder(csv_root: Path, include: str, exclude: str | None = None) -> Path:
    cands = sorted(d for d in csv_root.rglob("*") if d.is_dir() and re.search(include, d.name, re.I)
                   and not (exclude and re.search(exclude, d.name, re.I)))
    cands = [d for d in cands if any(p.suffix.lower() == ".csv" for p in d.iterdir() if p.is_file())]
    if not cands:
        raise Phase0Error("No se encontro carpeta candidata", f"busqueda recursiva en {csv_root}", "-",
                          f"carpeta cuyo nombre coincida con /{include}/ y contenga CSV",
                          "Verifique --root y los nombres reales de carpetas")
    if len(cands) > 1:
        raise Phase0Error("Multiples carpetas candidatas (ambiguo)", str(csv_root), "-",
                          "exactamente una carpeta", "Candidatas: " + "; ".join(map(str, cands)) +
                          ". Use --dir-2021/--dir-2024/--dir-va para fijar la ruta")
    return cands[0]


def pick_representative(folder: Path) -> Path:
    files = [p for p in _csvs(folder) if p.stat().st_size > 0]
    if not files:
        raise Phase0Error("Carpeta sin CSV validos (vacios o inexistentes)", str(folder), "-",
                          "al menos un CSV no vacio", "Revise el contenido de la carpeta")
    return files[0]  # criterio determinista: orden alfabetico, primer CSV no vacio


# ----------------------------------------------------------------- lectura
def detect_encoding(path: Path, probe_bytes=4 * 1024 * 1024):
    try:
        with open(path, "rb") as f:
            raw = f.read(probe_bytes)
    except PermissionError as e:
        raise Phase0Error("Permisos insuficientes", "apertura del archivo", str(path),
                          "permiso de lectura", f"Ajuste permisos ({e})")
    if not raw:
        raise Phase0Error("Archivo vacio", "lectura inicial", str(path), "contenido", "Revise la descarga/origen")
    bom = raw.startswith(codecs.BOM_UTF8)
    for enc in ENCODINGS:
        try:
            # tolerar corte multibyte al final del bloque de prueba
            raw.decode(enc) if len(raw) < probe_bytes else raw[:-4].decode(enc)
            note = "decodificacion estricta OK en bloque inicial"
            if enc == "latin-1":
                note = "ULTIMO RECURSO: latin-1 no falla nunca; encoding real incierto"
            return enc, note, bom
        except UnicodeDecodeError:
            continue
    raise Phase0Error("Encoding no reconocido", "deteccion", str(path), str(ENCODINGS), "Indique encoding manualmente")


def detect_delimiter(header_line: str, path: Path) -> str:
    counts = {d: header_line.count(d) for d in DELIMS}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise Phase0Error("No se detecto delimitador en la cabecera", "analisis de header", str(path),
                          f"alguno de {DELIMS}", "Archivo de una sola columna o delimitador inusual; revisar a mano")
    ties = [d for d, c in counts.items() if c == counts[best]]
    if len(ties) > 1:
        raise Phase0Error(f"Delimitador ambiguo {ties}", "analisis de header", str(path), "uno dominante",
                          "Indique delimitador manualmente")
    return best


def apparent_type(values):
    vals = [v for v in values if v.strip().upper() not in NULL_TOKENS]
    if not vals:
        return "vacio"
    if all(re.fullmatch(r"[+-]?\d+", v.strip()) for v in vals):
        return "entero"
    if all(re.fullmatch(r"[+-]?\d+[.]\d+", v.strip()) for v in vals):
        return "decimal(.)"
    if all(re.fullmatch(r"[+-]?\d+,\d+", v.strip()) for v in vals):
        return "decimal(,)"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}([ T].*)?|\d{1,2}/\d{1,2}/\d{2,4}.*", v.strip()) for v in vals):
        return "fecha?"
    return "texto"


def profile_file(path: Path, sample_rows: int, do_hash: bool) -> FileProfile:
    enc, note, bom = detect_encoding(path)
    try:
        with open(path, "r", encoding=enc, newline="") as f:
            header_line = f.readline()
            if not header_line.strip():
                raise Phase0Error("Primera linea vacia", "lectura de cabecera", str(path), "cabecera", "Revise el archivo")
            delim = detect_delimiter(header_line, path)
            f.seek(0)
            reader = csv.reader(f, delimiter=delim)
            header = [h.replace("\ufeff", "") for h in next(reader)]
            rows, mism = [], 0
            for i, r in enumerate(reader):
                if i >= sample_rows:
                    break
                rows.append(r)
                if len(r) != len(header):
                    mism += 1
    except UnicodeDecodeError as e:
        raise Phase0Error("Error de decodificacion", "lectura de muestra", str(path), f"texto valido en {enc}", str(e))
    except csv.Error as e:
        raise Phase0Error("CSV ilegible", "parsing", str(path), "CSV bien formado", str(e))

    seen, dups = set(), []
    for h in header:
        if h in seen and h not in dups:
            dups.append(h)
        seen.add(h)
    if dups:
        raise Phase0Error(f"Columnas duplicadas: {dups}", "validacion de cabecera", str(path),
                          "nombres unicos", "Decida regla de desduplicacion antes de continuar")

    cols_vals = {h: [r[i] for r in rows if i < len(r)] for i, h in enumerate(header)}
    types = {h: apparent_type(v) for h, v in cols_vals.items()}
    nulls = {h: (round(100 * sum(x.strip().upper() in NULL_TOKENS for x in v) / len(v), 1) if v else None)
             for h, v in cols_vals.items()}
    notes = []
    if enc == "latin-1":
        notes.append("Encoding incierto (latin-1 fallback)")
    if len(rows) < 3:
        notes.append(f"Solo {len(rows)} filas de datos disponibles")
    sha = None
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return FileProfile(
        path=str(path), size_bytes=path.stat().st_size, encoding=enc, encoding_note=note, bom=bom,
        delimiter=repr(delim), n_columns=len(header), columns=header, first_rows=rows[:3],
        sample_rows_read=len(rows), field_count_mismatches=mism, duplicate_columns=dups,
        empty_column_names=[i for i, h in enumerate(header) if not h.strip()],
        nonascii_columns=[h for h in header if not h.isascii()],
        whitespace_columns=[h for h in header if h != h.strip()],
        apparent_types=types, null_pct_sample=nulls,
        decimal_comma_suspects=[h for h, t in types.items() if t == "decimal(,)"],
        sha256=sha, notes=notes,
    )


def read_header_only(path: Path):
    enc, _, _ = detect_encoding(path)
    with open(path, "r", encoding=enc, newline="") as f:
        line = f.readline()
    d = detect_delimiter(line, path)
    return [h.replace("\ufeff", "") for h in next(csv.reader([line], delimiter=d))], enc, d


# ----------------------------------------------------------------- comparacion
def compare_schemas(a: list, b: list) -> dict:
    sa, sb = set(a), set(b)
    common = [c for c in a if c in sb]
    same_set = sa == sb
    same_order_common = common == [c for c in b if c in sa]
    moved = []
    if same_set and a != b:
        moved = [(c, a.index(c), b.index(c)) for c in a if a.index(c) != b.index(c)]
    return {
        "n_a": len(a), "n_b": len(b),
        "same_count": len(a) == len(b),
        "same_names_exact": same_set,
        "same_order": a == b,
        "same_order_among_common": same_order_common,
        "only_in_a": [c for c in a if c not in sb],
        "only_in_b": [c for c in b if c not in sa],
        "case_insensitive_matches": sorted({c.lower().strip() for c in sa} & {c.lower().strip() for c in sb}
                                           - {c.lower() for c in sa & sb}),
        "moved_columns(name,pos_a,pos_b)": moved,
        "schema_identical": a == b,
    }


def folder_header_consistency(folder: Path, rep_cols: list) -> dict:
    out = {"n_csv": 0, "identical_to_rep": 0, "different": {}, "errors": {}}
    for p in _csvs(folder):
        out["n_csv"] += 1
        try:
            cols, _, _ = read_header_only(p)
            if cols == rep_cols:
                out["identical_to_rep"] += 1
            else:
                out["different"][p.name] = compare_schemas(rep_cols, cols)
        except Phase0Error as e:
            out["errors"][p.name] = str(e)
    return out


# ----------------------------------------------------------------- main
def setup_logging(out_dir: Path, stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = out_dir / f"phase0_schema_check_{stamp}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(lp, encoding="utf-8", mode="x"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return lp


def show(label, p: FileProfile):
    log.info("=" * 78)
    log.info("%s: %s", label, p.path)
    log.info("  tamano=%.1f MB | encoding=%s (%s) | BOM=%s | delim=%s", p.size_bytes / 1e6, p.encoding,
             p.encoding_note, p.bom, p.delimiter)
    log.info("  n_columnas=%d | filas muestreadas=%d | filas con n campos distinto=%d", p.n_columns,
             p.sample_rows_read, p.field_count_mismatches)
    log.info("  columnas=%s", p.columns)
    for i, r in enumerate(p.first_rows, 1):
        preview = ", ".join(f"{c}={v!r}" for c, v in list(zip(p.columns, r))[:8])
        log.info("  fila %d (primeros 8 campos): %s%s", i, preview, " ..." if len(r) > 8 else "")
    if p.nonascii_columns: log.info("  columnas no-ASCII: %s", p.nonascii_columns)
    if p.whitespace_columns: log.info("  columnas con espacios en bordes: %s", p.whitespace_columns)
    if p.decimal_comma_suspects: log.info("  posibles decimales con coma: %s", p.decimal_comma_suspects)
    for n in p.notes: log.warning("  nota: %s", n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Prueba_2")
    ap.add_argument("--sample-rows", type=int, default=200)
    ap.add_argument("--hash", action="store_true", help="SHA-256 de cada archivo representativo (lee archivo completo)")
    ap.add_argument("--dir-2021"); ap.add_argument("--dir-2024"); ap.add_argument("--dir-va")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    csv_root = root / "CCPC"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "outputs"
    t0 = time.perf_counter(); tracemalloc.start()
    try:
        if not csv_root.is_dir():
            raise Phase0Error("Carpeta inexistente", "inicio", str(csv_root), "directorio CCPC bajo --root",
                              "Verifique --root (ruta que contiene CCPC/)")
        lp = setup_logging(out_dir, stamp)
        log.info("Fase 0 | inicio | root=%s | python=%s", root, sys.version.split()[0])

        d21 = Path(a.dir_2021) if a.dir_2021 else find_folder(csv_root, r"ConteosCensales2021$")
        d24 = Path(a.dir_2024) if a.dir_2024 else find_folder(csv_root, r"ConteosCensales2024$", r"Voto")
        dva = Path(a.dir_va) if a.dir_va else find_folder(csv_root, r"VotoAnticipado2024$")
        reps = {"2021": pick_representative(d21), "2024": pick_representative(d24), "VA": pick_representative(dva)}
        log.info("Representantes (orden alfabetico, primer CSV no vacio): %s", {k: str(v) for k, v in reps.items()})

        prof = {k: profile_file(p, a.sample_rows, a.hash) for k, p in reps.items()}
        for k, label in (("2021", "Archivo 2021"), ("2024", "Archivo 2024"), ("VA", "Archivo Voto Anticipado")):
            show(label, prof[k])

        cmp_21_24 = compare_schemas(prof["2021"].columns, prof["2024"].columns)
        cmp_21_va = compare_schemas(prof["2021"].columns, prof["VA"].columns)
        cmp_24_va = compare_schemas(prof["2024"].columns, prof["VA"].columns)
        consistency = {"2021": folder_header_consistency(d21, prof["2021"].columns),
                       "2024": folder_header_consistency(d24, prof["2024"].columns),
                       "VA": folder_header_consistency(dva, prof["VA"].columns)}

        log.info("=" * 78)
        log.info("COMPARACION 2021 vs 2024 estandar")
        for k, v in cmp_21_24.items():
            log.info("  %s: %s", k, v)
        log.info("  >> schema_identical = %s", cmp_21_24["schema_identical"])
        log.info("COMPARACION informativa 2021 vs VA: identical=%s | solo2021=%s | soloVA=%s",
                 cmp_21_va["schema_identical"], cmp_21_va["only_in_a"], cmp_21_va["only_in_b"])
        log.info("COMPARACION informativa 2024 vs VA: identical=%s | solo2024=%s | soloVA=%s",
                 cmp_24_va["schema_identical"], cmp_24_va["only_in_a"], cmp_24_va["only_in_b"])
        for k, c in consistency.items():
            log.info("Consistencia interna carpeta %s: %d CSV, %d identicos al representante, %d distintos, %d con error",
                     k, c["n_csv"], c["identical_to_rep"], len(c["different"]), len(c["errors"]))

        if cmp_21_24["schema_identical"]:
            interp = "Los encabezados de 2021 y 2024 (representantes) coinciden en cantidad, nombres y orden. Esto NO prueba equivalencia de contenido ni semantica."
        else:
            interp = ("Se detecto una diferencia estructural en el esquema. Se requiere analizar las columnas afectadas "
                      "y su semantica antes de determinar compatibilidad.")
        log.info("INTERPRETACION: %s", interp)
        log.info("Validaciones sugeridas Fase 1: tipos completos (DuckDB read_csv sniff), categorias de SEXO y otras "
                 "variables, nulos reales por columna, conteo de filas, equivalencia semantica de columnas, "
                 "formatos de fecha/decimal.")

        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        report = {"timestamp": stamp, "root": str(root), "representatives": {k: str(v) for k, v in reps.items()},
                  "profiles": {k: asdict(v) for k, v in prof.items()},
                  "cmp_2021_vs_2024": cmp_21_24, "cmp_2021_vs_VA": cmp_21_va, "cmp_2024_vs_VA": cmp_24_va,
                  "folder_consistency": consistency, "interpretation": interp,
                  "duration_s": round(elapsed, 3), "peak_python_mem_mb": round(peak, 2)}
        jp = out_dir / f"phase0_schema_check_{stamp}.json"
        with open(jp, "x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("Duracion=%.3fs | memoria pico Python=%.2f MB | log=%s | json=%s", elapsed, peak, lp, jp)
        return 0
    except Phase0Error as e:
        msg = f"FALLO CRITICO (fail-fast):{e}"
        (log.error if log.handlers else print)(msg)
        return 2
    except Exception as e:  # no ocultar errores inesperados
        (log.exception if log.handlers else print)(f"ERROR INESPERADO: {e!r}")
        return 3


if __name__ == "__main__":
    sys.exit(main())