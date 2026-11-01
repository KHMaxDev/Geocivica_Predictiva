#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 5 — GIS, integración cartográfica y mapa web interactivo
=============================================================

Une el resultado de la Fase 4 (escenarios 2024) con la cartografía municipal
local y genera un mapa Folium del riesgo de abstención PREDICHO por el modelo.

IMPORTANTE: el mapa muestra predicciones y un ajuste condicional (banda
contable-asociativa). No muestra abstención observada ni constituye una
estimación causal de una campaña o política real.

Ejecutar desde la raíz de Geocivica_Predictiva:

    python scripts/05_mapa_interactivo.py
    python scripts/05_mapa_interactivo.py --cartografia data/conjunto_de_datos/00mun.shp
    python scripts/05_mapa_interactivo.py --tolerancia 500 --browser-check

Rutas (relativas a la raíz del proyecto):
    entrada : scripts/salida/escenarios_finales_2024.csv   (Fase 4)
    carto   : data/conjunto_de_datos/**  (.shp .geojson .gpkg .zip)
    salidas : scripts/salida/mapa_riesgo_2024.html
              scripts/salida/reporte_mapa_2024.json
              scripts/salida/municipios_sin_geometria_2024.csv
              scripts/salida/logs/mapa_interactivo.log
    `data/Salida/` (Fases 1-2) y `scripts/salida/` (Fases 3-5) son directorios distintos; aquí solo se escribe en `scripts/salida/`.

Convención de impacto (documentada en el mapa):
    impacto_intervencion = prediccion_base − prediccion_simulada
    impacto_pp = impacto_intervencion × 100   (puntos porcentuales)
    + = el ajuste REDUCE el riesgo predicho ;  − = lo AUMENTA.
    El ajuste central es el punto medio de la banda contable-asociativa;
    no es un efecto causal identificado.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Rutas (derivadas de la ubicación del script = Geocivica_Predictiva/scripts/)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "salida"                         # scripts/salida
LOG_DIR = OUT_DIR / "logs"
HIST_DIR = OUT_DIR / "mapas_historial"
CSV_PATH = OUT_DIR / "escenarios_finales_2024.csv"
CROSSWALK_PATH = OUT_DIR / "crosswalk_ine_inegi_2024.csv"
CROSSWALK_REPORT = OUT_DIR / "reporte_crosswalk_2024.json"
HTML_PATH = OUT_DIR / "mapa_riesgo_2024.html"
REPORT_PATH = OUT_DIR / "reporte_mapa_2024.json"
MISSING_PATH = OUT_DIR / "municipios_sin_geometria_2024.csv"
SIN_INEGI_PATH = OUT_DIR / "municipios_ine_sin_geometria_2024.csv"
VENDOR_DIR = OUT_DIR / "vendor"
LOG_PATH = LOG_DIR / "mapa_interactivo.log"
CARTO_DIRS = [PROJECT_DIR / "data" / "conjunto_de_datos"]
CARTO_FALLBACK_DIRS = [PROJECT_DIR / "data" / "CCPC"]    # solo si la primera no tiene cartografía


def rel(p) -> str:
    """Ruta relativa a scripts/ (nunca absoluta) para logs y reportes."""
    try:
        return Path(os.path.relpath(Path(p).resolve(), SCRIPT_DIR)).as_posix()
    except ValueError:
        return Path(p).name


# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
YEAR = 2024
REQUIRED_CSV = ["EDOCVE", "MPIOCVE", "AELEC", "prediccion_riesgo_base",
                "prediccion_riesgo_simulado", "impacto_intervencion"]
OPTIONAL_CSV = ["riesgo_abstencion"]                    # valor observado (si existe)
MATCH_MIN = 0.95
MATCH_ABS_MIN = 2400
HTML_WARN_MB = 100.0
EXT = {".shp", ".geojson", ".gpkg", ".zip"}
ENT_W, MUN_W = 2, 3                                     # formato oficial: 2 y 3 dígitos
MX_UNIVERSE = (2000, 2800)                              # nº plausible de municipios de México
MX_BBOX = (-120.0, 14.0, -84.0, 33.5)                   # lon/lat razonables (incluye islas)
TARGET_VERTICES = 350_000                               # presupuesto de vértices para el HTML
TOL_CANDIDATES_M = [100, 250, 500, 1000, 2000]
WORK_CRS_MX = "EPSG:6372"                               # ITRF2008 / México LCC (métrico)
PRECISION_GRID_DEG = 1e-4                               # ~11 m, coherente con tolerancia ≥ 100 m

ENT_NAMES = ["CVE_ENT", "CVEENT", "CVE_EDO", "EDOCVE", "ID_ENT", "ID_ESTADO", "ENT",
             "CVE_ENTIDAD", "CODIGO_ENT", "ENTIDAD", "IDESTADO", "CVE_ESTADO"]
MUN_NAMES = ["CVE_MUN", "CVEMUN", "CVE_MPIO", "MPIOCVE", "ID_MUN", "ID_MUNICIPIO", "MUN",
             "CVE_MUNICIPIO", "CODIGO_MUN", "IDMUN", "CVE_MUNI", "MUNICIPIO"]
GEO_NAMES = ["CVEGEO", "CVE_GEO", "CVE_ENT_MUN", "CVEGEO_MUN"]
NAME_ENT = ["NOM_ENT", "NOMBRE_ENT", "NOM_ESTADO", "ESTADO", "NOMBRE_ESTADO", "NOM_EDO", "ENTIDAD"]
NAME_MUN = ["NOMGEO", "NOM_MUN", "NOMBRE_MUN", "NOM_MUNICIPIO", "MUNICIPIO", "NOM_MPIO", "NOMBRE"]
SUBMUN = {"CVE_LOC", "CVE_AGEB", "AGEB", "SECCION", "CVE_SECC", "DISTRITO", "CVE_DTO",
          "LOCALIDAD", "CVE_MZA", "MZA", "NOM_LOC", "ID_SECCION"}

log = logging.getLogger("fase5")
VALIDATIONS: dict[str, str] = {}
TIMES: dict[str, float] = {}


class MapError(RuntimeError):
    def __init__(self, msg: str, status: str = "CRITICAL"):
        super().__init__(msg)
        self.status = status


class _WarnCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0

    def emit(self, record):
        if record.levelno == logging.WARNING:
            self.n += 1


WARNINGS = _WarnCounter()


def check(cond: bool, msg: str, key: str | None = None) -> None:
    if key:
        if not cond:
            VALIDATIONS[key] = "FALLÓ"
        else:
            VALIDATIONS.setdefault(key, "OK")
    if not cond:
        log.critical("VALIDACIÓN FALLIDA: %s", msg)
        raise MapError(msg)
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
    WARNINGS.n = 0
    log.addHandler(WARNINGS)


class timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        TIMES[self.name] = round(time.perf_counter() - self.t, 3)
        log.info("Tiempo %-14s: %.2f s", self.name, TIMES[self.name])


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    tmp.replace(path)


def archive_previous(path: Path) -> None:
    if path.exists():
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        HIST_DIR.mkdir(parents=True, exist_ok=True)
        dest = HIST_DIR / f"{path.stem}_{stamp}{path.suffix}"
        k = 1
        while dest.exists():
            dest = HIST_DIR / f"{path.stem}_{stamp}_{k}{path.suffix}"
            k += 1
        shutil.move(str(path), str(dest))
        log.info("Resultado previo archivado: %s", rel(dest))


def versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for n in ("geopandas", "shapely", "folium", "branca", "pyogrio", "fiona", "pyproj"):
        try:
            v[n] = getattr(__import__(n), "__version__", "desconocida")
        except Exception:  # noqa: BLE001
            v[n] = None
    return v


# --------------------------------------------------------------------------
# Normalización de claves
# --------------------------------------------------------------------------
def canon_code(series: pd.Series, width: int) -> tuple[pd.Series, dict]:
    """Cadena canónica con ceros a la izquierda (sin convertir a entero a ciegas)."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\s+", "", regex=True).str.replace(r"\.0+$", "", regex=True)
    nulo = s.isna() | (s == "") | s.str.lower().isin(["nan", "none", "<na>", "null"])
    mal_formato = ~nulo & ~s.str.fullmatch(r"\d+").fillna(False)
    largo = ~nulo & ~mal_formato & (s.str.len() > width)
    ok = ~nulo & ~mal_formato & ~largo
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[ok] = s[ok].str.zfill(width)
    return out, {"nulos": int(nulo.sum()), "formato_invalido": int(mal_formato.sum()),
                 "longitud_excesiva": int(largo.sum()), "validos": int(ok.sum()),
                 "con_relleno_de_ceros": int((ok & (s.str.len() < width)).sum())}


def looks_code(s: pd.Series, maxlen: int) -> bool:
    v = s.dropna().astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    v = v[v != ""]
    return len(v) > 0 and bool(v.str.fullmatch(r"\d+").all()) and int(v.str.len().max()) <= maxlen


def has_letters(s: pd.Series) -> bool:
    v = s.dropna().astype("string")
    return len(v) > 0 and bool(v.str.contains(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]").mean() > 0.9)


# --------------------------------------------------------------------------
# 1. Escenarios (Fase 4)
# --------------------------------------------------------------------------
def load_scenarios(path: Path) -> tuple[pd.DataFrame, dict]:
    log.info("=== 1. DATASET DE ESCENARIOS ===")
    log.info("Archivo de escenarios: %s", rel(path))
    check(path.exists() and path.is_file(), f"Existe {rel(path)} (salida de Fase 4)", "dataset_2024")
    try:
        df = pd.read_csv(path, dtype={"EDOCVE": "string", "MPIOCVE": "string"})
    except Exception as e:  # noqa: BLE001
        raise MapError(f"CSV ilegible: {e}") from e
    log.info("Filas=%d | columnas=%s", len(df), list(df.columns))
    faltan = [c for c in REQUIRED_CSV if c not in df.columns]
    check(not faltan, f"Columnas requeridas presentes (faltan: {faltan})", "dataset_2024")
    aelec = pd.to_numeric(df["AELEC"], errors="coerce")
    check(aelec.notna().all() and aelec.nunique() == 1 and int(aelec.iloc[0]) == YEAR,
          f"El CSV contiene exclusivamente AELEC == {YEAR}", "dataset_2024")
    for c in ["prediccion_riesgo_base", "prediccion_riesgo_simulado", "impacto_intervencion"] + \
            [c for c in OPTIONAL_CSV if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nn = df[["prediccion_riesgo_base", "prediccion_riesgo_simulado", "impacto_intervencion"]]
    check(bool(np.isfinite(nn.to_numpy()).all()), "Predicciones e impacto finitos (sin NaN/inf)", "dataset_2024")
    mx = float(max(nn["prediccion_riesgo_base"].max(), nn["prediccion_riesgo_simulado"].max()))
    check(mx <= 1.5, f"Predicciones en escala 0–1 (máx={mx:.4f}); si ya fueran % el tooltip sería erróneo",
          "dataset_2024")

    # Claves
    log.info("Muestra EDOCVE crudo: %s | MPIOCVE crudo: %s",
             df["EDOCVE"].dropna().unique()[:5].tolist(), df["MPIOCVE"].dropna().unique()[:5].tolist())
    df["EDOCVE_JOIN"], st_e = canon_code(df["EDOCVE"], ENT_W)
    df["MPIOCVE_JOIN"], st_m = canon_code(df["MPIOCVE"], MUN_W)
    log.info("Normalización EDOCVE→%d díg.: %s", ENT_W, st_e)
    log.info("Normalización MPIOCVE→%d díg.: %s", MUN_W, st_m)
    bad = df["EDOCVE_JOIN"].isna() | df["MPIOCVE_JOIN"].isna()
    check(not bad.any(), f"Claves no nulas y normalizables (inválidas={int(bad.sum())})", "claves_normalizadas")
    imposibles = df[(df["EDOCVE_JOIN"].astype(int) < 1) | (df["EDOCVE_JOIN"].astype(int) > 32)
                    | (df["MPIOCVE_JOIN"].astype(int) < 1)]
    if len(imposibles):
        log.warning("%d filas con códigos fuera de rango (estado 1–32 / municipio ≥1); "
                    "quedarán como 'sin geometría' si no existen en la cartografía.", len(imposibles))

    dup = df.duplicated(subset=["EDOCVE_JOIN", "MPIOCVE_JOIN"], keep=False)
    if dup.any():
        d = df.loc[dup, ["EDOCVE", "MPIOCVE", "EDOCVE_JOIN", "MPIOCVE_JOIN"]].sort_values(["EDOCVE_JOIN", "MPIOCVE_JOIN"])
        d.to_csv(OUT_DIR / "diagnostico_duplicados_escenarios.csv", index=False)
        VALIDATIONS["unicidad_municipio"] = "FALLÓ"
        raise MapError(f"El CSV de escenarios tiene {int(dup.sum())} filas con clave municipal duplicada "
                       "(NO se eliminan automáticamente). Diagnóstico: salida/diagnostico_duplicados_escenarios.csv")
    VALIDATIONS["unicidad_municipio"] = "OK"
    log.info("OK  | Una fila por municipio en escenarios (%d)", len(df))
    return df, {"archivo": rel(path), "filas": int(len(df)), "ejercicio": YEAR,
                "municipios": int(len(df)), "normalizacion_EDOCVE": st_e,
                "normalizacion_MPIOCVE": st_m, "tiene_riesgo_observado": "riesgo_abstencion" in df.columns}


# --------------------------------------------------------------------------
# 1b. Crosswalk INE → INEGI
# --------------------------------------------------------------------------
def load_crosswalk() -> tuple[pd.DataFrame, dict]:
    log.info("=== 1b. CROSSWALK INE → INEGI ===")
    check(CROSSWALK_PATH.exists() and CROSSWALK_PATH.is_file(),
          f"Existe {rel(CROSSWALK_PATH)} (salida de Fase 4b)", "crosswalk_inegi")
    xw = pd.read_csv(CROSSWALK_PATH, dtype="string")
    req = ["EDOCVE", "MPIOCVE_INE", "MPIONOM_INE", "MPIOCVE_INEGI",
           "NOMGEO_INEGI", "CLAVE_INE", "CLAVE_INEGI", "metodo", "estado"]
    faltan = [c for c in req if c not in xw.columns]
    check(not faltan, f"Columnas del crosswalk presentes (faltan: {faltan})", "crosswalk_inegi")
    xw["EDOCVE_JOIN"], _ = canon_code(xw["EDOCVE"], ENT_W)
    xw["MPIOCVE_JOIN"], _ = canon_code(xw["MPIOCVE_INE"], MUN_W)
    xw["MPIOCVE_INEGI_JOIN"], _ = canon_code(xw["MPIOCVE_INEGI"], MUN_W)
    check(not xw["EDOCVE_JOIN"].isna().any() and not xw["MPIOCVE_JOIN"].isna().any(),
          "Claves INE del crosswalk normalizables", "crosswalk_inegi")
    check(xw["CLAVE_INE"].is_unique, "Una fila por clave INE en el crosswalk", "crosswalk_inegi")
    mapped = xw[xw["MPIOCVE_INEGI_JOIN"].notna()].copy()
    check(mapped["CLAVE_INEGI"].is_unique, "Una geometría INEGI por municipio INE", "crosswalk_inegi")
    check((xw["estado"] == "sin_geometria").sum() == xw["MPIOCVE_INEGI_JOIN"].isna().sum(),
          "Estado sin_geometria coherente con la ausencia de clave INEGI", "crosswalk_inegi")
    sin = xw[xw["MPIOCVE_INEGI_JOIN"].isna()]
    if len(sin):
        sin[["EDOCVE", "MPIOCVE_JOIN", "MPIONOM_INE", "metodo"]].to_csv(
            SIN_INEGI_PATH, index=False, encoding="utf-8")
    stats = {
        "archivo": rel(CROSSWALK_PATH),
        "filas_ine": int(len(xw)),
        "mapeadas": int(len(mapped)),
        "sin_geometria": int(len(sin)),
        "tasa_mapeo": round(len(mapped) / len(xw), 6) if len(xw) else 0.0,
        "metodos": xw["metodo"].value_counts().to_dict(),
        "sin_geometria_detalle": sin[["CLAVE_INE", "MPIONOM_INE", "metodo"]].to_dict("records"),
    }
    if CROSSWALK_REPORT.exists():
        try:
            stats["reporte"] = json.loads(CROSSWALK_REPORT.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("No se pudo leer %s: %s", rel(CROSSWALK_REPORT), e)
    log.info("Crosswalk: %d INE | %d mapeadas | %d sin geometría | tasa=%.2f%%",
             len(xw), len(mapped), len(sin), 100 * stats["tasa_mapeo"])
    VALIDATIONS["crosswalk_inegi"] = "OK"
    return xw, stats


# --------------------------------------------------------------------------
# 2. Cartografía
# --------------------------------------------------------------------------
@dataclass
class Cand:
    path: Path
    layer: str | None = None
    member: str | None = None          # miembro dentro de un ZIP

    @property
    def label(self) -> str:
        return " :: ".join(x for x in (rel(self.path), self.member, self.layer) if x)


def list_candidates(dirs: list[Path]) -> list[Cand]:
    out: list[Cand] = []
    for d in dirs:
        if not d.exists():
            log.info("Directorio inexistente: %s", rel(d))
            continue
        for p in sorted(d.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in EXT):
                continue
            s = p.suffix.lower()
            if s == ".zip":
                try:
                    with zipfile.ZipFile(p) as z:
                        names = z.namelist()
                except Exception as e:  # noqa: BLE001
                    log.warning("ZIP ilegible %s: %s", rel(p), e)
                    continue
                geo = [n for n in names if n.lower().endswith((".shp", ".gpkg", ".geojson"))
                       and not n.startswith("__MACOSX")]
                log.info("ZIP %s contiene %d archivos; capas geoespaciales: %s", rel(p), len(names), geo)
                for n in geo:
                    if n.lower().endswith(".shp"):      # un .shp válido necesita .dbf/.shx
                        base = n[:-4]
                        if not all(f"{base}{e}" in names or f"{base}{e.upper()}" in names for e in (".dbf", ".shx")):
                            log.warning("Shapefile incompleto en ZIP (falta .dbf/.shx): %s", n)
                            continue
                    out.append(Cand(p, member=n))
            elif s == ".gpkg":
                layers = _gpkg_layers(p)
                out.extend(Cand(p, layer=l) for l in layers) if layers else out.append(Cand(p))
            else:
                out.append(Cand(p))
    return out


def _gpkg_layers(p: Path) -> list[str]:
    try:
        import pyogrio
        return [str(x[0]) for x in pyogrio.list_layers(p)]
    except Exception:  # noqa: BLE001
        try:
            import fiona
            return list(fiona.listlayers(str(p)))
        except Exception:  # noqa: BLE001
            return []


def read_cand(c: Cand, rows: int | None, tmp: Path):
    import geopandas as gpd
    kw = {"rows": rows} if rows else {}
    if c.layer:
        kw["layer"] = c.layer
    if c.member is None:
        return gpd.read_file(c.path, **kw)
    attempts = [f"zip://{c.path.as_posix()}!{c.member}", f"/vsizip/{c.path.as_posix()}/{c.member}"]
    for uri in attempts:
        try:
            return gpd.read_file(uri, **kw)
        except Exception:  # noqa: BLE001
            continue
    # Extracción controlada a un directorio temporal (el ZIP original no se modifica)
    with zipfile.ZipFile(c.path) as z:
        base = c.member.rsplit(".", 1)[0]
        for n in z.namelist():
            if n.startswith(base + "."):
                z.extract(n, tmp)
    return gpd.read_file(tmp / c.member, **kw)


def analyze_layer(gdf, label: str) -> dict:
    gname = gdf.geometry.name
    cols = {str(c).upper(): c for c in gdf.columns if c != gname}
    info = {"candidato": label, "filas": int(len(gdf)), "ok": False, "motivos": [], "columnas": list(cols.values())}
    sub = [orig for up, orig in cols.items() if up in SUBMUN and gdf[orig].notna().any()]
    if sub:
        info["motivos"].append(f"columnas sub-municipales {sub}: no es nivel municipal")
    ent = next((cols[n] for n in ENT_NAMES if n in cols and looks_code(gdf[cols[n]], 2)), None)
    mun = next((cols[n] for n in MUN_NAMES if n in cols and looks_code(gdf[cols[n]], 3)), None)
    geo = next((cols[n] for n in GEO_NAMES if n in cols), None)
    if geo is not None:
        lens = set(gdf[geo].dropna().astype(str).str.strip().str.len().unique())
        info["longitudes_cvegeo"] = sorted(int(x) for x in lens)
        if lens == {2}:
            info["motivos"].append("CVEGEO de 2 dígitos: nivel estatal")
        elif lens and min(lens) >= 7:
            info["motivos"].append("CVEGEO ≥7 dígitos: localidad/AGEB/sección")
        elif lens == {5} and (ent is None or mun is None):
            s = gdf[geo].astype("string").str.strip()
            gdf = gdf.copy()
            gdf["__ENT"], gdf["__MUN"] = s.str[:2], s.str[2:]
            ent, mun = "__ENT", "__MUN"
    if ent is None or mun is None:
        info["motivos"].append("no se identificaron columnas de estado y municipio con contenido numérico")
    ne = next((cols[n] for n in NAME_ENT if n in cols and has_letters(gdf[cols[n]])), None)
    nm = next((cols[n] for n in NAME_MUN if n in cols and has_letters(gdf[cols[n]])), None)
    if not gdf.geometry.geom_type.dropna().isin(["Polygon", "MultiPolygon"]).all():
        info["motivos"].append("geometrías no poligonales")
    if ent and mun and not info["motivos"]:
        e, _ = canon_code(gdf[ent], ENT_W)
        m, _ = canon_code(gdf[mun], MUN_W)
        keys = pd.DataFrame({"e": e, "m": m}).dropna()
        n_unique = int(keys.drop_duplicates().shape[0])
        info.update(ent_col=ent, mun_col=mun, nombre_estado_col=ne, nombre_municipio_col=nm,
                    claves_unicas=n_unique, estados=int(keys["e"].nunique()),
                    filas_con_clave_repetida=int(len(keys) - n_unique))
        if not (MX_UNIVERSE[0] <= n_unique <= MX_UNIVERSE[1]):
            info["motivos"].append(f"{n_unique} claves únicas fuera del rango municipal esperado {MX_UNIVERSE}")
        elif info["estados"] > 33:
            info["motivos"].append(f"{info['estados']} estados (>33)")
        else:
            info["ok"] = True
    info["_gdf"] = gdf if info["ok"] else None
    return info


def guess_version(label: str, gdf) -> str:
    for c in gdf.columns:
        if str(c).upper() in ("ANIO", "AÑO", "YEAR", "VERSION", "FECHA", "FECHA_ACT"):
            v = gdf[c].dropna().astype(str).unique()[:3].tolist()
            if v:
                return f"columna {c}: {v}"
    yrs = re.findall(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)", label)
    return f"inferida del nombre: {yrs[-1]}" if yrs else "unknown"


def find_cartography(override: str | None):
    log.info("=== 2. CARTOGRAFÍA ===")
    tmp = Path(tempfile.mkdtemp(prefix="carto_"))
    try:
        if override:
            cands = [Cand(PROJECT_DIR / override)]
            check(cands[0].path.exists(), f"Existe la cartografía indicada {override}", "cartografia_municipal")
        else:
            cands = list_candidates(CARTO_DIRS)
            if not cands:
                log.warning("Sin cartografía en %s; se revisa %s (documentado).",
                            [rel(d) for d in CARTO_DIRS], [rel(d) for d in CARTO_FALLBACK_DIRS])
                cands = list_candidates(CARTO_FALLBACK_DIRS)
        if not cands:
            VALIDATIONS["cartografia_municipal"] = "FALLÓ"
            raise MapError("No hay cartografía local utilizable en data/conjunto_de_datos/ (ni en data/CCPC/). "
                           "Falta un insumo obligatorio (.shp/.geojson/.gpkg/.zip municipal). No se descarga nada.")
        log.info("Candidatos (%d):", len(cands))
        for c in cands:
            log.info("  - %s", c.label)

        good = []
        for c in cands:
            try:
                peek = read_cand(c, 5, tmp)
            except Exception as e:  # noqa: BLE001
                log.warning("No se pudo leer %s: %s", c.label, e)
                continue
            if peek.crs is None and len(peek) == 0:
                continue
            full = read_cand(c, None, tmp)
            a = analyze_layer(full, c.label)
            log.info("Evaluación %s -> filas=%d ok=%s motivos=%s", c.label, a["filas"], a["ok"], a["motivos"])
            if a["ok"]:
                a["_cand"] = c
                a["_crs"] = full.crs
                a["_version"] = guess_version(c.label, full)
                good.append(a)
        if not good:
            VALIDATIONS["cartografia_municipal"] = "FALLÓ"
            raise MapError("Ninguna capa local fue confirmada como cartografía MUNICIPAL "
                           "(revisa el log: puede ser estatal, localidades, AGEB, secciones, etc.).")

        def sort_key(a):
            yrs = re.findall(r"(19[89]\d|20[0-4]\d)", a["candidato"])
            return (-(int(yrs[-1]) if yrs else 0), a["candidato"])
        good.sort(key=sort_key)
        if len(good) > 1:
            log.warning("%d capas municipales válidas; se usa la de año más reciente en su nombre: %s. Otras: %s",
                        len(good), good[0]["candidato"], [g["candidato"] for g in good[1:]])
        sel = good[0]
        gdf = sel["_gdf"]
        sel_info = {k: v for k, v in sel.items() if not k.startswith("_")}
        sel_info["version"] = sel["_version"]
        sel_info["formato"] = sel["_cand"].path.suffix.lower().lstrip(".") + (
            f" (miembro {sel['_cand'].member})" if sel["_cand"].member else "")
        VALIDATIONS["cartografia_municipal"] = "OK"
        log.info("Cartografía encontrada: %s | Formato: %s | Versión: %s",
                 sel["candidato"], sel_info["formato"], sel["_version"])
        return gdf, sel_info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare_carto(gdf, sel_info: dict):
    import geopandas as gpd
    crs = gdf.crs
    check(crs is not None, "La cartografía tiene CRS definido (no se adivina)", "cartografia_municipal")
    log.info("CRS: %s | geográfico=%s", crs.to_string(), crs.is_geographic)
    g = gdf.copy()
    g["_EDO"], st_e = canon_code(g[sel_info["ent_col"]], ENT_W)
    g["_MPIO"], st_m = canon_code(g[sel_info["mun_col"]], MUN_W)
    log.info("Normalización cartografía estado: %s | municipio: %s", st_e, st_m)
    
    bad = g["_EDO"].isna() | g["_MPIO"].isna()
    check(not bad.any(), f"Claves de cartografía válidas (inválidas={int(bad.sum())})", "claves_normalizadas")
    
    g["_estado"] = g[sel_info["nombre_estado_col"]].astype("string") if sel_info.get("nombre_estado_col") else g["_EDO"]
    g["_mun"] = g[sel_info["nombre_municipio_col"]].astype("string") if sel_info.get("nombre_municipio_col") else g["_MPIO"]
    
    if not sel_info.get("nombre_estado_col"):
        log.info("Cartografía sin nombre de entidad; el estado se toma de EDONOM del crosswalk.")
    if not sel_info.get("nombre_municipio_col"):
        log.info("Cartografía sin nombre municipal; se usará la clave como respaldo.")

    # --- 1. PREVENCIÓN DE COLISIÓN DE COLUMNA GEOMETRY ---
    # Si existe una columna de metadatos llamada "geometry" que no es la activa, la borramos
    if "geometry" in g.columns and g.geometry.name != "geometry":
        g = g.drop(columns=["geometry"])
    # -----------------------------------------------------

    g = g[["_EDO", "_MPIO", "_estado", "_mun", g.geometry.name]]
    if g.geometry.name != "geometry":
        g = g.rename_geometry("geometry")

    dup = g.duplicated(subset=["_EDO", "_MPIO"], keep=False)
    dup_info = {"filas_con_clave_repetida": int(dup.sum())}
    
    if dup.any():
        d = g[dup]
        consistentes = (d.groupby(["_EDO", "_MPIO"])["_mun"].nunique() <= 1).all()
        log.warning("Cartografía: %d filas comparten clave municipal (%d claves). Nombres consistentes=%s",
                    int(dup.sum()), d[["_EDO", "_MPIO"]].drop_duplicates().shape[0], bool(consistentes))
        if not consistentes:
            VALIDATIONS["unicidad_municipio"] = "FALLÓ"
            raise MapError("Claves municipales repetidas en la cartografía con nombres distintos: causa no "
                           "determinada; se detiene para evitar un join incorrecto.")
        log.warning("Se disuelven (unión) las partes de un mismo municipio (p. ej., multipartes/islas); registrado.")
        
        # Al hacer dissolve, Geopandas unirá los polígonos que acabamos de renombrar
        g = g.dissolve(by=["_EDO", "_MPIO"], aggfunc={"_estado": "first", "_mun": "first"}, as_index=False)
        dup_info["disueltas"] = True
        
    g = g.reset_index(drop=True)
    check(not g.duplicated(subset=["_EDO", "_MPIO"]).any(), "Una geometría por clave municipal en cartografía (1:1)",
          "unicidad_municipio")
    
    return g, {"crs_original": crs.to_string(), "duplicados_cartografia": dup_info,
               "total_municipios_cartografia": int(len(g))}

# --------------------------------------------------------------------------
# 3. Join
# --------------------------------------------------------------------------
def join_scenarios(sc: pd.DataFrame, cg, xw: pd.DataFrame):
    log.info("=== 3. JOIN ESCENARIO → CROSSWALK INE/INEGI → CARTOGRAFÍA ===")
    x = xw[["EDOCVE_JOIN", "MPIOCVE_JOIN", "MPIOCVE_INEGI_JOIN", "CLAVE_INE",
            "CLAVE_INEGI", "MPIONOM_INE", "EDONOM", "metodo"]].rename(
                columns={"MPIOCVE_INEGI_JOIN": "MPIOCVE_INEGI"})
    try:
        sc = sc.merge(x, how="left", on=["EDOCVE_JOIN", "MPIOCVE_JOIN"],
                      indicator="_merge_xw", validate="one_to_one")
    except Exception as e:  # noqa: BLE001
        VALIDATIONS["crosswalk_inegi"] = "FALLÓ"
        raise MapError(f"El cruce escenario-crosswalk no es 1:1: {e}") from e
    sin_crosswalk = sc["_merge_xw"] != "both"
    check(not sin_crosswalk.any(),
          f"Todo municipio del escenario existe en el crosswalk (faltan={int(sin_crosswalk.sum())})",
          "crosswalk_inegi")
    sc = sc.drop(columns=["_merge_xw"])
    sin_geometria = sc["MPIOCVE_INEGI"].isna()
    if sin_geometria.any():
        log.warning("%d municipios del escenario no tienen geometría INEGI 2024: %s",
                    int(sin_geometria.sum()),
                    sc.loc[sin_geometria, "CLAVE_INE"].tolist()[:10])

    attrs = pd.DataFrame(cg.drop(columns="geometry"))
    attrs["_gi"] = np.arange(len(attrs))
    try:
        j = sc.merge(attrs, how="left", left_on=["EDOCVE_JOIN", "MPIOCVE_INEGI"],
                     right_on=["_EDO", "_MPIO"], indicator=True, validate="one_to_one")
    except Exception as e:  # noqa: BLE001
        VALIDATIONS["join_valido"] = "FALLÓ"
        raise MapError(f"El join cartográfico no es 1:1: {e}") from e
    check(len(j) == len(sc), f"El LEFT JOIN conserva todas las filas del escenario ({len(j)})", "join_valido")
    matched = j["_merge"] == "both"
    n_m, n_u = int(matched.sum()), int((~matched).sum())
    tot = int(len(sc))
    m_mun = int(j.loc[matched, "CLAVE_INEGI"].nunique())
    rate = n_m / tot if tot else 0.0
    xw_mapped = int(xw["MPIOCVE_INEGI_JOIN"].notna().sum())
    xw_sin = int(len(xw) - xw_mapped)
    stats = {
        "matched_rows": n_m,
        "unmatched_rows": n_u,
        "match_rate_filas": n_m / len(j) if len(j) else 0.0,
        "matched_municipalities": m_mun,
        "total_municipalities": tot,
        "municipality_match_rate": rate,
        "crosswalk_filas": int(len(xw)),
        "crosswalk_mapeadas": xw_mapped,
        "crosswalk_sin_geometria": xw_sin,
        "crosswalk_tasa_mapeo": round(xw_mapped / len(xw), 6) if len(xw) else 0.0,
        "scenario_sin_geometria": int(sin_geometria.sum()),
    }
    log.info("Escenario: %d | con geometría: %d | sin geometría: %d | cobertura=%.2f%% | crosswalk=%.2f%%",
             tot, n_m, n_u, 100 * rate, 100 * stats["crosswalk_tasa_mapeo"])
    solo_carto = len(cg) - m_mun
    stats["municipios_cartografia_sin_datos_2024"] = int(solo_carto)
    if solo_carto:
        log.info("Municipios de la cartografía sin fila en el escenario: %d (informativo)", solo_carto)

    miss = j.loc[~matched, ["EDOCVE", "MPIOCVE", "MPIOCVE_INEGI", "CLAVE_INE",
                            "CLAVE_INEGI", "MPIONOM_INE", "metodo"]].copy()
    if len(miss):
        miss["causa_probable"] = np.where(
            miss["MPIOCVE_INEGI"].isna(),
            "sin geometría INEGI 2024 (municipio nuevo o no incorporado al Marco Geoestadístico)",
            "clave INEGI no encontrada en la cartografía utilizada")
        archive_previous(MISSING_PATH)
        miss.sort_values("CLAVE_INE").to_csv(MISSING_PATH, index=False, encoding="utf-8")
        log.warning("%d municipios del escenario sin geometría -> %s | causas: %s",
                    len(miss), rel(MISSING_PATH), miss["causa_probable"].value_counts().to_dict())
        stats["causas_sin_match"] = miss["causa_probable"].value_counts().to_dict()
    else:
        archive_previous(MISSING_PATH)
        pd.DataFrame(columns=["EDOCVE", "MPIOCVE", "MPIOCVE_INEGI", "CLAVE_INE",
                              "CLAVE_INEGI", "MPIONOM_INE", "metodo", "causa_probable"]).to_csv(
                                  MISSING_PATH, index=False, encoding="utf-8")
        log.info("Todos los municipios del escenario encontraron geometría.")

    VALIDATIONS["match_rate_95"] = "OK" if rate >= MATCH_MIN else "FALLÓ"
    VALIDATIONS["join_valido"] = "OK"
    if rate < MATCH_MIN:
        raise MapError(f"municipality_match_rate={rate:.2%} < {MATCH_MIN:.0%}: no se genera el mapa final.")
    log.info("OK  | municipality_match_rate %.2f%% >= %.0f%%", 100 * rate, 100 * MATCH_MIN)
    if len(cg) >= MATCH_ABS_MIN and tot >= MATCH_ABS_MIN and m_mun < MATCH_ABS_MIN:
        log.warning("matched_municipalities=%d < %d (control secundario).", m_mun, MATCH_ABS_MIN)

    import geopandas as gpd
    mm = j.loc[matched].copy()
    geoms = cg.geometry.values[mm["_gi"].astype(int).to_numpy()]
    gm = gpd.GeoDataFrame(mm.drop(columns=["_merge", "_gi"]), geometry=geoms, crs=cg.crs).reset_index(drop=True)
    return gm, stats


# --------------------------------------------------------------------------
# 4. Geometrías
# --------------------------------------------------------------------------
def polygonal_only(geom):
    import shapely
    from shapely.geometry import MultiPolygon, Polygon
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        return Polygon()
    return shapely.union_all(parts)


def integrity(gm):
    import shapely
    log.info("=== 4. INTEGRIDAD GEOMÉTRICA ===")
    g = gm.geometry
    nulas = int(g.isna().sum())
    vacias = int((~g.isna() & g.is_empty).sum())
    inval = ~g.isna() & ~g.is_empty & ~g.is_valid
    n_inv = int(inval.sum())
    multi = int((g.geom_type == "MultiPolygon").sum())
    nv = shapely.get_num_coordinates(g.values)
    log.info("Geometrías nulas=%d vacías=%d inválidas=%d multiparte=%d | vértices: total=%d máx=%d",
             nulas, vacias, n_inv, multi, int(nv.sum()), int(nv.max()) if len(nv) else 0)
    reparadas = 0
    if n_inv:
        log.warning("Reparando %d geometrías inválidas con make_valid (se registra la transformación).", n_inv)
        fixed = g.copy()
        fixed[inval] = [polygonal_only(shapely.make_valid(x)) for x in g[inval]]
        still = ~fixed.isna() & ~fixed.is_empty & ~fixed.is_valid
        reparadas = int(n_inv - still[inval].sum())
        gm = gm.set_geometry(fixed)
        log.info("Reparadas=%d de %d", reparadas, n_inv)
    usable = ~gm.geometry.isna() & ~gm.geometry.is_empty & gm.geometry.is_valid
    unusable = int((~usable).sum())
    return gm, usable, {"nulas": nulas, "vacias": vacias, "invalidas": n_inv, "reparadas": reparadas,
                        "inutilizables": unusable, "multiparte": multi, "vertices_total": int(nv.sum())}


def simplify_for_web(gm, tol_forced: float | None):
    import shapely
    log.info("=== 5. SIMPLIFICACIÓN (solo copia para el HTML; originales intactos) ===")
    crs = gm.crs
    metric = (not crs.is_geographic) and crs.axis_info and "metre" in (crs.axis_info[0].unit_name or "").lower()
    work_crs = crs if metric else WORK_CRS_MX
    if not metric:
        log.info("CRS %s no es métrico: simplificando en %s y reproyectando a EPSG:4326 después.",
                 crs.to_string(), WORK_CRS_MX)
    w = gm.geometry.to_crs(work_crs)
    arr = w.values
    nv_before = int(shapely.get_num_coordinates(arr).sum())
    wkb_before = int(sum(len(b) for b in shapely.to_wkb(arr)))
    bounds = w.total_bounds
    log.info("Antes: vértices=%d | WKB=%.2f MB | extensión=%.0f x %.0f m", nv_before, wkb_before / 1e6,
             bounds[2] - bounds[0], bounds[3] - bounds[1])

    tol_used, result, method = 0.0, arr, "ninguno"
    cands = [tol_forced] if tol_forced else ([] if nv_before <= TARGET_VERTICES else TOL_CANDIDATES_M)
    for tol in cands:
        try:
            if hasattr(shapely, "coverage_simplify"):
                res = shapely.coverage_simplify(arr, tol)
                method = "shapely.coverage_simplify (preserva fronteras compartidas)"
            else:
                raise AttributeError
        except Exception:  # noqa: BLE001
            res = shapely.simplify(arr, tol, preserve_topology=True)
            method = "shapely.simplify(preserve_topology=True) (fronteras vecinas pueden separarse levemente)"
        vacio = shapely.is_empty(res) | shapely.is_missing(res)
        if vacio.any():
            res = np.where(vacio, arr, res)
        tol_used, result = float(tol), res
        nv = int(shapely.get_num_coordinates(result).sum())
        log.info("Tolerancia %s m -> vértices=%d", tol, nv)
        if nv <= TARGET_VERTICES:
            break
    else:
        if cands:
            log.warning("Con tolerancia %s m aún hay más de %d vértices; se usa la máxima.", tol_used, TARGET_VERTICES)
    n_simp = int((shapely.get_num_coordinates(result) < shapely.get_num_coordinates(arr)).sum())
    if tol_used == 0:
        log.info("No se simplifica: %d vértices ya está dentro del presupuesto (%d).", nv_before, TARGET_VERTICES)
    import geopandas as gpd
    out = gpd.GeoSeries(result, crs=work_crs, index=gm.index).to_crs("EPSG:4326")
    snapped = shapely.set_precision(out.values, PRECISION_GRID_DEG)
    ok = ~shapely.is_empty(snapped) & shapely.is_valid(snapped)
    final = np.where(ok, snapped, out.values)
    out = gpd.GeoSeries(final, crs="EPSG:4326", index=gm.index)
    inval = ~out.is_valid
    if inval.any():
        log.warning("%d geometrías inválidas tras simplificar: make_valid.", int(inval.sum()))
        out[inval] = [polygonal_only(shapely.make_valid(x)) for x in out[inval]]
    nv_after = int(shapely.get_num_coordinates(out.values).sum())
    wkb_after = int(sum(len(b) for b in shapely.to_wkb(out.values)))
    log.info("Después: vértices=%d | WKB=%.2f MB | tolerancia=%.0f m | método=%s", nv_after, wkb_after / 1e6,
             tol_used, method)
    return out, {"tolerancia_m": tol_used, "metodo": method, "vertices_antes": nv_before,
                 "vertices_despues": nv_after, "wkb_bytes_antes": wkb_before, "wkb_bytes_despues": wkb_after,
                 "geometrias_simplificadas": n_simp, "crs_trabajo": str(work_crs),
                 "precision_grid_grados": PRECISION_GRID_DEG}


def validate_after(out, n_expected: int) -> dict:
    log.info("=== 6. VALIDACIÓN POSTERIOR A SIMPLIFICACIÓN ===")
    check(len(out) == n_expected, f"Misma cantidad de municipios tras simplificar ({len(out)})")
    check(int(out.isna().sum()) == 0 and int(out.is_empty.sum()) == 0, "Sin geometrías nulas/vacías")
    n_inv = int((~out.is_valid).sum())
    if n_inv:
        log.warning("%d geometrías aún inválidas tras simplificar.", n_inv)
    minx, miny, maxx, maxy = out.total_bounds
    b = MX_BBOX
    check(b[0] <= minx and maxx <= b[2] and b[1] <= miny and maxy <= b[3],
          f"BBox razonable para México: lon[{minx:.2f},{maxx:.2f}] lat[{miny:.2f},{maxy:.2f}]")
    c = out.centroid
    fuera = int(((c.x < b[0]) | (c.x > b[2]) | (c.y < b[1]) | (c.y > b[3])).sum())
    check(fuera == 0, f"Ningún centroide desplazado fuera del territorio (fuera={fuera})")
    return {"bbox_4326": [float(minx), float(miny), float(maxx), float(maxy)], "invalidas_post": n_inv}


# --------------------------------------------------------------------------
# 5. Mapa
# --------------------------------------------------------------------------
def fmt_pct(v) -> str:
    return "s/d" if pd.isna(v) else f"{v * 100:.2f}%"


def fmt_pp(v) -> str:
    if pd.isna(v):
        return "s/d"
    x = v * 100
    return f"{x:+.2f} pp"


def scale_range(vals: np.ndarray) -> dict:
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    lo, hi = (float(np.quantile(vals, 0.01)), float(np.quantile(vals, 0.99)))
    robust = hi > lo and (vmax - vmin) > 1.25 * (hi - lo)
    a, b = (lo, hi) if robust else (vmin, vmax)
    if b <= a:
        b = a + 1e-9
    return {"min": vmin, "max": vmax, "escala_min": a, "escala_max": b, "robusta_p1_p99": bool(robust)}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%%TITULO%%</title>
<link rel="stylesheet" href="%%LEAFLET_CSS%%">
<style>
:root{--ink:#1d2733;--mut:#5b6673;--line:#d8dde3;--bg:#fff;--soft:#f3f5f7;--ac:#1f4e79}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{display:flex;font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:var(--ink);background:var(--bg)}
aside{width:350px;flex:none;height:100%;min-height:0;overflow-y:auto;border-right:1px solid var(--line);display:flex;flex-direction:column}
aside>*{padding:14px 18px;border-bottom:1px solid var(--line)}
#map{flex:1;min-width:0;height:100%;background:#e8ecef}
h1{font-size:19px;line-height:1.25;margin:0 0 6px}
h2{font-size:14px;margin:0 0 8px}
h3{font-size:14px;margin:0}
p{margin:0}
.sub,.nota,small,.vacio,footer{color:var(--mut)}
.sub{font-size:13px}
button,input{font:inherit;color:inherit}
.link{background:none;border:0;padding:0;margin-top:8px;color:var(--ac);text-decoration:underline;cursor:pointer}
.modo{display:flex;gap:8px;padding:7px 8px;margin:0 -8px;border-radius:4px;cursor:pointer}
.modo:has(input:checked){background:var(--soft)}
.modo span{display:flex;flex-direction:column}
.modo small{font-size:12.5px}
.barra{height:10px;border-radius:2px;margin-top:10px;border:1px solid rgba(0,0,0,.15)}
.ejes{display:flex;justify-content:space-between;font-size:12px;margin-top:3px}
.cats{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.cat{display:flex;align-items:center;gap:6px;font-size:12.5px}
.cat i{width:18px;height:10px;border-radius:2px;display:inline-block}
.nota{font-size:12.5px;margin-top:4px}
#q{width:100%;padding:7px 9px;border:1px solid #aab2bc;border-radius:4px;background:#fff}
#res{list-style:none;margin:4px 0 0;padding:0}
#res button,#top button{width:100%;text-align:left;background:none;border:0;padding:5px 4px;cursor:pointer;border-radius:3px;display:flex;justify-content:space-between;gap:8px}
#res button:hover,#top button:hover{background:var(--soft)}
dl{display:grid;grid-template-columns:1fr auto;gap:3px 12px;margin:8px 0 0}
dt{color:var(--mut)}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
ol{margin:6px 0 0;padding-left:22px}
#top li::marker{color:var(--mut)}
#top .v{font-variant-numeric:tabular-nums;font-weight:600}
summary{cursor:pointer;font-weight:600}
details p{margin-top:8px}
footer{margin-top:auto;font-size:12.5px;background:var(--soft);border-bottom:0}
footer p+p{margin-top:6px}
footer b{color:var(--ink)}
:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
.aviso{padding:24px}
@media(max-width:820px){body{flex-direction:column}aside{order:2;width:100%;height:auto;flex:1;border-right:0}#map{order:1;height:48dvh;flex:none}}
</style></head>
<body>
<aside>
<header>
<h1 id="t"></h1>
<p class="sub">Predicción del modelo para <span id="n"></span> municipios. No es abstención observada.</p>
<button class="link" id="todo" type="button">Ver todo el país</button>
</header>
<section><h2>Qué quieres ver</h2><div id="modos"></div><div id="leyenda"></div></section>
<section><h2>Buscar municipio</h2>
<input id="q" type="search" placeholder="Nombre del municipio o del estado" autocomplete="off" aria-label="Buscar municipio">
<ul id="res"></ul></section>
<section aria-live="polite"><h2>Municipio seleccionado</h2>
<div id="detalle"><p class="vacio">Haz clic en un municipio del mapa o búscalo por nombre.</p></div></section>
<section><h2>Resumen de lo que ves</h2><div id="resumen"></div>
<h3 id="rtit" style="margin-top:14px"></h3><ol id="top"></ol></section>
<section><details><summary>Cómo leer este mapa</summary>
<p>Cada municipio se colorea según la probabilidad que estima el modelo de que su población no participe en la elección de 2024. Son predicciones, no resultados de la votación.</p>
<p>El ajuste simula un aumento de 10 % en la tasa de voto joven (tope 100 %) manteniendo constante el voto adulto. El riesgo simulado usa el punto medio de una banda condicional: límite contable y asociación histórica 2024.</p>
<p>El impacto es el riesgo base menos el riesgo simulado, en puntos porcentuales (pp). Si es positivo, el ajuste reduce el riesgo predicho; si es negativo, lo aumenta. No es un efecto causal.</p>
<p>El semáforo usa cortes descriptivos: bajo &lt;40 %, medio 40–50 % y alto ≥50 %. No es una clasificación oficial.</p>
<p>El promedio y la mediana no están ponderados por población.</p>
</details></section>
<footer>
<p id="aviso"></p>
<p>Elaborado por <b>Wi'ijados.dev</b>. Generado el %%FECHA%%.</p>
</footer>
</aside>
<div id="map"></div>
<script id="datos" type="application/json">%%DATOS%%</script>
<script src="%%LEAFLET_JS%%"></script>
<script>
(()=>{
const D=JSON.parse(document.getElementById('datos').textContent),S=D.spec,N=D.m.length;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('t').textContent=S.title;$('n').textContent=N.toLocaleString('es-MX');$('aviso').textContent=S.disclaimer;
if(!window.L){$('map').innerHTML='<p class="aviso">No se pudo cargar Leaflet. Si el HTML se abrió sin la carpeta salida/vendor, usa conexión a Internet o vuelve a generar el mapa.</p>';return}
const V={base:D.b,sim:D.s,imp:D.i,semaforo:D.sem},KEYS=Object.keys(S.modes);
const fmt=(k,v)=>{const x=v/100;return(k==='imp'&&x>=0?'+':'')+x.toFixed(2)+(k==='imp'?' pp':'%')};
const hex=h=>[1,3,5].map(i=>parseInt(h.substr(i,2),16));
const ramp=m=>{const c=m.colors.map(hex),n=c.length-1;return x=>{let t=(x-m.vmin)/(m.vmax-m.vmin);t=t<0?0:t>1?1:t;const p=t*n,i=Math.min(Math.floor(p),n-1),f=p-i;return'rgb('+c[i].map((v,j)=>Math.round(v+(c[i+1][j]-v)*f)).join()+')'}};
const catColor=(m,x)=>{let c=m.cats[0];for(const q of m.cats){if(x>=q.min)c=q;else break}return c.color};
const COL={};KEYS.forEach(k=>{const m=S.modes[k],r=m.type==='cat'?null:ramp(m);COL[k]=V[k].map(v=>m.type==='cat'?catColor(m,v/100):r(v/100))});
const est=i=>D.e[D.ei[i]];
let mode=KEYS[0],sel=-1;
const BASE={weight:.3,color:'#6b7280'},HOVER={weight:1.6,color:'#111'};
const B=[[D.bbox[1],D.bbox[0]],[D.bbox[3],D.bbox[2]]];
const map=L.map('map',{preferCanvas:true,zoomSnap:.25});
map.attributionControl.setPrefix('Elaborado por Wi\'ijados.dev');
const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap contributors',maxZoom:12}).addTo(map);
map.fitBounds(B);
const ring=a=>{let x=a[0],y=a[1];const p=[[y/1e4,x/1e4]];for(let i=2;i<a.length;i+=2){x+=a[i];y+=a[i+1];p.push([y/1e4,x/1e4])}return p};
const tip=i=>'<b>'+esc(D.m[i])+'</b>, '+esc(est(i))+'<br>'+esc(S.modes[mode].label)+': '+fmt(mode,V[mode][i]);
const layers=D.g.map((geo,i)=>{
  const P=geo.map(p=>p.map(ring));
  const ly=L.polygon(P.length===1?P[0]:P,{...BASE,fillOpacity:.85,fillColor:COL[mode][i]});
  ly.on({mouseover:()=>{if(i!==sel)ly.setStyle(HOVER)},mouseout:()=>{if(i!==sel)ly.setStyle(BASE)},click:()=>select(i,false)});
  ly.bindTooltip(()=>tip(i),{sticky:true});
  return ly.addTo(map)});
function select(i,zoom){
  if(sel>=0&&sel!==i)layers[sel].setStyle(BASE);
  sel=i;layers[i].setStyle(HOVER);layers[i].bringToFront();
  if(zoom)map.fitBounds(layers[i].getBounds(),{maxZoom:9,padding:[30,30]});
  const o=D.o?D.o[i]:undefined,im=D.i[i];
  const row=(l,v)=>'<dt>'+l+'</dt><dd>'+v+'</dd>';
  $('detalle').innerHTML='<h3>'+esc(D.m[i])+'</h3><p class="sub">'+esc(est(i))+'. Clave INE '+esc(D.k[i])+' · INEGI '+esc(D.ki[i])+'</p><dl>'+
    (D.o?row('Riesgo observado 2024 (dato)',o==null?'s/d':fmt('base',o)):'')+
    row('Riesgo base predicho',fmt('base',D.b[i]))+row('Riesgo simulado',fmt('sim',D.s[i]))+
    row('Impacto del ajuste',fmt('imp',im))+'</dl><p class="nota">'+
    (im>0?'El ajuste reduce el riesgo predicho.':im<0?'El ajuste aumenta el riesgo predicho.':'El ajuste no cambia el riesgo predicho.')+'</p>';
}
function legend(){const m=S.modes[mode];
  if(m.type==='cat'){$('leyenda').innerHTML='<div class="cats">'+m.cats.map(c=>'<span class="cat"><i style="background:'+c.color+'"></i>'+esc(c.label)+'</span>').join('')+'</div><p class="nota">'+esc(m.note)+'</p>';return}
  $('leyenda').innerHTML='<div class="barra" style="background:linear-gradient(to right,'+m.colors.join()+')"></div><div class="ejes"><span>'+esc(m.lo)+'</span>'+(m.mid?'<span>'+esc(m.mid)+'</span>':'')+'<span>'+esc(m.hi)+'</span></div><p class="nota">'+esc(m.note)+'</p>'}
function summary(){
  const k=mode,v=V[k],s=[...v].sort((a,b)=>a-b),mean=v.reduce((a,b)=>a+b,0)/N,med=N%2?s[(N-1)/2]:(s[N/2-1]+s[N/2])/2;
  const row=(l,x)=>'<dt>'+l+'</dt><dd>'+x+'</dd>';
  $('resumen').innerHTML='<dl>'+row('Promedio',fmt(k,mean))+row('Mediana',fmt(k,med))+row('Mínimo',fmt(k,s[0]))+row('Máximo',fmt(k,s[N-1]))+
    (k==='imp'?row('Municipios donde baja el riesgo',v.filter(x=>x>0).length.toLocaleString('es-MX')+' de '+N.toLocaleString('es-MX')):'')+'</dl>';
  $('rtit').textContent=k==='imp'?'Los 10 municipios con mayor reducción':'Los 10 municipios con mayor riesgo';
  const top=v.map((x,i)=>i).sort((a,b)=>v[b]-v[a]).slice(0,10);
  $('top').innerHTML=top.map(i=>'<li><button type="button" data-i="'+i+'"><span>'+esc(D.m[i])+', '+esc(est(i))+'</span><span class="v">'+fmt(k,v[i])+'</span></button></li>').join('');
}
function setMode(k){mode=k;layers.forEach((l,i)=>l.setStyle({fillColor:COL[k][i]}));legend();summary()}
$('modos').innerHTML=KEYS.map((k,i)=>'<label class="modo"><input type="radio" name="modo" value="'+k+'"'+(i?'':' checked')+'><span><b>'+esc(S.modes[k].label)+'</b><small>'+esc(S.modes[k].help)+'</small></span></label>').join('');
$('modos').addEventListener('change',e=>setMode(e.target.value));
$('top').addEventListener('click',e=>{const b=e.target.closest('button');if(b)select(+b.dataset.i,true)});
$('todo').addEventListener('click',()=>{if(sel>=0){layers[sel].setStyle(BASE);sel=-1}map.fitBounds(B);$('detalle').innerHTML='<p class="vacio">Haz clic en un municipio del mapa o búscalo por nombre.</p>'});
const norm=s=>s.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const IDX=D.m.map((m,i)=>norm(m+' '+est(i)));
const buscar=()=>{const q=norm($('q').value.trim()),r=[];
  if(q.length>=2)for(let i=0;i<N&&r.length<8;i++)if(IDX[i].includes(q))r.push(i);
  $('res').innerHTML=r.map(i=>'<li><button type="button" data-i="'+i+'">'+esc(D.m[i])+', '+esc(est(i))+'</button></li>').join('');
  if(q.length>=2&&!r.length)$('res').innerHTML='<li class="vacio">Sin resultados. Prueba con otra parte del nombre.</li>';return r};
$('q').addEventListener('input',buscar);
$('q').addEventListener('keydown',e=>{if(e.key==='Enter'){const r=buscar();if(r.length)select(r[0],true)}});
$('res').addEventListener('click',e=>{const b=e.target.closest('button');if(b)select(+b.dataset.i,true)});
legend();summary();
window.__mapa={capas:layers.length};
})();
</script></body></html>
"""


def _enc_ring(coords) -> list[int]:
    """Anillo → enteros (1e-4°): primer punto absoluto y luego deltas (x, y). Sin punto de cierre."""
    pts = [(round(c[0] * 1e4), round(c[1] * 1e4)) for c in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    out = [pts[0][0], pts[0][1]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        out += [x1 - x0, y1 - y0]
    return out


def _enc_geom(g) -> list:
    polys = [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    return [[_enc_ring(p.exterior.coords)] + [_enc_ring(r.coords) for r in p.interiors]
            for p in polys if not p.is_empty]


LEAFLET_VERSION = "1.9.4"
LEAFLET_CDN = f"https://cdn.jsdelivr.net/npm/leaflet@{LEAFLET_VERSION}/dist"


def ensure_leaflet() -> tuple[str, str, bool]:
    """Copia Leaflet a salida/vendor para que el HTML funcione sin CDN."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    files = {"leaflet.css": f"{LEAFLET_CDN}/leaflet.css", "leaflet.js": f"{LEAFLET_CDN}/leaflet.js"}
    ok = True
    for name, url in files.items():
        p = VENDOR_DIR / name
        if p.exists() and p.stat().st_size > 1000:
            continue
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                data = r.read()
            if len(data) < 1000:
                raise ValueError(f"respuesta demasiado corta ({len(data)} bytes)")
            p.write_bytes(data)
        except Exception as e:  # noqa: BLE001
            ok = False
            log.warning("No se pudo guardar %s localmente: %s", name, e)
    if ok:
        log.info("Leaflet %s local disponible en %s", LEAFLET_VERSION, rel(VENDOR_DIR))
        return "vendor/leaflet.css", "vendor/leaflet.js", True
    log.warning("Leaflet no quedó local; el HTML usará CDN y necesitará Internet.")
    return f"{LEAFLET_CDN}/leaflet.css", f"{LEAFLET_CDN}/leaflet.js", False


def semaforo_summary(vals: np.ndarray) -> dict:
    n = int(len(vals))
    bajo = int(np.sum(vals < 0.40))
    alto = int(np.sum(vals >= 0.50))
    medio = n - bajo - alto
    return {
        "cortes_pct": {"bajo": 40.0, "medio": 50.0},
        "conteo": {"bajo": bajo, "medio": medio, "alto": alto},
        "porcentaje": {"bajo": round(100 * bajo / n, 2) if n else 0.0,
                       "medio": round(100 * medio / n, 2) if n else 0.0,
                       "alto": round(100 * alto / n, 2) if n else 0.0},
        "nota": ("Semáforo descriptivo con cortes fijos 40% y 50% sobre la predicción base; "
                 "no es una clasificación oficial del INE."),
    }


def build_map(gm, geom4326, sc_info: dict, simp: dict):
    from branca.colormap import linear
    log.info("=== 7. MAPA HTML (Leaflet + datos compactos) ===")
    df = gm.drop(columns="geometry").copy()
    base, sim, imp = (df["prediccion_riesgo_base"].to_numpy(), df["prediccion_riesgo_simulado"].to_numpy(),
                      df["impacto_intervencion"].to_numpy())
    sr = scale_range(np.concatenate([base, sim]))
    log.info("prediccion_riesgo_base: min=%.5f max=%.5f | escala común base/simulado: [%.5f, %.5f] robusta=%s",
             float(base.min()), float(base.max()), sr["escala_min"], sr["escala_max"], sr["robusta_p1_p99"])

    cm = linear.YlOrRd_09.scale(sr["escala_min"], sr["escala_max"])
    imp_pp = imp * 100
    q = np.quantile(imp_pp, [0.01, 0.99])
    a = float(max(abs(q[0]), abs(q[1]), 1e-6)) if np.abs(imp_pp).max() > 0 else 1e-6
    if np.abs(imp_pp).max() == 0:
        log.warning("Impacto = 0 en todos los municipios: la capa de impacto será de un solo color.")
    cm_imp = linear.RdBu_11.scale(-a, a)          # azul = reducción (+), rojo = aumento (−)

    def stops(c, lo, hi, n=9):
        return [c(lo + (hi - lo) * i / (n - 1))[:7] for i in range(n)]

    lo_s, hi_s = sr["escala_min"] * 100, sr["escala_max"] * 100
    rob = sr["robusta_p1_p99"]
    lo_txt = f"{lo_s:.1f}%" + (" o menos" if rob else "")
    hi_txt = f"{hi_s:.1f}%" + (" o más" if rob else "")
    spec = {
        "title": f"Geocívica Predictiva · Riesgo de abstención municipal {YEAR}",
        "modes": {
            "base": {"label": "Riesgo base predicho", "help": "Lo que estima hoy el modelo.",
                     "colors": stops(cm, sr["escala_min"], sr["escala_max"]), "vmin": lo_s, "vmax": hi_s,
                     "lo": lo_txt, "hi": hi_txt, "note": "Más intenso = mayor riesgo predicho."},
            "sim": {"label": "Riesgo simulado", "help": "Si la tasa de voto joven subiera 10 %.",
                    "colors": stops(cm, sr["escala_min"], sr["escala_max"]), "vmin": lo_s, "vmax": hi_s,
                    "lo": lo_txt, "hi": hi_txt,
                    "note": "Misma escala que el riesgo base: tasa de voto joven ×1.10 (tope 1.0)."},
            "imp": {"label": "Impacto del ajuste", "help": "Cuánto cambia el riesgo entre base y simulado.",
                    "colors": stops(cm_imp, -a, a, 11), "vmin": -a, "vmax": a,
                    "lo": f"{-a:+.2f} pp (aumenta)", "mid": "0", "hi": f"{a:+.2f} pp (reduce)",
                    "note": "Impacto = base − simulado. Azul: reducción predicha; rojo: aumento."},
            "semaforo": {"type": "cat", "label": "Semáforo de riesgo",
                         "help": "Bajo <40%, medio 40–50%, alto ≥50%.",
                         "cats": [{"label": "Bajo (<40%)", "min": 0.0, "color": "#2e7d32"},
                                  {"label": "Medio (40–50%)", "min": 40.0, "color": "#f9a825"},
                                  {"label": "Alto (≥50%)", "min": 50.0, "color": "#c62828"}],
                         "note": "Cortes descriptivos; no es una clasificación oficial."},
        },
        "disclaimer": ("Predicciones y ajuste condicional (banda contable-asociativa); "
                       "no es abstención observada ni evidencia causal de una política real."),
    }

    import geopandas as gpd
    estados = df["EDONOM"].astype(str) if "EDONOM" in df.columns else df["_estado"].astype(str)
    uniq = list(dict.fromkeys(estados))
    pos = {e: i for i, e in enumerate(uniq)}
    to_int = lambda arr: np.rint(np.asarray(arr, dtype=float) * 1e4).astype(int).tolist()   # centésimas de % / pp
    claves_ine = (df["EDOCVE_JOIN"] + "-" + df["MPIOCVE_JOIN"]).tolist()
    claves_inegi = (df["EDOCVE_JOIN"] + "-" + df["MPIOCVE_INEGI"]).tolist()
    minx, miny, maxx, maxy = geom4326.total_bounds
    data = {"spec": spec, "e": uniq, "ei": [pos[e] for e in estados], "m": df["_mun"].astype(str).tolist(),
            "k": claves_ine, "ki": claves_inegi, "b": to_int(base), "s": to_int(sim), "i": to_int(imp),
            "sem": to_int(base), "g": [_enc_geom(g) for g in geom4326.values],
            "bbox": [round(float(minx), 4), round(float(miny), 4), round(float(maxx), 4), round(float(maxy), 4)]}
    if "riesgo_abstencion" in df:
        data["o"] = [None if pd.isna(v) else int(round(float(v) * 1e4)) for v in df["riesgo_abstencion"]]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    leaflet_css, leaflet_js, leaflet_local = ensure_leaflet()
    html = (HTML_TEMPLATE.replace("%%TITULO%%", spec["title"])
            .replace("%%FECHA%%", datetime.now().strftime("%Y-%m-%d"))
            .replace("%%LEAFLET_CSS%%", leaflet_css)
            .replace("%%LEAFLET_JS%%", leaflet_js)
            .replace("%%DATOS%%", payload))

    mg = gpd.GeoDataFrame({"estado": estados.values, "municipio": data["m"],
                           "clave_ine": claves_ine, "clave_inegi": claves_inegi},
                          geometry=geom4326.values, crs="EPSG:4326")
    sem = semaforo_summary(base)
    imp_stats = {"promedio_pp": round(float(np.mean(imp_pp)), 4),
                 "mediana_pp": round(float(np.median(imp_pp)), 4),
                 "minimo_pp": round(float(np.min(imp_pp)), 4),
                 "maximo_pp": round(float(np.max(imp_pp)), 4)}
    return html, {"prediccion_base": {"min": sr["min"], "max": sr["max"]},
                  "escala_color": sr, "impacto_pp_escala_simetrica": a,
                  "impacto_pp": imp_stats, "semaforo": sem,
                  "paleta_riesgo": "YlOrRd (branca linear.YlOrRd_09)",
                  "paleta_impacto": "RdBu (azul = reducción, rojo = aumento)",
                  "convencion_impacto": "impacto_pp = (prediccion_base − prediccion_simulada) × 100; + = reducción",
                  "render": "Leaflet 1.9.4 local (scripts/salida/vendor) o CDN con datos compactos: geometrías en enteros delta, colores calculados en el navegador",
                  "leaflet_local": bool(leaflet_local),
                  "n_features": int(len(mg))}, mg


class _P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)


def verify_html(n_expected: int, browser_check: bool) -> dict:
    log.info("=== 8. VERIFICACIÓN DEL HTML ===")
    check(HTML_PATH.exists(), f"Existe {rel(HTML_PATH)}", "html_generado")
    size = HTML_PATH.stat().st_size
    check(size > 0, f"HTML > 0 bytes ({size})", "html_generado")
    txt = HTML_PATH.read_text(encoding="utf-8", errors="replace")
    p = _P()
    p.feed(txt)
    check({"html", "head", "body", "script"} <= p.tags and "</html>" in txt.lower(),
          "Estructura HTML válida (html/head/body/script y cierre)", "html_legible")
    mm = re.search(r'<script id="datos" type="application/json">(.*?)</script>', txt, re.S)
    check(mm is not None, "Datos embebidos presentes en el HTML", "html_legible")
    D = json.loads(mm.group(1))
    n_feat = len(D["g"])
    check(n_feat == n_expected and all(len(D[k]) == n_feat for k in ("m", "k", "ki", "b", "s", "i", "sem", "ei"))
          and all(len(g) > 0 for g in D["g"]),
          f"El HTML serializó {n_feat} geometrías == {n_expected} municipios (con sus valores)", "html_legible")
    ok_tip = (all(x in txt for x in ("Riesgo base predicho", "Riesgo simulado", "Impacto del ajuste",
                                     "Semáforo de riesgo"))
              and bool(D["e"]) and bool(D["m"]))
    check(ok_tip, "Vista con Estado, Municipio, Riesgo base, Riesgo simulado, Impacto y Semáforo",
          "tooltips_correctos")
    res = {"size_bytes": int(size), "size_mb": round(size / 1e6, 3), "features_en_html": n_feat}
    if size / 1e6 > HTML_WARN_MB:
        log.warning("HTML de %.1f MB > %.0f MB: aumentar la simplificación o reducir columnas.", size / 1e6, HTML_WARN_MB)
    if browser_check:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                try:
                    b = pw.chromium.launch()
                except Exception:
                    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                    if not chrome.exists():
                        raise
                    b = pw.chromium.launch(executable_path=str(chrome))
                pg = b.new_page()
                errs = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.goto(HTML_PATH.resolve().as_uri(), wait_until="load", timeout=60000)
                pg.wait_for_timeout(1200)
                n = pg.evaluate("window.__mapa ? window.__mapa.capas : 0")
                b.close()
            res["browser_paths"], res["browser_errors"] = n, errs
            if errs or n == 0:
                log.warning("Chequeo en navegador: polígonos=%d errores=%s (los CDN requieren Internet).", n, errs[:3])
            else:
                log.info("Chequeo en navegador: %d polígonos renderizados, sin errores JS.", n)
        except Exception as e:  # noqa: BLE001
            log.warning("Chequeo en navegador no disponible: %s", e)
            res["browser_check"] = f"no disponible: {e}"
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Fase 5 — GIS y mapa interactivo")
    ap.add_argument("--csv", default=None, help="CSV de escenarios (relativo a la raíz del proyecto)")
    ap.add_argument("--cartografia", default=None, help="Cartografía municipal (relativa a la raíz del proyecto)")
    ap.add_argument("--tolerancia", type=float, default=None, help="Tolerancia de simplificación en metros")
    ap.add_argument("--browser-check", action="store_true", help="Validar el HTML con Playwright si está instalado")
    args = ap.parse_args()

    t0 = time.perf_counter()
    started = datetime.now().isoformat(timespec="seconds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()
    log.info("##### FASE 5 | INICIO #####")
    csv_path = (PROJECT_DIR / args.csv) if args.csv else CSV_PATH
    archive_previous(HTML_PATH)       # el HTML/reporte canónicos siempre corresponden a ESTA ejecución
    archive_previous(REPORT_PATH)

    estado, error = "PASSED", None
    rep: dict = {"fecha_hora": started, "dataset_escenarios": rel(csv_path)}
    try:
        try:
            import branca  # noqa: F401
            import geopandas  # noqa: F401
            import shapely  # noqa: F401
        except ImportError as e:
            raise MapError(f"Falta una dependencia ({e}). Instala: pip install geopandas shapely branca pyogrio") from e
        log.info("Versiones: %s", json.dumps(versions()))

        p4 = None
        try:
            p4 = json.loads((OUT_DIR / "escenario_2024.json").read_text(encoding="utf-8"))
            log.info("Reporte Fase 4: estado=%s resultado=%s", p4.get("estado"),
                     (p4.get("resultado_hipotesis") or {}).get("estado"))
            if str(p4.get("estado", "")).startswith(("FAILED", "CRITICAL")):
                log.warning("El reporte de Fase 4 indica estado %s.", p4.get("estado"))
        except Exception:  # noqa: BLE001
            log.warning("No se pudo leer scripts/salida/escenario_2024.json (informativo).")

        with timer("carga"):
            sc, sc_info = load_scenarios(csv_path)
            xw, xw_info = load_crosswalk()
            cg_raw, sel = find_cartography(args.cartografia)
            cg, cinfo = prepare_carto(cg_raw, sel)
            del cg_raw
        rep.update(cartografia_utilizada=sel["candidato"], formato_cartografia=sel["formato"],
                   version_cartografica=sel["version"], crs_original=cinfo["crs_original"],
                   total_municipios_datos=sc_info["municipios"],
                   total_municipios_cartografia=cinfo["total_municipios_cartografia"],
                   claves_cartografia={"estado": sel["ent_col"], "municipio": sel["mun_col"],
                                       "nombre_estado": sel.get("nombre_estado_col"),
                                       "nombre_municipio": sel.get("nombre_municipio_col")},
                   crosswalk=xw_info)

        with timer("join"):
            gm, jstats = join_scenarios(sc, cg, xw)
        rep.update(municipios_con_match=jstats["matched_municipalities"],
                   municipios_sin_match=jstats["total_municipalities"] - jstats["matched_municipalities"],
                   municipios_ine_sin_geometria=jstats["crosswalk_sin_geometria"],
                   match_rate=jstats["municipality_match_rate"], join=jstats)

        gm, usable, gi = integrity(gm)
        if (~usable).any():
            log.warning("%d municipios con geometría inutilizable se excluyen del mapa (registrado).", int((~usable).sum()))
            pd.DataFrame(gm.loc[~usable, ["EDOCVE", "MPIOCVE"]]).assign(causa_probable="geometría ausente/inutilizable") \
                .to_csv(MISSING_PATH, mode="a", header=not MISSING_PATH.exists(), index=False)
            gm = gm.loc[usable].reset_index(drop=True)
        rate_final = len(gm) / sc_info["municipios"]
        check(rate_final >= MATCH_MIN, f"Cobertura final del mapa {rate_final:.2%} >= {MATCH_MIN:.0%}", "match_rate_95")
        rep.update(geometrias_invalidas=gi["invalidas"], geometrias_reparadas=gi["reparadas"], integridad_geometrica=gi)

        with timer("simplificacion"):
            geom4326, simp = simplify_for_web(gm, args.tolerancia)
        post = validate_after(geom4326, len(gm))
        VALIDATIONS["geometrias_validas"] = "OK" if post["invalidas_post"] == 0 else "WARNING"
        rep.update(geometrias_simplificadas=simp["geometrias_simplificadas"],
                   tolerancia_simplificacion=simp["tolerancia_m"], simplificacion=simp, validacion_post=post,
                   crs_final="EPSG:4326")

        # Correspondencia 1:1 mapa ↔ escenario ↔ geometría
        keys_ine = gm["CLAVE_INE"]
        keys_inegi = gm["CLAVE_INEGI"]
        check(not keys_ine.duplicated().any(), "Cada municipio INE aparece una sola vez en el mapa")
        check(not keys_inegi.duplicated().any(), "Cada geometría INEGI aparece una sola vez en el mapa")
        check(set(keys_ine) <= set(sc["EDOCVE_JOIN"] + "-" + sc["MPIOCVE_JOIN"]),
              "Todo municipio del mapa corresponde a una observación del escenario 2024")

        with timer("generacion"):
            html, minfo, mg = build_map(gm, geom4326, sc_info, simp)
            geo_bytes = len(mg[["geometry"]].to_json().encode("utf-8"))
            archive_previous(HTML_PATH)
            HTML_PATH.write_text(html, encoding="utf-8")
        vh = verify_html(len(gm), args.browser_check)

        rep.update(filas_finales=int(len(gm)), mapa=minfo, semaforo=minfo.get("semaforo"),
                   tamano_geojson_bytes=geo_bytes, html=vh,
                   html_size_bytes=vh["size_bytes"], html_size_mb=vh["size_mb"],
                   convencion_porcentajes="Valores en escala 0–1 mostrados ×100 con '%'; impacto en puntos porcentuales (pp)")
        if p4:
            rep["fase4"] = {"estado": p4.get("estado"),
                            "banda_pp": p4.get("banda_pp"),
                            "resumen_escenario": p4.get("resumen_escenario"),
                            "advertencia": p4.get("advertencia")}
        log.info("HTML generado: %s | Tamaño MB: %.2f", rel(HTML_PATH), vh["size_mb"])
    except MapError as e:
        estado, error = e.status, str(e)
        log.critical("FASE 5 DETENIDA [%s]: %s", e.status, e)
    except Exception as e:  # noqa: BLE001
        estado, error = "CRITICAL", f"{type(e).__name__}: {e}"
        log.exception("ERROR INESPERADO")

    if estado == "PASSED" and WARNINGS.n:
        estado = "PASSED + WARNINGS"
    rep.update(tiempos_s=dict(TIMES), duracion_total=round(time.perf_counter() - t0, 3), estado=estado,
               validaciones=dict(VALIDATIONS), versiones=versions(), advertencias=WARNINGS.n,
               advertencia_metodologica=("Mapa de predicciones y de un ajuste condicional "
                                         "(banda contable-asociativa); no es abstención observada "
                                         "ni una estimación causal."))
    if error:
        rep["error"] = error
        for k in ("html_size_bytes", "html_size_mb"):
            rep.setdefault(k, None)
    for k in ("cartografia_utilizada", "version_cartografica", "crs_original", "total_municipios_cartografia",
              "municipios_con_match", "municipios_sin_match", "municipios_ine_sin_geometria", "match_rate",
              "geometrias_invalidas", "geometrias_simplificadas", "tolerancia_simplificacion", "filas_finales"):
        rep.setdefault(k, None)
    rep.setdefault("crs_final", None)
    write_json(rep, REPORT_PATH)
    try:
        json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        log.info("Reporte generado y releído: %s", rel(REPORT_PATH))
    except Exception as e:  # noqa: BLE001
        log.error("Reporte ilegible: %s", e)

    log.info("Archivo de escenarios: %s | Municipios con match: %s | Sin match: %s | Match rate: %s | Estado: %s",
             rep["dataset_escenarios"], rep["municipios_con_match"], rep["municipios_sin_match"],
             f"{rep['match_rate']:.2%}" if rep.get("match_rate") is not None else "N/D", estado)
    log.info("##### FASE 5 | ESTADO FINAL: %s #####", estado)
    return 0 if estado in ("PASSED", "PASSED + WARNINGS") else 1


if __name__ == "__main__":
    sys.exit(main())