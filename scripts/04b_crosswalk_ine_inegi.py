#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT = Path(__file__).resolve().parent
P2 = SCRIPT.parent
RAW = P2 / "CCPC" / "DatosAbiertos_DECEyEC_ConteosCensales2024"
MUN = P2 / "conjunto_de_datos" / "00mun.shp"
OUT = SCRIPT / "salida"
XW = OUT / "crosswalk_ine_inegi_2024.csv"
REP = OUT / "reporte_crosswalk_2024.json"
SIN = OUT / "municipios_ine_sin_geometria_2024.csv"

ALIASES = {
    ("04", "006"): ("006", "alias_typo"),
    ("05", "007"): ("007", "alias_ortografia"),
    ("07", "038"): ("039", "alias_typo"),
    ("07", "107"): ("071", "alias_typo"),
    ("10", "006"): ("006", "alias_nombre_corto"),
    ("13", "050"): ("036", "alias_typo"),
    ("17", "013"): ("013", "alias_nombre_corto"),
    ("19", "010"): ("010", "alias_nombre_corto"),
    ("19", "014"): ("014", "alias_abreviatura"),
    ("19", "015"): ("015", "alias_abreviatura"),
    ("19", "016"): ("016", "alias_abreviatura"),
    ("19", "020"): ("020", "alias_abreviatura"),
    ("19", "021"): ("021", "alias_abreviatura"),
    ("19", "022"): ("022", "alias_abreviatura"),
    ("19", "023"): ("023", "alias_abreviatura"),
    ("19", "024"): ("024", "alias_abreviatura"),
    ("19", "025"): ("025", "alias_abreviatura"),
    ("20", "043"): ("043", "alias_nombre_oficial"),
    ("20", "121"): ("124", "alias_nombre_oficial"),
    ("20", "459"): ("459", "alias_nombre_oficial"),
    ("20", "562"): ("038", "alias_nombre_corto"),
    ("24", "001"): ("001", "alias_nombre_corto"),
    ("29", "037"): ("037", "alias_typo"),
    ("30", "010"): ("009", "alias_typo"),
    ("30", "047"): ("045", "alias_nombre_corto"),
}

DUPLICADOS = {
    ("20", "208"): "208",
    ("20", "209"): "209",
    ("20", "316"): "318",
    ("20", "317"): "319",
}

SIN_GEOMETRIA = {
    ("12", "082"): "Ñuu Savi",
    ("12", "083"): "Las Vigas",
    ("12", "084"): "San Nicolás",
    ("12", "085"): "Santa Cruz del Rincón",
    ("25", "019"): "Eldorado",
    ("25", "020"): "Juan José Ríos",
}

EJEMPLOS = {
    "21-115": "21-114",
    "21-116": "21-115",
    "09-007": "09-007",
    "20-316": "20-318",
}


def norm(s) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.casefold()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    files = sorted(RAW.glob("datosabiertos_deceyec_conteoscensales2024_*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron archivos 2024 en {RAW}")
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=["EDOCVE", "EDONOM", "MPIOCVE", "MPIONOM"], dtype=str)
        frames.append(d)
    ine = pd.concat(frames, ignore_index=True)
    ine["EDOCVE"] = ine["EDOCVE"].str.zfill(2)
    ine["MPIOCVE"] = ine["MPIOCVE"].str.zfill(3)
    ine["nname"] = ine["MPIONOM"].map(norm)
    ine = ine.drop_duplicates(["EDOCVE", "MPIOCVE", "MPIONOM"]).sort_values(["EDOCVE", "MPIOCVE"]).reset_index(drop=True)
    dup_codes = ine.duplicated(["EDOCVE", "MPIOCVE"], keep=False)
    if dup_codes.any():
        raise SystemExit(f"Claves INE duplicadas con nombres distintos: {int(dup_codes.sum())}")

    mun = gpd.read_file(MUN)[["CVE_ENT", "CVE_MUN", "NOMGEO"]].copy()
    mun["EDOCVE"] = mun["CVE_ENT"].astype(str).str.zfill(2)
    mun["MPIOCVE"] = mun["CVE_MUN"].astype(str).str.zfill(3)
    mun["nname"] = mun["NOMGEO"].map(norm)
    mun = mun.drop_duplicates(["EDOCVE", "MPIOCVE"]).sort_values(["EDOCVE", "MPIOCVE"]).reset_index(drop=True)
    if mun.duplicated(["EDOCVE", "MPIOCVE"]).any():
        raise SystemExit("La cartografía tiene claves municipales duplicadas")

    mun_idx = mun.set_index(["EDOCVE", "MPIOCVE"])
    rows = []
    for _, r in ine.iterrows():
        ent, ine_code, nombre = r["EDOCVE"], r["MPIOCVE"], r["MPIONOM"]
        key = (ent, ine_code)
        target = None
        metodo = ""
        if key in SIN_GEOMETRIA:
            metodo = "sin_geometria"
        elif key in ALIASES:
            target, metodo = ALIASES[key]
        elif key in DUPLICADOS:
            target, metodo = DUPLICADOS[key], "duplicado_orden"
        else:
            cand = mun[(mun["EDOCVE"] == ent) & (mun["nname"] == r["nname"])]
            if len(cand) == 1:
                target, metodo = cand["MPIOCVE"].iloc[0], "exacto"
            elif len(cand) > 1:
                raise SystemExit(f"Nombre ambiguo sin regla: {ent}-{ine_code} {nombre}")
            else:
                raise SystemExit(f"Sin match exacto ni alias: {ent}-{ine_code} {nombre}")

        inegi_name = ""
        if target is not None:
            tkey = (ent, target)
            if tkey not in mun_idx.index:
                raise SystemExit(f"Destino INEGI inexistente: {ent}-{target} para {ent}-{ine_code}")
            inegi_name = str(mun_idx.loc[tkey, "NOMGEO"])
        rows.append({
            "EDOCVE": ent,
            "EDONOM": r["EDONOM"],
            "MPIOCVE_INE": ine_code,
            "MPIONOM_INE": nombre,
            "MPIOCVE_INEGI": target,
            "NOMGEO_INEGI": inegi_name,
            "CLAVE_INE": f"{ent}-{ine_code}",
            "CLAVE_INEGI": f"{ent}-{target}" if target is not None else "",
            "metodo": metodo,
            "estado": "mapeado" if target is not None else "sin_geometria",
        })

    out = pd.DataFrame(rows)
    if not out["CLAVE_INE"].is_unique:
        raise SystemExit("El crosswalk no es único por clave INE")
    mapped = out[out["MPIOCVE_INEGI"].notna()].copy()
    if not mapped["CLAVE_INEGI"].is_unique:
        raise SystemExit("El crosswalk asigna una misma geometría a más de un municipio INE")
    if set(mapped["CLAVE_INEGI"]) != set(mun["EDOCVE"] + "-" + mun["MPIOCVE"]):
        faltan = sorted(set(mun["EDOCVE"] + "-" + mun["MPIOCVE"]) - set(mapped["CLAVE_INEGI"]))
        sobran = sorted(set(mapped["CLAVE_INEGI"]) - set(mun["EDOCVE"] + "-" + mun["MPIOCVE"]))
        raise SystemExit(f"Cobertura de geometrías incompleta. Faltan={faltan} Sobran={sobran}")

    for k, v in EJEMPLOS.items():
        got = out.loc[out["CLAVE_INE"] == k, "CLAVE_INEGI"]
        if len(got) != 1 or got.iloc[0] != v:
            raise SystemExit(f"Ejemplo de control falló: {k} -> {got.tolist()} (esperado {v})")

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(XW, index=False, encoding="utf-8")
    out[out["estado"] == "sin_geometria"][
        ["EDOCVE", "MPIOCVE_INE", "MPIONOM_INE", "metodo"]
    ].rename(columns={"MPIOCVE_INE": "MPIOCVE", "MPIONOM_INE": "MPIONOM"}).to_csv(
        SIN, index=False, encoding="utf-8"
    )

    sha = hashlib.sha256(XW.read_bytes()).hexdigest()
    report = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "estado": "PASSED",
        "fuente_ine": str(RAW),
        "fuente_inegi": str(MUN),
        "filas_ine": int(len(out)),
        "mapeadas": int(len(mapped)),
        "sin_geometria": int((out["estado"] == "sin_geometria").sum()),
        "tasa_mapeo": round(len(mapped) / len(out), 6),
        "geometrias_inegi": int(len(mun)),
        "geometrias_usadas": int(mapped["CLAVE_INEGI"].nunique()),
        "metodos": out["metodo"].value_counts().to_dict(),
        "sin_geometria_detalle": out.loc[out["estado"] == "sin_geometria",
                                          ["CLAVE_INE", "MPIONOM_INE", "metodo"]].to_dict("records"),
        "ejemplos": {k: out.loc[out["CLAVE_INE"] == k, "CLAVE_INEGI"].iloc[0] for k in EJEMPLOS},
        "sha256_crosswalk": sha,
    }
    REP.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
