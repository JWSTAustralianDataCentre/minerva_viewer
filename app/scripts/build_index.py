#!/usr/bin/env python3
"""Build parquet indices for the MINERVA viewer.

Outputs (to {MINERVA_INDEX_DIR}/{field}/):
  catalog.parquet   one row per SUPER_CATALOG object (294,126)
  spectra.parquet   one row per matched DJA spectrum
  meta.json         counts, ranges, band lists

Usage:
  cd app && /otdata2/themiya/grizli_rebels/bin/python3 -m scripts.build_index --field cosmos
  or:  python3 app/scripts/build_index.py --field cosmos

SUPER_CATALOG, zout.fits and eazy h5 are row-aligned 1:1 (verified) -> row index
joins all three, no id matching. DJA cross-match is SkyCoord nearest, max 0.5".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

# Make `server` importable whether run as module or as a script.
_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from server.config import settings  # noqa: E402
from server.fields import load_fields, FieldConfig  # noqa: E402

# Photometry band set + pivot wavelengths are derived PER FIELD from the
# default-template eazy h5 (cat/flux_columns + cat/pivot) so each field uses its
# own band list -- COSMOS has 32 bands, EGS 30 (f470n present; f336wu/f475w/
# f850lp absent). See _field_bands_pivots(). The list below is the historical
# COSMOS order, kept only as a documented reference / fallback default.
BANDS = [
    "f275wu", "f336wu", "f435w", "f475w", "f606w", "f814w", "f850lp", "f098m",
    "f105w", "f125w", "f140w", "f160w", "f070w", "f090w", "f115w", "f140m",
    "f150w", "f162m", "f182m", "f200w", "f210m", "f250m", "f277w", "f300m",
    "f335m", "f356w", "f360m", "f410m", "f430m", "f444w", "f460m", "f480m",
]


def _field_bands_pivots(field: FieldConfig) -> tuple[list[str], dict[str, float]]:
    """Derive the (fit-ordered) band list + pivot wavelengths (micron) from the
    default-template eazy h5 ``cat/flux_columns`` / ``cat/pivot`` (Angstrom).

    This is the authoritative, row-aligned band set the EAzY fit used, so it
    matches the /api/eazy photometry ordering. Replaces the hardcoded COSMOS
    32-band list; for COSMOS it reproduces BANDS exactly (verified)."""
    h5path = field.h5_path(field.default_template)
    with h5py.File(h5path, "r") as f:
        fcols = [
            c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
            for c in f["cat/flux_columns"][:]
        ]
        pivot_ang = np.asarray(f["cat/pivot"][:], dtype=float)
    bands = [c[2:] if c.startswith("f_") else c for c in fcols]
    pivots = {b: round(float(p) / 1e4, 6) for b, p in zip(bands, pivot_ang)}
    return bands, pivots

FLAG_COLS = [
    "use_phot", "flag_star", "flag_clean", "flag_kron", "flag_lowsnr", "flag_nophot",
]
MORPH_COLS = ["flux_radius", "a_image", "b_image", "theta_J2000", "kron_radius"]

# zout columns -> catalog column base name (before _{template} suffix).
ZOUT_MAP = {
    "z_phot": "z_phot",
    "z_ml": "z_ml",
    "z025": "z025",
    "z160": "z160",
    "z500": "z500",
    "z840": "z840",
    "z975": "z975",
    "z_phot_chi2": "chi2",
    "u_v": "u_v",
    "v_j": "v_j",
    "nusefilt": "nusefilt",
}

MATCH_RADIUS = 0.5  # arcsec
BBOX_PAD_ARCSEC = 30.0

DJA_USECOLS = [
    "file", "srcid", "ra", "dec", "grating", "filter", "root",
    "msamet", "grade", "z_best", "sn50", "exptime",
    # DJA reviewer verdict + per-disperser evidence (surfaced in the Inspector so
    # a second grader sees the original reviewer's note and z solutions).
    "zgrade", "reviewer", "comment", "z_prism", "z_grating", "sn_line",
]


def _flux_to_mag(f: np.ndarray) -> np.ndarray:
    """f in units of 10 nJy -> uJy = f*0.01 -> AB mag. NaN where flux<=0."""
    f = np.asarray(f, dtype=np.float64)
    ujy = f * 0.01
    mag = np.full(f.shape, np.nan, dtype=np.float64)
    good = f > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mag[good] = 23.9 - 2.5 * np.log10(ujy[good])
    return mag


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    good = x > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[good] = np.log10(x[good])
    return out


def build_catalog(
    field: FieldConfig, bands: list[str]
) -> tuple[pd.DataFrame, np.ndarray, dict, dict]:
    """Return (catalog_df, mag_f444w_array, {template: z_phot_array}, meta)."""
    print(f"[catalog] reading SUPER_CATALOG {field.catalog_path}")
    cat = Table.read(field.catalog_path, memmap=True)
    n = len(cat)
    print(f"[catalog] {n} rows; {len(bands)} bands")

    data: dict[str, np.ndarray] = {}
    data["id"] = np.asarray(cat["id"], dtype=np.int64)
    data["ra"] = np.asarray(cat["ra"], dtype=np.float64)
    data["dec"] = np.asarray(cat["dec"], dtype=np.float64)
    data["x"] = np.asarray(cat["x"], dtype=np.float64)
    data["y"] = np.asarray(cat["y"], dtype=np.float64)

    # per-band mags + snr
    for b in bands:
        f = np.asarray(cat[f"f_{b}"], dtype=np.float64)
        e = np.asarray(cat[f"e_{b}"], dtype=np.float64)
        data[f"mag_{b}"] = _flux_to_mag(f)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(e > 0, f / e, np.nan)
        data[f"snr_{b}"] = snr.astype(np.float64)

    # flags (bool)
    for c in FLAG_COLS:
        data[c] = np.asarray(cat[c], dtype=bool)

    # band counts
    data["n_bands"] = np.asarray(cat["n_bands_jwst"], dtype=np.int32)
    data["n_bands_mb"] = np.asarray(cat["n_bands_mb"], dtype=np.int32)

    # morphology
    for c in MORPH_COLS:
        data[c] = np.asarray(cat[c], dtype=np.float64)

    ra_range = [float(data["ra"].min()), float(data["ra"].max())]
    dec_range = [float(data["dec"].min()), float(data["dec"].max())]
    del cat

    # per-template zout columns
    zphot_by_tpl: dict[str, np.ndarray] = {}
    for tpl in field.eazy:
        zpath = field.zout_path(tpl)
        print(f"[catalog] reading zout[{tpl}] {zpath}")
        zt = Table.read(zpath, memmap=True)
        if len(zt) != n:
            raise ValueError(f"zout {tpl} nrows {len(zt)} != catalog {n}")
        for zcol, base in ZOUT_MAP.items():
            arr = np.asarray(zt[zcol])
            if base == "nusefilt":
                data[f"{base}_{tpl}"] = arr.astype(np.int32)
            else:
                data[f"{base}_{tpl}"] = arr.astype(np.float64)
        data[f"lmass_{tpl}"] = _safe_log10(np.asarray(zt["mass"], dtype=np.float64))
        data[f"lsfr_{tpl}"] = _safe_log10(np.asarray(zt["sfr"], dtype=np.float64))
        data[f"Av_{tpl}"] = np.asarray(zt["Av"], dtype=np.float64)
        data[f"uvj_{tpl}"] = np.asarray(zt["uvj_class"], dtype=np.float64).astype(np.int8)
        zphot_by_tpl[tpl] = data[f"z_phot_{tpl}"]
        del zt

    df = pd.DataFrame(data)
    meta = {"ra_range": ra_range, "dec_range": dec_range}
    return df, data["mag_f444w"], zphot_by_tpl, meta


def build_spectra(
    field: FieldConfig,
    cat_id: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    mag_f444w: np.ndarray,
    zphot_by_tpl: dict[str, np.ndarray],
    cat_ra_range: list[float],
    cat_dec_range: list[float],
) -> tuple[pd.DataFrame, float]:
    print(f"[spectra] reading DJA {field.dja_catalog_path}")
    dja = pd.read_csv(
        field.dja_catalog_path, compression="gzip", usecols=DJA_USECOLS,
        low_memory=False,
    )
    print(f"[spectra] DJA rows total: {len(dja)}")

    # bbox pre-cut with 30" pad
    dec_mid = 0.5 * (cat_dec_range[0] + cat_dec_range[1])
    dpad = BBOX_PAD_ARCSEC / 3600.0
    rpad = dpad / max(np.cos(np.radians(dec_mid)), 1e-6)
    m = (
        (dja["ra"] >= cat_ra_range[0] - rpad)
        & (dja["ra"] <= cat_ra_range[1] + rpad)
        & (dja["dec"] >= cat_dec_range[0] - dpad)
        & (dja["dec"] <= cat_dec_range[1] + dpad)
    )
    dja = dja[m].reset_index(drop=True)
    print(f"[spectra] DJA rows in bbox+{BBOX_PAD_ARCSEC}\": {len(dja)}")
    if len(dja) == 0:
        return pd.DataFrame(), float("nan")

    # SkyCoord match: each DJA spectrum -> nearest MINERVA source
    mcoord = SkyCoord(ra=cat_ra * u.deg, dec=cat_dec * u.deg)
    dcoord = SkyCoord(ra=dja["ra"].values * u.deg, dec=dja["dec"].values * u.deg)
    idx, d2d, _ = dcoord.match_to_catalog_sky(mcoord)
    sep_arcsec = d2d.arcsec
    keep = sep_arcsec < MATCH_RADIUS
    print(f"[spectra] within {MATCH_RADIUS}\": {int(keep.sum())} / {len(dja)}")

    dja = dja[keep].reset_index(drop=True)
    idx = idx[keep]
    sep_arcsec = sep_arcsec[keep]

    # parse file basename -> dja key, pid, grating token.
    # Well-formed DJA names are '{root}_{grating}_{pid}_{srcid}' (>=4 tokens,
    # pid an integer). A few UDS RUBIES background slits ship a 3-token
    # '{root}_{grating}_{srcid}' with no pid (e.g.
    # 'rubies-uds2-v4_prism-clear_b28'), so toks[-2] is the grating, not an int.
    # Parse pid defensively (sentinel -1 when non-integer) and take the grating
    # from toks[-2] in that case so the surfaced grating string stays correct.
    # Well-formed rows (all of COSMOS/EGS) are unchanged.
    basename = dja["file"].str.replace(".spec.fits", "", regex=False)
    toks = basename.str.split("_")
    pid_raw = pd.to_numeric(toks.str[-2], errors="coerce")
    no_pid = pid_raw.isna()
    pid = pid_raw.fillna(-1).astype(np.int64)
    grating_tok = toks.str[-3].where(~no_pid, toks.str[-2])

    metafile = dja["msamet"].astype("string")
    metafile = metafile.str.split("_").str[0].fillna("")

    grade = dja["grade"].fillna(0).astype(np.int8)

    out = pd.DataFrame(
        {
            "dja": basename.values,
            "mid": cat_id[idx].astype(np.int64),
            "ra": cat_ra[idx],
            "dec": cat_dec[idx],
            "dja_ra": dja["ra"].values.astype(np.float64),
            "dja_dec": dja["dec"].values.astype(np.float64),
            "sep": sep_arcsec.astype(np.float64),
            "zs": dja["z_best"].values.astype(np.float64),
            "grade": grade.values,
            "mag": mag_f444w[idx],
            "sn": dja["sn50"].values.astype(np.float64),
            "root": dja["root"].values,
            "file": dja["file"].values,
            "srcid": dja["srcid"].values.astype(np.int64),
            "pid": pid.values,
            "grating": grating_tok.values,
            "metafile": metafile.values,
            "exptime": dja["exptime"].values.astype(np.float64),
            # DJA reviewer verdict + per-disperser evidence.
            "reviewer": dja["reviewer"].astype("string").fillna("").values,
            "comment": dja["comment"].astype("string").fillna("").values,
            "zgrade": dja["zgrade"].values.astype(np.float64),
            "z_prism": dja["z_prism"].values.astype(np.float64),
            "z_grating": dja["z_grating"].values.astype(np.float64),
            "sn_line": dja["sn_line"].values.astype(np.float64),
        }
    )
    for tpl, zp in zphot_by_tpl.items():
        out[f"zp_{tpl}"] = zp[idx]

    # dedupe on dja key, keep first
    ndup = int(out["dja"].duplicated().sum())
    if ndup:
        print(f"[spectra] deduping {ndup} duplicate dja keys (keep first)")
        out = out.drop_duplicates(subset="dja", keep="first").reset_index(drop=True)

    # nspec: how many spectra per mid
    out["nspec"] = out.groupby("mid")["mid"].transform("count").astype(np.int32)

    # primary: per mid pick best grade, then prism-clear first, then exptime desc
    out["_is_prism"] = (out["grating"] == "prism-clear")
    out["_exptime_sort"] = out["exptime"].fillna(-np.inf)
    order = out.sort_values(
        by=["mid", "grade", "_is_prism", "_exptime_sort"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    primary_idx = order.groupby("mid", sort=False).head(1).index
    out["primary"] = False
    out.loc[primary_idx, "primary"] = True
    out = out.drop(columns=["_is_prism", "_exptime_sort"])

    median_sep = float(np.median(out["sep"].values))
    return out, median_sep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="cosmos")
    ap.add_argument("--out-dir", default=None, help="override index dir")
    args = ap.parse_args()

    t0 = time.time()
    fields = load_fields()
    if args.field not in fields:
        print(f"unknown field {args.field}; have {list(fields)}", file=sys.stderr)
        return 2
    field = fields[args.field]

    index_dir = Path(args.out_dir) if args.out_dir else settings.index_dir
    out_dir = index_dir / field.name
    out_dir.mkdir(parents=True, exist_ok=True)

    bands, pivots = _field_bands_pivots(field)
    print(f"[bands] {len(bands)} bands from {field.h5_path(field.default_template).name}")
    cat_df, mag_f444w, zphot_by_tpl, cmeta = build_catalog(field, bands)
    cat_path = out_dir / "catalog.parquet"
    cat_df.to_parquet(cat_path, index=False)
    print(f"[write] {cat_path} ({len(cat_df)} rows, {len(cat_df.columns)} cols)")

    spec_df, median_sep = build_spectra(
        field,
        cat_df["id"].values,
        cat_df["ra"].values,
        cat_df["dec"].values,
        mag_f444w,
        zphot_by_tpl,
        cmeta["ra_range"],
        cmeta["dec_range"],
    )
    spec_path = out_dir / "spectra.parquet"
    spec_df.to_parquet(spec_path, index=False)
    print(f"[write] {spec_path} ({len(spec_df)} rows, {len(spec_df.columns)} cols)")

    n_spec_matched = int(spec_df["mid"].nunique()) if len(spec_df) else 0
    meta = {
        "field": field.name,
        "n_objects": int(len(cat_df)),
        "n_spec_matched": n_spec_matched,
        "n_spectra": int(len(spec_df)),
        "ra_range": cmeta["ra_range"],
        "dec_range": cmeta["dec_range"],
        "bands": bands,
        "pivots": pivots,
        "templates": list(field.eazy.keys()),
        "default_template": field.default_template,
        "median_sep": median_sep,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[write] {meta_path}")
    print(f"[done] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
