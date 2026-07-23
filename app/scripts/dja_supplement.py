#!/usr/bin/env python3
"""Live DJA supplement sync for the MINERVA viewer.

Problem this solves
-------------------
The viewer's ``spectra.parquet`` index is built (scripts/build_index.py) from a
*frozen* DJA emission-line catalog snapshot (currently
``dja_msaexp_emission_lines_v4.5.csv.gz``, frozen 2026-02-23). New NIRSpec
extractions keep landing in the live ``nirspec_extractions`` service *after*
that freeze -- e.g. as of 2026-07-18 six COSMOS roots
(capers-cos68-v4, cos-barrufet-obs1/2/3-v4, jw09372001001-v4,
jw09381001001-v4; ~341 spectra) exist live but are absent from even the full
v4.5 CSV, and versions v4.6+ all 403 (no newer bulk CSV to re-index from).

This script pulls those live-but-missing spectra without re-downloading the
150 MB bulk catalog: it tiles the live ``nirspec_extractions`` API over the
field's ra/dec bounding box, diffs the returned ``(root, file)`` pairs against
what is already in ``spectra.parquet``, re-runs the *same* 0.5" SkyCoord
cross-match used at index-build on the genuinely new pairs, and writes a
supplementary parquet (``spectra_supplement.parquet``) with the cross-matched
(<=0.5") subset, in the *same schema* as ``spectra.parquet``.

It is standalone: it prints a diff report and writes the parquet. Wiring the
merge into CatalogStore is intentionally left out of this pass (a merge step
must recompute ``nspec``/``primary`` globally across the combined frame -- see
the note printed at the end).

Usage
-----
    cd app && /otdata2/themiya/grizli_rebels/bin/python3 scripts/dja_supplement.py --field cosmos

Options: ``--tile-size`` (arcsec half-width of each API box tile, default 250),
``--pad`` (arcsec bbox pad, default 30, matches build_index), ``--out-dir``
(override index dir), ``--dry-run`` (report only, do not write parquet),
``--timeout`` (per-request seconds), ``--retries`` (per-tile).

Idempotent: the output parquet is overwritten deterministically (pairs are
deduped by ``(root, file)`` and sorted by ``dja``), so re-running produces the
same content with no duplicates.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

# Make `server` + sibling scripts importable whether run as module or script.
_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from server.config import settings  # noqa: E402
from server.fields import load_fields  # noqa: E402
from scripts.build_index import MATCH_RADIUS, BBOX_PAD_ARCSEC  # noqa: E402

API_URL = "https://grizli-cutout.herokuapp.com/nirspec_extractions"
# CSV header the service emits (used to detect the "Nothing found for <SQL>"
# empty-tile response, which is NOT valid CSV).
_CSV_HEADER_PREFIX = "root,file,ra,dec"


# ---------------------------------------------------------------------------
# live API: tiled box queries over the field bbox
# ---------------------------------------------------------------------------
def _fetch_tile(ra_c, dec_c, size_as, timeout, retries):
    """One box query. Returns a DataFrame (possibly empty) of extractions.

    ``size_as`` is the box *half-width* in arcsec; the service builds a polygon
    [center +/- size/3600 deg] in BOTH ra and dec (no cos(dec) term). An empty
    tile returns a "Nothing found for <SQL>" message, not CSV -- treated as 0
    rows. Retries transient HTTP/parse failures.
    """
    params = {"coords": f"{ra_c:.6f},{dec_c:.6f}", "size": size_as,
              "output": "csv"}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            text = r.text
            if not text.lstrip().startswith(_CSV_HEADER_PREFIX):
                # "Nothing found for ..." (empty tile) or any non-CSV body.
                return pd.DataFrame()
            # comment is a free-text trailing column that can contain commas;
            # cap the split so stray commas fold into `comment` instead of
            # raising a tokenizer error.
            df = pd.read_csv(io.StringIO(text), engine="python",
                             on_bad_lines="warn")
            return df
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"[warn] tile ({ra_c:.5f},{dec_c:.5f}) failed after {retries} "
          f"retries: {last_exc}", file=sys.stderr)
    return pd.DataFrame()


def fetch_bbox(ra0, ra1, dec0, dec1, size_as, timeout, retries):
    """Tile the API over the (already padded) bbox, deduped by (root,file)."""
    step = (2.0 * size_as / 3600.0) * 0.9  # 10% overlap; service box is raw deg
    half = size_as / 3600.0
    ra_centers = np.arange(ra0 + half, ra1 + step, step)
    dec_centers = np.arange(dec0 + half, dec1 + step, step)
    n_tiles = len(ra_centers) * len(dec_centers)
    print(f"[api] tiling {len(ra_centers)}x{len(dec_centers)} = {n_tiles} "
          f"box queries (size={size_as}\") over "
          f"ra[{ra0:.5f},{ra1:.5f}] dec[{dec0:.5f},{dec1:.5f}]")
    frames, done = [], 0
    for rc in ra_centers:
        for dc in dec_centers:
            df = _fetch_tile(rc, dc, size_as, timeout, retries)
            if len(df):
                frames.append(df)
            done += 1
            print(f"  tile {done}/{n_tiles} -> {len(df)} rows", end="\r")
    print()
    if not frames:
        return pd.DataFrame()
    api = pd.concat(frames, ignore_index=True)
    api = api.dropna(subset=["root", "file", "ra", "dec"])
    api = api.drop_duplicates(subset=["root", "file"]).reset_index(drop=True)
    # keep strictly within the padded bbox (tiles overshoot the edges)
    m = ((api["ra"] >= ra0) & (api["ra"] <= ra1)
         & (api["dec"] >= dec0) & (api["dec"] <= dec1))
    api = api[m].reset_index(drop=True)
    return api


# ---------------------------------------------------------------------------
# cross-match new pairs against the MINERVA catalog (same 0.5" rule)
# ---------------------------------------------------------------------------
def cross_match(new, cat, templates):
    """SkyCoord nearest-match ``new`` DJA rows to MINERVA sources, keep <0.5".

    ``cat`` must carry id/ra/dec/mag_f444w/z_phot_{tpl}. Returns a DataFrame in
    the exact spectra.parquet schema (nspec/primary computed within this frame).
    """
    if new.empty:
        return pd.DataFrame()

    mcoord = SkyCoord(ra=cat["ra"].values * u.deg, dec=cat["dec"].values * u.deg)
    dcoord = SkyCoord(ra=new["ra"].values * u.deg, dec=new["dec"].values * u.deg)
    idx, d2d, _ = dcoord.match_to_catalog_sky(mcoord)
    sep = d2d.arcsec
    keep = sep < MATCH_RADIUS
    new = new[keep].reset_index(drop=True)
    idx = idx[keep]
    sep = sep[keep]
    if len(new) == 0:
        return pd.DataFrame()

    basename = new["file"].str.replace(".spec.fits", "", regex=False)
    toks = basename.str.split("_")
    pid = pd.to_numeric(toks.str[-2], errors="coerce").fillna(-1).astype(np.int64)
    grating_tok = toks.str[-3]

    if "msamet" in new.columns:
        metafile = new["msamet"].astype("string").str.split("_").str[0].fillna("")
    else:
        metafile = pd.Series([""] * len(new))

    grade = pd.to_numeric(new.get("grade"), errors="coerce").fillna(0).astype(np.int8)
    srcid = pd.to_numeric(new.get("srcid"), errors="coerce").fillna(-1).astype(np.int64)

    cat_id = cat["id"].values
    cat_ra = cat["ra"].values
    cat_dec = cat["dec"].values
    mag_f444w = cat["mag_f444w"].values

    out = pd.DataFrame({
        "dja": basename.values,
        "mid": cat_id[idx].astype(np.int64),
        "ra": cat_ra[idx].astype(np.float64),
        "dec": cat_dec[idx].astype(np.float64),
        "dja_ra": new["ra"].values.astype(np.float64),
        "dja_dec": new["dec"].values.astype(np.float64),
        "sep": sep.astype(np.float64),
        "zs": pd.to_numeric(new.get("z"), errors="coerce").values.astype(np.float64),
        "grade": grade.values,
        "mag": mag_f444w[idx].astype(np.float64),
        "sn": pd.to_numeric(new.get("sn50"), errors="coerce").values.astype(np.float64),
        "root": new["root"].values,
        "file": new["file"].values,
        "srcid": srcid.values,
        "pid": pid.values,
        "grating": grating_tok.values,
        "metafile": metafile.values,
        "exptime": pd.to_numeric(new.get("exptime"), errors="coerce").values.astype(np.float64),
    })
    for tpl in templates:
        zcol = f"z_phot_{tpl}"
        if zcol in cat.columns:
            out[f"zp_{tpl}"] = cat[zcol].values[idx].astype(np.float64)
        else:
            out[f"zp_{tpl}"] = np.nan

    # dedupe on dja key (defensive; box dedupe already covers this)
    out = out.drop_duplicates(subset="dja", keep="first").reset_index(drop=True)

    # nspec/primary within this supplement frame (a future merge into
    # spectra.parquet must recompute these globally across the combined set).
    out["nspec"] = out.groupby("mid")["mid"].transform("count").astype(np.int32)
    out["_is_prism"] = (out["grating"] == "prism-clear")
    out["_exptime_sort"] = out["exptime"].fillna(-np.inf)
    order = out.sort_values(
        by=["mid", "grade", "_is_prism", "_exptime_sort"],
        ascending=[True, False, False, False], kind="mergesort")
    primary_idx = order.groupby("mid", sort=False).head(1).index
    out["primary"] = False
    out.loc[primary_idx, "primary"] = True
    out = out.drop(columns=["_is_prism", "_exptime_sort"])

    out = out.sort_values("dja", kind="mergesort").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# diff report
# ---------------------------------------------------------------------------
def report_diff(api, existing_pairs, existing_roots):
    """Print the new-vs-existing diff; return the `new` (unmatched-yet) frame."""
    pairs = list(zip(api["root"], api["file"]))
    is_new = np.array([p not in existing_pairs for p in pairs])
    new = api[is_new].reset_index(drop=True)

    print("\n" + "=" * 68)
    print("DJA LIVE SUPPLEMENT — DIFF REPORT")
    print("=" * 68)
    print(f"live extractions in bbox : {len(api)}")
    print(f"already in spectra.parquet: {len(api) - len(new)}")
    print(f"NEW (root,file) pairs     : {len(new)}")

    if len(new):
        by_root = new.groupby("root").size().sort_values(ascending=False)
        brand_new_roots = [r for r in by_root.index if r not in existing_roots]
        n_brand_new = int(new["root"].isin(brand_new_roots).sum())
        print(f"\n  roots ENTIRELY ABSENT from the current index: "
              f"{len(brand_new_roots)}  ({n_brand_new} spectra)")
        for r in sorted(brand_new_roots):
            print(f"    {r:<28} {int(by_root[r]):>5} spectra")
        seen_root_new = [r for r in by_root.index if r in existing_roots]
        if seen_root_new:
            n_seen = int(new[new["root"].isin(seen_root_new)].shape[0])
            print(f"\n  new files under already-indexed roots: {n_seen} "
                  f"(unmatched at build time or added since)")
    print("=" * 68)
    return new


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="cosmos")
    ap.add_argument("--out-dir", default=None, help="override index dir")
    ap.add_argument("--tile-size", type=float, default=250.0,
                    help="box half-width per API tile, arcsec")
    ap.add_argument("--pad", type=float, default=BBOX_PAD_ARCSEC,
                    help="bbox pad in arcsec (default matches build_index)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not write the supplement parquet")
    args = ap.parse_args()

    t0 = time.time()
    fields = load_fields()
    if args.field not in fields:
        print(f"unknown field {args.field}; have {list(fields)}", file=sys.stderr)
        return 2
    field = fields[args.field]
    templates = list(field.eazy.keys())

    index_dir = Path(args.out_dir) if args.out_dir else settings.index_dir
    field_dir = index_dir / field.name
    spectra_path = field_dir / "spectra.parquet"
    catalog_path = field_dir / "catalog.parquet"
    meta_path = field_dir / "meta.json"
    for p in (spectra_path, catalog_path, meta_path):
        if not p.exists():
            print(f"missing index artifact: {p} (run build_index first)",
                  file=sys.stderr)
            return 2

    with open(meta_path) as fh:
        meta = json.load(fh)
    ra0, ra1 = meta["ra_range"]
    dec0, dec1 = meta["dec_range"]
    dpad = args.pad / 3600.0
    ra0, ra1, dec0, dec1 = ra0 - dpad, ra1 + dpad, dec0 - dpad, dec1 + dpad

    # existing (root,file) pairs + roots from the built index
    sp = pd.read_parquet(spectra_path, columns=["root", "file"])
    existing_pairs = set(zip(sp["root"], sp["file"]))
    existing_roots = set(sp["root"])
    print(f"[index] spectra.parquet: {len(sp)} rows, "
          f"{len(existing_roots)} roots")

    # live tiled fetch
    api = fetch_bbox(ra0, ra1, dec0, dec1, args.tile_size,
                     args.timeout, args.retries)
    if api.empty:
        print("[api] no live extractions returned; nothing to do.")
        return 0

    new = report_diff(api, existing_pairs, existing_roots)
    if new.empty:
        print("\n[done] no new spectra; index is up to date.")
        return 0

    # cross-match the new pairs against the MINERVA catalog (same 0.5" rule)
    cat_cols = ["id", "ra", "dec", "mag_f444w"] + [f"z_phot_{t}" for t in templates]
    cat = pd.read_parquet(catalog_path,
                          columns=[c for c in cat_cols if c])
    supp = cross_match(new, cat, templates)
    print(f"\n[match] {len(new)} new pairs -> {len(supp)} cross-matched "
          f"(<= {MATCH_RADIUS}\") to a MINERVA source "
          f"({supp['mid'].nunique() if len(supp) else 0} unique ids)")
    if len(supp):
        mroots = supp.groupby("root").size().sort_values(ascending=False)
        for r, n in mroots.items():
            print(f"    {r:<28} {int(n):>5} matched")

    if args.dry_run:
        print("\n[dry-run] not writing parquet.")
        return 0

    out_path = field_dir / "spectra_supplement.parquet"
    # align column order to spectra.parquet for a clean future concat/merge
    ref_cols = list(pd.read_parquet(spectra_path).columns)
    for c in ref_cols:
        if c not in supp.columns:
            supp[c] = pd.NA
    supp = supp[ref_cols]
    supp.to_parquet(out_path, index=False)
    print(f"\n[write] {out_path} ({len(supp)} rows, {len(supp.columns)} cols)")
    print("[note] merge caveat: concatenating this into spectra.parquet "
          "requires recomputing nspec/primary across the combined frame "
          "(CatalogStore recomputes 'primary' at load but serves the stored "
          "'nspec').")
    print(f"[done] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
