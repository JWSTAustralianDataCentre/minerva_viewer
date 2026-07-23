#!/usr/bin/env python3
"""Build the MINERVA Field Map (/map) Leaflet tiles with FitsMap.

Generates, for a field (currently ``cosmos``):
  * ``rgb``   -- a Lupton asinh RGB composite (R=f444w, G=f277w, B=f115w),
                 written as one big PNG and tiled by FitsMap's PIL engine.
  * one grayscale layer per band (f115w f150w f200w f277w f444w) tiled by
    FitsMap's matplotlib engine with a per-band asinh normalisation that is
    computed once from the full frame (so it is identical at every zoom).
  * ``minerva`` catalog markers (use_phot sources, ellipse markers with a
    popup that links to the Catalog Explorer ``/?id=``).
  * ``spectra`` catalog markers (DJA spectra, popup links to the Redshift
    Inspector ``/inspector/?field=cosmos&sel=`` and the Explorer).

The emitted ``index.html`` uses *relative* tile URLs (``rgb/{z}/{y}/{x}.png``)
so it works when served under a ``/map/`` prefix, and its ``js/urlCoords.js``
carries the mosaic WCS so ``?ra=&dec=&zoom=`` deep links pan correctly.

Usage
-----
    python build_map.py --field cosmos --test                 # crop smoke test
    python build_map.py --field cosmos                        # full build
    python build_map.py --field cosmos --layers rgb,f444w     # subset

Run the ``--test`` build first; it verifies the tile tree end-to-end and
prints a scaled estimate + the exact full-build command.  See
build_map_README.md next to this file for details.

FitsMap API facts this script relies on (fitsmap 0.11.2, read from source):
  * ``convert.files_to_map(files, out_dir, title, task_procs, procs_per_task,
    catalog_delim, cat_wcs_fits_file, max_catalog_zoom, norm_kwargs,
    n_columns, catalog_starts_at_one, cluster_*)``.  Image inputs .fits/.png;
    catalog inputs must end in ``.cat``.  Layer name == input file basename
    (sans ext, ``.-()`` -> ``_``) -> we feed short-named symlinks/files so
    layers are ``rgb``, ``f444w`` ... ``minerva`` ``spectra``.
  * FITS layers render via matplotlib ``simple_norm(**norm_kwargs)`` + gray
    cmap; ``norm_kwargs`` may be keyed by file basename for per-band norms.
    The norm is built once on the whole (padded) array -> zoom-consistent.
  * PNG/JPG layers render via PIL and DO accept 3-channel colour; module sets
    ``Image.MAX_IMAGE_PIXELS = sys.maxsize`` so the huge composite is allowed.
    Empirically a colour PNG must be saved ``np.flipud``'d to line up with the
    FITS layers and the ra/dec markers (verified: corr 0.995 vs 0.0).
  * Catalog .cat: comma-delimited, header row; needs ``id,ra,dec`` (lowercase
    kept as-is); optional ``a,b,theta`` (pixels/deg) turn circle markers into
    ellipses.  Every other column shows in the popup; popup values are written
    to the table as innerHTML, so an HTML ``<a>`` in a column becomes a real
    link.  Popups fetch per-source ``catalog_assets/<layer>/<id>.cbor``.
  * Tiles are ``{layer}/{z}/{y}/{x}.png``; markers ``{layer}/{z}/{y}/{x}.pbf``
    with ``<layer>.columns`` + ``catalog_assets/<layer>/*.cbor``.  Resume is
    per-layer: a layer whose dir already exists in out_dir is skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow the huge composite PNG (fitsmap also sets this)

# --------------------------------------------------------------------------- #
# Field configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/otdata2/themiya/minerva/data")

FIELDS = {
    "cosmos": dict(
        images=DATA_ROOT / "images/cosmos",
        img_tmpl="cosmos-grizli-v8.0-minerva-v3.0-40mas-{band}-clear_drc_sci.fits",
        index=DATA_ROOT / "viewer_index/cosmos",
        out=DATA_ROOT / "map_tiles/cosmos",
        out_test=DATA_ROOT / "map_tiles/cosmos_test",
        title="MINERVA COSMOS",
    ),
}

BANDS = ["f115w", "f150w", "f200w", "f277w", "f444w"]
RGB_BANDS = ("f444w", "f277w", "f115w")  # -> R, G, B
IMG_LAYERS = ["rgb"] + BANDS
CAT_LAYERS = ["minerva", "spectra"]
ALL_LAYERS = IMG_LAYERS + CAT_LAYERS

# per-band asinh norm knobs (simple_norm)
NORM_PCT = 99.5      # bright clip percentile (over positive pixels)
NORM_ASINH_A = 0.1   # asinh softening
# Lupton RGB knobs (channels are pre-normalised to their own NORM_PCT first)
RGB_STRETCH = 0.5
RGB_Q = 8


def band_file(cfg: dict, band: str) -> Path:
    return cfg["images"] / cfg["img_tmpl"].format(band=band)


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def positive_scale(sample: np.ndarray, pct: float = NORM_PCT) -> float:
    """Bright-clip value = ``pct`` percentile over strictly-positive pixels."""
    pos = sample[np.isfinite(sample) & (sample > 0)]
    if pos.size == 0:
        return 1.0
    v = float(np.percentile(pos, pct))
    return v if v > 0 else 1.0


def sample_full_band(path: Path, target_px: int = 4_000_000) -> np.ndarray:
    """Row-strided sample of a full mosaic (memmap; avoids loading 2.5 GB)."""
    data = fits.getdata(path, memmap=True)
    stride = max(1, data.shape[0] // max(1, target_px // data.shape[1]))
    return np.asarray(data[::stride]).astype(np.float32)


def band_norm_kwargs(vmax: float) -> dict:
    """simple_norm kwargs for one grayscale band (asinh, fixed vmin/vmax)."""
    return dict(stretch="asinh", asinh_a=NORM_ASINH_A, vmin=0.0, vmax=vmax, clip=True)


# --------------------------------------------------------------------------- #
# RGB composite
# --------------------------------------------------------------------------- #
def build_rgb_png(cfg: dict, out_png: Path, region: tuple | None) -> dict:
    """Write the flipud(Lupton asinh RGB) composite PNG. Returns per-channel scales."""
    chans = []
    scales = {}
    for band in RGB_BANDS:
        p = band_file(cfg, band) if region is None else None
        if region is None:
            arr = np.asarray(fits.getdata(p)).astype(np.float32)
            samp = arr[:: max(1, arr.shape[0] // 4000)]  # sample for the scale
        else:
            y0, x0, sz = region
            arr = np.asarray(
                fits.getdata(band_file(cfg, band), memmap=True)[y0:y0 + sz, x0:x0 + sz]
            ).astype(np.float32)
            samp = arr
        sc = positive_scale(samp)
        scales[band] = sc
        chans.append(np.nan_to_num(arr, nan=0.0) / sc)
    rgb = make_lupton_rgb(chans[0], chans[1], chans[2],
                          minimum=0.0, stretch=RGB_STRETCH, Q=RGB_Q)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(rgb)).save(out_png)
    return scales


# --------------------------------------------------------------------------- #
# Crop FITS (test mode) -- copies WCS with an offset CRPIX so markers align
# --------------------------------------------------------------------------- #
def write_crop_fits(cfg: dict, band: str, region: tuple, out_fits: Path) -> None:
    y0, x0, sz = region
    with fits.open(band_file(cfg, band), memmap=True) as hdul:
        hdr = hdul[0].header.copy()
        data = np.asarray(hdul[0].data[y0:y0 + sz, x0:x0 + sz]).astype(np.float32)
    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = hdr["CRPIX1"] - x0
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = hdr["CRPIX2"] - y0
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=hdr).writeto(out_fits, overwrite=True)


# --------------------------------------------------------------------------- #
# Catalog .cat writers
# --------------------------------------------------------------------------- #
def _region_xy_mask(df, region: tuple | None):
    if region is None:
        return np.ones(len(df), dtype=bool)
    y0, x0, sz = region
    return ((df["x"] > x0) & (df["x"] <= x0 + sz)
            & (df["y"] > y0) & (df["y"] <= y0 + sz)).to_numpy()


def write_minerva_cat(cfg: dict, out_cat: Path, region: tuple | None, limit: int | None) -> int:
    import pandas as pd
    cols = ["id", "ra", "dec", "x", "y", "a_image", "b_image", "theta_J2000",
            "mag_f444w", "z_phot_sfhz_blue_agn", "use_phot"]
    df = pd.read_parquet(cfg["index"] / "catalog.parquet", columns=cols)
    df = df[df["use_phot"]].copy()
    df = df[_region_xy_mask(df, region)]
    if limit:
        df = df.iloc[:limit]

    out_cat.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ra", "dec", "a", "b", "theta", "mag_f444w", "z_phot", "info"])
        for r in df.itertuples(index=False):
            a, b, th = r.a_image, r.b_image, r.theta_J2000
            if not (np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0):
                a, b, th = -1, -1, 0                      # -> circle marker in JS
            if not np.isfinite(th):
                th = 0
            mag = "" if not np.isfinite(r.mag_f444w) else f"{r.mag_f444w:.2f}"
            z = r.z_phot_sfhz_blue_agn
            zs = "" if (not np.isfinite(z) or z < 0) else f"{z:.3f}"
            info = f"<a href='/?id={r.id}' target='_blank'>explorer</a>"
            w.writerow([r.id, f"{r.ra:.7f}", f"{r.dec:.7f}",
                        f"{a:.3f}", f"{b:.3f}", f"{th:.2f}", mag, zs, info])
    return len(df)


def write_spectra_cat(cfg: dict, out_cat: Path, region: tuple | None, limit: int | None) -> int:
    import pandas as pd
    df = pd.read_parquet(cfg["index"] / "spectra.parquet",
                         columns=["dja", "mid", "ra", "dec", "zs", "grade",
                                  "grating", "mag", "sn"])
    # spectra.parquet has no x/y -> filter by ra/dec box derived from region
    if region is not None:
        lo_ra, hi_ra, lo_dec, hi_dec = _region_radec_box(cfg, region)
        m = ((df["ra"] >= lo_ra) & (df["ra"] <= hi_ra)
             & (df["dec"] >= lo_dec) & (df["dec"] <= hi_dec))
        df = df[m.to_numpy()].copy()
    if limit:
        df = df.iloc[:limit]

    out_cat.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ra", "dec", "zs", "grade", "grating", "mag", "sn", "info"])
        for r in df.itertuples(index=False):
            zs = "" if not np.isfinite(r.zs) else f"{r.zs:.4f}"
            mag = "" if not np.isfinite(r.mag) else f"{r.mag:.2f}"
            sn = "" if not np.isfinite(r.sn) else f"{r.sn:.1f}"
            grating = "" if pd.isna(r.grating) else str(r.grating)
            info = (f"<a href='/inspector/?field=cosmos&sel={r.dja}' target='_blank'>inspector</a>"
                    f" | <a href='/?id={r.mid}' target='_blank'>explorer</a>")
            w.writerow([r.dja, f"{r.ra:.7f}", f"{r.dec:.7f}", zs,
                        int(r.grade), grating, mag, sn, info])
    return len(df)


def _region_radec_box(cfg: dict, region: tuple):
    from astropy.wcs import WCS
    y0, x0, sz = region
    w = WCS(fits.getheader(band_file(cfg, "f444w")))
    corners = np.array([[x0, y0], [x0 + sz, y0], [x0, y0 + sz], [x0 + sz, y0 + sz]], float)
    sky = w.wcs_pix2world(corners, 0)
    ra, dec = sky[:, 0], sky[:, 1]
    return ra.min(), ra.max(), dec.min(), dec.max()


# --------------------------------------------------------------------------- #
# Staging: short-named inputs so fitsmap layer names are clean
# --------------------------------------------------------------------------- #
def stage_inputs(cfg: dict, staging: Path, layers: list[str], region: tuple | None,
                 cat_limit: int | None) -> tuple[list[Path], Path, dict]:
    """Materialise the input files fitsmap will consume; return (files, wcs_fits, info)."""
    staging.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    info: dict = {"scales": {}, "n_minerva": 0, "n_spectra": 0, "norm": {}}

    # grayscale band FITS (symlink to real for full; crop copy for test)
    for band in BANDS:
        if band not in layers:
            continue
        dst = staging / f"{band}.fits"
        if region is None:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(band_file(cfg, band))
            samp = sample_full_band(band_file(cfg, band))
        else:
            write_crop_fits(cfg, band, region, dst)
            samp = np.asarray(fits.getdata(dst)).astype(np.float32)
        info["norm"][f"{band}.fits"] = band_norm_kwargs(positive_scale(samp))
        files.append(dst)

    # RGB composite PNG
    if "rgb" in layers:
        rgb_png = staging / "rgb.png"
        info["scales"] = build_rgb_png(cfg, rgb_png, region)
        files.insert(0, rgb_png)  # first -> default base layer

    # WCS reference fits for catalogs / urlCoords deep links
    wcs_fits = staging / "f444w.fits"
    if not wcs_fits.exists():
        if region is None:
            if wcs_fits.is_symlink():
                wcs_fits.unlink()
            wcs_fits.symlink_to(band_file(cfg, "f444w"))
        else:
            write_crop_fits(cfg, "f444w", region, wcs_fits)

    # catalog .cat files
    if "minerva" in layers:
        c = staging / "minerva.cat"
        info["n_minerva"] = write_minerva_cat(cfg, c, region, cat_limit)
        files.append(c)
    if "spectra" in layers:
        c = staging / "spectra.cat"
        info["n_spectra"] = write_spectra_cat(cfg, c, region, cat_limit)
        files.append(c)

    return files, wcs_fits, info


# --------------------------------------------------------------------------- #
# Drive fitsmap
# --------------------------------------------------------------------------- #
def run_fitsmap(files: list[Path], out_dir: Path, wcs_fits: Path, title: str,
                norm: dict, task_procs: int, procs_per_task: int) -> float:
    import fitsmap.convert as convert
    t0 = time.time()
    convert.files_to_map(
        [str(f) for f in files],
        out_dir=str(out_dir),
        title=title,
        task_procs=task_procs,
        procs_per_task=procs_per_task,
        catalog_delim=",",
        cat_wcs_fits_file=str(wcs_fits),
        norm_kwargs=norm,
        n_columns=1,
        catalog_starts_at_one=True,
    )
    return time.time() - t0


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _count_pngs(layer_dir: Path) -> int:
    return sum(1 for _ in layer_dir.rglob("*.png"))


def _deepest_tile(layer_dir: Path) -> Path | None:
    """Cheaply find one PNG at the max zoom (deepest z dir) of a layer."""
    zs = [int(p.name) for p in layer_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not zs:
        return None
    zdir = layer_dir / str(max(zs))
    for ydir in sorted(zdir.iterdir()):
        for tile in sorted(ydir.glob("*.png")):
            return tile
    return None


def verify(out_dir: Path, layers: list[str]) -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []

    def check(cond, msg):
        nonlocal ok
        ok = ok and cond
        lines.append(("  OK  " if cond else " FAIL ") + msg)

    idx = out_dir / "index.html"
    check(idx.exists(), f"index.html present ({idx})")
    check((out_dir / "js" / "index.js").exists(), "js/index.js present")
    check((out_dir / "js" / "urlCoords.js").exists(), "js/urlCoords.js present")

    # relative tile URLs + WCS deep links
    if (out_dir / "js" / "index.js").exists():
        js = (out_dir / "js" / "index.js").read_text()
        has_abs = 'tileLayer("/' in js or 'tileLayer("http' in js
        check(not has_abs, "tile URLs are relative (work under /map/ prefix)")
    if (out_dir / "js" / "urlCoords.js").exists():
        uc = (out_dir / "js" / "urlCoords.js").read_text()
        check("const is_ra_dec = 1" in uc, "urlCoords.js has WCS (is_ra_dec=1) for ?ra=&dec=&zoom= links")

    # image layers
    for layer in [l for l in layers if l in IMG_LAYERS]:
        d = out_dir / layer
        n = _count_pngs(d) if d.exists() else 0
        check(d.is_dir() and n > 0, f"image layer '{layer}': {n} png tiles")
        if n > 0:
            # sample a deep-zoom tile and confirm it is a valid, non-blank PNG
            sample = _deepest_tile(d)
            try:
                a = np.asarray(Image.open(sample).convert("RGBA")).astype(float)
                check(a.shape[:2] == (256, 256) and a[..., :3].std() > 0,
                      f"  sample {sample.relative_to(out_dir)} valid, std={a[...,:3].std():.1f}")
            except Exception as e:  # noqa
                check(False, f"  sample tile unreadable: {e}")

    # catalog layers
    for layer in [l for l in layers if l in CAT_LAYERS]:
        d = out_dir / layer
        npbf = sum(1 for _ in d.rglob("*.pbf")) if d.exists() else 0
        check(d.is_dir() and npbf > 0, f"marker layer '{layer}': {npbf} pbf tiles")
        check((out_dir / f"{layer}.columns").exists(), f"  {layer}.columns present")
        assets = out_dir / "catalog_assets" / layer
        ncbor = sum(1 for _ in assets.rglob("*.cbor")) if assets.exists() else 0
        check(ncbor > 0, f"  catalog_assets/{layer}: {ncbor} cbor popups")
        check((out_dir / "js" / "tiledMarkers.min.js").exists(), "  js/tiledMarkers.min.js present")

    return ok, lines


# --------------------------------------------------------------------------- #
# Estimate + reporting
# --------------------------------------------------------------------------- #
def full_zoom_slots(long_px: int, tile: int = 256) -> int:
    """Total tile slots fitsmap generates for a frame (balanced to a power of two)."""
    side = 1 << int(np.ceil(np.log2(long_px)))
    max_zoom = int(np.log2(side / tile))
    return int(sum(4 ** i for i in range(0, max_zoom + 1)))


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Build MINERVA Field Map tiles (FitsMap).")
    ap.add_argument("--field", default="cosmos", choices=list(FIELDS))
    ap.add_argument("--test", action="store_true", help="central crop smoke test")
    ap.add_argument("--layers", default=",".join(ALL_LAYERS),
                    help="comma list subset of: " + ",".join(ALL_LAYERS))
    ap.add_argument("--crop-size", type=int, default=4096, help="test crop side (px)")
    ap.add_argument("--task-procs", type=int, default=1)
    ap.add_argument("--procs-per-task", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--cat-limit", type=int, default=None, help="cap markers per catalog")
    ap.add_argument("--out-dir", default=None, help="override output tile dir")
    ap.add_argument("--clean", action="store_true", help="wipe out_dir before building (default for --test)")
    ap.add_argument("--no-clean", action="store_true", help="never wipe out_dir (force resume even in --test)")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    cfg = FIELDS[args.field]
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    bad = [l for l in layers if l not in ALL_LAYERS]
    if bad:
        print(f"unknown layers: {bad}", file=sys.stderr)
        return 2

    # sanity: inputs exist
    for band in BANDS:
        if band in layers or "rgb" in layers:
            if not band_file(cfg, band).exists():
                print(f"missing input: {band_file(cfg, band)}", file=sys.stderr)
                return 2

    region = None
    if args.test:
        with fits.open(band_file(cfg, "f444w"), memmap=True) as h:
            H, W = h[0].data.shape
        sz = args.crop_size
        region = ((H - sz) // 2, (W - sz) // 2, sz)   # (y0, x0, size)

    out_dir = Path(args.out_dir) if args.out_dir else (cfg["out_test"] if args.test else cfg["out"])
    staging = out_dir.parent / (out_dir.name + "__staging")
    title = cfg["title"] + (" (test)" if args.test else "")

    # test: clean by default (fresh each run); full: keep by default (per-layer resume).
    clean = args.clean or (args.test and not args.no_clean)
    if clean and out_dir.exists():
        print(f"cleaning {out_dir}")
        shutil.rmtree(out_dir)
    if args.test and staging.exists():
        shutil.rmtree(staging)

    print(f"[{args.field}] mode={'TEST' if args.test else 'FULL'} layers={layers}")
    print(f"  out_dir = {out_dir}")
    print(f"  staging = {staging}")
    if region:
        print(f"  crop    = {sz}x{sz} at (x0={region[1]}, y0={region[0]})")

    print("staging inputs ...")
    files, wcs_fits, info = stage_inputs(cfg, staging, layers, region, args.cat_limit)
    print(f"  files: {[f.name for f in files]}")
    if "rgb" in layers:
        print(f"  rgb channel scales (10 nJy): "
              f"{ {b: round(s, 4) for b, s in info['scales'].items()} }")
    for k, v in info["norm"].items():
        print(f"  norm {k}: vmax={v['vmax']:.4f} asinh_a={v['asinh_a']}")
    if "minerva" in layers:
        print(f"  minerva markers: {info['n_minerva']}")
    if "spectra" in layers:
        print(f"  spectra markers: {info['n_spectra']}")

    # ---- build (combined -> correct index.html) ----
    print("building fitsmap (combined) ...")
    t_total = run_fitsmap(files, out_dir, wcs_fits, title,
                          info["norm"], args.task_procs, args.procs_per_task)
    print(f"  combined build wall = {t_total:.1f} s")

    # ---- verify ----
    ok, lines = verify(out_dir, layers)
    print("VERIFY:")
    for l in lines:
        print(l)

    # ---- per-layer tile counts + disk ----
    img_layers_built = [l for l in layers if l in IMG_LAYERS]
    img_tiles = {l: _count_pngs(out_dir / l) for l in img_layers_built if (out_dir / l).exists()}
    img_disk = sum(dir_bytes(out_dir / l) for l in img_tiles)
    print("tile counts:", {**img_tiles,
                            **{l: sum(1 for _ in (out_dir / l).rglob('*.pbf'))
                               for l in layers if l in CAT_LAYERS and (out_dir / l).exists()}})

    # ---- estimate full build (test mode only) ----
    if args.test:
        _estimate_and_command(args.field, cfg, region, layers, t_total, img_tiles, img_disk, info)

    if not args.keep_staging and args.test:
        pass  # keep test staging around for inspection; harmless

    print("DONE" if ok else "DONE (verification FAILED)")
    return 0 if ok else 1


def _estimate_and_command(field, cfg, region, layers, t_total, img_tiles, img_disk, info):
    y0, x0, sz = region
    with fits.open(band_file(cfg, "f444w"), memmap=True) as h:
        H, W = h[0].data.shape
    long_full, long_crop = max(H, W), sz
    slots_full = full_zoom_slots(long_full)
    slots_crop = full_zoom_slots(long_crop)
    slot_ratio = slots_full / slots_crop
    area_ratio = (H * W) / (sz * sz)

    n_img = max(1, len([l for l in layers if l in IMG_LAYERS]))
    # image tiling wall scales ~ per-image tile slots; catalogs are a small tail.
    est_img_time = t_total * slot_ratio                     # conservative: full has more blank (skipped) tiles
    est_img_disk = img_disk * slot_ratio

    print("\n================ FULL-BUILD ESTIMATE ================")
    print(f"  frame {W}x{H}; crop {sz}x{sz}; per-image tile-slot ratio = {slot_ratio:.1f}; area ratio = {area_ratio:.1f}")
    print(f"  test combined wall           = {t_total/60:.2f} min ({len(img_tiles)} img layers"
          f" + {info['n_minerva']}+{info['n_spectra']} markers)")
    print(f"  est. full image tiling       ~ {est_img_time/3600:.2f} h  (upper bound; blank sky tiles are skipped)")
    print(f"  est. full image tile disk    ~ {est_img_disk/1e9:.1f} GB")
    print(f"  + one-time RGB composite PNG build ~ minutes (Lupton over {H*W/1e6:.0f} Mpx x3) + ~1-2 GB PNG in staging")
    print(f"  + full catalog markers: minerva {218602} + spectra {6515} sources (few min clustering/tiling)")
    print("  NOTE: resume is per-layer -- rerun the same command to skip finished layers;")
    print("        delete a layer dir under the out dir to force its rebuild.")
    py = sys.executable
    script = os.path.abspath(__file__)
    print("\n  EXACT FULL-BUILD COMMAND:")
    print(f"    {py} {script} --field {field} "
          f"--procs-per-task {os.cpu_count() or 32}")
    print("  (writes to " + str(cfg["out"]) + ")")
    print("====================================================\n")


if __name__ == "__main__":
    raise SystemExit(main())
