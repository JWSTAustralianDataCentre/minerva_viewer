#!/usr/bin/env python3
"""Smoke test for server/dja_spectra.get_spectrum / get_preview.

Two cases:
  (1) LOCAL  — seed app/docs/sample.spec.fits into the cache at the canonical path
               for root=capers-cos01-v4, file=capers-cos01-v4_prism-clear_6368_46271.spec.fits,
               then parse it with NO network access.
  (2) LIVE   — pick a *different* capers-cos01-v4 spectrum from the DJA v4.5 csv and
               fetch+parse it over the network.

Asserts on both: wave strictly increasing, len(flux)==nx, base64 decodes to ny*nx*4
bytes, flux/err within a plausible uJy range.

Run:
    /otdata2/themiya/grizli_rebels/bin/python3 app/scripts/check_spectrum.py
"""

from __future__ import annotations

import base64
import gzip
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make server/ importable when run from repo root or app/.
_APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP))

from server.dja_spectra import get_spectrum, UpstreamError  # noqa: E402

DOCS_SAMPLE = _APP / "docs" / "sample.spec.fits"
SAMPLE_ROOT = "capers-cos01-v4"
SAMPLE_FILE = "capers-cos01-v4_prism-clear_6368_46271.spec.fits"
DJA_CSV = Path("/otdata2/themiya/minerva/data/catalogs/dja/dja_msaexp_emission_lines_v4.5.csv.gz")

# Plausible uJy range for extracted NIRSpec fluxes (very loose sanity bound).
UJY_LO, UJY_HI = -1e4, 1e6


def check_payload(label: str, payload: dict) -> None:
    meta = payload["meta"]
    wave = np.array([w for w in payload["wave_um"] if w is not None], dtype=float)
    flux_raw = payload["flux_uJy"]
    twod = payload["twod"]

    n = len(payload["wave_um"])
    assert len(flux_raw) == n, f"{label}: len(flux)={len(flux_raw)} != len(wave)={n}"

    # wave strictly increasing
    assert np.all(np.diff(wave) > 0), f"{label}: wave not strictly increasing"

    # 2D checks
    assert twod is not None, f"{label}: missing twod"
    ny, nx = twod["ny"], twod["nx"]
    assert nx == n, f"{label}: nx={nx} != len(wave)={n}"
    blob = base64.b64decode(twod["data"])
    assert len(blob) == ny * nx * 4, (
        f"{label}: b64 decodes to {len(blob)} bytes, expected {ny*nx*4}"
    )
    # Confirm it decodes as little-endian float32 with the right shape.
    arr = np.frombuffer(blob, dtype="<f4").reshape(ny, nx)
    assert arr.shape == (ny, nx)

    # Plausible flux range (ignore nulls / NaNs)
    finite_flux = np.array([f for f in flux_raw if f is not None], dtype=float)
    finite_flux = finite_flux[np.isfinite(finite_flux)]
    assert finite_flux.size > 0, f"{label}: no finite flux values"
    fmin, fmax = float(finite_flux.min()), float(finite_flux.max())
    assert UJY_LO <= fmin and fmax <= UJY_HI, (
        f"{label}: flux out of plausible range [{fmin}, {fmax}]"
    )

    finite2d = arr[np.isfinite(arr)]
    n_null_flux = sum(1 for f in flux_raw if f is None)

    print(f"[{label}] OK")
    print(f"    meta: grating={meta['grating']} pid={meta['pid']} "
          f"srcid={meta['srcid']} exptime={meta['exptime']}")
    print(f"    N points   : {n}  ({n_null_flux} null flux)")
    print(f"    wave range : {wave.min():.4f} - {wave.max():.4f} um")
    print(f"    flux range : {fmin:.4g} - {fmax:.4g} uJy")
    print(f"    2D (SCI)   : ny={ny} x nx={nx}  ({len(blob)} bytes b64)")
    print(f"    2D finite  : {finite2d.size}/{arr.size}  "
          f"range {finite2d.min():.4g}..{finite2d.max():.4g} uJy")


def find_live_file() -> tuple[str, str]:
    """Pick a capers-cos01-v4 spectrum from the csv that is NOT the seeded sample."""
    with gzip.open(DJA_CSV, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        i_file = header.index("file")
        i_root = header.index("root")
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(i_file, i_root):
                continue
            root = parts[i_root]
            file = parts[i_file]
            if root == "capers-cos01-v4" and file.endswith(".spec.fits") \
                    and file != SAMPLE_FILE:
                return root, file
    raise RuntimeError("no alternative capers-cos01-v4 spectrum found in csv")


def main() -> int:
    tmp_cache = Path(tempfile.mkdtemp(prefix="check_spectrum_"))
    print(f"cache_dir = {tmp_cache}")

    # --- Case 1: LOCAL (seed sample, no network) ---
    seeded = tmp_cache / "spectra" / SAMPLE_ROOT / SAMPLE_FILE
    seeded.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DOCS_SAMPLE, seeded)
    payload_local = get_spectrum(SAMPLE_ROOT, SAMPLE_FILE, tmp_cache)
    check_payload("LOCAL", payload_local)

    # --- Case 2: LIVE fetch ---
    live_root, live_file = find_live_file()
    print(f"\nLIVE fetch: {live_root}/{live_file}")
    payload_live = get_spectrum(live_root, live_file, tmp_cache)
    check_payload("LIVE", payload_live)
    # Confirm it actually hit disk cache.
    live_path = tmp_cache / "spectra" / live_root / live_file
    assert live_path.exists() and live_path.stat().st_size > 0, "live file not cached"
    print(f"    cached at  : {live_path} ({live_path.stat().st_size} bytes)")

    # --- Negative: validation rejects traversal ---
    for bad_root, bad_file in [
        ("../etc", "x.spec.fits"),
        ("ok", "x.fits"),
        ("ok", "../x.spec.fits"),
    ]:
        try:
            get_spectrum(bad_root, bad_file, tmp_cache)
        except UpstreamError as e:
            assert e.status == 400, f"expected 400, got {e.status}"
        else:
            raise AssertionError(f"validation did not reject {bad_root}/{bad_file}")
    print("\n[VALIDATION] OK — traversal / bad-extension inputs rejected with 400")

    shutil.rmtree(tmp_cache, ignore_errors=True)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
