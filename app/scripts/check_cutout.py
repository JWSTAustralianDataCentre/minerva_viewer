#!/usr/bin/env /otdata2/themiya/grizli_rebels/bin/python3
"""Live smoke test for server/cutouts.py.

Fetches one cutout per mode (rgb, slit, grid) for ra=150.13, dec=2.328,
verifies PNG magic bytes, and verifies a second call for the same params is
served from the disk cache. For `slit` mode a metafile is required; this
script derives one by querying grizli-cutout.herokuapp.com/nirspec_extractions
for a nearby DJA source's msamet, falling back to rgb+grid-only (noting why)
if that lookup fails.

Run:
    cd app && /otdata2/themiya/grizli_rebels/bin/python3 scripts/check_cutout.py
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from server.cutouts import get_cutout, _cache_path, _cache_key, _canonical_query, _validate_params

CACHE_DIR = Path("/tmp/claude-1005/-otdata2-themiya-minerva/fcc0bdc0-01de-4a5b-92a4-a69c924bb4c7/scratchpad/a4/cache_test")
RA, DEC = 150.13, 2.328


def derive_metafile() -> str | None:
    """Query nirspec_extractions for a msamet near (RA, DEC) and derive
    metafile = msamet.split('_')[0], per ARCHITECTURE.md."""
    url = f"https://grizli-cutout.herokuapp.com/nirspec_extractions?coords={RA},{DEC}&size=5&output=csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[warn] nirspec_extractions query failed: {exc}")
        return None
    text = resp.text
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception as exc:
        print(f"[warn] could not parse nirspec_extractions CSV: {exc}")
        return None
    if not rows or "msamet" not in (rows[0].keys() if rows else []):
        print(f"[warn] nirspec_extractions returned no usable rows (n={len(rows)})")
        return None
    msamet = rows[0].get("msamet")
    if not msamet:
        print("[warn] first row has empty msamet")
        return None
    metafile = msamet.split("_")[0]
    print(f"[info] derived metafile={metafile!r} from msamet={msamet!r} (root={rows[0].get('root')})")
    return metafile


def check_mode(mode: str, params: dict) -> None:
    print(f"\n=== mode={mode} params={params} ===")

    q = _validate_params(mode, params)
    canonical_qs = _canonical_query(q)
    key = _cache_key(canonical_qs)
    path = _cache_path(CACHE_DIR, key)
    if path.exists():
        path.unlink()  # start clean so the first fetch below is a real miss

    t0 = time.time()
    data1 = get_cutout(mode, params, CACHE_DIR)
    dt1 = time.time() - t0

    assert data1.startswith(b"\x89PNG\r\n\x1a\n"), f"{mode}: missing PNG magic bytes"
    assert len(data1) > 1000, f"{mode}: suspiciously small ({len(data1)} bytes)"
    assert path.exists(), f"{mode}: cache file not written at {path}"
    print(f"[ok] first fetch: {len(data1)} bytes in {dt1:.2f}s, cached at {path}")

    t1 = time.time()
    data2 = get_cutout(mode, params, CACHE_DIR)
    dt2 = time.time() - t1
    assert data2 == data1, f"{mode}: cached bytes differ from first fetch"
    print(f"[ok] second fetch (cache hit): {len(data2)} bytes in {dt2:.3f}s (should be much faster than {dt1:.2f}s)")


def main() -> int:
    results = {}
    failures = []

    try:
        check_mode("rgb", {"ra": RA, "dec": DEC, "size": 3.0})
        results["rgb"] = "ok"
    except Exception as exc:
        print(f"[FAIL] rgb: {exc}")
        failures.append(("rgb", exc))

    try:
        check_mode("grid", {"ra": RA, "dec": DEC, "size": 2.0})
        results["grid"] = "ok"
    except Exception as exc:
        print(f"[FAIL] grid: {exc}")
        failures.append(("grid", exc))

    metafile = derive_metafile()
    if metafile:
        try:
            check_mode("slit", {"ra": RA, "dec": DEC, "size": 3.0, "metafile": metafile})
            results["slit"] = "ok"
        except Exception as exc:
            print(f"[FAIL] slit: {exc}")
            failures.append(("slit", exc))
    else:
        print("\n[note] skipping slit mode: could not derive a metafile from "
              "nirspec_extractions; verified rgb+grid only.")
        results["slit"] = "skipped (no metafile derivable)"

    print("\n=== summary ===")
    for mode, status in results.items():
        print(f"  {mode}: {status}")

    if failures:
        print(f"\n{len(failures)} mode(s) FAILED")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
