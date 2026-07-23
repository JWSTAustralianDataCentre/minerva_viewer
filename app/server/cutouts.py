"""grizli-cutout.herokuapp.com /thumb proxy with disk cache.

Public interface (per ARCHITECTURE.md):
    get_cutout(mode: str, params: dict, cache_dir: Path) -> bytes   # PNG

Modes (per API.md):
    rgb  (default): filters=f115w-clear,f277w-clear,f444w-clear, scl=2.0, asinh=True,
                     rgb_scl=1.5,0.74,1.3, pl=2
    slit: filters=f444w-clear (grey), invert=True, nirspec=True, metafile=<msamet
          root>, dpi_scale=6, nrs_lw=0.5, nrs_alpha=0.8  (metafile is required)
    grid: all_filters=True, filters=<field's band list, caller-supplied or our
          15-band NIRCam default>

Extra passthrough params honored on top of the mode defaults: filters, scl,
metafile, invert.

UpstreamError is defined in server.dja_spectra and reused here (per
ARCHITECTURE.md "same UpstreamError pattern (import from dja_spectra to avoid
duplication)").
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import requests

from server.dja_spectra import UpstreamError

THUMB_URL = "https://grizli-cutout.herokuapp.com/thumb"

# Default 15-band NIRCam medium+wide filter list used for `grid` mode when the
# caller (main.py, via FieldConfig.cutout_filters) doesn't supply one.
DEFAULT_GRID_FILTERS = (
    "f090w-clear,f115w-clear,f140m-clear,f150w-clear,f182m-clear,f200w-clear,"
    "f210m-clear,f250w-clear,f277w-clear,f300m-clear,f356w-clear,f360m-clear,"
    "f410m-clear,f444w-clear,f460m-clear"
)

DEFAULT_RGB_FILTERS = "f115w-clear,f277w-clear,f444w-clear"

VALID_MODES = ("rgb", "slit", "grid")

_TIMEOUT_S = 30
_MIN_PNG_BYTES = 1000
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_SESSION = requests.Session()


def _validate_float(name: str, value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float, got {value!r}")
    if f != f:  # NaN
        raise ValueError(f"{name} must be finite, got NaN")
    return f


def _validate_params(mode: str, params: dict) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if "ra" not in params or "dec" not in params:
        raise ValueError("ra and dec are required")

    ra = _validate_float("ra", params["ra"])
    dec = _validate_float("dec", params["dec"])
    if not (-90.0 <= dec <= 90.0):
        raise ValueError(f"dec out of range: {dec}")

    size = _validate_float("size", params.get("size", 1.5))
    if not (0.3 < size < 30):
        raise ValueError(f"size must be in (0.3, 30), got {size}")

    q = {"ra": ra, "dec": dec, "size": size, "output": "png"}

    if mode == "rgb":
        q["filters"] = params.get("filters") or DEFAULT_RGB_FILTERS
        q["scl"] = params.get("scl", 2.0)
        q["asinh"] = "True"
        q["rgb_scl"] = "1.5,0.74,1.3"
        q["pl"] = 2
    elif mode == "slit":
        metafile = params.get("metafile")
        if not metafile:
            raise ValueError("metafile is required for mode=slit")
        q["filters"] = params.get("filters") or "f444w-clear"
        q["invert"] = "True" if params.get("invert", True) else "False"
        q["nirspec"] = "True"
        q["metafile"] = metafile
        q["dpi_scale"] = 6
        q["nrs_lw"] = 0.5
        q["nrs_alpha"] = 0.8
        if "scl" in params:
            q["scl"] = params["scl"]
    elif mode == "grid":
        q["all_filters"] = "True"
        q["filters"] = params.get("filters") or DEFAULT_GRID_FILTERS
        if "scl" in params:
            q["scl"] = params["scl"]

    # generic passthrough (only if explicitly given and not already set above)
    if params.get("invert") is not None and "invert" not in q:
        q["invert"] = "True" if params["invert"] else "False"

    return q


def _canonical_query(q: dict) -> str:
    # sorted, stable string -> stable cache key regardless of dict insertion order
    items = sorted((str(k), str(v)) for k, v in q.items())
    return urlencode(items)


def _cache_key(canonical_qs: str) -> str:
    return hashlib.sha1(canonical_qs.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "cutouts" / key[:2] / f"{key}.png"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".png")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _sane_png(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    if "image/png" not in ctype:
        return False
    if len(resp.content) <= _MIN_PNG_BYTES:
        return False
    if not resp.content.startswith(_PNG_MAGIC):
        return False
    return True


def _fetch(url: str) -> bytes:
    last_exc = None
    for attempt in range(2):  # one retry
        try:
            resp = _SESSION.get(url, timeout=_TIMEOUT_S)
        except requests.RequestException as exc:
            last_exc = exc
            continue
        if resp.status_code != 200:
            last_exc = UpstreamError(resp.status_code, f"grizli-cutout returned {resp.status_code}")
            continue
        if not _sane_png(resp):
            last_exc = UpstreamError(
                resp.status_code,
                f"grizli-cutout returned non-PNG or too-small payload "
                f"({len(resp.content)} bytes, Content-Type={resp.headers.get('Content-Type')!r})",
            )
            continue
        return resp.content
    if isinstance(last_exc, UpstreamError):
        raise last_exc
    raise UpstreamError(502, f"grizli-cutout request failed: {last_exc}")


def get_cutout(mode: str, params: dict, cache_dir: Path) -> bytes:
    """Return PNG bytes for a cutout, per API.md /api/cutout.

    mode: 'rgb' | 'slit' | 'grid'
    params: dict with ra, dec (required floats), size (arcsec, default 1.5,
        must be in (0.3, 30)), and mode-specific / passthrough keys
        (filters, scl, metafile, invert).
    cache_dir: base cache directory; cutouts are cached under
        {cache_dir}/cutouts/{sha1[:2]}/{sha1}.png
    """
    mode = (mode or "rgb").lower()
    q = _validate_params(mode, params or {})
    canonical_qs = _canonical_query(q)
    key = _cache_key(canonical_qs)
    path = _cache_path(cache_dir, key)

    if path.exists():
        try:
            data = path.read_bytes()
            if data.startswith(_PNG_MAGIC) and len(data) > _MIN_PNG_BYTES:
                return data
        except OSError:
            pass  # fall through and refetch

    url = f"{THUMB_URL}?{canonical_qs}"
    data = _fetch(url)
    _atomic_write(path, data)
    return data
