"""DJA NIRSpec spec.fits fetch + parse + disk cache.

Fetches per-spectrum extraction products from the public msaexp-nirspec S3 bucket,
caches them on local disk (atomic write), and parses spec.fits into the /api/spectrum
payload (see app/API.md). Also proxies the .fnu.png / .flam.png quicklook previews.

Public interface (imported by server/main.py and server/cutouts.py):
    class UpstreamError(Exception)          # .status (int), .detail (str)
    def get_spectrum(root, file, cache_dir) -> dict
    def get_preview(root, basename, cache_dir) -> tuple[bytes, str]

Modules built in parallel import UpstreamError from here; keep it stable.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_BASE = "https://s3.amazonaws.com/msaexp-nirspec/extractions"

# Security: root / file / basename token whitelist. Anything outside this set
# (or containing a path traversal) is rejected before any network / disk access.
_NAME_RE = re.compile(r"^[A-Za-z0-9._+-]+$")

_TIMEOUT = 30           # seconds per request
_RETRIES = 2            # retries AFTER the first attempt (3 attempts total)
_BACKOFF = 1.5          # seconds; multiplied by attempt index

# spec.fits filename pattern: <root>_<grating>-<filter>_<pid>_<srcid>.spec.fits
_SPECFILE_RE = re.compile(r"_(?P<pid>\d+)_(?P<srcid>\d+)\.spec\.fits$")


class UpstreamError(Exception):
    """Raised for validation failures (4xx) and upstream fetch failures (5xx).

    Attributes:
        status: HTTP status code the caller should surface.
        detail: human-readable message (safe to return in JSON body).
    """

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = int(status)
        self.detail = str(detail)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_token(value: str, label: str) -> None:
    if not value or not _NAME_RE.match(value):
        raise UpstreamError(400, f"invalid {label!r}: must match [A-Za-z0-9._+-]+")
    if ".." in value:
        raise UpstreamError(400, f"invalid {label!r}: path traversal rejected")
    # Belt-and-suspenders: the regex already excludes these, but be explicit.
    if "/" in value or "\\" in value or "\x00" in value:
        raise UpstreamError(400, f"invalid {label!r}: illegal character")


def _validate_spec(root: str, file: str) -> None:
    _validate_token(root, "root")
    _validate_token(file, "file")
    if not file.endswith(".spec.fits"):
        raise UpstreamError(400, "invalid 'file': must end with .spec.fits")


def _validate_preview(root: str, basename: str) -> None:
    _validate_token(root, "root")
    _validate_token(basename, "basename")
    if not (basename.endswith(".fnu.png") or basename.endswith(".flam.png")):
        raise UpstreamError(400, "invalid 'basename': must end with .fnu.png or .flam.png")


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

def _atomic_download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically (tmp file then os.replace).

    Retries transient failures with linear backoff. Raises UpstreamError on
    exhaustion. Leaves no partial file at ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp.{os.getpid()}"

    last_exc: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=_TIMEOUT) as resp:
                sc = resp.status_code
                # A missing object in this bucket returns 403 (no ListBucket),
                # not 404 — treat all definitive client-side 4xx as not-found and
                # do NOT retry them (retrying burns a threadpool worker ~6.9s and
                # is a DoS-amplification vector on user-supplied spec names).
                if sc in (401, 403, 404, 410):
                    raise UpstreamError(404, f"upstream object not found: {url}")
                if sc != 200:
                    raise UpstreamError(
                        502, f"upstream returned HTTP {sc} for {url}"
                    )
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            os.replace(tmp, dest)
            return
        except UpstreamError as exc:
            # 404 (incl. mapped 401/403/410) is definitive; don't retry it.
            if exc.status == 404:
                _unlink_quiet(tmp)
                raise
            last_exc = exc
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
        _unlink_quiet(tmp)
        if attempt < _RETRIES:
            time.sleep(_BACKOFF * (attempt + 1))

    raise UpstreamError(502, f"failed to fetch {url}: {last_exc}")


def _unlink_quiet(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def _ensure_cached(url: str, dest: Path) -> Path:
    """Return ``dest``, downloading it first if not already cached.

    Spec/preview objects are immutable, so a present file is always reused.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _atomic_download(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean_1d(arr, sig: int | None = None) -> list:
    """numpy 1D -> python list, non-finite -> None (JSON null).

    ``sig`` optionally rounds finite values to that many significant figures to
    keep the JSON payload compact (per API.md: fluxes 4 sig figs).
    """
    out: list = []
    for v in np.asarray(arr, dtype=np.float64):
        if not np.isfinite(v):
            out.append(None)
        elif sig is None:
            out.append(float(v))
        else:
            out.append(_round_sig(float(v), sig))
    return out


def _round_sig(v: float, sig: int) -> float:
    if v == 0.0:
        return 0.0
    from math import floor, log10
    return round(v, -int(floor(log10(abs(v)))) + (sig - 1))


def _column_or_none(table, name: str):
    """Return the named SPEC1D column as float64, or None if absent."""
    if name in (c.lower() for c in table.columns.names):
        # Match case-insensitively.
        for cname in table.columns.names:
            if cname.lower() == name:
                return np.asarray(table[cname], dtype=np.float64)
    return None


def _srcid_from_filename(file: str) -> int | None:
    m = _SPECFILE_RE.search(file)
    if m:
        try:
            return int(m.group("srcid"))
        except (TypeError, ValueError):
            return None
    return None


def _pid_from_header(hdr, file: str) -> int | None:
    prog = hdr.get("PROGRAM")
    if prog is not None:
        try:
            return int(str(prog).strip())
        except (TypeError, ValueError):
            pass
    m = _SPECFILE_RE.search(file)
    if m:
        try:
            return int(m.group("pid"))
        except (TypeError, ValueError):
            return None
    return None


def _grating_from_header(hdr) -> str:
    grating = str(hdr.get("GRATING", "") or "").strip().lower()
    filt = str(hdr.get("FILTER", "") or "").strip().lower()
    parts = [p for p in (grating, filt) if p]
    return "-".join(parts)


def _srcid_from_header(hdr, file: str) -> int | None:
    for key in ("SRCID", "SOURCEID", "SRCALIAS"):
        val = hdr.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return _srcid_from_filename(file)


def _parse_spec_fits(path: Path, root: str, file: str) -> dict:
    with fits.open(path, memmap=False) as hdul:
        try:
            spec1d = hdul["SPEC1D"]
        except KeyError:
            raise UpstreamError(502, "spec.fits missing SPEC1D HDU")
        table = spec1d.data

        wave = _column_or_none(table, "wave")
        flux = _column_or_none(table, "flux")
        err = _column_or_none(table, "err")
        sky = _column_or_none(table, "sky")
        if wave is None or flux is None:
            raise UpstreamError(502, "SPEC1D missing required wave/flux columns")
        n = len(wave)
        if err is None:
            err = np.full(n, np.nan)
        if sky is None:
            sky = np.full(n, np.nan)

        # 2D rectified spectrum from SCI.
        twod = None
        try:
            sci = hdul["SCI"]
        except KeyError:
            sci = None
        if sci is not None and sci.data is not None:
            data = np.asarray(sci.data, dtype="<f4")  # little-endian float32
            if data.ndim != 2:
                raise UpstreamError(502, f"SCI HDU is not 2D (ndim={data.ndim})")
            ny, nx = int(data.shape[0]), int(data.shape[1])
            if nx != n:
                raise UpstreamError(
                    502, f"SCI nx={nx} does not match 1D wave length {n}"
                )
            # Row-major (C order) bytes; NaNs preserved in the b64 blob.
            blob = np.ascontiguousarray(data).tobytes(order="C")
            twod = {
                "ny": ny,
                "nx": nx,
                "data": base64.b64encode(blob).decode("ascii"),
                "wave_um": _clean_1d(wave, sig=6),
            }
            hdr = sci.header
        else:
            hdr = spec1d.header

        meta = {
            "root": root,
            "file": file,
            "grating": _grating_from_header(hdr),
            "pid": _pid_from_header(hdr, file),
            "srcid": _srcid_from_header(hdr, file),
            "exptime": _finite_or_none(hdr.get("EFFEXPTM")),
        }

        return {
            "meta": meta,
            "wave_um": _clean_1d(wave, sig=6),
            "flux_uJy": _clean_1d(flux, sig=4),
            "err_uJy": _clean_1d(err, sig=4),
            "sky_uJy": _clean_1d(sky, sig=4),
            "twod": twod,
        }


def _finite_or_none(val):
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_spectrum(root: str, file: str, cache_dir) -> dict:
    """Fetch (S3 -> disk cache) and parse a DJA spec.fits into /api/spectrum payload.

    Args:
        root:      DJA reduction root, e.g. "capers-cos01-v4".
        file:      spec.fits basename, e.g. "capers-cos01-v4_prism-clear_6368_46271.spec.fits".
        cache_dir: base cache directory (files land under {cache_dir}/spectra/{root}/{file}).

    Raises:
        UpstreamError: on validation (400), missing upstream (404) or fetch/parse (502).
    """
    _validate_spec(root, file)
    cache_dir = Path(cache_dir)
    dest = cache_dir / "spectra" / root / file
    url = f"{S3_BASE}/{root}/{file}"
    _ensure_cached(url, dest)
    try:
        return _parse_spec_fits(dest, root, file)
    except UpstreamError:
        raise
    except Exception as exc:  # corrupt cache / astropy failure
        raise UpstreamError(502, f"failed to parse {file}: {exc}")


def get_preview(root: str, basename: str, cache_dir) -> tuple[bytes, str]:
    """Fetch (S3 -> disk cache) the .fnu.png / .flam.png quicklook preview.

    Returns:
        (png_bytes, "image/png")

    Raises:
        UpstreamError: on validation (400), missing upstream (404) or fetch (502).
    """
    _validate_preview(root, basename)
    cache_dir = Path(cache_dir)
    dest = cache_dir / "previews" / root / basename
    url = f"{S3_BASE}/{root}/{basename}"
    _ensure_cached(url, dest)
    try:
        data = dest.read_bytes()
    except OSError as exc:
        raise UpstreamError(502, f"failed to read cached preview {basename}: {exc}")
    if not data:
        raise UpstreamError(502, f"empty preview {basename}")
    return data, "image/png"
