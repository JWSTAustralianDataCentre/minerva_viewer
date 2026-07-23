"""MINERVA Viewer v2 — FastAPI application.

Composes the per-field CatalogStore + EazyStore + DJA spectrum fetch +
grizli cutout proxy + QC decision store behind the API.md contract, and
serves the three build-free frontends as static files.

Run from the app/ directory:
    /otdata2/themiya/grizli_rebels/bin/python3 -m uvicorn server.main:app --port 8321
"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import starlette.middleware.gzip as _gzip_mod
from starlette.middleware.gzip import GZipMiddleware

from server.config import settings
from server.fields import load_fields
from server.catalog import CatalogStore
from server.decisions import append_decisions, load_decisions
from server.eazy_products import EazyStore
from server.dja_spectra import get_spectrum, get_preview, UpstreamError
from server.cutouts import get_cutout

APP_DIR = Path(__file__).resolve().parent.parent          # .../app
WEB_DIR = APP_DIR / "web"

IMMUTABLE = "public, max-age=31536000, immutable"
DEFAULT_FIELD = "cosmos"
DEFAULT_TEMPLATE = "sfhz_blue_agn"


# ---------------------------------------------------------------------------
# settings accessors (defensive: exact attr names live in sibling config.py)
# ---------------------------------------------------------------------------
def _setting(name, env, default):
    val = getattr(settings, name, None)
    if val:
        return val
    return os.environ.get(env, default)


def index_dir() -> Path:
    return Path(_setting("index_dir", "MINERVA_INDEX_DIR",
                         "/otdata2/themiya/minerva/data/viewer_index"))


def cache_dir() -> Path:
    return Path(_setting("cache_dir", "MINERVA_CACHE_DIR",
                         "/otdata2/themiya/minerva/data/cache"))


# ---------------------------------------------------------------------------
# rounding / JSON-safety helper (API.md: non-finite floats -> null)
# ---------------------------------------------------------------------------
def _sanitize(o):
    """Recursively make a payload JSON-safe: NaN/inf -> None, numpy -> native."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    return o


def json_ok(data, headers=None):
    return JSONResponse(_sanitize(data), headers=headers)


# ---------------------------------------------------------------------------
# lifespan: load fields + build stores once (parquet load only; eazy lazy)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    fields = load_fields()
    idx = index_dir()
    catalog_stores: dict[str, CatalogStore] = {}
    eazy_stores: dict[str, EazyStore] = {}
    for name, fcfg in fields.items():
        try:
            catalog_stores[name] = CatalogStore(fcfg, idx)
        except Exception as exc:  # missing index should not kill the server
            print(f"[startup] catalog load failed for {name}: {exc}")
        try:
            eazy_stores[name] = EazyStore(fcfg)   # lazy-opens h5 on first use
        except Exception as exc:
            print(f"[startup] eazy init failed for {name}: {exc}")
    app.state.fields = fields
    app.state.catalog_stores = catalog_stores
    app.state.eazy_stores = eazy_stores
    # Warm the per-template EAzY shared arrays (zgrid/pivot/template list) so the
    # first real /api/eazy request per template doesn't pay the ~780ms lazy
    # h5-open + template-rebuild cost. Per-object results stay lazy (LRU).
    for name, fcfg in fields.items():
        store = eazy_stores.get(name)
        if store is None:
            continue
        for tpl in getattr(fcfg, "eazy", {}):
            try:
                store._shared(tpl)  # noqa: SLF001 -- intentional startup warm
                print(f"[startup] warmed eazy {name}/{tpl}")
            except Exception as exc:  # a bad template dir must not kill startup
                print(f"[startup] eazy warm failed for {name}/{tpl}: {exc}")
    print(f"[startup] fields={list(fields)} "
          f"catalog={list(catalog_stores)} eazy={list(eazy_stores)}")
    yield


app = FastAPI(title="MINERVA Viewer v2", lifespan=lifespan)

# Compress large text/JSON responses (catalog/query, inspect/list, HTML shells).
# The minimum_size gate skips tiny bodies; we additionally exclude "image/*" so
# the already-compressed PNGs from /api/cutout and /api/dja/preview are NOT
# re-gzipped when a browser sends Accept-Encoding: gzip (Starlette only excludes
# text/event-stream by default; the responder reads this module constant at
# send time, so extending it here covers both the identity and gzip paths).
if "image/" not in _gzip_mod.DEFAULT_EXCLUDED_CONTENT_TYPES:
    _gzip_mod.DEFAULT_EXCLUDED_CONTENT_TYPES = (
        _gzip_mod.DEFAULT_EXCLUDED_CONTENT_TYPES + ("image/",)
    )
app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Optional shared-credential gate for internet exposure (tunnel / reverse proxy).
# Off by default (localhost use). Set BOTH env vars to enable:
#   MINERVA_AUTH_USER=minerva MINERVA_AUTH_PASS=<secret>  -> HTTP Basic auth on
# everything except /healthz. Browsers prompt natively; no frontend changes.
# ---------------------------------------------------------------------------
_AUTH_USER = os.environ.get("MINERVA_AUTH_USER") or None
_AUTH_PASS = os.environ.get("MINERVA_AUTH_PASS") or None
if _AUTH_USER and _AUTH_PASS:
    import base64
    import secrets as _secrets

    _EXPECTED = "Basic " + base64.b64encode(
        f"{_AUTH_USER}:{_AUTH_PASS}".encode()).decode()

    @app.middleware("http")
    async def _basic_auth(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)
        supplied = request.headers.get("authorization", "")
        if not _secrets.compare_digest(supplied, _EXPECTED):
            return Response(status_code=401, headers={
                "WWW-Authenticate": 'Basic realm="MINERVA viewer"'})
        return await call_next(request)


@app.exception_handler(UpstreamError)
async def _upstream_handler(request: Request, exc: UpstreamError):
    status = getattr(exc, "status", 502) or 502
    detail = getattr(exc, "detail", str(exc))
    return JSONResponse({"detail": detail}, status_code=status)


def _catalog(field: str) -> CatalogStore:
    store = app.state.catalog_stores.get(field)
    if store is None:
        raise _not_found(f"unknown field '{field}'")
    return store


def _eazy(field: str) -> EazyStore:
    store = app.state.eazy_stores.get(field)
    if store is None:
        raise _not_found(f"eazy products unavailable for field '{field}'")
    return store


class _HTTPError(Exception):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail


def _not_found(detail):
    return _HTTPError(404, detail)


@app.exception_handler(_HTTPError)
async def _http_error_handler(request: Request, exc: _HTTPError):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status)


# ===========================================================================
# Catalog / Explorer
# ===========================================================================
@app.get("/api/fields")
def api_fields():
    entries = [store.info() for store in app.state.catalog_stores.values()]
    return json_ok({"fields": entries})


@app.get("/api/catalog/query")
def api_query(
    field: str = DEFAULT_FIELD,
    template: str | None = None,
    ids: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_arcsec: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    lmass_min: float | None = None,
    lmass_max: float | None = None,
    mag_min: float | None = None,
    mag_max: float | None = None,
    mag_band: str = "f444w",
    has_spec: int | None = None,
    grade_min: int | None = None,
    use_phot: int = 1,
    no_star: int = 1,
    uvj: str | None = None,
    sort: str | None = None,
    limit: int = Query(500, ge=0, le=5000),
    offset: int = Query(0, ge=0),
):
    store = _catalog(field)
    total, rows = store.query(
        template=template, ids=ids, ra=ra, dec=dec, radius_arcsec=radius_arcsec,
        z_min=z_min, z_max=z_max, lmass_min=lmass_min, lmass_max=lmass_max,
        mag_min=mag_min, mag_max=mag_max, mag_band=mag_band, has_spec=has_spec,
        grade_min=grade_min, use_phot=use_phot, no_star=no_star, uvj=uvj,
        sort=sort, limit=limit, offset=offset,
    )
    return json_ok({"total": total, "rows": rows})


@app.get("/api/object/{field}/{obj_id}")
def api_object(field: str, obj_id: int):
    store = _catalog(field)
    rec = store.object(obj_id)
    if rec is None:
        raise _not_found(f"object {obj_id} not found in field '{field}'")
    return json_ok(rec)


# ===========================================================================
# EAzY products
# ===========================================================================
@app.get("/api/eazy/{field}/{template}/{obj_id}")
def api_eazy(field: str, template: str, obj_id: int):
    store = _eazy(field)
    try:
        payload = store.get_products(template, obj_id)
    except KeyError:
        # EazyStore raises KeyError for an unknown object id or template set.
        raise _not_found(
            f"eazy products not found for {field}/{template}/{obj_id}")
    if payload is None:
        raise _not_found(f"eazy products not found for {field}/{template}/{obj_id}")
    return json_ok(payload)


# ===========================================================================
# Bulk prefetch (Inspector navigation: next-3 / prev-1)
# ===========================================================================
@app.get("/api/prefetch")
def api_prefetch(
    ids: str,
    field: str = DEFAULT_FIELD,
    template: str | None = None,
):
    """Bundle /api/object + /api/eazy for several ids into one response.

    Additive to the per-id endpoints: each id's ``object`` block equals the body
    of GET /api/object/{field}/{id} and its ``eazy`` block equals the body of
    GET /api/eazy/{field}/{template}/{id}. Missing ids/products serialise as
    null blocks rather than failing the whole batch. Compounds with gzip.
    """
    store = _catalog(field)
    eazy = app.state.eazy_stores.get(field)
    tpl = template or store.default_template

    id_list, seen = [], set()
    for tok in str(ids).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            iv = int(float(tok))
        except ValueError:
            continue
        if iv not in seen:
            seen.add(iv)
            id_list.append(iv)

    # The inspector prefetches next-3/prev-1; cap defends the synchronous
    # object+eazy loop against unbounded single-GET CPU/IO amplification.
    id_list = id_list[:8]

    items = []
    for iv in id_list:
        obj = store.object(iv)
        ez = None
        if eazy is not None:
            try:
                ez = eazy.get_products(tpl, iv)
            except KeyError:
                ez = None
        items.append({"id": iv, "object": obj, "eazy": ez})

    return json_ok({"field": field, "template": tpl, "objects": items})


# ===========================================================================
# Spectra
# ===========================================================================
@app.get("/api/inspect/list")
def api_inspect_list(field: str = DEFAULT_FIELD, template: str | None = None):
    store = _catalog(field)
    return json_ok({"objects": store.inspect_list(template)})


@app.get("/api/spectrum/{root}/{file}")
def api_spectrum(root: str, file: str):
    payload = get_spectrum(root, file, cache_dir())
    # immutable per-object product: long cache
    return json_ok(payload, headers={"Cache-Control": IMMUTABLE})


# ===========================================================================
# Cutouts
# ===========================================================================
@app.get("/api/cutout")
def api_cutout(request: Request):
    params = dict(request.query_params)
    mode = params.pop("mode", "rgb")
    try:
        png = get_cutout(mode, params, cache_dir())
    except ValueError as exc:
        # get_cutout raises ValueError for invalid ra/dec/size/mode/metafile.
        raise _HTTPError(400, str(exc))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": IMMUTABLE})


@app.get("/api/dja/preview/{root}/{basename}")
def api_dja_preview(root: str, basename: str):
    png, media_type = get_preview(root, basename, cache_dir())
    return Response(content=png, media_type=media_type or "image/png",
                    headers={"Cache-Control": IMMUTABLE})


# ===========================================================================
# QC decisions
# ===========================================================================
@app.post("/api/decisions")
async def api_post_decisions(request: Request):
    # Malformed / empty / non-JSON body -> 400 (not 500). json.loads (used by
    # Starlette) accepts NaN/Infinity, so those survive parsing and are caught
    # below at append time; a truly unparseable body raises here.
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        raise _HTTPError(400, "malformed JSON body")
    if not isinstance(payload, dict):
        raise _HTTPError(400, "decisions payload must be a JSON object")
    field = payload.get("field", DEFAULT_FIELD)
    # Off-load the blocking append to the threadpool so concurrent POSTs don't
    # serialize on the event loop. Non-finite floats (inf/NaN) make
    # json.dumps(allow_nan=False) raise ValueError -> 400 (file untouched).
    try:
        n = await run_in_threadpool(append_decisions, field, payload, cache_dir())
    except ValueError as exc:
        raise _HTTPError(400, f"invalid decisions payload: {exc}")
    return json_ok({"ok": True, "n": n})


@app.get("/api/decisions")
def api_get_decisions(field: str = DEFAULT_FIELD):
    return json_ok(load_decisions(field, cache_dir()))


# ===========================================================================
# Static mounts (registered last so /api/* wins). check_dir=False tolerates
# frontends not yet built by sibling agents.
# ===========================================================================
# The inspector StaticFiles mount (html=True) serves its index only at the
# canonical "/inspector/" path; the explorer nav links to the bare "/inspector".
# Register an explicit redirect BEFORE the mounts so the link resolves.
@app.get("/inspector")
def _inspector_redirect():
    return RedirectResponse(url="/inspector/", status_code=307)


# Health check MUST be registered before the catch-all "/" mount below; Starlette
# matches routes in registration order and StaticFiles(html=True) at "/" matches
# every path, so a later /healthz route is unreachable.
@app.get("/healthz")
def healthz():
    return {"ok": True, "fields": list(getattr(app.state, "catalog_stores", {}))}


class _CachedStaticFiles(StaticFiles):
    """StaticFiles that stamps a fixed Cache-Control on every file response.

    Starlette's StaticFiles emits ETag + Last-Modified but no Cache-Control, so
    each asset costs a conditional request. Immutable, per-deploy assets (vendor
    bundles, fonts, design tokens) get a long max-age; the mutable HTML shells
    get "no-cache" so a redeploy is picked up on the next load.
    """

    def __init__(self, *args, cache_control: str | None = None, **kwargs):
        self._cache_control = cache_control
        super().__init__(*args, **kwargs)

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        if self._cache_control:
            resp.headers["Cache-Control"] = self._cache_control
        return resp


# Shell HTML is mutable per deploy -> revalidate every load.
SHELL_CACHE = "no-cache"


def _mount(path, subdir, html=False, cache_control=None):
    app.mount(path, _CachedStaticFiles(directory=str(WEB_DIR / subdir),
                                       html=html, check_dir=False,
                                       cache_control=cache_control),
              name=subdir)


# /inspector and / are HTML shells (mutable) -> no-cache.
# /vendor (react, runtime) and /shared (fonts, tokens) are immutable per deploy.
_mount("/inspector", "inspector", html=True, cache_control=SHELL_CACHE)
_mount("/shared", "shared", cache_control=IMMUTABLE)
_mount("/vendor", "vendor", cache_control=IMMUTABLE)

# Field Map: pre-generated fitsmap tile trees (scripts/build_map.py), one dir per
# field under MINERVA_MAP_DIR. Mounted only when tiles exist; tiles are immutable.
_MAP_DIR = Path(os.environ.get("MINERVA_MAP_DIR",
                               "/otdata2/themiya/minerva/data/map_tiles"))
if _MAP_DIR.is_dir() and any(p.is_dir() and not p.name.endswith("_test")
                             for p in _MAP_DIR.iterdir()):
    app.mount("/map", _CachedStaticFiles(directory=str(_MAP_DIR), html=True,
                                         check_dir=False,
                                         cache_control=IMMUTABLE),
              name="map")

_mount("/", "explorer", html=True, cache_control=SHELL_CACHE)   # root last
