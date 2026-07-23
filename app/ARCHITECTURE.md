# MINERVA Viewer v2 — Architecture

One local FastAPI server, three decoupled surfaces. Frontend is build-free (no Node on this host).

```
minerva_viewer/app/
  server/
    main.py            # FastAPI app: routes, static mounts, startup index load
    config.py          # env-overridable settings (paths, port, cache dir)
    fields.py          # field manifest loader (fields/*.json)
    fields/cosmos.json # per-field data paths + template sets
    catalog.py         # parquet-backed catalog store + query engine
    eazy_products.py   # h5 row slices: p(z), chi2, SED reconstruction, photometry
    dja_spectra.py     # DJA S3 spec.fits fetch + parse + disk cache
    cutouts.py         # grizli thumb proxy + disk cache
    decisions.py       # server-side QC decision store (JSONL)
  scripts/
    build_index.py     # SUPER catalog + zout + DJA cross-match -> parquet indices
    debundle.py        # handoff/index.html ds-bundle -> plain static files
    validate_eazy.py   # offline: our SED/p(z) vs eazy-py show_fit reference
  web/
    _bundle_src/       # decoded prototype sources (template.html, app.js, runtime, fonts)
    inspector/         # de-bundled, wired Redshift Inspector (served at /inspector)
    explorer/          # Catalog/Image Explorer (served at /)
    shared/            # design tokens css, fonts, small shared js
  docs/                # domain reports: dataSchemas.md, prototype.md, djaAws.md, oldViewer.md
```

## Surfaces
1. **Redshift Inspector** (`/inspector`) — the handoff prototype (design frozen), wired to real data.
   One object at a time; prefetch next 3 / prev 1 (same pattern as old zgui_panel).
2. **Catalog Explorer** (`/`) — query panel (ID / cone / z / mass / mag / flags) → table →
   per-object detail (multiband cutout strip, SED quicklook, link to Inspector + external map).
   Per-object cutouts only; never loads mosaics wholesale.
3. **Field Map** (`/map`, DEFERRED module) — pre-generated static tiles from local mosaics.
   App is complete without it; links out to minerva.colorado.edu meanwhile.

## Data facts (verified 2026-07-18; details in docs/dataSchemas.md, docs/djaAws.md)
- COSMOS SUPER catalog, zout.fits, and eazy h5 `cat/id` are **row-aligned 1:1** (294,126 rows,
  np.array_equal verified). Row index joins all three — no id matching.
- Catalog fluxes `f_{band}` are in units of **10 nJy** (×0.01 = µJy; mAB = 23.9 − 2.5·log10(µJy)).
- zout has z_phot/z_ml + percentiles z025..z975, mass, sfr, restU/B/V/J, u_v, v_j, uvj_class.
- p(z) is NOT stored: compute exp(−(chi2−min)/2) from h5 `fit/chi2_fit[i]` (607-pt log zgrid;
  explicit grid in eazy.data.fits HDU 'ZGRID'). APPLY_PRIOR=0.
- Template sets locally available: `sfhz_blue_agn` (default) and `larson`, both SUPER + D020.
- DJA: **v4.5** (2026-02-23, 89,303 spectra, 219 roots) downloaded to
  `/otdata2/themiya/minerva/data/catalogs/dja/dja_msaexp_emission_lines_v4.5.csv.gz`.
  Per-spectrum files `https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}`
  (public GET, **no ListBucket**, **no CORS** → server proxies). spec.fits HDUs: SPEC1D
  (wave µm, flux µJy, err µJy…), SCI 2D (31 spatial × 473 wave, µJy), WHT, PROFILE, SLITS.
- Cutouts: grizli-cutout.herokuapp.com/thumb (png or fits; nirspec slit overlay via
  metafile = msamet.split('_')[0]; no CORS → proxied + disk-cached).
- Cross-match: SkyCoord nearest, max 0.5″, done at index-build; keep ALL DJA rows within 0.5″
  per MINERVA source (one row per *spectrum*), sep surfaced everywhere.
- DJA `grade`: 3=robust, 2=maybe, 1=bad, 0/NaN=ungraded (~47% NaN) → serve NaN as grade 0.

## Non-negotiables
- Inspector UI/UX per handoff (fonts/colors from extracted Classical tokens). The one sanctioned
  addition: an "ungraded" (grade 0) filter chip and a spectrum selector when an object has
  multiple DJA spectra — data necessities, styled with existing tokens.
- Server binds 127.0.0.1:8321 by default (user port-forwards; override via env MINERVA_HOST/PORT).
- RAM discipline: FITS via memmap; h5 row slices only; bounded LRUs; parquet index loaded once.
- Disk cache under `/otdata2/themiya/minerva/data/cache/{spectra,cutouts}/…`; immutable objects
  get long Cache-Control headers.
- Graceful upstream degradation: S3/heroku failures → cached copy or 502 JSON, never a hung worker.
- Python: `/otdata2/themiya/grizli_rebels/bin/python3` (astropy 7.2, h5py, pandas 3.0, eazy-py 0.8.5,
  fastapi, uvicorn). No new heavyweight deps without checking they're installed.

## Module interfaces (imports must match exactly — modules are built in parallel)
- `server/config.py`: `settings` singleton; env-overridable: `MINERVA_INDEX_DIR`
  (default `/otdata2/themiya/minerva/data/viewer_index`), `MINERVA_CACHE_DIR`
  (default `/otdata2/themiya/minerva/data/cache`), `MINERVA_HOST` (127.0.0.1), `MINERVA_PORT` (8321),
  `MINERVA_FIELDS_DIR` (default `<app>/server/fields`).
- `server/fields.py`: `load_fields() -> dict[str, FieldConfig]`; `FieldConfig` dataclass:
  `name, title, catalog_path, eazy: dict[str, Path]  # template -> ZPiter dir`,
  `default_template, dja_catalog_path, cutout_filters: str, map_link: str`.
- `server/catalog.py`: `class CatalogStore(field: FieldConfig, index_dir: Path)` —
  `.query(**params) -> tuple[int, list[dict]]`, `.object(obj_id: int) -> dict`,
  `.inspect_list(template: str) -> list[dict]`, `.info() -> dict` (the /api/fields entry).
  Reads parquet written by scripts/build_index.py:
  `{index_dir}/{field}/catalog.parquet` (one row per object: id, ra, dec, mags, flags, morph,
  per-template z/lmass/... columns suffixed `_{template}`),
  `{index_dir}/{field}/spectra.parquet` (one row per matched DJA spectrum, /api/inspect/list schema
  + `zp_{template}` columns), `{index_dir}/{field}/meta.json` (counts, ranges, band lists).
- `server/eazy_products.py`: `class EazyStore(field: FieldConfig)` —
  `.get_products(template: str, obj_id: int) -> dict` (the /api/eazy payload).
  Lazy-opens h5 per template; caches zgrid, pivots, templates; bounded LRU on per-object results.
- `server/dja_spectra.py`: `def get_spectrum(root: str, file: str, cache_dir: Path) -> dict`
  (the /api/spectrum payload); raises `UpstreamError(status, detail)` defined in the same module.
- `server/cutouts.py`: `def get_cutout(mode: str, params: dict, cache_dir: Path) -> bytes` (PNG);
  same `UpstreamError` pattern (import from dja_spectra to avoid duplication).
- `server/decisions.py`: `def append_decisions(field, payload, cache_dir) -> int`;
  `def load_decisions(field, cache_dir) -> dict`.
- `server/main.py`: composes all of the above per API.md; `uvicorn server.main:app`.
  Run from `app/` dir: `cd app && /otdata2/themiya/grizli_rebels/bin/python3 -m uvicorn server.main:app --port 8321`.
