# MINERVA Viewer

Interactive quality-assessment suite for **JWST MINERVA survey** photometric redshifts,
built around spectroscopic redshifts from the public
[DAWN JWST Archive (DJA)](https://dawn-cph.github.io/dja/). One lightweight FastAPI server,
three linked browser surfaces, no frontend build step.

**MINERVA (Medium-band Imaging of NIRCam for Extragalactic Research and Vision Advantage)**
is a JWST NIRCam medium-band imaging survey over the Hubble legacy fields, delivering
high-precision photometric redshifts and galaxy characterization —
see [Muzzin et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250719706M).

| Surface | URL | Purpose |
|---|---|---|
| **Catalog Explorer** | `/` | Query the photometric catalog (ID / cone / z / mass / mag / flags), browse multiband cutouts, SED + p(z) quicklooks |
| **Redshift Inspector** | `/inspector/` | Grade spec-z × photo-z agreement object by object: 1D+2D NIRSpec spectra with line markers, EAzY SED and p(z)·χ², MSA slit cutouts, keyboard-first QC workflow |
| **Field Map** | `/map/{field}/` | Pan/zoom the full mosaics (pre-generated [fitsmap](https://github.com/ryanhausen/fitsmap) tiles) with catalog + spectra overlays |

Fields currently wired: **COSMOS** (294,126 objects) and **EGS** (520,875 objects), with
19,000+ matched DJA spectra from 45+ programs (CAPERS, RUBIES, CEERS, GTO-wide, BlueJay, …).

## Quick start

```bash
cd app && bash run.sh          # serves http://127.0.0.1:8321 (localhost only)
```

Requires the lab Python env (astropy ≥7, h5py, pandas ≥2, pyarrow, eazy-py, fastapi, uvicorn —
see `app/requirements.txt`) and the MINERVA data tree (below). First start warms the EAzY
stores (~30 s); wait for `/healthz` → 200.

**Share with collaborators** (Basic-auth gate + ngrok tunnel):

```bash
cd app && bash run_public.sh   # prints the public URL; credentials in app/.secrets/public_auth
```

Never expose the server without `MINERVA_AUTH_USER`/`MINERVA_AUTH_PASS` set (`run_public.sh`
handles this). QC decisions sync server-side per inspector initials, so several people can
grade in parallel and `Export decisions` captures everyone's work.

## Inspector workflow (the daily driver)

Keyboard: `3`/`2`/`1` grade good/unsure/bad (auto-advance), `f` flag, `0` clear, `n`/`p` navigate,
`m` cycle line-marker mode (z_spec / z_phot / both / free-slider), `r` reset zoom, `?` cheat-sheet.
Mouse: scroll-zoom + drag-pan on the 1D spectrum (2D strip follows) and on the p(z) panel;
drag on the cutout = ds9-style bias/contrast, wheel = cutout size.
Spectra display in fλ (default) or fν; sky spectrum overlay toggleable; the DJA reviewer's own
grade/comment/per-disperser redshifts are shown alongside; star/quality flags from the MINERVA
catalog warn about brown-dwarf-type interlopers. Decisions persist to localStorage **and** the
server; CSV export follows the `dja_id,…,inspector` schema in `app/API.md`.

## Data layout (server side)

The app never modifies survey data; it reads a `data/` tree (paths set in
`app/server/fields/{field}.json` and env vars, defaults in `app/server/config.py`):

```
data/catalogs/{field}/…SUPER_CATALOG.fits            photometric catalog
data/catalogs/{field}/EAzY/{TEMPLATE}/SUPER/ZPiter/  eazy-py .zout.fits + .h5 per template set
data/catalogs/dja/dja_msaexp_emission_lines_v4.5.csv.gz   DJA release catalog
data/viewer_index/{field}/                           parquet indices (built, see below)
data/cache/{spectra,cutouts,previews,decisions}/     disk caches + QC decision log
data/map_tiles/{field}/                              fitsmap tiles (optional)
```

Spectra are fetched on demand from DJA S3 (`msaexp-nirspec` bucket) and cached; cutouts proxy
the grizli cutout service. Both lack CORS — that is why the server proxies them.

## Operations

```bash
# Rebuild a field's index (after a catalog or DJA release update):
python3 app/scripts/build_index.py --field cosmos

# Pull public DJA spectra newer than the CSV release (live nirspec_extractions API):
python3 app/scripts/dja_supplement.py --field cosmos     # then restart the server

# Verify (run after any change):
python3 app/scripts/smoke_api.py        # every endpoint, real data, must be all-PASS
python3 app/scripts/validate_eazy.py    # SED/p(z) reconstruction vs eazy-py ground truth

# Rebuild the field map (uses local mosaics; ~1h, ~6GB tiles):
python3 app/scripts/build_map.py --field cosmos --procs-per-task 32

# Reassemble the Inspector page after editing its sources (app.js / template.html):
python3 app/scripts/debundle.py
```

### Adding a field
1. Drop the catalog + EAzY products into `data/catalogs/{field}/` (COSMOS layout).
2. Write `app/server/fields/{field}.json` (copy `egs.json`; explicit `h5`/`zout` paths override
   filename globbing when naming conventions differ).
3. `build_index.py --field {field}` (verifies catalog/zout/h5 row alignment), optionally
   `dja_supplement.py`, restart. The UI picks the field up automatically.

## Architecture, contracts, provenance

- `app/ARCHITECTURE.md` — design, data facts, module interfaces. **Read first.**
- `app/API.md` — the HTTP contract (endpoints, shapes, units). Frontends code against this only.
- `app/docs/` — verified schema/domain reports (catalog & h5 layouts, DJA S3 facts, mosaic
  manifest, prototype anatomy) written during the 2026-07 build.
- `handoff/` — the frozen, approved UI design (self-contained prototype). The Inspector's visual
  design follows it exactly; `app/web/_bundle_src/` holds its decoded sources and
  `debundle.py` assembles the production page from them (including the documented
  `MINERVA-SVG-TEXT-FIX` runtime patch and content-hashed asset URLs).

### Notable implementation facts
- Catalog fluxes are 10 nJy units (×0.01 → µJy; AB zp 23.9); all displayed line wavelengths
  are **vacuum**; p(z) is recomputed from the h5 χ²(z) grids (no prior), validated to ≲0.01%
  against `eazy-py show_fit`.
- DJA `z_best = -1` and negative eazy `z_phot` are "no measurement" sentinels → served as null.
- Cross-match: nearest-neighbour ≤0.5″, one row per **spectrum**, separation always displayed.
- The legacy v1 tools (Panel GUI, batch plots) were removed 2026-07-23; see git history.

## License

MIT (see `LICENSE`). MINERVA collaboration / JWST Australian Data Centre.
Data: DAWN JWST Archive (DJA) is maintained by the Cosmic Dawn Center; please follow DJA
citation guidance when publishing results derived from its spectra.
