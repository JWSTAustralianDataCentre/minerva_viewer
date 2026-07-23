# DJA Redshift Inspector — handoff to Claude Code

`index.html` is a fully working, self-contained UI prototype (open it in any browser — no build, no server). All data in it is **synthetic**. The task is to replace the mock data layer with real DJA + MINERVA data while keeping the UI/UX exactly as designed.

## What the UI already does
- Linked triage: z_spec vs z_phot scatter + sortable/filterable object table; user-editable outlier threshold |Δz|/(1+z_spec) (default 0.15); grade 1/2/3 filters; search.
- Inspector: scroll-zoom/pan 1D prism spectrum with emission-line markers (modes: z_spec / z_phot / both / free z-slider), aligned 2D spectrum strip, eazy SED + photometry panel, p(z)+χ²(z) panel, image cutout with MSA crosshair + MINERVA-match circle, separation warning (>0.5″).
- Grading: keyboard 3/2/1 (good/unsure/bad), f flag, 0 clear, n/p nav, m marker-mode cycle, r reset zoom; comment + corrected-z per object; inspector initials; decisions persist to localStorage; CSV export.

## Data layer to build (replace the mock generators)
The mock lives in the logic class: `makeObjects()` (catalog), `spec()` (1D+err), `drawAll()` (2D + cutout), SED/p(z) synthesis in `renderVals()`. Everything downstream consumes plain arrays — swap the sources, keep the shapes.

### 1. DJA spectroscopic catalog
- `https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.csv.gz` (+ `.columns.csv` for metadata). Columns used: `ra, dec, z_best, grade, root, file, msamet`.
- Per-object spectra: `https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}` (`*.spec.fits`; 1D flux/err/wave + 2D SCI extension). Publicly served — test CORS; if it allows browser fetch, the whole app can stay a static page (ideal for community distribution). Otherwise add a thin FastAPI proxy.

### 2. MINERVA photo-z
- Cross-match catalog assumption: MINERVA `id`, `z_phot`, plus RA/Dec — but the app does its **own** nearest-neighbor match on RA/Dec (à la `minerva_viewer.py`: SkyCoord match, 0.5″ max sep) and surfaces `sep` per object so bad matches are diagnosable.
- eazy-py outputs: `.zout.fits` for z_phot/χ²; `.h5` (eazy-py HDF5) for p(z) grids, template SED + observed photometry per id. Field naming per repo: `MINERVA-{FIELD}_*_SUPER_*CATALOG_{template}.zout.fits`, template sets SFHZ / SFHZ_BLUE_AGN / LARSON / LARSON_MIRI. For a static app, pre-extract the needed per-object slices (p(z), template SED, photometry) to compact JSON with a small Python export script rather than reading HDF5 in the browser.

### 3. Cutouts
- Grizli cutout service, as in `minerva_viewer.py`:
  - RGB: `https://grizli-cutout.herokuapp.com/thumb?size=1.5&scl=2.0&asinh=True&filters=f115w-clear,f277w-clear,f444w-clear&...&coord={ra},{dec}`
  - Slit overlay: same service with `nirspec=True&metafile={msamet.split('_')[0]}`.

### 4. Wiring map (mock → real)
| Mock (logic class) | Real source |
|---|---|
| `makeObjects()` | DJA csv.gz × MINERVA zout, RA/Dec matched |
| `o.zs`, `o.grade` | `z_best`, `grade` (DJA) |
| `o.zp` | `z_phot` (eazy zout, chosen template set) |
| `spec(o)` → `{w,f,e}` | `*.spec.fits` 1D ext (μm, μJy) |
| 2D canvas | `*.spec.fits` 2D SCI, same wavelength WCS as 1D (keep the shared zoom) |
| `sedPts` / `sedTemplPath` | eazy h5: observed fnu + best template |
| `P(z)` / χ² | eazy h5 `lnp`/`chi2` grid |
| cutout canvas | grizli thumb (RGB + slit overlay) |
| CSV export | keep; add server-side session store if multi-user |

### 5. Priorities
1. CORS probe on the S3 + grizli endpoints → decide static vs proxy.
2. FITS in browser: use `fitsjs`-class reader or pre-convert; spectra are small (~100s KB).
3. Keep interaction latency low: prefetch next N objects in the filtered list (repo's zgui_panel.py does LRU + background prefetch — same idea).
4. Field switcher (COSMOS/UDS/EGS) just swaps catalog paths.

## Constraints
- Do not redesign the UI; it is the approved spec. Fonts/colors come from the embedded Classical tokens.
- Decisions format (CSV): `dja_id, minerva_id, ra, dec, z_spec, z_phot, grade_dja, dz_1pz, sep_arcsec, decision, flag, z_corrected, comment, inspector`.
