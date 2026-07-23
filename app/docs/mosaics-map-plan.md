# Field Map module plan (deferred; probe findings 2026-07-18)

## MINERVA COSMOS mosaics on S3 (bucket grizli-v2, public GetObject, NO ListBucket)
- Prefix: `MINERVA/mosaics/cosmos/40mas-v{1.0,2.0,3.0}/` — **v3.0 is latest (2026-06-04)**; 40mas is
  the only pixel scale. UDS: `uds/40mas-v2.3`; EGS: `egs/40mas-v2.0` (fresh, 2026-07-12); GOODS-N not yet.
- Naming: `cosmos-grizli-v8.0-minerva-v2.0-40mas-{filter}[-clear]_drc_{sci|wht|exp|var}.fits.gz` + `_wcs.csv`
  (v3.0 presumably `minerva-v3.0-...` — verify exact names before download).
- 25 bands with sci mosaics: 19 NIRCam (F070W…F480M) + 6 HST (F336WU, F435W, F475W, F606W, F814W, F850LP).
  **F162M missing on S3** (tiled on colorado.edu but no FITS found); COSMOS MIRI not on S3.
- Sizes: Σ sci = 35.2 GB (25 files; per-band sizes in cosmos_v2_manifest.txt); Σ wht = 33.1 GB;
  everything ≈ 102 GB. Disk free on /otdata2: ~490 GB.

## minerva.colorado.edu viewer
- fitsmap v0.11.1 (Leaflet; tiles `{layer}/{z}/{y}/{x}.png`, catalog markers as `.pbf` vector tiles;
  maxNativeZoom 8). `/cosmos/` responded fully at HTTP level during probe (all layers/tiles 200) —
  user-reported breakage likely a preview-build or client-side issue, not missing tiles.
- COSMOS layers there: rgb, 3 medium-band composites, segmentation, 19 NIRCam, 6 MIRI, 6 HST,
  + marker layers minerva / sed_plots / spectra.

## Plan (when activated)
1. Download v3.0 `_drc_sci.fits.gz` for a starter set: F115W, F277W, F444W (RGB), + F150W, F200W
   (~13 GB), decompress to `data/images/cosmos/`.
2. `pip install fitsmap` (pure Python, ray-parallel — 32 cores ideal). Generate: `rgb` composite
   layer + per-band grayscale layers + a `minerva` catalog marker layer from catalog.parquet
   (id/ra/dec/a/b/theta + mag_f444w; marker popup links `/inspector?sel=` and explorer `?id=`).
3. Serve output dir statically at `/map` from the existing FastAPI app; deep links
   `/map/?ra={ra}&dec={dec}&zoom=8` from explorer + inspector (replace external colorado.edu links).
4. Extend to all 25 bands + segmentation later if useful; watch inode count (~10⁵–10⁶ tiles).
CANFAR is an alternate download source if S3 throttles (user has access).
