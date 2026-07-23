# build_map.py — MINERVA Field Map tiles (FitsMap)

Generates the pre-rendered Leaflet tile tree served at `/map` from the local
COSMOS mosaics. Output is static (`{layer}/{z}/{y}/{x}.png` image tiles,
`.pbf` marker tiles, `index.html`, `js/`, `css/`). Nothing here touches
`app/server/` or `app/web/`.

## Quick start

```bash
PY=/otdata2/themiya/grizli_rebels/bin/python3
cd /otdata2/themiya/minerva/minerva_viewer/app/scripts

# 1. crop smoke test (central 4096², writes data/map_tiles/cosmos_test/, verifies, prints estimate)
$PY build_map.py --field cosmos --test

# 2. full build (writes data/map_tiles/cosmos/) — hours, do on purpose
$PY build_map.py --field cosmos --procs-per-task 32
```

CLI: `--field cosmos`, `--test`, `--layers rgb,f444w,minerva,...` (subset),
`--crop-size N`, `--task-procs`, `--procs-per-task`, `--cat-limit N`,
`--out-dir PATH`, `--clean` / `--no-clean`.

## Layers

| layer | source | engine | notes |
|-------|--------|--------|-------|
| `rgb` | f444w/f277w/f115w | PIL (colour PNG) | Lupton asinh (`make_lupton_rgb`), default base layer |
| `f115w`…`f444w` | band FITS | matplotlib gray | per-band asinh `simple_norm`, `vmin=0`, `vmax=p99.5(+)`, `asinh_a=0.1` |
| `minerva` | `catalog.parquet` (use_phot) | markers | ellipse (a_image/b_image/theta_J2000); popup links `/?id=` |
| `spectra` | `spectra.parquet` | markers | popup links `/inspector/?field=cosmos&sel=<dja>` + `/?id=<mid>` |

## How it works / FitsMap facts

- Drives `fitsmap.convert.files_to_map(...)`. Layer names come from the input
  file basename, so the script stages short-named inputs in
  `<out>__staging/` (symlinks to the real FITS for the full build; small crop
  copies for `--test`) → clean layer names `rgb`, `f444w`, `minerva`, …
- The RGB composite is written **`np.flipud(make_lupton_rgb(...))`** — that
  orientation was verified to line up with the FITS layers and the ra/dec
  markers (luminance corr ≈ 0.99; un-flipped ≈ 0.0).
- Grayscale norms are computed once on the full frame and passed via
  `norm_kwargs` keyed by file basename, so a band looks identical at every zoom.
- Catalog `.cat` files are comma-delimited with a header (`id,ra,dec,a,b,theta,
  …,info`). The `info` column holds an HTML `<a>` (single-quoted attrs, no
  commas) which FitsMap renders as a real link in the popup.
- `cat_wcs_fits_file` is the staged `f444w.fits`; it also seeds
  `js/urlCoords.js` so `?ra=&dec=&zoom=` deep links pan correctly.
- Emitted tile URLs are **relative** (`rgb/{z}/{y}/{x}.png`) → the page works
  unchanged under a `/map/` prefix. (The HTML still pulls Leaflet/plugins from
  CDNs, needed at view time.)

## Resume

Resume is **per layer**: rerun the same command and any layer whose directory
already exists is skipped; `index.html` is regenerated over all requested
layers. To rebuild one layer, delete its dir under the out dir and rerun.
The full build keeps existing output by default (`--test` wipes by default;
override with `--clean` / `--no-clean`).

## Serving

Point a static mount at `data/map_tiles/cosmos/` under `/map`. Deep link from
explorer/inspector as `/map/?ra={ra}&dec={dec}&zoom=8`.
Note the large tile/inode count on the full build — plan filesystem accordingly.
