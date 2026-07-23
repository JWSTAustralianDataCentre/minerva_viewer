# MINERVA Viewer v2 — API contract (v1)

All endpoints return JSON unless noted. Errors: `{"detail": str}` with 4xx/5xx.
Floats rounded to sensible precision (coords 6dp, z 4dp, fluxes 4 sig figs).
`field` defaults to `cosmos`; `template` defaults to `sfhz_blue_agn` (other: `larson`).

## Catalog / Explorer

### GET /api/fields
```
{ "fields": [ { "name":"cosmos", "title":"COSMOS", "n_objects":294126,
    "templates":["sfhz_blue_agn","larson"], "default_template":"sfhz_blue_agn",
    "n_spec_matched": <int>, "bands":[...32 band names...],
    "ra_range":[min,max], "dec_range":[min,max] } ] }
```

### GET /api/catalog/query
Params (all optional): `field`, `template`, `ids` (comma-sep MINERVA ids),
`ra`,`dec`,`radius_arcsec` (cone), `z_min`,`z_max` (on z_phot),
`lmass_min`,`lmass_max` (log10 M*), `mag_min`,`mag_max` (AB, band `mag_band` default f444w),
`has_spec` (0/1), `grade_min` (on matched spec grade), `use_phot` (default 1; 0=no filter),
`no_star` (default 1), `uvj` (`q`|`sf`), `sort` (`id|ra|dec|z_phot|lmass|mag|sep|dist`, prefix `-` desc),
`limit` (default 500, max 5000), `offset`.
```
{ "total": <int matching before limit>, "rows": [ {
  "id":int, "ra":float, "dec":float, "z_phot":float, "z160":float, "z840":float,
  "chi2":float, "lmass":float, "lsfr":float, "mag":float, "flux_radius":float,
  "n_bands":int, "uvj":int, "u_v":float, "v_j":float,
  "dist_arcsec":float,   # cone queries ONLY: distance (2dp) from the cone center
  "spec": null | {"dja":str, "zs":float, "grade":int, "sep":float, "grating":str}
} ] }
```
`lmass`/`lsfr` are log10; `mag` is AB in `mag_band`; `uvj` = uvj_class as int.
`dist_arcsec` is present on a row **only when the query supplied `ra`+`dec`+`radius_arcsec`**
(the cos-dec-scaled angular distance from the cone center, ascending default); non-cone
queries omit the field. `sort=dist` orders by it (ascending; falls back to `id` with no cone).
`spec` = primary matched spectrum (best grade, then prism first, then highest exptime).

### GET /api/object/{field}/{id}
Full single-object record:
```
{ "id":…, "ra":…, "dec":…, "flags":{use_phot,flag_star,flag_clean,…}, "n_bands":…,
  "morph":{flux_radius,a_image,b_image,theta,kron_radius},
  "phot":[ {"band":"f444w","lam_um":4.421,"fnu_uJy":…,"efnu_uJy":…,"mag":…,"ok":bool}, … ],
  "zout":{ "sfhz_blue_agn":{z_phot,z_ml,chi2,z025,z160,z500,z840,z975,lmass,lsfr,Av,u_v,v_j,uvj},
           "larson":{…} },
  "spectra":[ {spectrum row as /api/inspect/list} ] }
```

## EAzY products (Inspector SED + p(z) panels)

### GET /api/eazy/{field}/{template}/{id}
```
{ "id":…, "z_phot":float, "z_ml":float, "chi2_min":float,
  "zgrid":[607], "chi2":[607], "pz":[607],          # pz normalized ∫pz dz = 1
  "phot":{ "band":[str]*32, "lam_um":[32], "fnu_uJy":[32], "efnu_uJy":[32], "ok":[bool]*32 },
  "templ":{ "lam_um":[N], "fnu_uJy":[N] },          # best-fit template SED at z_ml, IGM applied,
                                                    # trimmed to 0.05–30 µm observed
  "templ_z": float }                                 # redshift the template was evaluated at
```
Units: catalog fnu ×0.01 → µJy (catalog unit is 10 nJy, ABZP 28.9). Template scaled consistently
(validated offline against eazy-py `show_fit` — see scripts/validate_eazy.py).

## Spectra (Inspector)

### GET /api/inspect/list?field=&template=
Precomputed at index build. One row per matched DJA **spectrum** (not per galaxy), shaped for the
prototype's object records:
```
{ "objects": [ {
  "dja":str,        # file basename minus .spec.fits — UNIQUE key (o.dja)
  "mid":int,        # MINERVA id (o.mid)
  "ra":float,"dec":float,   # MINERVA position
  "dja_ra":float,"dja_dec":float,  # DJA spectrum position (cutout centers here; MINERVA circle offset)
  "sep":float,      # match separation arcsec
  "zs":float,       # DJA z_best
  "zp":float,       # MINERVA z_phot for chosen template
  "grade":int,      # 3/2/1, 0 = ungraded (NaN in DJA csv)
  "mag":float,      # f444w AB
  "sn":float,       # DJA sn50
  "root":str,"file":str,"srcid":int,"pid":int,
  "grating":str,    # e.g. prism-clear, g395m-f290lp
  "metafile":str,   # msamet.split('_')[0], for slit overlay; may be ""
  "nspec":int,      # how many spectra this mid has (for the spectrum selector)
  "exptime":int,    # effective exposure (s); null if absent
  # DJA reviewer verdict + per-disperser evidence (null when ungraded / supplement):
  "reviewer":str|null, "comment":str|null,   # reviewer initials + free-text note
  "zgrade":float|null,                        # reviewer's redshift at grade time
  "z_prism":float|null, "z_grating":float|null,  # per-disperser z (−1 sentinel → null)
  "sn_line":float|null                        # strongest-line S/N
} ] }
```

### GET /api/spectrum/{root}/{file}
Fetches (S3 → disk cache) and parses spec.fits:
```
{ "meta":{root,file,grating,pid,srcid,exptime}, 
  "wave_um":[N], "flux_uJy":[N], "err_uJy":[N], "sky_uJy":[N],
  "twod":{ "ny":int, "nx":int, "data":"<base64 little-endian float32, row-major, ny rows>",
           "wave_um":[nx] } }    # SCI HDU; nx aligns with the 1D wave grid
```
Non-finite → null in 1D arrays; NaN allowed inside the b64 2D payload.

## Cutouts

### GET /api/cutout  → image/png
Params: `ra`,`dec` (required), `size` (arcsec, default 1.5), `mode`:
- `rgb` (default): filters=f115w-clear,f277w-clear,f444w-clear, scl=2.0, asinh, rgb_scl=1.5,0.74,1.3, pl=2
- `slit`: adds nirspec=True&metafile={metafile}&dpi_scale=6&nrs_lw=0.5&nrs_alpha=0.8, invert, f444w grey
- `grid`: all_filters=True with `filters` param (default the field's 15-band list), used by Explorer strips
Extra passthrough params: `filters`, `scl`, `metafile`, `invert`.
Proxied to grizli-cutout.herokuapp.com/thumb, disk-cached by URL hash, 30s timeout,
`Cache-Control: public, max-age=31536000` on success.

### GET /api/dja/preview/{root}/{basename}  → image/png  (proxied .fnu.png/.flam.png fallback)

## QC decisions

### POST /api/decisions   body: {"field":str,"inspector":str,"decisions":{dja_id:{v,flag,zNew,comment,by,t}}}
Appends one JSONL line per call to `data/cache/decisions/{field}.jsonl`; returns {"ok":true,"n":int}.
### GET /api/decisions?field=  → {"decisions":{dja_id:{…latest…}}, "n":int}
Merged by dja_id, latest t wins. UI still keeps localStorage; server store is sync/backup.

## Static
- `/` → web/explorer/, `/inspector` → web/inspector/, `/shared/*`, `/vendor/*` (react, runtime, fonts).
- `/map/{field}/` → pre-generated fitsmap tile tree (data/map_tiles/{field}, built by
  scripts/build_map.py; mounted only when tiles exist). Deep links: `/map/cosmos/?ra=&dec=&zoom=`.
  Field manifests' `map_link` points here; layers: rgb + f115w/f150w/f200w/f277w/f444w grayscale +
  `minerva` (use_phot ellipses, popup links to explorer/inspector) + `spectra` marker layers.
