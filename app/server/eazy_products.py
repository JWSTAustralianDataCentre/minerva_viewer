"""
EAzY products service.

Reconstructs, per object and per template set, the payload served at
``GET /api/eazy/{field}/{template}/{id}`` (see app/API.md):

  * p(z)  -- computed as exp(-(chi2 - chi2_min)/2) over the 607-pt log zgrid,
             normalised to unit integral (APPLY_PRIOR=0 for these products, so
             no prior is applied), read via cheap h5py row slices.
  * chi2  -- the per-object chi2 vs redshift curve (h5 ``fit/chi2_fit`` row).
  * SED   -- the best-fit *observed-frame* template SED at z_ml, reconstructed
             from ``fit/coeffs_best`` and the template arrays, with IGM
             absorption applied (APPLY_IGM=1), scaled to microJy and trimmed to
             0.05-30 microns observed.
  * phot  -- the observed photometry the fit was performed against, in microJy.

Design / RAM discipline
-----------------------
* Shared arrays (zgrid, pivots, band names, tempfilt scale, zeropoints, the
  id->row index map, and the reconstructed template list) are read *once* per
  template set and cached on the instance.
* Per-object data (chi2 row, coeffs_best row, fnu/efnu/ok rows, zbest/zml) are
  read as single h5py row slices -- the big ``fit/chi2_fit`` array
  (294126 x 607) is never loaded whole.
* ``get_products`` is wrapped in an ``functools.lru_cache(maxsize=256)`` so the
  Inspector's prefetch pattern (next 3 / prev 1) re-serves without recomputing.

The reconstruction convention was verified offline against eazy-py 0.8.5
``PhotoZ.show_fit`` (``scripts/validate_eazy.py``): the observed-frame template
flux in microJy is

    fnu_uJy(lam_obs) = (coeffs_best . tempflux) * igm(z) * 10**(-0.4*(ABZP+48.6)) * 1e29

with ``tempflux[t] = templates[t].flux_fnu(z=z_ml) * tempfilt_scale[t]`` on the
wavelength grid of ``templates[0]`` and ``lam_obs = templates[0].wave*(1+z_ml)``.
For ABZP = PRIOR_ABZP = 28.9 the constant factor equals 1e-2, i.e. one catalog
flux unit (10 nJy) maps to 0.01 microJy, consistent with the photometry points
(cat/fnu x zeropoint x 0.01).
"""

from __future__ import annotations

import functools
import math
import threading
from pathlib import Path

import numpy as np
import h5py

# eazy is used only to rebuild the (small) template list and its IGM model;
# it is imported lazily inside _shared() so importing this module stays cheap.

# Observed-frame wavelength window for the served template SED (microns).
_LAM_MIN_UM = 0.05
_LAM_MAX_UM = 30.0
# Maximum number of points in the served template SED (payload-size guard).
_MAX_TEMPL_PTS = 1500
# eazy uses the "nearest" redshift index when evaluating redshift-dependent
# (SFH-z) templates -- see eazy.photoz.TEMPLATE_REDSHIFT_TYPE.
_REDSHIFT_TYPE = "nearest"


def _decode(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def _sig(x, n=4):
    """Round to ``n`` significant figures; non-finite -> None (JSON-safe)."""
    if x is None:
        return None
    xf = float(x)
    if not math.isfinite(xf):
        return None
    if xf == 0.0:
        return 0.0
    return round(xf, -int(math.floor(math.log10(abs(xf)))) + (n - 1))


def _sig_list(a, n=4):
    a = np.asarray(a, dtype=float)
    return [_sig(v, n) for v in a]


class EazyStore:
    """Per-field EAzY products, lazy-opened per template set.

    Parameters
    ----------
    field : FieldConfig
        Needs ``field.eazy`` -- a ``dict[str, Path]`` mapping template-set name
        (e.g. ``"sfhz_blue_agn"``, ``"larson"``) to the ZPiter directory that
        holds ``*.eazy.h5`` and ``*.eazy.data.fits``.
    """

    def __init__(self, field):
        self.field = field
        # template-set -> dict of shared, once-loaded arrays / objects
        self._shared_cache: dict[str, dict] = {}
        # Guards the expensive lazy build so two concurrent first-hits don't both
        # open the h5 / rebuild the template list (redundant work + 2x RAM spike).
        self._shared_lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def _zpiter_dir(self, template: str) -> Path:
        try:
            d = self.field.eazy[template]
        except Exception as exc:  # noqa: BLE001
            raise KeyError(f"unknown template set {template!r}") from exc
        return Path(d)

    def _h5_path(self, template: str) -> Path:
        # Prefer the field manifest's resolver (honours explicit per-template h5
        # paths for fields whose filenames don't match the COSMOS glob, e.g.
        # EGS '*.eazypy.h5'). Fall back to globbing the ZPiter dir for minimal
        # shim FieldConfigs (e.g. scripts/validate_eazy.py) that lack the method.
        getter = getattr(self.field, "h5_path", None)
        if callable(getter):
            return Path(getter(template))
        d = self._zpiter_dir(template)
        hits = sorted(d.glob("*.eazy.h5")) or sorted(d.glob("*.eazypy.h5"))
        if not hits:
            raise FileNotFoundError(f"no eazy h5 in {d}")
        return hits[0]

    def _data_fits_path(self, template: str) -> Path | None:
        # Explicit manifest 'data_fits' path wins; otherwise glob the ZPiter dir.
        # Returns None when absent (e.g. EGS ships no *.eazy.data.fits) -> the
        # zgrid is regenerated from the h5 fit attrs in _load_zgrid.
        explicit = getattr(self.field, "data_fits", None)
        if isinstance(explicit, dict) and template in explicit:
            return Path(explicit[template])
        d = self._zpiter_dir(template)
        hits = sorted(d.glob("*.eazy.data.fits"))
        return hits[0] if hits else None

    # --------------------------------------------------------------- shared
    def _shared(self, template: str) -> dict:
        """Load and cache the arrays/objects shared across all objects."""
        cached = self._shared_cache.get(template)
        if cached is not None:
            return cached

        # Double-checked locking: only one thread builds a given template set.
        with self._shared_lock:
            cached = self._shared_cache.get(template)
            if cached is not None:
                return cached
            return self._build_shared(template)

    def _build_shared(self, template: str) -> dict:
        h5path = self._h5_path(template)
        with h5py.File(h5path, "r") as f:
            ids = f["cat/id"][:].astype(np.int64)
            pivot_ang = f["cat/pivot"][:].astype(float)          # Angstrom
            zp = f["cat/zp"][:].astype(float)                    # zeropoint corr
            flux_columns = [_decode(c) for c in f["cat/flux_columns"][:]]
            tempfilt_scale = f["fit/tempfilt_scale"][:].astype(float)
            fit_attrs = {k: f["fit"].attrs[k] for k in f["fit"].attrs}
            prior_abzp = float(fit_attrs.get("PRIOR_ABZP", 28.9))

        # Band names: strip the catalog 'f_' flux-column prefix.
        bands = [c[2:] if c.startswith("f_") else c for c in flux_columns]

        # id -> row index (row-aligned with catalog & zout; built once).
        id_to_row = {int(v): i for i, v in enumerate(ids)}

        # 607-pt log redshift grid: authoritative copy is eazy.data.fits HDU
        # 'ZGRID'; fall back to regenerating the eazy log grid from attrs.
        zgrid = self._load_zgrid(template, fit_attrs)

        # Rebuild the (small) template list + IGM model exactly as the fit did.
        import eazy.hdf5  # lazy

        igm_kwargs = self._igm_kwargs(fit_attrs)
        templates = eazy.hdf5.templates_from_hdf5(
            str(h5path), igm_kwargs=igm_kwargs, verbose=False
        )

        # Catalog-unit -> microJy factor for the reconstructed template, matching
        # show_fit's `fnu_factor * flam_spec` for show_fnu=1 (fnu in microJy).
        uJy_factor = 10 ** (-0.4 * (prior_abzp + 48.6)) * 1e29

        shared = dict(
            h5path=h5path,
            id_to_row=id_to_row,
            n_obj=int(ids.shape[0]),
            zgrid=zgrid,
            n_z=int(zgrid.shape[0]),
            pivot_um=pivot_ang / 1e4,
            zp=zp,
            bands=bands,
            n_band=len(bands),
            tempfilt_scale=tempfilt_scale,
            templates=templates,
            n_temp=len(templates),
            uJy_factor=uJy_factor,
        )
        self._shared_cache[template] = shared
        return shared

    def _load_zgrid(self, template: str, fit_attrs: dict) -> np.ndarray:
        data_fits = self._data_fits_path(template)
        if data_fits is not None:
            try:
                from astropy.io import fits

                with fits.open(data_fits, memmap=False) as hd:
                    return np.asarray(hd["ZGRID"].data, dtype=float)
            except Exception:  # noqa: BLE001  -- fall back below
                pass
        # Fallback: regenerate the eazy log grid from the fit parameters.
        import eazy.utils

        zmin = float(fit_attrs.get("Z_MIN", 0.01))
        zmax = float(fit_attrs.get("Z_MAX", 20.0))
        zstep = float(fit_attrs.get("Z_STEP", 0.005))
        return np.asarray(eazy.utils.log_zgrid([zmin, zmax], zstep), dtype=float)

    @staticmethod
    def _igm_kwargs(fit_attrs: dict) -> dict:
        """IGM model keywords used by the fit (Asada24 defaults + attrs)."""
        true_vals = {"y", "yes", "true", "1", 1, True}
        add_cgm = fit_attrs.get("ADD_CGM", "y")
        add_cgm = (add_cgm.decode() if isinstance(add_cgm, bytes) else add_cgm)
        kwargs = {
            "scale_tau": float(fit_attrs.get("IGM_SCALE_TAU", 1.0)),
            "add_cgm": (str(add_cgm).lower() in {"y", "yes", "true", "1"}),
            "sigmoid_params": (
                float(fit_attrs.get("SIGMOID_PARAM1", 3.5918)),
                float(fit_attrs.get("SIGMOID_PARAM2", 1.8414)),
                float(fit_attrs.get("SIGMOID_PARAM3", 18.001)),
            ),
        }
        return kwargs

    # ------------------------------------------------------------- per object
    def get_products(self, template: str, obj_id: int) -> dict:
        """Return the /api/eazy payload dict for one object (LRU-cached)."""
        return self._get_products_cached(str(template), int(obj_id))

    @functools.lru_cache(maxsize=256)  # noqa: B019 -- intentional per-instance LRU
    def _get_products_cached(self, template: str, obj_id: int) -> dict:
        shared = self._shared(template)
        row = shared["id_to_row"].get(int(obj_id))
        if row is None:
            raise KeyError(f"object {obj_id} not found in template set {template!r}")

        # --- cheap per-object h5py row slices (never load full arrays) -------
        with h5py.File(shared["h5path"], "r") as f:
            chi2 = f["fit/chi2_fit"][row, :].astype(np.float64)      # (607,)
            coeffs_best = f["fit/coeffs_best"][row, :].astype(np.float64)  # (15,)
            fnu = f["cat/fnu"][row, :].astype(np.float64)           # (32,)
            efnu = f["cat/efnu_orig"][row, :].astype(np.float64)    # (32,)
            ok = f["cat/ok_data"][row, :].astype(bool)             # (32,)
            z_ml = float(f["fit/zml"][row])
            z_phot = float(f["fit/zbest"][row])

        zgrid = shared["zgrid"]

        # --- p(z): exp(-(chi2 - chi2_min)/2), trapz-normalised (no prior) ----
        finite = np.isfinite(chi2)
        chi2_min = float(np.min(chi2[finite])) if finite.any() else float("nan")
        lnp = np.full_like(chi2, -np.inf)
        lnp[finite] = -0.5 * (chi2[finite] - chi2_min)
        pz = np.exp(lnp)
        norm = np.trapz(pz, zgrid)
        if norm > 0 and math.isfinite(norm):
            pz = pz / norm

        # --- photometry (microJy), zeropoint-corrected as the fit saw it -----
        zp = shared["zp"]
        fnu_uJy = fnu * zp * 0.01
        efnu_uJy = efnu * zp * 0.01

        # --- best-fit observed-frame template SED at z_ml -------------------
        templ_lam_um, templ_fnu_uJy = self._reconstruct_sed(
            shared, coeffs_best, z_ml
        )

        payload = {
            "id": int(obj_id),
            "z_phot": _sig(z_phot, 6) if math.isfinite(z_phot) else None,
            "z_ml": _sig(z_ml, 6) if math.isfinite(z_ml) else None,
            "chi2_min": _sig(chi2_min, 5),
            "zgrid": _sig_list(zgrid, 6),
            "chi2": _sig_list(chi2, 5),
            "pz": _sig_list(pz, 5),
            "phot": {
                "band": list(shared["bands"]),
                "lam_um": _sig_list(shared["pivot_um"], 6),
                "fnu_uJy": _sig_list(fnu_uJy, 4),
                "efnu_uJy": _sig_list(efnu_uJy, 4),
                "ok": [bool(v) for v in ok],
            },
            "templ": {
                "lam_um": _sig_list(templ_lam_um, 5),
                "fnu_uJy": _sig_list(templ_fnu_uJy, 4),
            },
            "templ_z": _sig(z_ml, 6) if math.isfinite(z_ml) else None,
        }
        return payload

    # ---------------------------------------------------------- reconstruction
    def _reconstruct_sed(self, shared: dict, coeffs_best: np.ndarray, z: float):
        """Observed-frame best-fit SED -> (lam_um, fnu_uJy), trimmed & resampled.

        Mirrors ``eazy.photoz.PhotoZ.show_fit`` (get_spec=True, show_fnu=1):
        each template is evaluated in f-nu at ``z`` (nearest redshift index for
        the SFH-z grids), scaled by ``tempfilt_scale``, combined with
        ``coeffs_best``, multiplied by the IGM transmission, and converted to
        microJy.
        """
        templates = shared["templates"]
        scale = shared["tempfilt_scale"]
        n_temp = shared["n_temp"]

        if not math.isfinite(z) or z < 0:
            return np.asarray([]), np.asarray([])

        base = templates[0]
        base_wave = base.wave  # rest-frame Angstrom
        tempflux = np.zeros((n_temp, base_wave.shape[0]), dtype=np.float64)
        for t in range(n_temp):
            fnu_t = templates[t].flux_fnu(z=z, redshift_type=_REDSHIFT_TYPE)
            fnu_t = np.asarray(fnu_t, dtype=np.float64) * scale[t]
            if fnu_t.shape[0] == base_wave.shape[0]:
                tempflux[t, :] = fnu_t
            else:
                tempflux[t, :] = np.interp(base_wave, templates[t].wave, fnu_t)

        igmz = np.asarray(base.igm_absorption(z=z), dtype=np.float64)
        templ_fnu = np.dot(coeffs_best, tempflux) * igmz          # catalog units
        templ_fnu_uJy = templ_fnu * shared["uJy_factor"]         # microJy
        lam_obs_um = base_wave * (1.0 + z) / 1e4                  # microns

        # Trim to the observed window.
        m = (lam_obs_um >= _LAM_MIN_UM) & (lam_obs_um <= _LAM_MAX_UM)
        lam = lam_obs_um[m]
        flx = templ_fnu_uJy[m]

        # Ensure ascending wavelength for interp/serialisation.
        order = np.argsort(lam)
        lam = lam[order]
        flx = flx[order]

        # Resample to <= _MAX_TEMPL_PTS points (log-spaced) for payload size.
        if lam.shape[0] > _MAX_TEMPL_PTS and lam.shape[0] > 1:
            grid = np.logspace(
                np.log10(lam[0]), np.log10(lam[-1]), _MAX_TEMPL_PTS
            )
            flx = np.interp(grid, lam, flx)
            lam = grid

        return lam, flx
