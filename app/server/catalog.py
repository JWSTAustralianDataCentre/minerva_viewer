"""Parquet-backed catalog store + query engine for the MINERVA viewer.

Reads the three per-field index artifacts written by scripts/build_index.py:

  {index_dir}/{field}/catalog.parquet  -- one row per MINERVA object
  {index_dir}/{field}/spectra.parquet  -- one row per matched DJA spectrum
  {index_dir}/{field}/meta.json        -- counts, ranges, band list, templates

The parquet column conventions this module codes against (build_index must match):

catalog.parquet
  id (int64), ra, dec (float64, deg)
  flux_radius, a_image, b_image, theta, kron_radius (morphology)
  use_phot, flag_star, flag_clean (bool flags)
  n_bands (int)
  mag_{band}   AB magnitude per band (from f_{band}); NaN when unmeasured
  snr_{band}   S/N per band (f_{band}/e_{band}); <=0 or NaN when unmeasured
  per template T (e.g. sfhz_blue_agn, larson), suffixed "_{T}":
    z_phot_{T}, z_ml_{T}, chi2_{T},
    z025_{T}, z160_{T}, z500_{T}, z840_{T}, z975_{T},
    lmass_{T} (log10 M*), lsfr_{T} (log10 SFR),
    Av_{T}, u_v_{T}, v_j_{T}, uvj_{T} (uvj_class as int)

spectra.parquet (one row per matched DJA spectrum; /api/inspect/list schema)
  dja, mid (int MINERVA id), ra, dec, dja_ra, dja_dec, sep, zs,
  zp_{T} (MINERVA z_phot for template T), grade (int, 0=ungraded),
  mag, sn, root, file, srcid, pid, grating, metafile, nspec, exptime (optional)

Cone search uses simple deg math with cos(dec) scaling, adequate at arcsec scales.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pivot wavelengths (micron) for the 32 COSMOS SUPER bands.
# These are PIVOT WAVELENGTHS in micron, taken from the eazy filter set
# (h5 cat/pivot, Angstrom -> um), i.e. the effective bandpass pivot used for
# plotting the observed SED. Keyed by band name (no "f_" prefix).
# ---------------------------------------------------------------------------
PIVOT_UM: dict[str, float] = {
    "f275wu": 0.2708, "f336wu": 0.3354, "f435w": 0.4319, "f475w": 0.4747,
    "f606w": 0.5921, "f814w": 0.8057, "f850lp": 0.9033, "f098m": 0.9867,
    "f105w": 1.0544, "f125w": 1.2471, "f140w": 1.3924, "f160w": 1.5397,
    "f070w": 0.7043, "f090w": 0.9023, "f115w": 1.1543, "f140m": 1.4054,
    "f150w": 1.5007, "f162m": 1.6272, "f182m": 1.8452, "f200w": 1.9886,
    "f210m": 2.0955, "f250m": 2.5037, "f277w": 2.7623, "f300m": 2.9895,
    "f335m": 3.3623, "f356w": 3.5682, "f360m": 3.6243, "f410m": 4.0821,
    "f430m": 4.2813, "f444w": 4.4037, "f460m": 4.6302, "f480m": 4.8156,
}

SORT_WHITELIST = {"id", "ra", "dec", "z_phot", "lmass", "mag", "sep", "dist"}
_DEFAULT_MAG_BAND = "f444w"


# ---------------------------------------------------------------------------
# rounding helpers (API precision: coords 6dp, z 4dp, fluxes 4 sig figs)
# ---------------------------------------------------------------------------
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _rnd(x, ndigits):
    """Round to fixed decimals; non-finite -> None (JSON null)."""
    if not _finite(x):
        return None
    return round(float(x), ndigits)


def round_sig(x, sig=4):
    """Round to `sig` significant figures; non-finite -> None."""
    if not _finite(x):
        return None
    v = float(x)
    if v == 0.0:
        return 0.0
    d = sig - int(math.floor(math.log10(abs(v)))) - 1
    return round(v, d)


def _int_or_none(x):
    if not _finite(x):
        return None
    return int(round(float(x)))


class CatalogStore:
    """Query engine over one field's parquet index."""

    def __init__(self, field, index_dir):
        self.field = field
        self.name = getattr(field, "name", str(field))
        base = Path(index_dir) / self.name

        self.catalog = pd.read_parquet(base / "catalog.parquet")
        self.spectra = pd.read_parquet(base / "spectra.parquet")
        self._merge_supplement(base / "spectra_supplement.parquet")
        # DJA uses z_best = -1 as a "no measured redshift" sentinel, and eazy
        # z_phot can be negative for unfittable objects. Serve both as missing
        # (null) so no client ever renders a -1 redshift or an Infinity dz.
        zs = pd.to_numeric(self.spectra["zs"], errors="coerce")
        self.spectra["zs"] = zs.where(zs >= 0)
        for c in self.spectra.columns:
            if c.startswith("zp_"):
                v = pd.to_numeric(self.spectra[c], errors="coerce")
                self.spectra[c] = v.where(v >= 0)
        # DJA per-disperser / graded redshifts use -1 as "no measurement"; serve
        # those (and any negative) as missing so no client renders a -1 redshift.
        for c in ("z_prism", "z_grating", "zgrade"):
            if c in self.spectra.columns:
                v = pd.to_numeric(self.spectra[c], errors="coerce")
                self.spectra[c] = v.where(v >= 0)
        with open(base / "meta.json") as fh:
            self.meta = json.load(fh)

        self.bands: list[str] = list(self.meta.get("bands", []))
        # Per-field pivot wavelengths (um) written by build_index from the h5
        # cat/pivot. Used as a fallback for bands absent from the hardcoded
        # COSMOS PIVOT_UM table (e.g. EGS f470n); COSMOS bands all resolve via
        # PIVOT_UM first, so its output is unchanged.
        self._pivots_meta: dict[str, float] = dict(self.meta.get("pivots", {}) or {})
        # template list: prefer field.eazy keys, fall back to meta
        try:
            self.templates = list(field.eazy.keys())
        except AttributeError:
            self.templates = list(self.meta.get("templates", []))
        self.default_template = getattr(
            field, "default_template",
            self.meta.get("default_template",
                          self.templates[0] if self.templates else None),
        )

        # fast id lookup
        self._id_arr = self.catalog["id"].to_numpy()
        self._id_index = pd.Index(self._id_arr)

        # Numeric-column cache: the query hot path filters/sorts on the same
        # columns on every request; converting them to float numpy arrays once
        # (parquet columns are already numeric, so this is just a stable view)
        # avoids re-running pd.to_numeric over the full ~294k-row frame per call.
        self._num_cache: dict[str, np.ndarray] = {}

        # Boolean flag arrays (fillna(False)) precomputed once for the flag gate.
        self._use_phot_bool = (
            self.catalog["use_phot"].fillna(False).to_numpy().astype(bool)
            if "use_phot" in self.catalog.columns else None)
        self._flag_star_bool = (
            self.catalog["flag_star"].fillna(False).to_numpy().astype(bool)
            if "flag_star" in self.catalog.columns else None)

        # Memoized /api/inspect/list result per template (spectra.parquet is
        # static for the process lifetime, so the sorted rows are deterministic).
        self._inspect_cache: dict[str, list] = {}

        # per-mid spectrum aggregates for query joins (also builds row-aligned
        # arrays for the has_spec / grade_min / sort-by-sep joins).
        self._build_spec_aggregates()

    def _merge_supplement(self, path):
        """Fold scripts/dja_supplement.py output (spectra newer than the DJA CSV
        release, found via the live nirspec_extractions API) into the served
        spectra table. Supplement rows never displace CSV-built rows; nspec and
        the one-primary-per-mid election are recomputed over the union. Any
        problem skips the supplement rather than breaking startup."""
        if not Path(path).exists():
            return
        try:
            sup = pd.read_parquet(path)
            # Column-align to the served table: missing columns (e.g. the newer
            # DJA reviewer fields the supplement builder predates) fill with NaN,
            # extra columns drop. Supplement spectra carry no CSV review anyway.
            sup = sup.reindex(columns=self.spectra.columns)
            sup = sup[~sup["dja"].isin(set(self.spectra["dja"]))]
            if not len(sup):
                return
            comb = pd.concat([self.spectra, sup], ignore_index=True)
            comb["nspec"] = comb.groupby("mid")["dja"].transform("size") \
                                .astype(self.spectra["nspec"].dtype)
            is_prism = comb["grating"].astype(str).str.startswith("prism")
            elect = comb.assign(_g=-pd.to_numeric(comb["grade"], errors="coerce").fillna(0),
                                _p=~is_prism,
                                _e=-pd.to_numeric(comb["exptime"], errors="coerce").fillna(0))
            first = elect.sort_values(["mid", "_g", "_p", "_e"],
                                      kind="mergesort").groupby("mid").head(1).index
            comb["primary"] = False
            comb.loc[first, "primary"] = True
            self.spectra = comb
            print(f"[catalog] {self.name}: merged {len(sup)} supplement spectra "
                  f"(total {len(comb)})")
        except Exception as exc:  # never break startup on a bad supplement file
            print(f"[catalog] {self.name}: supplement skipped: {exc!r}")

    # -- spectrum aggregates -------------------------------------------------
    def _build_spec_aggregates(self):
        """Precompute, per MINERVA id: primary spectrum + best grade + min sep."""
        self._primary: dict[int, dict] = {}
        self._best_grade: dict[int, int] = {}
        self._min_sep: dict[int, float] = {}
        sp = self.spectra
        if sp.empty or "mid" not in sp.columns:
            # No spectra: still build row-aligned arrays so query joins vectorize.
            self._best_grade_arr = np.full(len(self.catalog), -1, dtype=np.int64)
            self._has_primary_arr = np.zeros(len(self.catalog), dtype=bool)
            self._min_sep_arr = np.full(len(self.catalog), math.inf)
            return
        has_exptime = "exptime" in sp.columns
        for mid, grp in sp.groupby("mid"):
            grades = pd.to_numeric(grp.get("grade"), errors="coerce").fillna(0)
            self._best_grade[int(mid)] = int(grades.max())
            seps = pd.to_numeric(grp.get("sep"), errors="coerce")
            if seps.notna().any():
                self._min_sep[int(mid)] = float(seps.min())
            # primary: best grade, then prism first, then highest exptime (or sn)
            grating = grp.get("grating", pd.Series([""] * len(grp))).fillna("")
            is_prism = grating.str.contains("prism", case=False, na=False)
            tiebreak = (pd.to_numeric(grp["exptime"], errors="coerce")
                        if has_exptime else
                        pd.to_numeric(grp.get("sn"), errors="coerce")).fillna(-1)
            order = pd.DataFrame({
                "grade": grades.to_numpy(),
                "prism": is_prism.to_numpy().astype(int),
                "tie": tiebreak.to_numpy(),
            }, index=grp.index)
            best_idx = order.sort_values(
                ["grade", "prism", "tie"], ascending=[False, False, False]
            ).index[0]
            row = grp.loc[best_idx]
            self._primary[int(mid)] = {
                "dja": row.get("dja"),
                "zs": _rnd(row.get("zs"), 4),
                "grade": _int_or_none(row.get("grade")) or 0,
                "sep": _rnd(row.get("sep"), 4),
                "grating": (None if pd.isna(row.get("grating"))
                            else row.get("grating")),
            }

        # Row-aligned arrays (indexed by catalog position) so the has_spec /
        # grade_min / sort-by-sep joins are vectorized numpy lookups instead of
        # per-row dict comprehensions over all 294k ids on every query.
        self._best_grade_arr = np.array(
            [self._best_grade.get(int(i), -1) for i in self._id_arr],
            dtype=np.int64)
        self._has_primary_arr = np.array(
            [int(i) in self._primary for i in self._id_arr], dtype=bool)
        self._min_sep_arr = np.array(
            [self._min_sep.get(int(i), math.inf) for i in self._id_arr],
            dtype=np.float64)

    # -- numeric column cache -----------------------------------------------
    def _numcol(self, col):
        """Return a cached float64 numpy view of a catalog column (coerced)."""
        if col is None:
            return None
        arr = self._num_cache.get(col)
        if arr is None:
            arr = pd.to_numeric(self.catalog[col], errors="coerce").to_numpy()
            self._num_cache[col] = arr
        return arr

    # -- template column resolution -----------------------------------------
    def _tcol(self, base_name, template):
        """Resolve a per-template column name, tolerating missing template."""
        col = f"{base_name}_{template}"
        if col in self.catalog.columns:
            return col
        # fall back to default template so query never hard-fails
        alt = f"{base_name}_{self.default_template}"
        return alt if alt in self.catalog.columns else None

    # -- query ---------------------------------------------------------------
    def query(self, *, template=None, ids=None, ra=None, dec=None,
              radius_arcsec=None, z_min=None, z_max=None,
              lmass_min=None, lmass_max=None, mag_min=None, mag_max=None,
              mag_band=None, has_spec=None, grade_min=None, use_phot=1,
              no_star=1, uvj=None, sort=None, limit=500, offset=0,
              **_ignored):
        """Return (total_before_limit, rows) per /api/catalog/query.

        Hot path: filters and sorts operate on cached float/bool numpy arrays and
        positional indices; only the (paginated) page is materialised into row
        dicts. No pd.to_numeric or iterrows over the full frame per request.
        """
        template = template or self.default_template
        mag_band = mag_band or _DEFAULT_MAG_BAND
        cols = self.catalog.columns

        z_col = self._tcol("z_phot", template)
        lmass_col = self._tcol("lmass", template)
        mag_col = f"mag_{mag_band}"
        if mag_col not in cols:
            mag_col = f"mag_{_DEFAULT_MAG_BAND}"

        mask = np.ones(len(self.catalog), dtype=bool)

        # ids list (comma-sep or list)
        if ids is not None:
            id_list = _parse_int_list(ids)
            if id_list:
                mask &= np.isin(self._id_arr, id_list)

        # flag filters
        if _truthy(use_phot, default=True) and self._use_phot_bool is not None:
            mask &= self._use_phot_bool
        if _truthy(no_star, default=True) and self._flag_star_bool is not None:
            mask &= ~self._flag_star_bool

        # ranges (cached numeric arrays; NaN comparisons -> False as before)
        if z_col:
            zc = self._numcol(z_col)
            if z_min is not None:
                mask &= zc >= float(z_min)
            if z_max is not None:
                mask &= zc <= float(z_max)
        if lmass_col:
            mc = self._numcol(lmass_col)
            if lmass_min is not None:
                mask &= mc >= float(lmass_min)
            if lmass_max is not None:
                mask &= mc <= float(lmass_max)
        if mag_col in cols:
            gc = self._numcol(mag_col)
            if mag_min is not None:
                mask &= gc >= float(mag_min)
            if mag_max is not None:
                mask &= gc <= float(mag_max)

        # uvj
        if uvj:
            uv_col = self._tcol("uvj", template)
            if uv_col:
                uvv = self._numcol(uv_col)
                if uvj == "q":
                    mask &= uvv == 1
                elif uvj == "sf":
                    mask &= uvv == 0

        # cone search. When active, keep the per-row angular distance (cos-dec
        # scaled arcsec) so it can be surfaced as "dist_arcsec" and sorted on.
        # Non-cone queries leave dist_by_row None -> field omitted downstream.
        dist_by_row = None
        if ra is not None and dec is not None and radius_arcsec is not None:
            ra0, dec0, rad = float(ra), float(dec), float(radius_arcsec)
            cosd = math.cos(math.radians(dec0))
            dra = (self._numcol("ra") - ra0) * cosd
            ddec = self._numcol("dec") - dec0
            sep_by_row = np.sqrt(dra ** 2 + ddec ** 2) * 3600.0
            dist_by_row = sep_by_row
            with np.errstate(invalid="ignore"):
                mask &= sep_by_row <= rad

        # has_spec / grade_min (row-aligned join arrays)
        want_spec = _truthy(has_spec, default=None)
        if want_spec is True or grade_min is not None:
            if grade_min is not None:
                mask &= self._best_grade_arr >= int(grade_min)
            else:
                mask &= self._has_primary_arr
        elif want_spec is False:
            mask &= ~self._has_primary_arr

        pos = np.nonzero(mask)[0]  # positional indices of matching rows
        total = int(pos.size)

        # sort (position-based; preserves the previous stable-argsort ordering)
        sort_key, desc = _parse_sort(sort)
        pos = self._sort_positions(pos, sort_key, desc, template, mag_col,
                                   dist_by_row)

        # pagination
        off = max(int(offset or 0), 0)
        lim = min(max(int(limit if limit is not None else 500), 0), 5000)
        page_pos = pos[off: off + lim]

        rows = self._rows_from_positions(page_pos, template, mag_col,
                                         dist_by_row)
        return total, rows

    def _sort_positions(self, pos, sort_key, desc, template, mag_col,
                        dist_by_row=None):
        if pos.size == 0:
            return pos
        if sort_key == "ra":
            key = self._numcol("ra")[pos]
        elif sort_key == "dec":
            key = self._numcol("dec")[pos]
        elif sort_key == "z_phot":
            c = self._numcol(self._tcol("z_phot", template))
            key = c[pos] if c is not None else self._id_arr[pos]
        elif sort_key == "lmass":
            c = self._numcol(self._tcol("lmass", template))
            key = c[pos] if c is not None else self._id_arr[pos]
        elif sort_key == "mag":
            key = self._numcol(mag_col)[pos]
        elif sort_key == "sep":
            key = self._min_sep_arr[pos]
        elif sort_key == "dist":
            # cone-center distance; only defined when a cone is active. With no
            # cone, fall back to id order so sort=dist never hard-fails.
            key = dist_by_row[pos] if dist_by_row is not None else self._id_arr[pos]
        else:  # "id" and any fallback
            key = self._id_arr[pos]
        order = np.argsort(key, kind="stable")
        if desc:
            order = order[::-1]
        return pos[order]

    def _rows_from_positions(self, page_pos, template, mag_col,
                             dist_by_row=None):
        """Build the row dicts for a page of positional indices (no iterrows).

        When a cone was active, `dist_by_row` carries the cos-dec-scaled angular
        distance (arcsec) from the cone center; each row gets a "dist_arcsec"
        (2dp) field. Non-cone pages omit the field entirely.
        """
        if page_pos.size == 0:
            return []
        ids = self._id_arr[page_pos]
        ra = self._numcol("ra")[page_pos]
        dec = self._numcol("dec")[page_pos]
        dist = dist_by_row[page_pos] if dist_by_row is not None else None

        def _slice(base):
            arr = self._numcol(self._tcol(base, template))
            return arr[page_pos] if arr is not None else None

        z_phot = _slice("z_phot")
        z160 = _slice("z160")
        z840 = _slice("z840")
        chi2 = _slice("chi2")
        lmass = _slice("lmass")
        lsfr = _slice("lsfr")
        uvj = _slice("uvj")
        u_v = _slice("u_v")
        v_j = _slice("v_j")
        mag = self._numcol(mag_col)[page_pos]
        fr_arr = self._numcol("flux_radius") if "flux_radius" in self.catalog.columns else None
        flux_radius = fr_arr[page_pos] if fr_arr is not None else None
        nb_arr = self._numcol("n_bands") if "n_bands" in self.catalog.columns else None
        n_bands = nb_arr[page_pos] if nb_arr is not None else None

        def _at(arr, i):
            return None if arr is None else arr[i]

        rows = []
        for i in range(page_pos.size):
            mid = int(ids[i])
            row = {
                "id": mid,
                "ra": _rnd(ra[i], 6),
                "dec": _rnd(dec[i], 6),
                "z_phot": _rnd(_at(z_phot, i), 4),
                "z160": _rnd(_at(z160, i), 4),
                "z840": _rnd(_at(z840, i), 4),
                "chi2": round_sig(_at(chi2, i), 4),
                "lmass": _rnd(_at(lmass, i), 4),
                "lsfr": _rnd(_at(lsfr, i), 4),
                "mag": _rnd(mag[i], 4),
                "flux_radius": _rnd(_at(flux_radius, i), 4),
                "n_bands": _int_or_none(_at(n_bands, i)),
                "uvj": _int_or_none(_at(uvj, i)),
                "u_v": _rnd(_at(u_v, i), 4),
                "v_j": _rnd(_at(v_j, i), 4),
                "spec": self._primary.get(mid),
            }
            if dist is not None:
                row["dist_arcsec"] = _rnd(dist[i], 2)
            rows.append(row)
        return rows

    # -- single object -------------------------------------------------------
    def object(self, obj_id):
        """Full single-object record per /api/object/{field}/{id}."""
        obj_id = int(obj_id)
        pos = self._id_index.get_indexer([obj_id])[0]
        if pos < 0:
            return None
        r = self.catalog.iloc[pos]

        phot = []
        for band in self.bands:
            mag = _num(r.get(f"mag_{band}"))
            snr = _num(r.get(f"snr_{band}"))
            ok = _finite(snr) and snr > 0 and _finite(mag)
            fnu = 10.0 ** ((23.9 - mag) / 2.5) if _finite(mag) else None
            efnu = (fnu / snr) if (ok and fnu is not None) else None
            phot.append({
                "band": band,
                "lam_um": PIVOT_UM.get(band, self._pivots_meta.get(band)),
                "fnu_uJy": round_sig(fnu, 4),
                "efnu_uJy": round_sig(efnu, 4),
                "mag": _rnd(mag, 4),
                "ok": bool(ok),
            })

        zout = {}
        for t in self.templates:
            zout[t] = {
                "z_phot": _rnd(_g(r, self._tcol("z_phot", t)), 4),
                "z_ml": _rnd(_g(r, self._tcol("z_ml", t)), 4),
                "chi2": round_sig(_g(r, self._tcol("chi2", t)), 4),
                "z025": _rnd(_g(r, self._tcol("z025", t)), 4),
                "z160": _rnd(_g(r, self._tcol("z160", t)), 4),
                "z500": _rnd(_g(r, self._tcol("z500", t)), 4),
                "z840": _rnd(_g(r, self._tcol("z840", t)), 4),
                "z975": _rnd(_g(r, self._tcol("z975", t)), 4),
                "lmass": _rnd(_g(r, self._tcol("lmass", t)), 4),
                "lsfr": _rnd(_g(r, self._tcol("lsfr", t)), 4),
                "Av": _rnd(_g(r, self._tcol("Av", t)), 4),
                "u_v": _rnd(_g(r, self._tcol("u_v", t)), 4),
                "v_j": _rnd(_g(r, self._tcol("v_j", t)), 4),
                "uvj": _int_or_none(_g(r, self._tcol("uvj", t))),
            }

        spectra = self._spectra_for_mid(obj_id, self.default_template)

        return {
            "id": obj_id,
            "ra": _rnd(r.get("ra"), 6),
            "dec": _rnd(r.get("dec"), 6),
            "flags": {
                "use_phot": _flag(r.get("use_phot")),
                "flag_star": _flag(r.get("flag_star")),
                "flag_clean": _flag(r.get("flag_clean")),
            },
            "n_bands": _int_or_none(r.get("n_bands")),
            "morph": {
                "flux_radius": _rnd(r.get("flux_radius"), 4),
                "a_image": _rnd(r.get("a_image"), 4),
                "b_image": _rnd(r.get("b_image"), 4),
                "theta": _rnd(r.get("theta", r.get("theta_J2000")), 4),
                "kron_radius": _rnd(r.get("kron_radius"), 4),
            },
            "phot": phot,
            "zout": zout,
            "spectra": spectra,
        }

    def _spectra_for_mid(self, mid, template):
        sp = self.spectra
        if sp.empty or "mid" not in sp.columns:
            return []
        grp = sp[sp["mid"] == int(mid)]
        return [self._inspect_row(r, template) for _, r in grp.iterrows()]

    # -- inspect list --------------------------------------------------------
    def inspect_list(self, template=None):
        """All matched spectra shaped for /api/inspect/list, sorted by dja asc.

        Memoized per template: spectra.parquet is static for the process
        lifetime, so the sorted rows are fully deterministic. Subsequent calls
        return the cached list (json_ok rebuilds the payload on serialize, so
        sharing the cached list across requests is safe).
        """
        template = template or self.default_template
        cached = self._inspect_cache.get(template)
        if cached is not None:
            return cached
        sp = self.spectra
        if sp.empty:
            self._inspect_cache[template] = []
            return []
        rows = [self._inspect_row(r, template) for _, r in sp.iterrows()]
        rows.sort(key=lambda o: (o.get("dja") or ""))
        self._inspect_cache[template] = rows
        return rows

    def _inspect_row(self, r, template):
        zp_col = f"zp_{template}"
        if zp_col not in self.spectra.columns:
            zp_col = f"zp_{self.default_template}"
        return {
            "dja": r.get("dja"),
            "mid": _int_or_none(r.get("mid")),
            "ra": _rnd(r.get("ra"), 6),
            "dec": _rnd(r.get("dec"), 6),
            "dja_ra": _rnd(r.get("dja_ra"), 6),
            "dja_dec": _rnd(r.get("dja_dec"), 6),
            "sep": _rnd(r.get("sep"), 4),
            "zs": _rnd(r.get("zs"), 4),
            "zp": _rnd(r.get(zp_col) if zp_col in self.spectra.columns else None, 4),
            "grade": _int_or_none(r.get("grade")) or 0,
            "mag": _rnd(r.get("mag"), 4),
            "sn": round_sig(r.get("sn"), 4),
            "root": r.get("root"),
            "file": r.get("file"),
            "srcid": _int_or_none(r.get("srcid")),
            "pid": _int_or_none(r.get("pid")),
            "grating": None if pd.isna(r.get("grating")) else r.get("grating"),
            "metafile": "" if pd.isna(r.get("metafile")) else r.get("metafile"),
            "nspec": _int_or_none(r.get("nspec")) or 1,
            "exptime": _int_or_none(r.get("exptime")),
            # DJA reviewer verdict + per-disperser evidence (may be absent on
            # supplement rows / pre-reindex parquet -> None).
            "reviewer": _str_or_none(r.get("reviewer")),
            "comment": _str_or_none(r.get("comment")),
            "zgrade": _rnd(r.get("zgrade"), 4),
            "z_prism": _rnd(r.get("z_prism"), 4),
            "z_grating": _rnd(r.get("z_grating"), 4),
            "sn_line": round_sig(r.get("sn_line"), 4),
        }

    # -- field info ----------------------------------------------------------
    def info(self):
        """The /api/fields entry for this field."""
        rr = self.meta.get("ra_range")
        dr = self.meta.get("dec_range")
        if rr is None and "ra" in self.catalog.columns:
            ra = pd.to_numeric(self.catalog["ra"], errors="coerce")
            rr = [_rnd(ra.min(), 6), _rnd(ra.max(), 6)]
        if dr is None and "dec" in self.catalog.columns:
            de = pd.to_numeric(self.catalog["dec"], errors="coerce")
            dr = [_rnd(de.min(), 6), _rnd(de.max(), 6)]
        return {
            "name": self.name,
            "title": getattr(self.field, "title", self.meta.get("title", self.name)),
            "n_objects": int(self.meta.get("n_objects", len(self.catalog))),
            "templates": self.templates,
            "default_template": self.default_template,
            "n_spec_matched": int(self.meta.get("n_spec_matched", len(self.spectra))),
            "bands": self.bands,
            "ra_range": rr,
            "dec_range": dr,
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _g(row, col):
    return row.get(col) if col else None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _flag(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return bool(x)


def _str_or_none(x):
    """Non-empty stripped string, else None (NaN/''/whitespace -> None)."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    return s or None


def _parse_int_list(ids):
    if ids is None:
        return []
    if isinstance(ids, (list, tuple)):
        raw = ids
    else:
        raw = str(ids).split(",")
    out = []
    for tok in raw:
        tok = str(tok).strip()
        if tok:
            try:
                out.append(int(float(tok)))
            except ValueError:
                pass
    return out


def _truthy(v, default):
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(int(v)) if isinstance(v, (int, float)) else bool(v)


def _parse_sort(sort):
    if not sort:
        return "id", False
    s = str(sort).strip()
    desc = s.startswith("-")
    key = s[1:] if desc else s
    if key not in SORT_WHITELIST:
        return "id", False
    return key, desc


# ---------------------------------------------------------------------------
# Synthetic fixture generator + query tests (run: python -m server.catalog)
# ---------------------------------------------------------------------------
def _write_fixture(index_dir, field_name="cosmos",
                   templates=("sfhz_blue_agn", "larson"), n=200, seed=42):
    """Write a 200-row fake index (catalog + spectra + meta) to index_dir."""
    rng = np.random.default_rng(seed)
    bands = list(PIVOT_UM.keys())
    base = Path(index_dir) / field_name
    base.mkdir(parents=True, exist_ok=True)

    ids = np.arange(1000, 1000 + n)
    ra = 150.0 + rng.uniform(-0.1, 0.1, n)
    dec = 2.0 + rng.uniform(-0.1, 0.1, n)
    cat = {
        "id": ids, "ra": ra, "dec": dec,
        "flux_radius": rng.uniform(0.05, 0.5, n),
        "a_image": rng.uniform(1, 5, n), "b_image": rng.uniform(1, 4, n),
        "theta": rng.uniform(-90, 90, n), "kron_radius": rng.uniform(2, 6, n),
        "use_phot": rng.random(n) > 0.1,
        "flag_star": rng.random(n) < 0.05,
        "flag_clean": rng.random(n) > 0.2,
        "n_bands": rng.integers(10, 32, n),
    }
    for b in bands:
        mag = rng.uniform(20, 30, n)
        mag[rng.random(n) < 0.1] = np.nan  # some unmeasured
        cat[f"mag_{b}"] = mag
        snr = rng.uniform(-1, 50, n)
        cat[f"snr_{b}"] = snr
    for t in templates:
        zp = rng.uniform(0.1, 8.0, n)
        cat[f"z_phot_{t}"] = zp
        cat[f"z_ml_{t}"] = zp + rng.normal(0, 0.05, n)
        cat[f"chi2_{t}"] = rng.uniform(5, 500, n)
        cat[f"z025_{t}"] = zp - rng.uniform(0.1, 0.5, n)
        cat[f"z160_{t}"] = zp - rng.uniform(0.05, 0.2, n)
        cat[f"z500_{t}"] = zp
        cat[f"z840_{t}"] = zp + rng.uniform(0.05, 0.2, n)
        cat[f"z975_{t}"] = zp + rng.uniform(0.1, 0.5, n)
        cat[f"lmass_{t}"] = rng.uniform(7, 12, n)
        cat[f"lsfr_{t}"] = rng.uniform(-2, 3, n)
        cat[f"Av_{t}"] = rng.uniform(0, 3, n)
        cat[f"u_v_{t}"] = rng.uniform(0, 2.5, n)
        cat[f"v_j_{t}"] = rng.uniform(0, 2.5, n)
        cat[f"uvj_{t}"] = rng.integers(0, 2, n)
    pd.DataFrame(cat).to_parquet(base / "catalog.parquet")

    # spectra: ~60 spectra spread across a subset of objects (some multi)
    n_spec = 60
    mids = rng.choice(ids, n_spec)
    gratings = ["prism-clear", "g395m-f290lp", "g140m-f070lp", "g235m-f170lp"]
    spec = {
        "dja": [f"root{m%5}-v4_x_{i}" for i, m in enumerate(mids)],
        "mid": mids,
        "ra": ra[mids - 1000] if False else rng.uniform(150, 150.2, n_spec),
        "dec": rng.uniform(1.9, 2.1, n_spec),
        "dja_ra": rng.uniform(150, 150.2, n_spec),
        "dja_dec": rng.uniform(1.9, 2.1, n_spec),
        "sep": rng.uniform(0.0, 0.5, n_spec),
        "zs": rng.uniform(0.1, 8, n_spec),
        "grade": rng.choice([0, 1, 2, 3], n_spec),
        "mag": rng.uniform(20, 28, n_spec),
        "sn": rng.uniform(1, 40, n_spec),
        "root": [f"root{m%5}-v4" for m in mids],
        "file": [f"root{m%5}-v4_prism-clear_{i}.spec.fits" for i, m in enumerate(mids)],
        "srcid": rng.integers(1, 99999, n_spec),
        "pid": rng.integers(1000, 5000, n_spec),
        "grating": rng.choice(gratings, n_spec),
        "metafile": ["jw0000_msa" for _ in range(n_spec)],
        "exptime": rng.uniform(1000, 10000, n_spec),
    }
    sp_df = pd.DataFrame(spec)
    # correct ra/dec of MINERVA position from the catalog, and nspec counts
    pos = {int(i): (r, d) for i, r, d in zip(ids, ra, dec)}
    sp_df["ra"] = sp_df["mid"].map(lambda m: pos[int(m)][0])
    sp_df["dec"] = sp_df["mid"].map(lambda m: pos[int(m)][1])
    counts = sp_df["mid"].value_counts()
    sp_df["nspec"] = sp_df["mid"].map(counts)
    for t in templates:
        sp_df[f"zp_{t}"] = sp_df["mid"].map(
            {int(i): z for i, z in zip(ids, cat[f"z_phot_{t}"])})
    sp_df.to_parquet(base / "spectra.parquet")

    meta = {
        "n_objects": n,
        "n_spec_matched": int(sp_df["mid"].nunique()),
        "bands": bands,
        "templates": list(templates),
        "default_template": templates[0],
        "title": "COSMOS (fixture)",
        "ra_range": [round(float(ra.min()), 6), round(float(ra.max()), 6)],
        "dec_range": [round(float(dec.min()), 6), round(float(dec.max()), 6)],
    }
    with open(base / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return base


class _FakeField:
    def __init__(self, name, templates):
        self.name = name
        self.title = "COSMOS (fixture)"
        self.default_template = templates[0]
        self.eazy = {t: Path(f"/dev/null/{t}") for t in templates}


def _selftest():
    import tempfile
    templates = ("sfhz_blue_agn", "larson")
    tmp = Path(tempfile.mkdtemp(prefix="minerva_cat_"))
    _write_fixture(tmp, "cosmos", templates, n=200)
    field = _FakeField("cosmos", templates)
    store = CatalogStore(field, tmp)

    def show(label, total, rows):
        print(f"[{label}] total={total} returned={len(rows)}"
              + (f" first_id={rows[0]['id']}" if rows else ""))

    print("== info ==")
    info = store.info()
    print({k: info[k] for k in ("name", "n_objects", "n_spec_matched",
                                "templates", "default_template")})
    assert info["n_objects"] == 200
    assert len(info["bands"]) == 32

    print("== baseline query (use_phot=1, no_star=1) ==")
    total, rows = store.query()
    show("baseline", total, rows)
    assert total <= 200 and len(rows) == total

    print("== no filters (use_phot=0, no_star=0) ==")
    t0, r0 = store.query(use_phot=0, no_star=0)
    show("nofilter", t0, r0)
    assert t0 == 200

    print("== ids list ==")
    t, r = store.query(ids="1000,1005,1010", use_phot=0, no_star=0)
    show("ids", t, r)
    assert t == 3 and {x["id"] for x in r} == {1000, 1005, 1010}

    print("== z range ==")
    t, r = store.query(z_min=1.0, z_max=3.0, use_phot=0, no_star=0)
    show("zrange", t, r)
    assert all(1.0 <= x["z_phot"] <= 3.0 for x in r)

    print("== lmass range ==")
    t, r = store.query(lmass_min=9.0, lmass_max=11.0, use_phot=0, no_star=0)
    show("lmass", t, r)
    assert all(9.0 <= x["lmass"] <= 11.0 for x in r)

    print("== mag range on f444w ==")
    t, r = store.query(mag_min=22, mag_max=25, mag_band="f444w",
                       use_phot=0, no_star=0)
    show("mag", t, r)
    assert all(22 <= x["mag"] <= 25 for x in r if x["mag"] is not None)

    print("== cone search ==")
    c_ra, c_dec = 150.0, 2.0
    t, r = store.query(ra=c_ra, dec=c_dec, radius_arcsec=200,
                       use_phot=0, no_star=0)
    show("cone", t, r)
    # verify all returned within radius
    for x in r:
        cosd = math.cos(math.radians(c_dec))
        sep = math.hypot((x["ra"] - c_ra) * cosd, x["dec"] - c_dec) * 3600
        assert sep <= 200.0001, sep

    print("== has_spec=1 ==")
    t, r = store.query(has_spec=1, use_phot=0, no_star=0)
    show("has_spec", t, r)
    assert all(x["spec"] is not None for x in r)
    assert t == info["n_spec_matched"]

    print("== has_spec=0 ==")
    t, r = store.query(has_spec=0, use_phot=0, no_star=0)
    show("no_spec", t, r)
    assert all(x["spec"] is None for x in r)

    print("== grade_min=2 ==")
    t, r = store.query(grade_min=2, use_phot=0, no_star=0)
    show("grade_min", t, r)
    for x in r:
        assert x["spec"] is not None

    print("== uvj q / sf ==")
    tq, rq = store.query(uvj="q", use_phot=0, no_star=0)
    tsf, rsf = store.query(uvj="sf", use_phot=0, no_star=0)
    show("uvj_q", tq, rq)
    show("uvj_sf", tsf, rsf)
    assert tq + tsf == 200

    print("== sort id asc/desc ==")
    _, ra_asc = store.query(sort="id", use_phot=0, no_star=0, limit=5000)
    _, ra_desc = store.query(sort="-id", use_phot=0, no_star=0, limit=5000)
    ids_asc = [x["id"] for x in ra_asc]
    ids_desc = [x["id"] for x in ra_desc]
    assert ids_asc == sorted(ids_asc)
    assert ids_desc == sorted(ids_desc, reverse=True)

    print("== sort z_phot desc ==")
    _, rz = store.query(sort="-z_phot", use_phot=0, no_star=0, limit=5000)
    zs = [x["z_phot"] for x in rz]
    assert zs == sorted(zs, reverse=True)

    print("== sort sep asc (has_spec) ==")
    _, rsep = store.query(sort="sep", has_spec=1, use_phot=0, no_star=0, limit=5000)
    seps = [store._min_sep[x["id"]] for x in rsep]
    assert seps == sorted(seps)

    print("== pagination ==")
    t_all, all_rows = store.query(sort="id", use_phot=0, no_star=0, limit=5000)
    _, p1 = store.query(sort="id", use_phot=0, no_star=0, limit=10, offset=0)
    _, p2 = store.query(sort="id", use_phot=0, no_star=0, limit=10, offset=10)
    assert [x["id"] for x in p1] == [x["id"] for x in all_rows[:10]]
    assert [x["id"] for x in p2] == [x["id"] for x in all_rows[10:20]]
    assert t_all == 200

    print("== combined: cone + z + sort + pagination ==")
    t, r = store.query(ra=150.0, dec=2.0, radius_arcsec=300, z_min=0.5,
                       sort="-lmass", use_phot=0, no_star=0, limit=5, offset=2)
    show("combined", t, r)
    lm = [x["lmass"] for x in r]
    assert lm == sorted(lm, reverse=True)

    print("== object() ==")
    obj = store.object(1000)
    assert obj["id"] == 1000
    assert len(obj["phot"]) == 32
    for p in obj["phot"]:
        if p["ok"]:
            assert p["fnu_uJy"] is not None and p["efnu_uJy"] is not None
            # verify inversion: mag = 23.9 - 2.5 log10(fnu)
            m = 23.9 - 2.5 * math.log10(p["fnu_uJy"])
            assert abs(m - p["mag"]) < 1e-3, (m, p["mag"])
    assert set(obj["zout"].keys()) == set(templates)
    print("  phot[0]:", obj["phot"][29])  # f444w
    print("  zout sfhz:", obj["zout"]["sfhz_blue_agn"])
    print("  n spectra:", len(obj["spectra"]))

    print("== object() missing id ==")
    assert store.object(999999) is None

    print("== inspect_list sorted by dja asc ==")
    il = store.inspect_list("sfhz_blue_agn")
    djas = [o["dja"] for o in il]
    assert djas == sorted(djas)
    print("  n inspect rows:", len(il), "first:", il[0]["dja"] if il else None)
    assert len(il) == 60

    print("\nALL CATALOG SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
