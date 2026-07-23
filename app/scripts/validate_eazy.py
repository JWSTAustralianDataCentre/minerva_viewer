#!/usr/bin/env python3
"""
Offline validation of server/eazy_products.py against the eazy-py ground truth.

Loads the full ``eazy.photoz.PhotoZ`` object for the COSMOS ``sfhz_blue_agn``
SUPER products via ``eazy.hdf5.initialize_from_hdf5`` (slow, ~90 s, fine
offline), picks 5 objects spread in redshift (including one z>4), and for each
compares the EazyStore payload against ``PhotoZ.show_fit(..., get_spec=True)``:

  (1) p(z)      : our normalised p(z) vs the prior-free reference
                  exp(-(chi2-chi2_min)/2) built from ``PhotoZ.chi2_fit`` (the
                  fit used APPLY_PRIOR=0, so no prior is applied).  Note:
                  ``initialize_from_hdf5`` re-applies a prior to ``PhotoZ.lnp``
                  by default (ZML_WITH_PRIOR=True in the file), so we build the
                  reference straight from ``chi2_fit`` rather than ``lnp``.
  (2) template  : our observed-frame template f-nu vs show_fit ``templz/templf``
                  (show_fnu=1, microJy), median fractional deviation over
                  0.5-10 microns where templf>0 must be < 2%.
  (3) photometry: our phot f-nu vs show_fit ``fobs`` (microJy) on valid bands.

Run (from anywhere)::

    /otdata2/themiya/grizli_rebels/bin/python3 app/scripts/validate_eazy.py

The eazy reference-object initialisation needs the eazy data tree on disk; this
script cd's into it so the relative FILTERS_RES/template paths resolve.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --- locate paths ---------------------------------------------------------
# Default target: COSMOS sfhz_blue_agn SUPER ZPiter. Override with --zpiter /
# --h5 / --template to validate another field's h5 (e.g. EGS, whose products
# are named '*.eazypy.h5' with no *.eazy.data.fits, so the served zgrid is the
# regenerated log grid -- this script cross-checks it against pz.zgrid).
ZPITER = Path(
    "/otdata2/themiya/minerva/data/catalogs/cosmos/ACS+WEBB_chi-mean/"
    "EAzY/SFHZ_BLUE_AGN/SUPER/ZPiter"
)
APP_DIR = Path(__file__).resolve().parents[1]  # .../app
sys.path.insert(0, str(APP_DIR))

TEMPL_MED_FRAC_THRESH = 0.02   # (2) median frac dev over 0.5-10 um
PZ_ATOL = 1e-4                 # (1) max abs pz diff on the grid
PHOT_MED_FRAC_THRESH = 1e-2    # (3) median frac dev on valid bands


def _eazy_data_dir() -> Path:
    import eazy.utils

    return Path(eazy.utils.DATA_PATH)


def _find_h5(zpiter: Path) -> Path:
    hits = sorted(zpiter.glob("*.eazy.h5")) or sorted(zpiter.glob("*.eazypy.h5"))
    if not hits:
        raise FileNotFoundError(f"no eazy h5 in {zpiter}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate eazy_products vs eazy-py.")
    ap.add_argument("--zpiter", default=str(ZPITER),
                    help="ZPiter dir holding the eazy h5 (default: COSMOS sfhz)")
    ap.add_argument("--h5", default=None,
                    help="explicit eazy h5 path (else globbed inside --zpiter)")
    ap.add_argument("--template", default="sfhz_blue_agn",
                    help="template-set name key (default: sfhz_blue_agn)")
    ap.add_argument("--n-targets", type=int, default=5,
                    help="how many redshift-spread objects to test (default 5)")
    args = ap.parse_args()

    zpiter = Path(args.zpiter)
    template = args.template
    h5path = Path(args.h5) if args.h5 else _find_h5(zpiter)

    from server.eazy_products import EazyStore  # local module under test

    # Build a minimal FieldConfig shim: EazyStore only needs .eazy (+ optional
    # h5_path). Provide an explicit h5_path so the store loads THIS h5 exactly.
    field = SimpleNamespace(
        name="validate",
        eazy={template: zpiter},
        default_template=template,
        h5_path=lambda t, _p=h5path: _p,
    )
    store = EazyStore(field)

    # --- ground truth: full PhotoZ (slow) ---------------------------------
    cwd0 = os.getcwd()
    os.chdir(_eazy_data_dir())  # so eazy's relative data paths resolve
    try:
        import eazy.hdf5

        print(f"initialize_from_hdf5({h5path.name}) ... (~90s)", flush=True)
        pz = eazy.hdf5.initialize_from_hdf5(str(h5path), verbose=False)
    finally:
        os.chdir(cwd0)

    zbest = np.asarray(pz.zbest)
    good = np.where((zbest > 0.05) & (zbest < 12))[0]
    zg = zbest[good]
    _all_targets = [0.2, 0.8, 1.7, 2.8, 4.5]  # last is the z>4 requirement
    targets = _all_targets[:max(1, args.n_targets)]
    ids = [int(pz.OBJID[good[np.argmin(np.abs(zg - t))]]) for t in targets]

    print(f"\nselected ids: {ids}")
    print(
        f"selected z  : "
        f"{[round(float(zbest[pz.idx[pz.OBJID == i][0]]), 3) for i in ids]}\n"
    )

    rows = []
    all_pass = True
    for oid in ids:
        ix = int(pz.idx[pz.OBJID == oid][0])
        z = float(zbest[ix])

        # reference from show_fit (microJy)
        ref = pz.show_fit(oid, get_spec=True, show_fnu=1)
        rtz = np.asarray(ref["templz"], dtype=float)   # Angstrom
        rtf = np.asarray(ref["templf"], dtype=float)   # microJy
        rfobs = np.asarray(ref["fobs"], dtype=float)   # microJy
        rvalid = np.asarray(ref["valid"], dtype=bool)

        # prior-free reference p(z) from chi2_fit (APPLY_PRIOR=0)
        chi2 = np.asarray(pz.chi2_fit[ix], dtype=float)
        zgrid = np.asarray(pz.zgrid, dtype=float)
        pz_ref = np.exp(-(chi2 - np.nanmin(chi2)) / 2.0)
        pz_ref /= np.trapz(pz_ref, zgrid)

        # --- our payload --------------------------------------------------
        out = store.get_products(template, oid)
        our_pz = np.array([np.nan if v is None else v for v in out["pz"]], float)
        our_zgrid = np.asarray(out["zgrid"], float)
        our_lam = np.asarray(out["templ"]["lam_um"], float)      # microns
        our_flx = np.array(
            [np.nan if v is None else v for v in out["templ"]["fnu_uJy"]], float
        )
        our_phot = np.array(
            [np.nan if v is None else v for v in out["phot"]["fnu_uJy"]], float
        )
        our_ok = np.asarray(out["phot"]["ok"], bool)

        # (1) p(z)
        pz_max_abs = float(np.nanmax(np.abs(our_pz - pz_ref)))
        grid_ok = np.allclose(our_zgrid, zgrid, rtol=0, atol=1e-4)
        pz_pass = grid_ok and (pz_max_abs < PZ_ATOL)

        # (2) template over 0.5-10 um where templf > 0
        wlo, whi = 0.5, 10.0
        m = (our_lam >= wlo) & (our_lam <= whi)
        ref_on_our = np.interp(our_lam, rtz / 1e4, rtf)
        mm = m & (ref_on_our > 0) & np.isfinite(our_flx)
        if mm.sum() > 0:
            templ_med = float(
                np.median(np.abs(our_flx[mm] - ref_on_our[mm]) / ref_on_our[mm])
            )
        else:
            templ_med = np.nan
        templ_pass = np.isfinite(templ_med) and (templ_med < TEMPL_MED_FRAC_THRESH)

        # (3) photometry on valid bands
        pm = rvalid & our_ok & np.isfinite(rfobs) & (np.abs(rfobs) > 0)
        if pm.sum() > 0:
            phot_med = float(
                np.median(np.abs(our_phot[pm] - rfobs[pm]) / np.abs(rfobs[pm]))
            )
        else:
            phot_med = np.nan
        phot_pass = np.isfinite(phot_med) and (phot_med < PHOT_MED_FRAC_THRESH)

        ok = pz_pass and templ_pass and phot_pass
        all_pass = all_pass and ok
        rows.append(
            (oid, z, pz_max_abs, pz_pass, templ_med, templ_pass,
             phot_med, phot_pass, ok)
        )

    # --- table ------------------------------------------------------------
    print("=" * 92)
    print(
        f"{'id':>9} {'z':>6} | {'pz max|d|':>10} {'pz':>4} | "
        f"{'templ med':>9} {'tmpl':>4} | {'phot med':>9} {'phot':>4} | {'RESULT':>6}"
    )
    print("-" * 92)
    for (oid, z, pzd, pzp, tmed, tp, pmed, pp, ok) in rows:
        print(
            f"{oid:>9} {z:>6.3f} | {pzd:>10.2e} {'OK' if pzp else 'XX':>4} | "
            f"{tmed:>9.4f} {'OK' if tp else 'XX':>4} | "
            f"{pmed:>9.2e} {'OK' if pp else 'XX':>4} | "
            f"{'PASS' if ok else 'FAIL':>6}"
        )
    print("=" * 92)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
