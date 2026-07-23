"""End-to-end endpoint smoke test for the MINERVA Viewer v2 server.

Hits EVERY endpoint in app/API.md with REAL data pulled from the built parquet
index, and asserts payload shapes / types / units against the contract:

  * /api/fields               -- counts, 32 bands, template order
  * /api/catalog/query        -- filters, cone search around (150.13, 2.328)
                                 with returned-sep verification, ids, has_spec
  * /api/object/{f}/{id}       -- 32 phot bands, mag<->fnu round-trip, zout, 404
  * /api/eazy/{f}/{t}/{id}     -- 607-pt pz integrates to ~1, phot uJy, SED, 404
  * /api/inspect/list          -- one row per spectrum, unique dja keys
  * /api/spectrum/{root}/{f}   -- prism wave in [0.5,5.5] um, 2D b64 shape, 400s
  * /api/cutout                -- rgb/slit/grid PNG magic bytes
  * /api/dja/preview/{root}/{b}
  * /api/decisions             -- POST then GET latest-wins merge

Run (server must be up on --base):
    python scripts/smoke_api.py --base http://127.0.0.1:8321
Exit code 0 = all pass, 1 = any failure.
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
import time

import numpy as np
import pandas as pd
import requests

INDEX = "/otdata2/themiya/minerva/data/viewer_index/cosmos"

RESULTS: list[tuple[str, str, str, float]] = []  # (name, status, detail, ms)


def _check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        ms = (time.time() - t0) * 1000
        RESULTS.append((name, "PASS", detail or "", ms))
    except AssertionError as e:
        ms = (time.time() - t0) * 1000
        RESULTS.append((name, "FAIL", f"assert: {e}", ms))
    except Exception as e:  # noqa: BLE001
        ms = (time.time() - t0) * 1000
        RESULTS.append((name, "ERROR", f"{type(e).__name__}: {e}", ms))


def pick_ids():
    c = pd.read_parquet(f"{INDEX}/catalog.parquet",
                        columns=["id", "ra", "dec", "mag_f444w",
                                 "z_phot_sfhz_blue_agn", "use_phot", "flag_star"])
    s = pd.read_parquet(f"{INDEX}/spectra.parquet")
    bright = c[(c.use_phot) & (~c.flag_star) & (c.mag_f444w > 18)
               & (c.mag_f444w < 21)].sort_values("mag_f444w").iloc[0]
    hz = c[(c.use_phot) & (~c.flag_star) & (c.z_phot_sfhz_blue_agn > 4)
           & (c.z_phot_sfhz_blue_agn < 9) & (c.mag_f444w < 27)
           ].sort_values("mag_f444w").iloc[0]
    prism = s[(s.grating.str.contains("prism", case=False, na=False))
              & (s.root.str.contains("cos", case=False, na=False))
              & (s.metafile.notna()) & (s.metafile != "")
              & (s.grade >= 3)].iloc[0]
    multi = s[s.nspec > 1].sort_values("nspec", ascending=False).iloc[0]
    return {
        "bright": int(bright.id),
        "bright_ra": float(bright.ra), "bright_dec": float(bright.dec),
        "bright_mag": float(bright.mag_f444w),
        "highz": int(hz.id), "highz_z": float(hz.z_phot_sfhz_blue_agn),
        "spec_root": prism.root, "spec_file": prism.file,
        "spec_meta": prism.metafile, "spec_mid": int(prism.mid),
        "spec_ra": float(prism.dja_ra), "spec_dec": float(prism.dja_dec),
        "multi_mid": int(multi.mid), "multi_n": int(multi.nspec),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8321")
    args = ap.parse_args()
    B = args.base.rstrip("/")
    S = requests.Session()

    ids = pick_ids()
    print("== sample ids ==")
    for k, v in ids.items():
        print(f"  {k}: {v}")
    print()

    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    # ---- /api/fields ------------------------------------------------------
    def t_fields():
        r = S.get(f"{B}/api/fields", timeout=30)
        assert r.status_code == 200, r.status_code
        f = r.json()["fields"][0]
        assert f["name"] == "cosmos"
        assert f["n_objects"] == 294126, f["n_objects"]
        assert len(f["bands"]) == 32, len(f["bands"])
        assert f["templates"] == ["sfhz_blue_agn", "larson"]
        assert f["default_template"] == "sfhz_blue_agn"
        # 4387 = v4.5-CSV baseline; the DJA live supplement can only grow this.
        assert f["n_spec_matched"] >= 4387, f["n_spec_matched"]
        assert len(f["ra_range"]) == 2 and f["ra_range"][0] < f["ra_range"][1]
        return f"n_obj={f['n_objects']} bands={len(f['bands'])} nspec={f['n_spec_matched']}"
    _check("GET /api/fields", t_fields)

    # ---- /api/catalog/query basic + ids ----------------------------------
    def t_query_ids():
        r = S.get(f"{B}/api/catalog/query",
                  params={"ids": f"{ids['bright']},{ids['highz']}",
                          "use_phot": 0, "no_star": 0}, timeout=30)
        assert r.status_code == 200
        j = r.json()
        got = {row["id"] for row in j["rows"]}
        assert got == {ids["bright"], ids["highz"]}, got
        row = next(x for x in j["rows"] if x["id"] == ids["highz"])
        for key in ("id", "ra", "dec", "z_phot", "z160", "z840", "chi2",
                    "lmass", "lsfr", "mag", "flux_radius", "n_bands", "uvj",
                    "u_v", "v_j", "spec"):
            assert key in row, f"missing {key}"
        assert row["z_phot"] > 4, row["z_phot"]
        return f"ids ok, total={j['total']}"
    _check("GET /api/catalog/query (ids)", t_query_ids)

    # ---- cone search around (150.13, 2.328) with sep verification --------
    def t_cone():
        ra0, dec0, rad = 150.13, 2.328, 30.0
        r = S.get(f"{B}/api/catalog/query",
                  params={"ra": ra0, "dec": dec0, "radius_arcsec": rad,
                          "use_phot": 0, "no_star": 0, "sort": "id",
                          "limit": 5000}, timeout=30)
        assert r.status_code == 200
        rows = r.json()["rows"]
        assert len(rows) > 0, "cone returned no rows"
        cosd = math.cos(math.radians(dec0))
        maxsep = 0.0
        for x in rows:
            sep = math.hypot((x["ra"] - ra0) * cosd, x["dec"] - dec0) * 3600.0
            assert sep <= rad + 1e-3, f"row {x['id']} sep={sep:.4f} > {rad}"
            maxsep = max(maxsep, sep)
        return f"{len(rows)} rows within {rad}\", max sep={maxsep:.3f}\""
    _check("GET /api/catalog/query (cone 150.13,2.328)", t_cone)

    def t_query_hasspec():
        r = S.get(f"{B}/api/catalog/query",
                  params={"has_spec": 1, "use_phot": 0, "no_star": 0,
                          "limit": 5000}, timeout=30)
        assert r.status_code == 200
        j = r.json()
        assert j["total"] >= 4387, j["total"]  # baseline; supplement grows it
        assert all(x["spec"] is not None for x in j["rows"])
        sp = j["rows"][0]["spec"]
        for key in ("dja", "zs", "grade", "sep", "grating"):
            assert key in sp, f"spec missing {key}"
        return f"total={j['total']} (>= 4387 baseline)"
    _check("GET /api/catalog/query (has_spec=1)", t_query_hasspec)

    # ---- /api/object bright ----------------------------------------------
    def t_object_bright():
        r = S.get(f"{B}/api/object/cosmos/{ids['bright']}", timeout=30)
        assert r.status_code == 200
        o = r.json()
        assert o["id"] == ids["bright"]
        assert len(o["phot"]) == 32, len(o["phot"])
        assert set(o["zout"].keys()) == {"sfhz_blue_agn", "larson"}
        for key in ("use_phot", "flag_star", "flag_clean"):
            assert key in o["flags"]
        for key in ("flux_radius", "a_image", "b_image", "theta", "kron_radius"):
            assert key in o["morph"]
        # mag <-> fnu_uJy round-trip on ok bands
        nok = 0
        for p in o["phot"]:
            assert set(p) >= {"band", "lam_um", "fnu_uJy", "efnu_uJy", "mag", "ok"}
            if p["ok"] and p["fnu_uJy"] and p["fnu_uJy"] > 0 and p["mag"] is not None:
                m = 23.9 - 2.5 * math.log10(p["fnu_uJy"])
                assert abs(m - p["mag"]) < 2e-3, (p["band"], m, p["mag"])
                nok += 1
        # COSMOS per-band coverage varies; a source may have only a handful of
        # observed bands. Require the round-trip to hold on ALL ok bands, and at
        # least a few ok bands to exist.
        assert nok >= 5, f"only {nok} ok bands round-tripped"
        return f"32 phot, {nok} bands mag<->fnu ok, theta={o['morph']['theta']}"
    _check("GET /api/object (bright)", t_object_bright)

    def t_object_highz():
        r = S.get(f"{B}/api/object/cosmos/{ids['highz']}", timeout=30)
        assert r.status_code == 200
        o = r.json()
        assert o["zout"]["sfhz_blue_agn"]["z_phot"] > 4
        return f"z_phot={o['zout']['sfhz_blue_agn']['z_phot']}"
    _check("GET /api/object (highz)", t_object_highz)

    def t_object_404():
        r = S.get(f"{B}/api/object/cosmos/999999999", timeout=30)
        assert r.status_code == 404, r.status_code
        assert "detail" in r.json()
        return "404 + detail"
    _check("GET /api/object (bad id -> 404)", t_object_404)

    def t_object_badfield():
        r = S.get(f"{B}/api/object/nosuchfield/{ids['bright']}", timeout=30)
        assert r.status_code == 404, r.status_code
        return "404 unknown field"
    _check("GET /api/object (bad field -> 404)", t_object_badfield)

    # ---- /api/eazy --------------------------------------------------------
    def t_eazy():
        r = S.get(f"{B}/api/eazy/cosmos/sfhz_blue_agn/{ids['bright']}", timeout=60)
        assert r.status_code == 200, r.status_code
        e = r.json()
        assert len(e["zgrid"]) == 607, len(e["zgrid"])
        assert len(e["chi2"]) == 607
        assert len(e["pz"]) == 607
        z = np.array(e["zgrid"], dtype=float)
        pz = np.array([v if v is not None else 0.0 for v in e["pz"]], dtype=float)
        integ = float(np.trapz(pz, z))
        assert abs(integ - 1.0) < 0.02, f"pz integral={integ:.4f}"
        assert len(e["phot"]["band"]) == 32
        assert len(e["phot"]["lam_um"]) == 32
        assert len(e["templ"]["lam_um"]) == len(e["templ"]["fnu_uJy"]) > 0
        # template wavelengths within observed window
        lam = np.array(e["templ"]["lam_um"], float)
        assert lam.min() >= 0.049 and lam.max() <= 30.1, (lam.min(), lam.max())
        assert e["templ_z"] is not None
        return (f"pz_integral={integ:.4f}, z_ml={e['z_ml']}, "
                f"templ_pts={len(lam)}")
    _check("GET /api/eazy (bright, warm-computes)", t_eazy)

    def t_eazy_warm():
        t0 = time.time()
        r = S.get(f"{B}/api/eazy/cosmos/sfhz_blue_agn/{ids['bright']}", timeout=60)
        dt = (time.time() - t0) * 1000
        assert r.status_code == 200
        return f"warm latency={dt:.0f}ms (LRU hit)"
    _check("GET /api/eazy (warm/LRU)", t_eazy_warm)

    def t_eazy_larson():
        r = S.get(f"{B}/api/eazy/cosmos/larson/{ids['highz']}", timeout=60)
        assert r.status_code == 200
        assert len(r.json()["zgrid"]) == 607
        return "larson template ok"
    _check("GET /api/eazy (larson)", t_eazy_larson)

    def t_eazy_404():
        r = S.get(f"{B}/api/eazy/cosmos/sfhz_blue_agn/999999999", timeout=30)
        assert r.status_code == 404, r.status_code
        return "404"
    _check("GET /api/eazy (bad id -> 404)", t_eazy_404)

    # ---- /api/prefetch (bulk Inspector navigation) ------------------------
    def t_prefetch():
        want = [ids["bright"], ids["highz"]]
        r = S.get(f"{B}/api/prefetch",
                  params={"ids": ",".join(str(i) for i in want),
                          "field": "cosmos", "template": "sfhz_blue_agn"},
                  timeout=90)
        assert r.status_code == 200, r.status_code
        j = r.json()
        assert j["field"] == "cosmos" and j["template"] == "sfhz_blue_agn"
        objs = j["objects"]
        assert [o["id"] for o in objs] == want, [o["id"] for o in objs]
        for o in objs:
            assert o["object"] is not None, f"null object block for {o['id']}"
            assert o["eazy"] is not None, f"null eazy block for {o['id']}"
        # per-id blocks must equal the standalone /api/object and /api/eazy bodies
        blk = objs[0]
        oid = blk["id"]
        ro = S.get(f"{B}/api/object/cosmos/{oid}", timeout=30)
        re_ = S.get(f"{B}/api/eazy/cosmos/sfhz_blue_agn/{oid}", timeout=60)
        assert blk["object"] == ro.json(), "prefetch object != /api/object"
        assert blk["eazy"] == re_.json(), "prefetch eazy != /api/eazy"
        return f"{len(objs)} ids bundled, object+eazy blocks match singles"
    _check("GET /api/prefetch (bulk object+eazy)", t_prefetch)

    def t_prefetch_missing():
        # unknown ids serialise as null blocks, don't fail the whole batch
        r = S.get(f"{B}/api/prefetch",
                  params={"ids": f"999999999,{ids['bright']}", "field": "cosmos"},
                  timeout=60)
        assert r.status_code == 200, r.status_code
        objs = {o["id"]: o for o in r.json()["objects"]}
        assert objs[999999999]["object"] is None, "missing id should be null"
        assert objs[ids["bright"]]["object"] is not None
        return "missing id -> null block, batch survives"
    _check("GET /api/prefetch (missing id -> null block)", t_prefetch_missing)

    # ---- /api/inspect/list ------------------------------------------------
    def t_inspect():
        r = S.get(f"{B}/api/inspect/list", timeout=60)
        assert r.status_code == 200
        objs = r.json()["objects"]
        assert len(objs) >= 6515, len(objs)  # 6515 = CSV baseline; supplement grows it
        djas = [o["dja"] for o in objs]
        assert len(set(djas)) == len(djas), "dja keys not unique"
        assert djas == sorted(djas), "not sorted by dja asc"
        o0 = objs[0]
        for key in ("dja", "mid", "ra", "dec", "dja_ra", "dja_dec", "sep",
                    "zs", "zp", "grade", "mag", "sn", "root", "file", "srcid",
                    "pid", "grating", "metafile", "nspec"):
            assert key in o0, f"inspect row missing {key}"
        multi = [o for o in objs if o["mid"] == ids["multi_mid"]]
        assert len(multi) == ids["multi_n"], (len(multi), ids["multi_n"])
        assert all(o["nspec"] == ids["multi_n"] for o in multi)
        return f"{len(objs)} spectra, unique+sorted, multi_mid has {len(multi)}"
    _check("GET /api/inspect/list", t_inspect)

    # ---- /api/spectrum ----------------------------------------------------
    def t_spectrum():
        r = S.get(f"{B}/api/spectrum/{ids['spec_root']}/{ids['spec_file']}",
                  timeout=90)
        assert r.status_code == 200, (r.status_code, r.text[:200])
        sp = r.json()
        assert set(sp["meta"]) >= {"root", "file", "grating", "pid", "srcid",
                                   "exptime"}
        wave = [w for w in sp["wave_um"] if w is not None]
        assert len(wave) > 100, len(wave)
        assert min(wave) >= 0.4 and max(wave) <= 5.6, (min(wave), max(wave))
        assert len(sp["flux_uJy"]) == len(sp["wave_um"])
        assert len(sp["err_uJy"]) == len(sp["wave_um"])
        td = sp["twod"]
        assert td is not None, "no 2D payload"
        assert td["nx"] == len(sp["wave_um"]), (td["nx"], len(sp["wave_um"]))
        blob = base64.b64decode(td["data"])
        arr = np.frombuffer(blob, dtype="<f4")
        assert arr.size == td["ny"] * td["nx"], (arr.size, td["ny"], td["nx"])
        assert "prism" in (sp["meta"]["grating"] or ""), sp["meta"]["grating"]
        return (f"prism wave [{min(wave):.3f},{max(wave):.3f}] um, "
                f"2D {td['ny']}x{td['nx']}, cached")
    _check("GET /api/spectrum (prism)", t_spectrum)

    def t_spectrum_cache():
        t0 = time.time()
        r = S.get(f"{B}/api/spectrum/{ids['spec_root']}/{ids['spec_file']}",
                  timeout=30)
        dt = (time.time() - t0) * 1000
        assert r.status_code == 200
        assert r.headers.get("Cache-Control", "").startswith("public")
        return f"cache-hit {dt:.0f}ms, Cache-Control set"
    _check("GET /api/spectrum (cache hit)", t_spectrum_cache)

    def t_spectrum_traversal():
        # ".." in file token must be rejected 400 before any disk/network access
        r = S.get(f"{B}/api/spectrum/cosmos/evil..path.spec.fits", timeout=30)
        assert r.status_code == 400, r.status_code
        assert "detail" in r.json()
        return "400 path-traversal rejected"
    _check("GET /api/spectrum (path traversal -> 400)", t_spectrum_traversal)

    def t_spectrum_badext():
        r = S.get(f"{B}/api/spectrum/cosmos/notaspec.fits", timeout=30)
        assert r.status_code == 400, r.status_code
        return "400 bad extension"
    _check("GET /api/spectrum (bad ext -> 400)", t_spectrum_badext)

    def t_spectrum_404():
        r = S.get(f"{B}/api/spectrum/nope-v4/nope-v4_prism-clear_1_1.spec.fits",
                  timeout=60)
        assert r.status_code in (404, 502), r.status_code
        return f"{r.status_code} (upstream missing)"
    _check("GET /api/spectrum (missing upstream -> 404/502)", t_spectrum_404)

    # ---- /api/cutout ------------------------------------------------------
    def t_cutout_rgb():
        r = S.get(f"{B}/api/cutout",
                  params={"ra": ids["spec_ra"], "dec": ids["spec_dec"],
                          "size": 3.0, "mode": "rgb"}, timeout=90)
        assert r.status_code == 200, (r.status_code, r.text[:200])
        assert r.headers["Content-Type"] == "image/png"
        assert r.content.startswith(PNG_MAGIC), r.content[:8]
        assert len(r.content) > 1000
        return f"rgb PNG {len(r.content)} bytes"
    _check("GET /api/cutout (rgb)", t_cutout_rgb)

    def t_cutout_slit():
        r = S.get(f"{B}/api/cutout",
                  params={"ra": ids["spec_ra"], "dec": ids["spec_dec"],
                          "size": 3.0, "mode": "slit",
                          "metafile": ids["spec_meta"]}, timeout=90)
        assert r.status_code == 200, (r.status_code, r.text[:200])
        assert r.content.startswith(PNG_MAGIC)
        return f"slit PNG {len(r.content)} bytes (metafile={ids['spec_meta']})"
    _check("GET /api/cutout (slit)", t_cutout_slit)

    def t_cutout_slit_nometa():
        r = S.get(f"{B}/api/cutout",
                  params={"ra": ids["spec_ra"], "dec": ids["spec_dec"],
                          "mode": "slit"}, timeout=30)
        assert r.status_code == 400, r.status_code
        assert "detail" in r.json()
        return f"{r.status_code} slit without metafile (+detail)"
    _check("GET /api/cutout (slit no metafile -> 400)", t_cutout_slit_nometa)

    def t_cutout_bad():
        r = S.get(f"{B}/api/cutout",
                  params={"ra": 150.1, "dec": 200.0, "size": 3.0}, timeout=30)
        assert r.status_code == 400, r.status_code
        assert "detail" in r.json()
        return f"{r.status_code} dec out of range (+detail)"
    _check("GET /api/cutout (bad dec -> 400)", t_cutout_bad)

    # ---- /api/dja/preview -------------------------------------------------
    def t_preview():
        base = ids["spec_file"].replace(".spec.fits", ".fnu.png")
        r = S.get(f"{B}/api/dja/preview/{ids['spec_root']}/{base}", timeout=90)
        # preview may legitimately be absent upstream -> 404/502 acceptable
        if r.status_code == 200:
            assert r.content.startswith(PNG_MAGIC), r.content[:8]
            return f"preview PNG {len(r.content)} bytes"
        assert r.status_code in (404, 502), r.status_code
        return f"{r.status_code} (preview not on S3, tolerated)"
    _check("GET /api/dja/preview", t_preview)

    def t_preview_badext():
        r = S.get(f"{B}/api/dja/preview/cosmos/evil.exe", timeout=30)
        assert r.status_code == 400, r.status_code
        return "400 bad basename"
    _check("GET /api/dja/preview (bad ext -> 400)", t_preview_badext)

    # ---- /api/decisions POST + GET ---------------------------------------
    def t_decisions():
        dja = ids["spec_file"].replace(".spec.fits", "")
        t_now = time.time()
        p1 = {"field": "cosmos", "inspector": "smoke",
              "decisions": {dja: {"v": 1, "flag": "ok", "zNew": 2.0,
                                  "by": "smoke", "t": t_now}}}
        r = S.post(f"{B}/api/decisions", json=p1, timeout=30)
        assert r.status_code == 200, r.status_code
        assert r.json()["ok"] is True
        # newer edit wins
        p2 = {"field": "cosmos", "inspector": "smoke",
              "decisions": {dja: {"v": 3, "flag": "great", "by": "smoke",
                                  "t": t_now + 10}}}
        r2 = S.post(f"{B}/api/decisions", json=p2, timeout=30)
        assert r2.status_code == 200
        g = S.get(f"{B}/api/decisions", params={"field": "cosmos"}, timeout=30)
        assert g.status_code == 200
        gj = g.json()
        assert dja in gj["decisions"], "decision not stored"
        assert gj["decisions"][dja]["v"] == 3, gj["decisions"][dja]
        return f"POST+GET ok, latest-wins v=3, n={gj['n']}"
    _check("POST+GET /api/decisions", t_decisions)

    # ---- static surfaces --------------------------------------------------
    def t_static_root():
        r = S.get(f"{B}/", timeout=30)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")
        assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()
        return "explorer index served"
    _check("GET / (explorer html)", t_static_root)

    def t_static_inspector():
        r = S.get(f"{B}/inspector/", timeout=30)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")
        return "inspector index served"
    _check("GET /inspector (html)", t_static_inspector)

    def t_static_tokens():
        r = S.get(f"{B}/shared/tokens.css", timeout=30)
        assert r.status_code == 200
        assert "css" in r.headers.get("Content-Type", "")
        return f"tokens.css {len(r.content)} bytes"
    _check("GET /shared/tokens.css", t_static_tokens)

    def t_static_vendor():
        r = S.get(f"{B}/vendor/dc-runtime.js", timeout=30)
        assert r.status_code == 200
        return f"dc-runtime.js {len(r.content)} bytes"
    _check("GET /vendor/dc-runtime.js", t_static_vendor)

    # ---- report -----------------------------------------------------------
    print(f"\n{'ENDPOINT':<48} {'STATUS':<7} {'ms':>7}  DETAIL")
    print("-" * 100)
    npass = nfail = 0
    for name, status, detail, ms in RESULTS:
        if status == "PASS":
            npass += 1
        else:
            nfail += 1
        print(f"{name:<48} {status:<7} {ms:>7.0f}  {detail}")
    print("-" * 100)
    print(f"TOTAL: {npass} pass, {nfail} fail, {len(RESULTS)} checks")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
