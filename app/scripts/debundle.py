#!/usr/bin/env python3
"""De-bundle the DJA Redshift Inspector prototype into plain served static files.

The handoff prototype (``handoff/index.html``) is a "Classical" ds-bundle: a JSON
manifest of gzip+base64 resources (React 18.3.1 UMD, ReactDOM UMD, the dc-runtime
framework, 12 embedded woff2 fonts) wrapping a JSON-escaped HTML template.  The
decoded parts are already checked in under ``app/web/_bundle_src/`` (see
``extract.py`` there for the decode recipe).

This script turns those decoded parts into a *plain* static page that a browser
can serve and an engineer can edit:

  vendor/react.js  vendor/react-dom.js  vendor/dc-runtime.js   (verbatim UMD/runtime)
  shared/fonts/*.woff2  +  shared/fonts.css                     (Cormorant Garamond + Lora)
  shared/tokens.css                                             (:root design tokens + component classes)
  inspector/app.js         (verbatim logic class — the editable data-layer seam)
  inspector/template.html  (the <x-dc> {{ }}-bound markup)
  inspector/inspector.css  (inspector-only page/layout styles)
  inspector/index.html     (ASSEMBLED from the four sources above — reproducible)

Why this works without the manifest: the dc-runtime locates its markup by
``document.querySelector("x-dc")`` and its logic by ``script[data-dc-script]``
(see ``parseDcDocument`` in the runtime).  It only reaches for the manifest /
unpkg CDN inside ``loadReactUmd()`` when ``window.React``/``window.ReactDOM`` are
missing.  We load the vendored React UMD bundles *before* the runtime, so those
globals exist and no network fetch happens.

app.js and template.html are the maintained sources: they are extracted once
(create-if-missing) and never overwritten unless ``--force-sources`` is given, so
the next engineer's edits to the data layer survive a rebuild.  index.html and
the derived vendor/shared assets are always regenerated from disk.

Usage:
    python3 scripts/debundle.py            # build everything
    python3 scripts/debundle.py --force-sources   # also re-extract app.js/template.html
    python3 scripts/debundle.py --check    # verify emitted HTML references resolve
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent          # .../app
SRC = APP / "web" / "_bundle_src"
WEB = APP / "web"
VENDOR = WEB / "vendor"
SHARED = WEB / "shared"
FONTS_DIR = SHARED / "fonts"
INSPECTOR = WEB / "inspector"

FAMILY_SLUG = {"Cormorant Garamond": "CormorantGaramond", "Lora": "Lora"}


# ----------------------------------------------------------------------------- vendor
def classify_out(path: Path) -> str | None:
    """Sniff a decoded .out file: 'react', 'react-dom', 'runtime', 'font', or None."""
    head = path.read_bytes()[:4096]
    if head[:4] == b"wOF2":
        return "font"
    text = head.decode("latin-1", "replace")
    if "react-dom.production.min.js" in text:
        return "react-dom"
    if "react.production.min.js" in text:
        return "react"
    if "dc-runtime" in text or "StreamableLogic" in text or "parseDcDocument" in text:
        return "runtime"
    return None


def build_vendor() -> dict:
    VENDOR.mkdir(parents=True, exist_ok=True)
    found = {}
    for out in sorted(SRC.glob("*.out")):
        kind = classify_out(out)
        if kind in ("react", "react-dom", "runtime"):
            found[kind] = out
    missing = {"react", "react-dom", "runtime"} - set(found)
    if missing:
        raise SystemExit(f"debundle: could not locate vendor bundles: {sorted(missing)}")
    mapping = {"react": "react.js", "react-dom": "react-dom.js", "runtime": "dc-runtime.js"}
    written = []
    for kind, out in found.items():
        dst = VENDOR / mapping[kind]
        shutil.copyfile(out, dst)
        if kind == "runtime":
            patch_runtime_svg_text(dst)
        written.append(dst)
    return {"written": written, "runtime_uuid": found["runtime"].stem}


# --------------------------------------------------------------------- runtime patch
# MINERVA-SVG-TEXT-FIX — the vendored dc-runtime wraps every interpolated {{ }} text
# value in an HTML <span class="sc-interp">. React namespaces that span into SVG when
# the nearest host ancestor is <svg>, and an SVG-namespaced <span> is a non-rendering
# unknown element — so every bound axis-tick / marker / band label in the Inspector's
# SVG panels (scatter, 1D spectrum, SED, p(z)) is invisible while literal captions and
# childless shapes render fine. This re-applies the fix on every rebuild so `debundle`
# never silently clobbers it. Keep in lockstep with the same edits in web/vendor/dc-runtime.js.
_SVG_TEXT_MARKER = "/*MINERVA-SVG-TEXT-FIX*/"
_SVG_TEXT_PATCHES = [
    (
        "  function walkText(node) {\n"
        "    const txt = node.nodeValue ?? \"\";\n"
        "    if (!txt.includes(\"{{\")) {\n"
        "      if (!txt.trim() && !txt.includes(\" \")) return null;\n"
        "      return () => txt;\n"
        "    }\n"
        "    const parts = txt.split(/\\{\\{([\\s\\S]+?)\\}\\}/g);",
        "  function isSvgTextCtx(node) {" + _SVG_TEXT_MARKER + "\n"
        "    let el = node.parentNode;\n"
        "    while (el && el.nodeType === 1) {\n"
        "      const ln = el.localName;\n"
        "      if (ln === \"foreignObject\") return false;\n"
        "      if (ln === \"svg\") return true;\n"
        "      el = el.parentNode;\n"
        "    }\n"
        "    return false;\n"
        "  }\n"
        "  function walkText(node) {\n"
        "    const txt = node.nodeValue ?? \"\";\n"
        "    if (!txt.includes(\"{{\")) {\n"
        "      if (!txt.trim() && !txt.includes(\" \")) return null;\n"
        "      return () => txt;\n"
        "    }\n"
        "    const svgCtx = isSvgTextCtx(node);" + _SVG_TEXT_MARKER + "\n"
        "    const parts = txt.split(/\\{\\{([\\s\\S]+?)\\}\\}/g);",
    ),
    (
        "              return h(\n"
        "                \"span\",\n"
        "                { key: i, className: \"sc-interp sc-unresolved\" },\n"
        "                \"{{ \" + p.trim() + \" }}\"\n"
        "              );",
        "              return svgCtx ? \"{{ \" + p.trim() + \" }}\" : h(\n"
        "                \"span\",\n"
        "                { key: i, className: \"sc-interp sc-unresolved\" },\n"
        "                \"{{ \" + p.trim() + \" }}\"\n"
        "              );",
    ),
    (
        "          return h(\n"
        "            \"span\",\n"
        "            { key: i, className: \"sc-interp sc-missing\" },\n"
        "            p.trim()\n"
        "          );",
        "          return svgCtx ? p.trim() : h(\n"
        "            \"span\",\n"
        "            { key: i, className: \"sc-interp sc-missing\" },\n"
        "            p.trim()\n"
        "          );",
    ),
    (
        "        return h(\"span\", { key: i, className: \"sc-interp\" }, String(v));",
        "        return svgCtx ? String(v) : h(\"span\", { key: i, className: \"sc-interp\" }, String(v));" + _SVG_TEXT_MARKER,
    ),
]


def patch_runtime_svg_text(dst: Path) -> None:
    """Idempotently re-apply the SVG interpolated-text fix to the copied runtime."""
    text = dst.read_text()
    if _SVG_TEXT_MARKER in text:
        return  # already patched (source already carried it)
    for old, new in _SVG_TEXT_PATCHES:
        if old not in text:
            raise SystemExit(
                "debundle: MINERVA-SVG-TEXT-FIX anchor not found in dc-runtime — "
                "the vendored runtime changed; re-derive the patch against the new source."
            )
        text = text.replace(old, new, 1)
    dst.write_text(text)


# ----------------------------------------------------------------------------- css split
FONTFACE_RE = re.compile(r"@font-face\s*\{(.*?)\}", re.S)
COMMENT_BEFORE_RE = re.compile(r"/\*\s*([a-z0-9\- ]+?)\s*\*/\s*$")


def read_template_css() -> tuple[str, str, str, str]:
    """Return (fonts_css, tokens_css, layout_css, markup) parsed from template.html."""
    tpl = (SRC / "template.html").read_text()

    # The <helmet> holds two <style> blocks: [0]=fonts+tokens+components, [1]=layout.
    helmet = re.search(r"<helmet>(.*?)</helmet>", tpl, re.S).group(1)
    styles = re.findall(r"<style>(.*?)</style>", helmet, re.S)
    if len(styles) != 2:
        raise SystemExit(f"debundle: expected 2 <style> blocks in helmet, got {len(styles)}")
    css_main, layout_css = styles[0], styles[1]

    root_at = css_main.index(":root")
    fonts_css = css_main[:root_at]
    tokens_css = css_main[root_at:]

    # Markup = everything in <x-dc> after </helmet>, before </x-dc>.
    xdc = re.search(r"<x-dc>(.*)</x-dc>", tpl, re.S).group(1)
    markup = xdc.split("</helmet>", 1)[1].strip()

    return fonts_css, tokens_css.strip(), layout_css.strip(), markup


def parse_font_map(fonts_css: str) -> dict[str, str]:
    """Map each font-resource uuid -> filename slug (Family-subset.woff2)."""
    uuid_slug: dict[str, str] = {}
    for m in FONTFACE_RE.finditer(fonts_css):
        block = m.group(1)
        pre = fonts_css[: m.start()]
        cm = COMMENT_BEFORE_RE.search(pre)
        subset = cm.group(1).strip() if cm else "font"
        fam = re.search(r"font-family:\s*'([^']+)'", block).group(1)
        uuid = re.search(r'url\("([0-9a-f\-]+)"\)', block).group(1)
        slug = f"{FAMILY_SLUG.get(fam, fam.replace(' ', ''))}-{subset}"
        uuid_slug.setdefault(uuid, slug)
    return uuid_slug


def build_fonts(fonts_css: str) -> list[Path]:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    uuid_slug = parse_font_map(fonts_css)
    if len(uuid_slug) != 12:
        print(f"debundle: warning — expected 12 fonts, mapped {len(uuid_slug)}", file=sys.stderr)
    written = []
    css = fonts_css
    for uuid, slug in uuid_slug.items():
        src = SRC / f"{uuid}.out"
        dst = FONTS_DIR / f"{slug}.woff2"
        shutil.copyfile(src, dst)
        written.append(dst)
        css = css.replace(f'url("{uuid}")', f'url("/shared/fonts/{slug}.woff2")')
    header = (
        "/* Cormorant Garamond + Lora @font-face rules — de-bundled from the\n"
        "   Classical prototype. Weights 400/600, self-hosted woff2 (no network).\n"
        "   Shared by /inspector and /explorer. Generated by scripts/debundle.py. */\n\n"
    )
    out = SHARED / "fonts.css"
    out.write_text(header + css.strip() + "\n")
    written.append(out)
    return written


def build_tokens(tokens_css: str) -> Path:
    header = (
        "/* Classical design-system tokens + component classes — de-bundled from the\n"
        "   prototype helmet <style>. Source of truth for the shared look; imported by\n"
        "   BOTH /inspector and /explorer. Generated by scripts/debundle.py. */\n\n"
    )
    out = SHARED / "tokens.css"
    out.write_text(header + tokens_css + "\n")
    return out


# ----------------------------------------------------------------------------- sources
def extract_sources(markup: str, layout_css: str, force: bool) -> list[Path]:
    INSPECTOR.mkdir(parents=True, exist_ok=True)
    written = []

    appjs_src = (SRC / "app.js").read_text()
    appjs_dst = INSPECTOR / "app.js"
    if force or not appjs_dst.exists():
        appjs_dst.write_text(appjs_src)
        written.append(appjs_dst)

    tpl_dst = INSPECTOR / "template.html"
    if force or not tpl_dst.exists():
        banner = (
            "<!-- Redshift Inspector markup (dc-runtime {{ }} bindings).\n"
            "     Maintained source — assembled into index.html by scripts/debundle.py.\n"
            "     The <script data-dc-script> logic lives in app.js. -->\n"
        )
        tpl_dst.write_text(banner + markup + "\n")
        written.append(tpl_dst)

    css_dst = INSPECTOR / "inspector.css"
    css_header = "/* Inspector-only page/layout styles (full-height, no body scroll). */\n"
    css_dst.write_text(css_header + layout_css + "\n")
    written.append(css_dst)

    return written


# ----------------------------------------------------------------------------- assemble
def read_data_props() -> str:
    tpl = (SRC / "template.html").read_text()
    m = re.search(r'data-dc-script=""\s+data-props="([^"]*)"', tpl)
    return m.group(1) if m else ""


def strip_banner(text: str) -> str:
    # Drop a leading HTML comment banner (our own) so it is not double-nested.
    return re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.S)


def _asset_hash(url_path: str) -> str:
    """8-char content hash of a /vendor/... or /shared/... asset on disk."""
    p = WEB / url_path.lstrip("/")
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


def version_refs(html: str) -> str:
    """Rewrite immutable-mounted asset URLs to content-hashed ?v= forms.

    /vendor and /shared are served with Cache-Control: immutable, so any change
    to those files (e.g. the dc-runtime SVG text patch) MUST change the URL or
    browsers keep executing the stale cached copy forever. Idempotent."""
    return re.sub(r'(?:src|href)="((?:/vendor|/shared)/[^"?]+)(?:\?v=[0-9a-f]+)?"',
                  lambda m: f'{"src" if m.group(0).startswith("src") else "href"}='
                            f'"{m.group(1)}?v={_asset_hash(m.group(1))}"',
                  html)


def version_explorer() -> None:
    """Apply the same ?v= content-hash rewrite to the explorer page in place."""
    idx = WEB / "explorer" / "index.html"
    if idx.exists():
        idx.write_text(version_refs(idx.read_text()))


def assemble_index() -> Path:
    markup = strip_banner((INSPECTOR / "template.html").read_text()).strip()
    appjs = (INSPECTOR / "app.js").read_text()
    data_props = read_data_props()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DJA Redshift Inspector · MINERVA</title>
<!-- GENERATED by scripts/debundle.py from inspector/template.html + inspector/app.js.
     Do not edit this file directly; edit the sources and re-run the script. -->
<link rel="stylesheet" href="/shared/fonts.css">
<link rel="stylesheet" href="/shared/tokens.css">
<link rel="stylesheet" href="/inspector/inspector.css">
<!-- React globals must exist before dc-runtime boots, so it never reaches for the CDN. -->
<script src="/vendor/react.js"></script>
<script src="/vendor/react-dom.js"></script>
<script src="/vendor/dc-runtime.js"></script>
</head>
<body>
<x-dc>
{markup}
</x-dc>
<script type="text/x-dc" data-dc-script="" data-props="{data_props}">
{appjs}
</script>
</body>
</html>
"""
    out = INSPECTOR / "index.html"
    out.write_text(version_refs(html))
    version_explorer()
    return out


# ----------------------------------------------------------------------------- check
ASSET_REF_RE = re.compile(r'(?:src|href)="(/[^"]+)"')


def check_index() -> int:
    """Parse index.html for local src/href and stat each target under web/."""
    idx = INSPECTOR / "index.html"
    if not idx.exists():
        print("check: index.html missing", file=sys.stderr)
        return 1
    refs = ASSET_REF_RE.findall(idx.read_text())
    bad = 0
    for ref in refs:
        target = WEB / ref.split("?")[0].lstrip("/")  # ?v= cache-buster is not part of the path
        ok = target.is_file()
        size = target.stat().st_size if ok else 0
        print(f"  [{'OK ' if ok else 'MISS'}] {ref:38s} {size:>8d} B  -> {target}")
        if not ok:
            bad += 1
    print(f"check: {len(refs)} references, {bad} missing")
    return 1 if bad else 0


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-sources", action="store_true",
                    help="re-extract app.js and template.html, overwriting local edits")
    ap.add_argument("--check", action="store_true", help="only verify index.html references resolve")
    args = ap.parse_args()

    if args.check:
        return check_index()

    fonts_css, tokens_css, layout_css, markup = read_template_css()

    vendor = build_vendor()
    fonts = build_fonts(fonts_css)
    tokens = build_tokens(tokens_css)
    sources = extract_sources(markup, layout_css, args.force_sources)
    index = assemble_index()

    print("vendor:")
    for p in vendor["written"]:
        print(f"  {p}  ({p.stat().st_size} B)")
    print(f"fonts: {len([f for f in fonts if f.suffix == '.woff2'])} woff2 + fonts.css")
    print(f"tokens: {tokens} ({tokens.stat().st_size} B)")
    print("sources:")
    for p in sources:
        print(f"  {p}")
    print(f"index: {index} ({index.stat().st_size} B)")
    print()
    rc = check_index()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
