"""Field manifest loader.

Interface (ARCHITECTURE.md "Module interfaces"):
    load_fields() -> dict[str, FieldConfig]

FieldConfig dataclass fields:
    name, title, catalog_path, eazy: dict[str, Path]  # template -> ZPiter dir,
    default_template, dja_catalog_path, cutout_filters: str, map_link: str

Manifests live in MINERVA_FIELDS_DIR (server/fields/*.json). Each JSON:
{
  "name": "cosmos",
  "title": "COSMOS",
  "catalog_path": "<abs path to SUPER_CATALOG.fits>",
  "eazy": {"sfhz_blue_agn": "<abs ZPiter dir>", "larson": "<abs ZPiter dir>"},
  "default_template": "sfhz_blue_agn",
  "dja_catalog_path": "<abs path to dja csv.gz>",
  "cutout_filters": "f090w-clear,...",
  "map_link": "https://minerva.colorado.edu/cosmos/?ra={ra}&dec={dec}&zoom=8"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class FieldConfig:
    name: str
    title: str
    catalog_path: Path
    eazy: dict[str, Path]  # template name -> ZPiter directory
    default_template: str
    dja_catalog_path: Path
    cutout_filters: str
    map_link: str
    # Optional explicit per-template product paths. When present these override
    # the ZPiter-dir glob below -- necessary because the eazy product filenames
    # are not uniform across fields (COSMOS: '*.eazy.h5'/'*.eazy.zout.fits';
    # EGS: '*.eazypy.h5'/'*_{template}.zout.fits'). Fields without these keys
    # (e.g. cosmos.json) keep working via the legacy glob.
    h5: dict[str, Path] = field(default_factory=dict)
    zout: dict[str, Path] = field(default_factory=dict)

    def zout_path(self, template: str) -> Path:
        """Resolve the zout.fits for a template: explicit manifest path first,
        else glob the ZPiter dir (legacy COSMOS naming)."""
        explicit = self.zout.get(template)
        if explicit is not None:
            return Path(explicit)
        d = self.eazy[template]
        hits = sorted(d.glob("*.eazy.zout.fits")) or sorted(d.glob("*.zout.fits"))
        if not hits:
            raise FileNotFoundError(
                f"no zout.fits in {d} and no explicit 'zout' path for {template!r}")
        return hits[0]

    def h5_path(self, template: str) -> Path:
        """Resolve the eazy h5 for a template: explicit manifest path first,
        else glob the ZPiter dir (legacy COSMOS naming)."""
        explicit = self.h5.get(template)
        if explicit is not None:
            return Path(explicit)
        d = self.eazy[template]
        hits = sorted(d.glob("*.eazy.h5")) or sorted(d.glob("*.eazypy.h5"))
        if not hits:
            raise FileNotFoundError(
                f"no eazy h5 in {d} and no explicit 'h5' path for {template!r}")
        return hits[0]


def _parse(raw: dict) -> FieldConfig:
    return FieldConfig(
        name=raw["name"],
        title=raw["title"],
        catalog_path=Path(raw["catalog_path"]),
        eazy={k: Path(v) for k, v in raw["eazy"].items()},
        default_template=raw["default_template"],
        dja_catalog_path=Path(raw["dja_catalog_path"]),
        cutout_filters=raw["cutout_filters"],
        map_link=raw["map_link"],
        h5={k: Path(v) for k, v in raw.get("h5", {}).items()},
        zout={k: Path(v) for k, v in raw.get("zout", {}).items()},
    )


def load_fields(fields_dir: Path | None = None) -> dict[str, FieldConfig]:
    d = Path(fields_dir) if fields_dir is not None else settings.fields_dir
    out: dict[str, FieldConfig] = {}
    for jf in sorted(Path(d).glob("*.json")):
        with open(jf) as fh:
            raw = json.load(fh)
        cfg = _parse(raw)
        out[cfg.name] = cfg
    return out
