"""Env-overridable settings singleton for the MINERVA viewer server.

Interface (ARCHITECTURE.md "Module interfaces"):
    from server.config import settings
    settings.index_dir   # MINERVA_INDEX_DIR
    settings.cache_dir   # MINERVA_CACHE_DIR
    settings.host        # MINERVA_HOST
    settings.port        # MINERVA_PORT
    settings.fields_dir  # MINERVA_FIELDS_DIR
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Directory holding this file: <repo>/app/server/
_SERVER_DIR = Path(__file__).resolve().parent

_DEFAULT_INDEX_DIR = "/otdata2/themiya/minerva/data/viewer_index"
_DEFAULT_CACHE_DIR = "/otdata2/themiya/minerva/data/cache"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8321
_DEFAULT_FIELDS_DIR = str(_SERVER_DIR / "fields")


@dataclass(frozen=True)
class Settings:
    index_dir: Path
    cache_dir: Path
    host: str
    port: int
    fields_dir: Path

    @property
    def spectra_cache_dir(self) -> Path:
        return self.cache_dir / "spectra"

    @property
    def cutouts_cache_dir(self) -> Path:
        return self.cache_dir / "cutouts"

    @property
    def decisions_cache_dir(self) -> Path:
        return self.cache_dir / "decisions"


def _load() -> Settings:
    return Settings(
        index_dir=Path(os.environ.get("MINERVA_INDEX_DIR", _DEFAULT_INDEX_DIR)),
        cache_dir=Path(os.environ.get("MINERVA_CACHE_DIR", _DEFAULT_CACHE_DIR)),
        host=os.environ.get("MINERVA_HOST", _DEFAULT_HOST),
        port=int(os.environ.get("MINERVA_PORT", _DEFAULT_PORT)),
        fields_dir=Path(os.environ.get("MINERVA_FIELDS_DIR", _DEFAULT_FIELDS_DIR)),
    )


# Singleton, read once at import.
settings = _load()
