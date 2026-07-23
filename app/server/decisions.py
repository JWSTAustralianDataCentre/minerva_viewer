"""Server-side QC decision store (append-only JSONL).

One JSONL line is appended per POST /api/decisions call to
  {cache_dir}/decisions/{field}.jsonl
Each line records the full payload plus a server receive timestamp.

Reading merges all lines by dja_id, latest `t` wins (the decision's own
timestamp, falling back to the line's server timestamp). The UI still keeps
localStorage; this store is a sync/backup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _decisions_path(field: str, cache_dir) -> Path:
    d = Path(cache_dir) / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    # guard against path traversal / weird field names
    safe = "".join(c for c in str(field) if c.isalnum() or c in "-_") or "cosmos"
    return d / f"{safe}.jsonl"


def append_decisions(field: str, payload: dict, cache_dir) -> int:
    """Append one JSONL line for this call; return number of decisions in it."""
    decisions = (payload or {}).get("decisions") or {}
    line = {
        "field": (payload or {}).get("field", field),
        "inspector": (payload or {}).get("inspector"),
        "t": time.time(),
        "decisions": decisions,
    }
    path = _decisions_path(field, cache_dir)
    with open(path, "a") as fh:
        fh.write(json.dumps(line, allow_nan=False) + "\n")
    return len(decisions)


def load_decisions(field: str, cache_dir) -> dict:
    """Merge every appended line by dja_id, latest `t` wins.

    Returns {"decisions": {dja_id: {...latest...}}, "n": int}.
    """
    path = _decisions_path(field, cache_dir)
    merged: dict[str, dict] = {}
    if not path.exists():
        return {"decisions": {}, "n": 0}

    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            line_t = line.get("t", 0)
            for dja_id, dec in (line.get("decisions") or {}).items():
                if not isinstance(dec, dict):
                    continue
                dec_t = dec.get("t", line_t)
                prev = merged.get(dja_id)
                prev_t = prev.get("__t", 0) if prev else float("-inf")
                if prev is None or dec_t >= prev_t:
                    rec = dict(dec)
                    rec["__t"] = dec_t
                    merged[dja_id] = rec

    for rec in merged.values():
        rec.pop("__t", None)
    return {"decisions": merged, "n": len(merged)}


# ---------------------------------------------------------------------------
# selftest: python -m server.decisions
# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    cache = Path(tempfile.mkdtemp(prefix="minerva_dec_"))
    field = "cosmos"

    n1 = append_decisions(field, {
        "field": field, "inspector": "jadc",
        "decisions": {
            "specA": {"v": 1, "flag": "ok", "zNew": 2.1, "comment": "clean",
                      "by": "jadc", "t": 100},
            "specB": {"v": 0, "flag": "bad", "by": "jadc", "t": 100},
        },
    }, cache)
    assert n1 == 2, n1

    # later edit to specA (higher t) should win; specC new
    n2 = append_decisions(field, {
        "field": field, "inspector": "jadc",
        "decisions": {
            "specA": {"v": 2, "flag": "great", "zNew": 2.15, "by": "jadc", "t": 200},
            "specC": {"v": 3, "by": "jadc", "t": 150},
        },
    }, cache)
    assert n2 == 2, n2

    # an older edit to specB (lower t than existing) must NOT overwrite
    append_decisions(field, {
        "field": field, "inspector": "other",
        "decisions": {"specB": {"v": 9, "by": "other", "t": 50}},
    }, cache)

    out = load_decisions(field, cache)
    print("merged:", json.dumps(out, indent=2))
    assert out["n"] == 3, out["n"]
    assert out["decisions"]["specA"]["v"] == 2       # latest t won
    assert out["decisions"]["specA"]["flag"] == "great"
    assert out["decisions"]["specB"]["v"] == 0       # older edit ignored
    assert out["decisions"]["specC"]["v"] == 3
    assert "__t" not in out["decisions"]["specA"]

    # empty field
    empty = load_decisions("nonexistent", cache)
    assert empty == {"decisions": {}, "n": 0}, empty

    print("\nALL DECISIONS SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
