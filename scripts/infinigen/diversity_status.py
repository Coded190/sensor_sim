"""Record which full-quality seeds exist. Fast solves are listed and excluded."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("dataset/qualification/asset_realism/source_runs")
REPORT = Path("dataset/qualification/asset_realism/reports/room_diversity.json")
PLANNED = [
    ("LivingRoom", 21),
    ("Kitchen", 31),
    ("Bedroom", 41),
    ("DiningRoom", 51),
    ("Bathroom", 61),
    ("Office", 71),
    ("LivingRoom", 22),
    ("LivingRoom", 23),
    ("Kitchen", 32),
    ("Kitchen", 33),
    ("Bedroom", 42),
    ("Bedroom", 43),
    ("DiningRoom", 52),
    ("DiningRoom", 53),
    ("Bathroom", 62),
    ("Bathroom", 63),
    ("Office", 72),
    ("Office", 73),
]


ALIASES = {("LivingRoom", 21): ROOT / "LivingRoom_full_21_retry"}


def main() -> None:
    rows = []
    for room, seed in PLANNED:
        directory = ALIASES.get((room, seed), ROOT / f"{room}_full_{seed}")
        blend = directory / "scene.blend"
        log = directory / "solver_logs" / "coarse.log"
        fast_hits = 0
        if log.is_file():
            fast_hits = log.read_text(encoding="utf-8", errors="replace").count("fast_solve")
        rows.append({
            "room": room,
            "seed": seed,
            "blend": blend.is_file(),
            "blend_bytes": blend.stat().st_size if blend.is_file() else 0,
            "fast_solve_mentions_in_log": fast_hits,
            "counts_as_full": blend.is_file() and fast_hits == 0,
        })
    summary = {
        "planned": len(rows),
        "full_blends": sum(1 for row in rows if row["counts_as_full"]),
        "definition": "A scene counts when scene.blend exists and the coarse log does not mention fast_solve. Different seeds are not diversity until dimensions, counts, and hashes are compared.",
        "rows": rows,
    }
    REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("planned", "full_blends")}))


if __name__ == "__main__":
    main()
