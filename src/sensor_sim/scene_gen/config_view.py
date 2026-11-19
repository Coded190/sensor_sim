"""Read-only view of a scene spec. This does not change generation."""

from __future__ import annotations

import hashlib
import json


def config_view(spec: dict) -> dict:
    """Split one spec into assets, scene placement, and sensor settings."""
    return {
        "assets": {
            "families": list(spec.get("assets", [])),
            "strata": spec.get("strata"),
        },
        "scene": {
            "mode": spec.get("mode"),
            "seed": spec.get("seed"),
            "num_objects": spec.get("num_objects"),
            "materials": list(spec.get("materials", [])),
            "gap": spec.get("gap"),
            "placement_attempts": spec.get("placement_attempts"),
        },
        "sensor": {
            "resolution": spec.get("resolution"),
            "spp": spec.get("spp"),
            "max_depth": spec.get("max_depth"),
            "variant": spec.get("variant"),
            "camera": spec.get("camera"),
            "light": spec.get("light"),
        },
    }


def config_view_hash(spec: dict) -> str:
    payload = json.dumps(config_view(spec), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
