"""Recorded physics settings for the offline bridge.

These are test parameters. They are not measured material properties and they
are not written into ``asset_extension_v1``.
"""

from __future__ import annotations

import hashlib

from sensor_sim.scene_gen.asset_package import canonical_json_bytes

TIMESTEP_S = 1.0 / 120.0
SAMPLE_HZ = 60
SAMPLE_EVERY_STEPS = 2
DURATION_S = 0.5
INITIAL_VELOCITY_M_S = [0.20, 0.0, 0.0]
STATIC_FRICTION = 0.05
DYNAMIC_FRICTION = 0.05
RESTITUTION = 0.0
FRICTION_COMBINE_MODE = "average"
SNAPSHOTS_S = {"t0": 0.0, "t1": 0.10, "t2": 0.40}
DENSITY_KG_M3 = 400.0
DENSITY_PROVENANCE = "simulation_default_not_a_measurement"


def physics_document() -> dict:
    return {
        "timestep_s": TIMESTEP_S,
        "sample_hz": SAMPLE_HZ,
        "sample_every_steps": SAMPLE_EVERY_STEPS,
        "duration_s": DURATION_S,
        "initial_velocity_m_s": list(INITIAL_VELOCITY_M_S),
        "initial_velocity_frame": "canonical_world",
        "static_friction": STATIC_FRICTION,
        "dynamic_friction": DYNAMIC_FRICTION,
        "restitution": RESTITUTION,
        "friction_combine_mode": FRICTION_COMBINE_MODE,
        "snapshots_s": dict(SNAPSHOTS_S),
        "density_kg_m3": DENSITY_KG_M3,
        "density_provenance": DENSITY_PROVENANCE,
        "gravity_m_s2": [0.0, 0.0, -9.81],
    }


BRIDGE_REVISION = "2-enable-cameras-visual-collision-off"


def run_hash(manifest_sha256: str, catalog_sha256: str, backend_versions: dict) -> str:
    """Identity of one physics run. A different config or backend must not overwrite another."""
    payload = {
        "manifest_sha256": manifest_sha256,
        "catalog_sha256": catalog_sha256,
        "physics": physics_document(),
        "backend_versions": backend_versions,
        "bridge_revision": BRIDGE_REVISION,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
