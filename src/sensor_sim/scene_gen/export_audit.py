"""Check that an exported Mitsuba scene still contains every logical object.

Mitsuba 3.9.1 ``shape_index`` names a shape inside a shapegroup. It is not an
instance id. ``SurfaceInteraction.instance`` is a different pointer and is
``None`` when the hit is not an instance. A shapegroup has one BSDF, so two
logical objects that need different materials are exported as distinct shapes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def xml_shape_report(path: Path) -> dict:
    """Ids and BSDF references actually written in the XML."""
    root = ET.parse(path).getroot()
    shapes = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "shape":
            continue
        shapes.append(
            {
                "type": node.get("type"),
                "id": node.get("id") or "",
                "name": node.get("name") or "",
            }
        )
    empty = [item for item in shapes if item["id"] == "" and item["name"] == ""]
    return {"shapes": shapes, "empty": empty}


def require_logical_shapes(scene, names: list[str]) -> dict:
    """Every logical object must be a named shape. An empty id fails the sample."""
    found = []
    for shape in scene.shapes():
        ident = str(shape.id() or "")
        found.append({"id": ident, "class": shape.class_name()})
    empty = [item for item in found if item["id"] == ""]
    missing = []
    for name in names:
        token = f"mesh-{name}"
        if not any(item["id"] == token or item["id"].endswith(token) for item in found):
            missing.append(name)
    if empty or missing:
        raise RuntimeError(f"exported shapes {[item['id'] for item in found]} missing={missing} empty={len(empty)}")
    return {"shapes": found, "missing": missing, "empty": empty}
