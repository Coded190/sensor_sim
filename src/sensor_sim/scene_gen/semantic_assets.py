"""Procedural furniture and tools for the semantic-scene pilot.

Meshes are Z-up meters. A hollow asset is an open shell of solid parts, not a
dark face. Packages are written only after the preflight freezes the sizes.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.scene_gen.asset_validate import triangulate, validate_mesh
from sensor_sim.scene_gen.structural import box_mesh, cylinder_mesh

SEGMENTS = 12


def _shift(vertices: np.ndarray, offset) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64) + np.asarray(offset, dtype=np.float64)


def _box(name: str, minimum, maximum, axis=(0.0, 0.0, 1.0), profile: str = "closed_solid") -> dict:
    vertices, polygons = box_mesh(minimum, maximum)
    return {"id": name, "vertices": vertices, "polygons": polygons, "thickness_axis": list(axis), "profile": profile}


def _cylinder(name: str, radius: float, height: float, origin, segments: int = SEGMENTS) -> dict:
    vertices, polygons = cylinder_mesh(radius, height, segments)
    return {
        "id": name,
        "vertices": _shift(vertices, origin),
        "polygons": polygons,
        "thickness_axis": [0.0, 0.0, 1.0],
        "profile": "closed_solid",
    }


def _open_shell(prefix: str, outer, wall: float, height: float, floor: float) -> list[dict]:
    """Rectangular open vessel. The cavity is the space inside the walls."""
    x0, y0, x1, y1 = outer
    return [
        _box(f"{prefix}_bottom", [x0, y0, 0.0], [x1, y1, floor]),
        _box(f"{prefix}_wall_x0", [x0, y0, floor], [x0 + wall, y1, height]),
        _box(f"{prefix}_wall_x1", [x1 - wall, y0, floor], [x1, y1, height]),
        _box(f"{prefix}_wall_y0", [x0 + wall, y0, floor], [x1 - wall, y0 + wall, height]),
        _box(f"{prefix}_wall_y1", [x0 + wall, y1 - wall, floor], [x1 - wall, y1, height]),
    ]


def _legs(length: float, depth: float, top_z: float, leg: float = 0.06) -> list[dict]:
    pieces = []
    for index, (x_value, y_value) in enumerate(((0.0, 0.0), (length - leg, 0.0), (0.0, depth - leg), (length - leg, depth - leg))):
        pieces.append(_box(f"leg_{index}", [x_value, y_value, 0.0], [x_value + leg, y_value + leg, top_z], (1.0, 0.0, 0.0), "structural_solid"))
    return pieces


def _workbench() -> dict:
    length, depth, height = 1.80, 0.80, 0.90
    top_t = 0.04
    pieces = _legs(length, depth, height - top_t)
    pieces.append(_box("top", [0.0, 0.0, height - top_t], [length, depth, height], profile="structural_solid"))
    pieces.append(_box("lower_shelf", [0.08, 0.08, 0.28], [length - 0.08, depth - 0.08, 0.31], profile="structural_solid"))
    return _record(
        "furniture_workbench",
        pieces,
        (length, depth, height),
        "furniture",
        {"type": "diffuse", "reflectance": [0.45, 0.30, 0.16], "research_role": "depolarizing_reference"},
        support=(0.0, 0.0, length, depth, height),
        interior=False,
        zones=_bench_zones(length, depth),
    )


def _bench_zones(length: float, depth: float) -> list[dict]:
    """Leave a clear column on the right. The left and middle hold the workpiece and tools."""
    return [
        _zone("assembly", 0.04, 0.04, 0.72, 0.50, "workpiece and the material beside it", ["tool", "manipulation_target", "material"]),
        _zone("tools", 0.76, 0.04, 1.36, 0.42, "hand tools", ["tool"]),
        _zone("staging", 0.04, 0.52, 0.78, 0.76, "stock and fixtures", ["material", "tool"]),
        _zone("small_parts", 0.82, 0.46, 1.36, 0.76, "trays, offcuts, and small tools", ["container", "tool", "context", "material"]),
        _zone("free_workspace", 1.40, 0.04, 1.76, 0.76, "open working area", []),
    ]


def _zone(zone_id, x0, y0, x1, y1, purpose, roles) -> dict:
    return {
        "id": zone_id,
        "support_surface": "top",
        "rectangle_m": [x0, y0, x1, y1],
        "purpose": purpose,
        "roles": roles,
    }


def _counter() -> dict:
    length, depth, height = 1.60, 0.60, 0.90
    top_t = 0.04
    pieces = _legs(length, depth, height - top_t)
    pieces.append(_box("top", [0.0, 0.0, height - top_t], [length, depth, height], profile="structural_solid"))
    return _record(
        "furniture_counter",
        pieces,
        (length, depth, height),
        "furniture",
        {"type": "diffuse", "reflectance": [0.72, 0.72, 0.70], "research_role": "depolarizing_reference"},
        support=(0.0, 0.0, length, depth, height),
        interior=False,
        zones=[
            _zone("preparation", 0.05, 0.05, 0.70, 0.55, "food preparation", ["manipulation_target", "tool", "material"]),
            _zone("ingredients", 0.75, 0.05, 1.20, 0.55, "ingredients", ["material", "manipulation_target"]),
            _zone("storage", 1.25, 0.05, 1.55, 0.55, "storage at the counter", ["container", "material", "manipulation_target"]),
        ],
    )


def _desk() -> dict:
    length, depth, height = 1.40, 0.70, 0.75
    top_t = 0.04
    pieces = _legs(length, depth, height - top_t)
    pieces.append(_box("top", [0.0, 0.0, height - top_t], [length, depth, height], profile="structural_solid"))
    return _record(
        "furniture_desk",
        pieces,
        (length, depth, height),
        "furniture",
        {"type": "diffuse", "reflectance": [0.55, 0.42, 0.30], "research_role": "depolarizing_reference"},
        support=(0.0, 0.0, length, depth, height),
        interior=False,
        zones=[
            _zone("main_work", 0.05, 0.05, 0.55, 0.65, "primary work area", ["manipulation_target", "context"]),
            _zone("input_devices", 0.60, 0.08, 1.05, 0.62, "keyboard and pointing", ["manipulation_target", "context"]),
            _zone("writing", 1.08, 0.08, 1.35, 0.62, "writing tools", ["context", "manipulation_target"]),
        ],
    )


def _screwdriver() -> dict:
    pieces = [
        _cylinder("handle", 0.012, 0.09, [0.0, 0.0, 0.0]),
        _cylinder("shaft", 0.0035, 0.11, [0.0, 0.0, 0.09]),
        _box("tip", [-0.004, -0.0012, 0.19], [0.004, 0.0012, 0.205]),
    ]
    return _tool("prop_screwdriver", pieces, "tool", _metal_handle())


def _pliers() -> dict:
    pieces = [
        _box("handle_a", [0.0, -0.012, 0.0], [0.09, -0.004, 0.012]),
        _box("handle_b", [0.0, 0.004, 0.0], [0.09, 0.012, 0.012]),
        _cylinder("pivot", 0.008, 0.016, [-0.008, -0.008, 0.0]),
        _box("jaw_a", [0.09, -0.008, 0.002], [0.15, -0.002, 0.010]),
        _box("jaw_b", [0.09, 0.002, 0.002], [0.15, 0.008, 0.010]),
    ]
    return _tool("prop_pliers", pieces, "tool", _metal())


def _wrench() -> dict:
    pieces = [
        _box("handle", [0.0, -0.008, 0.0], [0.14, 0.008, 0.008]),
        _box("jaw_outer", [0.14, -0.016, 0.0], [0.18, 0.016, 0.008]),
        _box("jaw_gap_fill_near", [0.145, -0.016, 0.0], [0.175, -0.008, 0.008]),
    ]
    return _tool("prop_wrench", pieces, "tool", _metal())


def _hammer() -> dict:
    pieces = [
        _box("handle", [-0.012, -0.16, 0.0], [0.012, 0.02, 0.024]),
        _box("head", [-0.035, 0.0, 0.004], [0.035, 0.03, 0.028]),
    ]
    return _tool("prop_hammer", pieces, "tool", _metal_handle())


def _clamp() -> dict:
    pieces = [
        _box("spine", [0.0, 0.0, 0.0], [0.012, 0.09, 0.10]),
        _box("jaw", [0.0, 0.0, 0.088], [0.08, 0.09, 0.10]),
        _cylinder("screw", 0.006, 0.07, [0.03, 0.035, 0.02]),
        _box("pad", [0.02, 0.03, 0.07], [0.05, 0.06, 0.082]),
    ]
    return _tool("prop_clamp", pieces, "tool", _metal())


def _hex_keys() -> dict:
    pieces = []
    for index, reach in enumerate((0.04, 0.055, 0.07)):
        x_value = index * 0.018
        pieces.append(_box(f"shank_{index}", [x_value, 0.0, 0.0], [x_value + 0.004, reach, 0.004]))
        pieces.append(_box(f"bend_{index}", [x_value, reach - 0.004, 0.0], [x_value + 0.004, reach, 0.025]))
    return _tool("prop_hex_key_group", pieces, "tool", _metal())


def _spool() -> dict:
    pieces = [
        _cylinder("core", 0.012, 0.05, [-0.012, -0.012, 0.0]),
        _cylinder("winding", 0.034, 0.03, [-0.034, -0.034, 0.01]),
    ]
    return _tool("prop_wire_spool", pieces, "material", {"type": "conductor", "material": "Cu"})


def _cable(asset_id: str) -> dict:
    pieces = []
    for index in range(10):
        angle = index * 0.45
        pieces.append(_box(f"segment_{index}", [0.012 * index, 0.018 * np.sin(angle), 0.0], [0.012 * index + 0.014, 0.018 * np.sin(angle) + 0.008, 0.008]))
    material = {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.15, 0.15, 0.16], "alpha": 0.1, "distribution": "beckmann"}
    role = "context" if asset_id == "prop_cable_group" else "material"
    return _tool(asset_id, pieces, role, material)


def _bracket() -> dict:
    pieces = [
        _box("flange_a", [0.0, 0.0, 0.0], [0.08, 0.04, 0.004]),
        _box("flange_b", [0.0, 0.0, 0.0], [0.004, 0.04, 0.06]),
    ]
    return _tool("prop_metal_bracket", pieces, "material", _metal())


def _notebook() -> dict:
    pieces = [
        _box("cover", [0.0, 0.0, 0.0], [0.20, 0.14, 0.006]),
        _box("pages", [0.008, 0.006, 0.006], [0.19, 0.134, 0.012]),
    ]
    return _tool("prop_notebook", pieces, "manipulation_target", {"type": "diffuse", "reflectance": [0.86, 0.84, 0.78], "research_role": "depolarizing_reference"})


def _pen() -> dict:
    return _tool("prop_pen", [_cylinder("body", 0.004, 0.14, [-0.004, -0.004, 0.0])], "context", {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.12, 0.18, 0.45], "alpha": 0.1, "distribution": "beckmann"})


def _keyboard() -> dict:
    pieces = [_box("base", [0.0, 0.0, 0.0], [0.32, 0.12, 0.012]), _box("keys", [0.01, 0.01, 0.012], [0.31, 0.11, 0.018])]
    return _tool("prop_keyboard", pieces, "manipulation_target", {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.25, 0.25, 0.27], "alpha": 0.1, "distribution": "beckmann"})


def _board() -> dict:
    return _tool("prop_cutting_board", [_box("slab", [0.0, 0.0, 0.0], [0.30, 0.20, 0.012])], "manipulation_target", {"type": "diffuse", "reflectance": [0.72, 0.55, 0.32], "research_role": "depolarizing_reference"})


def _pot() -> dict:
    pieces = _open_shell("pot", (0.0, 0.0, 0.16, 0.16), 0.004, 0.09, 0.004)
    pieces.append(_box("handle", [0.16, 0.06, 0.05], [0.22, 0.10, 0.062]))
    built = _tool("prop_pot", pieces, "manipulation_target", _metal())
    built["interior"] = True
    built["interior_point_m"] = [0.08, 0.08, 0.05]
    return built


def _spatula() -> dict:
    pieces = [_cylinder("handle", 0.006, 0.12, [-0.006, -0.006, 0.0]), _box("blade", [-0.025, -0.006, 0.12], [0.025, 0.05, 0.125])]
    return _tool("prop_spatula", pieces, "tool", {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.75, 0.75, 0.72], "alpha": 0.1, "distribution": "beckmann"})


def _cloth() -> dict:
    return _tool("prop_folded_cloth", [_box("fold", [0.0, 0.0, 0.0], [0.18, 0.18, 0.008])], "context", {"type": "diffuse", "reflectance": [0.75, 0.78, 0.82], "research_role": "depolarizing_reference"})


def _dish() -> dict:
    built = _tool("prop_serving_dish", _open_shell("dish", (0.0, 0.0, 0.22, 0.16), 0.006, 0.03, 0.004), "manipulation_target", {"type": "diffuse", "reflectance": [0.90, 0.90, 0.88], "research_role": "depolarizing_reference"})
    built["interior"] = True
    built["interior_point_m"] = [0.11, 0.08, 0.02]
    return built


def _fork() -> dict:
    pieces = [_box("handle", [0.0, -0.006, 0.0], [0.10, 0.006, 0.004])]
    for index in range(3):
        y_value = -0.006 + index * 0.005
        pieces.append(_box(f"tine_{index}", [0.10, y_value, 0.0], [0.15, y_value + 0.002, 0.003]))
    return _tool("prop_fork", pieces, "context", _metal())


def _tray() -> dict:
    pieces = _open_shell("tray", (0.0, 0.0, 0.16, 0.11), 0.004, 0.025, 0.004)
    for index in range(4):
        pieces.append(_box(f"fastener_{index}", [0.03 + (index % 2) * 0.04, 0.03 + (index // 2) * 0.03, 0.004], [0.038 + (index % 2) * 0.04, 0.038 + (index // 2) * 0.03, 0.012]))
    built = _tool("prop_fastener_tray", pieces, "container", _metal())
    built["interior"] = True
    built["interior_point_m"] = [0.08, 0.055, 0.018]
    return built


def _work_panel() -> dict:
    return _tool(
        "prop_work_panel",
        [_box("panel", [0.0, 0.0, 0.0], [0.52, 0.30, 0.018])],
        "manipulation_target",
        {"type": "diffuse", "reflectance": [0.62, 0.45, 0.28], "research_role": "depolarizing_reference"},
    )


def _fixture_block() -> dict:
    return _tool(
        "prop_fixture_block",
        [_box("fixture", [0.0, 0.0, 0.0], [0.28, 0.14, 0.040])],
        "material",
        {"type": "diffuse", "reflectance": [0.55, 0.42, 0.30], "research_role": "depolarizing_reference"},
    )


def _stock_plate() -> dict:
    return _tool("prop_stock_plate", [_box("plate", [0.0, 0.0, 0.0], [0.34, 0.16, 0.008])], "material", _metal())


def _offcut() -> dict:
    return _tool("prop_wood_offcut", [_box("plank", [0.0, 0.0, 0.0], [0.22, 0.08, 0.02])], "material", {"type": "diffuse", "reflectance": [0.62, 0.45, 0.28], "research_role": "depolarizing_reference"})


def _bin() -> dict:
    built = _tool("prop_parts_bin", _open_shell("bin", (0.0, 0.0, 0.14, 0.10), 0.004, 0.07, 0.004), "container", {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "diffuse_reflectance": [0.20, 0.35, 0.55], "alpha": 0.1, "distribution": "beckmann"})
    built["interior"] = True
    built["interior_point_m"] = [0.07, 0.05, 0.04]
    return built


def _block() -> dict:
    return _tool("prop_component_block", [_box("block", [0.0, 0.0, 0.0], [0.08, 0.05, 0.04])], "material", _metal())


def _metal() -> dict:
    return {"type": "conductor", "material": "Al"}


def _metal_handle() -> dict:
    return {"type": "roughconductor", "material": "Al", "distribution": "beckmann", "alpha": 0.1}


def _tool(asset_id: str, pieces: list[dict], role: str, material: dict) -> dict:
    return _record(asset_id, pieces, None, role, material, support=None, interior=False, zones=[])


def _record(asset_id, pieces, expected, role, material, support, interior, zones) -> dict:
    vertices, polygons, components = _combine(pieces)
    profile = "multi_component" if len(pieces) > 1 else pieces[0]["profile"]
    if profile == "multi_component":
        report = validate_mesh(vertices, polygons, profile, components=components)
    else:
        report = validate_mesh(vertices, polygons, profile, thickness_axis=pieces[0]["thickness_axis"])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    size = high - low
    return {
        "asset_id": asset_id,
        "pieces": pieces,
        "vertices": vertices,
        "polygons": polygons,
        "components": components,
        "faces": triangulate(polygons),
        "profile": profile,
        "report": report,
        "expected_m": None if expected is None else [float(value) for value in expected],
        "measured_m": [float(value) for value in size],
        "origin_m": [float(value) for value in low],
        "role": role,
        "material": material,
        "support": None if support is None else {"id": "top", "rectangle_m": [support[0], support[1], support[2], support[3]], "z_m": support[4]},
        "interior": interior,
        "zones": zones,
        "triangle_count": int(report["triangle_count"]),
        "footprint_m2": float(size[0] * size[1]),
    }


def _combine(pieces: list[dict]):
    vertices = []
    polygons: list[list[int]] = []
    components = []
    offset = 0
    for piece in pieces:
        local = [[int(index) + offset for index in polygon] for polygon in piece["polygons"]]
        start = len(polygons)
        polygons.extend(local)
        vertices.append(np.asarray(piece["vertices"], dtype=np.float64))
        profile = piece["profile"]
        component = {"id": piece["id"], "profile": profile, "polygon_indexes": list(range(start, start + len(local)))}
        if profile == "structural_solid":
            component["thickness_axis"] = piece["thickness_axis"]
        components.append(component)
        offset += len(piece["vertices"])
    return np.vstack(vertices), polygons, components


BUILDERS = {
    "furniture_workbench": _workbench,
    "furniture_counter": _counter,
    "furniture_desk": _desk,
    "prop_screwdriver": _screwdriver,
    "prop_pliers": _pliers,
    "prop_wrench": _wrench,
    "prop_hammer": _hammer,
    "prop_clamp": _clamp,
    "prop_hex_key_group": _hex_keys,
    "prop_wire_spool": _spool,
    "prop_cable": lambda: _cable("prop_cable"),
    "prop_cable_group": lambda: _cable("prop_cable_group"),
    "prop_metal_bracket": _bracket,
    "prop_notebook": _notebook,
    "prop_pen": _pen,
    "prop_keyboard": _keyboard,
    "prop_cutting_board": _board,
    "prop_pot": _pot,
    "prop_spatula": _spatula,
    "prop_folded_cloth": _cloth,
    "prop_serving_dish": _dish,
    "prop_fork": _fork,
    "prop_fastener_tray": _tray,
    "prop_work_panel": _work_panel,
    "prop_fixture_block": _fixture_block,
    "prop_stock_plate": _stock_plate,
    "prop_wood_offcut": _offcut,
    "prop_parts_bin": _bin,
    "prop_component_block": _block,
}


def build_asset(asset_id: str) -> dict:
    if asset_id not in BUILDERS:
        raise KeyError(asset_id)
    built = BUILDERS[asset_id]()
    if not built["report"]["passed"]:
        raise RuntimeError(f"{asset_id} geometry failed: {built['report']['issues']}")
    if built["triangle_count"] > 2500:
        raise RuntimeError(f"{asset_id} triangle_count={built['triangle_count']}")
    return built


def interior_is_open(built: dict) -> bool:
    if not built.get("interior"):
        return False
    point = np.asarray(built["interior_point_m"], dtype=np.float64)
    for piece in built["pieces"]:
        low = piece["vertices"].min(axis=0)
        high = piece["vertices"].max(axis=0)
        if np.all(point >= low - 1e-9) and np.all(point <= high + 1e-9):
            return False
    return True
