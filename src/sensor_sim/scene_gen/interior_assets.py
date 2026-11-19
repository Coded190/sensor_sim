"""Recognizable furniture and appliances for the diverse-interior review.

Each appliance is more than one box. Doors, handles, burners, and screens are
separate parts. Packages are written only into the diverse-interior catalog.
"""

from __future__ import annotations

from sensor_sim.scene_gen.semantic_assets import _box, _cylinder, _metal, _record

WOOD = {"type": "diffuse", "reflectance": [0.45, 0.30, 0.16], "research_role": "depolarizing_reference"}
WHITE = {"type": "diffuse", "reflectance": [0.90, 0.90, 0.88], "research_role": "depolarizing_reference"}
CREAM = {"type": "diffuse", "reflectance": [0.86, 0.82, 0.74], "research_role": "depolarizing_reference"}
FABRIC = {"type": "diffuse", "reflectance": [0.35, 0.38, 0.42], "research_role": "depolarizing_reference"}
DARK = {"type": "diffuse", "reflectance": [0.08, 0.08, 0.09], "research_role": "depolarizing_reference"}
SCREEN = {"type": "diffuse", "reflectance": [0.15, 0.18, 0.22], "research_role": "depolarizing_reference"}


def _furn(asset_id, pieces, role, material, support):
    built = _record(asset_id, pieces, None, role, material, support, False, [])
    if not built["report"]["passed"]:
        raise RuntimeError(f"{asset_id} geometry failed: {built['report']['issues']}")
    return built


def _base_cabinet() -> dict:
    pieces = [
        _box("toe", [0.04, 0.04, 0.00], [1.16, 0.50, 0.08]),
        _box("carcass", [0.02, 0.02, 0.08], [1.18, 0.56, 0.86]),
        _box("door", [0.06, 0.56, 0.12], [1.14, 0.60, 0.84]),
        _box("handle", [0.90, 0.58, 0.50], [1.04, 0.62, 0.54]),
        _box("countertop", [0.00, 0.00, 0.86], [1.20, 0.62, 0.90]),
    ]
    return _furn("furn_base_cabinet", pieces, "furniture", WHITE, (0.0, 0.0, 1.20, 0.62, 0.90))


def _upper_cabinet() -> dict:
    pieces = [
        _box("carcass", [0.02, 0.02, 0.02], [1.18, 0.32, 0.66]),
        _box("door", [0.06, 0.32, 0.06], [1.14, 0.36, 0.62]),
        _box("handle", [0.90, 0.34, 0.30], [1.02, 0.37, 0.34]),
    ]
    return _furn("furn_upper_cabinet", pieces, "furniture", WHITE, None)


def _sink() -> dict:
    pieces = [
        _box("basin", [0.02, 0.02, 0.00], [0.58, 0.42, 0.16]),
        _box("rim", [0.00, 0.00, 0.16], [0.60, 0.46, 0.19]),
        _cylinder("faucet", 0.012, 0.22, [0.30, 0.08, 0.19]),
        _box("spout", [0.26, 0.08, 0.38], [0.32, 0.28, 0.42]),
    ]
    return _furn("furn_sink", pieces, "furniture", _metal(), None)


def _cooktop() -> dict:
    pieces = [_box("body", [0.0, 0.0, 0.0], [0.60, 0.52, 0.04])]
    for index, (x_value, y_value) in enumerate(((0.15, 0.14), (0.45, 0.14), (0.15, 0.36), (0.45, 0.36))):
        pieces.append(_cylinder(f"burner_{index}", 0.07, 0.015, [x_value, y_value, 0.04]))
    return _furn("furn_cooktop", pieces, "furniture", DARK, (0.0, 0.0, 0.60, 0.52, 0.055))


def _oven() -> dict:
    pieces = [
        _box("body", [0.0, 0.0, 0.0], [0.60, 0.56, 0.90]),
        _box("door", [0.04, 0.56, 0.16], [0.56, 0.62, 0.72]),
        _box("window", [0.16, 0.60, 0.32], [0.44, 0.64, 0.60]),
        _box("panel", [0.04, 0.56, 0.76], [0.56, 0.62, 0.88]),
        _cylinder("knob_0", 0.015, 0.02, [0.14, 0.61, 0.80]),
        _cylinder("knob_1", 0.015, 0.02, [0.24, 0.61, 0.80]),
    ]
    return _furn("furn_oven", pieces, "furniture", WHITE, (0.0, 0.0, 0.60, 0.62, 0.90))


def _microwave() -> dict:
    pieces = [
        _box("body", [0.0, 0.0, 0.0], [0.52, 0.34, 0.32]),
        _box("door", [0.02, 0.34, 0.02], [0.40, 0.38, 0.30]),
        _box("window", [0.08, 0.36, 0.08], [0.32, 0.40, 0.24]),
        _box("controls", [0.40, 0.34, 0.04], [0.50, 0.38, 0.28]),
    ]
    return _furn("furn_microwave", pieces, "furniture", WHITE, None)


def _fridge() -> dict:
    pieces = [
        _box("body", [0.0, 0.0, 0.0], [0.75, 0.68, 1.80]),
        _box("door_left", [0.02, 0.68, 0.04], [0.36, 0.75, 1.76]),
        _box("door_right", [0.39, 0.68, 0.04], [0.73, 0.75, 1.76]),
        _box("handle_left", [0.30, 0.74, 0.70], [0.34, 0.78, 1.30]),
        _box("handle_right", [0.40, 0.74, 0.70], [0.44, 0.78, 1.30]),
    ]
    return _furn("furn_fridge", pieces, "furniture", WHITE, None)


def _island() -> dict:
    pieces = [
        _box("base", [0.04, 0.04, 0.0], [1.36, 0.66, 0.86]),
        _box("top", [0.0, 0.0, 0.86], [1.40, 0.70, 0.92]),
    ]
    return _furn("furn_island", pieces, "furniture", CREAM, (0.0, 0.0, 1.40, 0.70, 0.92))


def _sideboard() -> dict:
    pieces = [
        _box("plinth", [0.04, 0.04, 0.00], [1.36, 0.36, 0.08]),
        _box("body", [0.0, 0.0, 0.08], [1.40, 0.42, 0.78]),
        _box("door_left", [0.04, 0.42, 0.12], [0.68, 0.46, 0.74]),
        _box("door_right", [0.72, 0.42, 0.12], [1.36, 0.46, 0.74]),
        _box("top", [0.0, 0.0, 0.78], [1.40, 0.46, 0.82]),
    ]
    return _furn("furn_sideboard", pieces, "furniture", WOOD, (0.0, 0.0, 1.40, 0.46, 0.82))


def _sofa() -> dict:
    pieces = [
        _box("leg_a", [0.12, 0.12, 0.00], [0.20, 0.20, 0.28]),
        _box("leg_b", [1.80, 0.60, 0.00], [1.88, 0.68, 0.28]),
        _box("seat", [0.10, 0.08, 0.28], [1.90, 0.70, 0.48]),
        _box("back", [0.10, 0.62, 0.48], [1.90, 0.86, 0.84]),
        _box("arm_left", [0.0, 0.08, 0.28], [0.12, 0.78, 0.62]),
        _box("arm_right", [1.88, 0.08, 0.28], [2.00, 0.78, 0.62]),
        _box("cushion", [0.20, 0.16, 0.48], [1.00, 0.58, 0.56]),
    ]
    return _furn("furn_sofa", pieces, "furniture", FABRIC, None)


def _armchair() -> dict:
    pieces = [
        _box("leg_a", [0.10, 0.10, 0.00], [0.18, 0.18, 0.28]),
        _box("leg_b", [0.62, 0.54, 0.00], [0.70, 0.62, 0.28]),
        _box("seat", [0.08, 0.08, 0.28], [0.72, 0.62, 0.48]),
        _box("back", [0.08, 0.56, 0.48], [0.72, 0.80, 0.84]),
        _box("arm_left", [0.0, 0.08, 0.28], [0.10, 0.70, 0.60]),
        _box("arm_right", [0.70, 0.08, 0.28], [0.80, 0.70, 0.60]),
    ]
    return _furn("furn_armchair", pieces, "furniture", FABRIC, None)


def _coffee() -> dict:
    pieces = [
        _box("top", [0.0, 0.0, 0.38], [1.10, 0.60, 0.42]),
        _box("leg", [0.08, 0.08, 0.0], [0.14, 0.14, 0.38]),
        _box("leg_b", [0.96, 0.46, 0.0], [1.02, 0.52, 0.38]),
    ]
    return _furn("furn_coffee", pieces, "furniture", WOOD, (0.0, 0.0, 1.10, 0.60, 0.42))


def _media() -> dict:
    pieces = [
        _box("plinth", [0.04, 0.04, 0.00], [1.36, 0.32, 0.05]),
        _box("body", [0.0, 0.0, 0.05], [1.40, 0.38, 0.46]),
        _box("door", [0.06, 0.38, 0.10], [1.34, 0.42, 0.42]),
        _box("top", [0.0, 0.0, 0.46], [1.40, 0.42, 0.50]),
    ]
    return _furn("furn_media", pieces, "furniture", WOOD, (0.0, 0.0, 1.40, 0.42, 0.50))


def _tv() -> dict:
    pieces = [
        _box("bezel", [0.0, 0.0, 0.0], [1.10, 0.06, 0.64]),
        _box("screen", [0.04, 0.04, 0.04], [1.06, 0.08, 0.60]),
        _box("stand", [0.48, 0.0, -0.08], [0.62, 0.08, 0.0]),
    ]
    return _furn("furn_tv", pieces, "furniture", DARK, None)


def _bookcase() -> dict:
    pieces = [
        _box("side_l", [0.0, 0.0, 0.0], [0.03, 0.32, 1.80]),
        _box("side_r", [0.77, 0.0, 0.0], [0.80, 0.32, 1.80]),
        _box("back", [0.0, 0.0, 0.0], [0.80, 0.02, 1.80]),
    ]
    for index, z_value in enumerate((0.05, 0.50, 0.95, 1.40, 1.74)):
        pieces.append(_box(f"shelf_{index}", [0.03, 0.02, z_value], [0.77, 0.32, z_value + 0.03]))
    return _furn("furn_bookcase", pieces, "furniture", WOOD, (0.03, 0.02, 0.77, 0.32, 1.43))


def _bed() -> dict:
    pieces = [
        _box("leg_a", [0.08, 0.08, 0.00], [0.16, 0.16, 0.10]),
        _box("leg_b", [1.88, 1.36, 0.00], [1.96, 1.44, 0.10]),
        _box("frame", [0.0, 0.0, 0.10], [2.05, 1.55, 0.32]),
        _box("mattress", [0.06, 0.06, 0.32], [1.98, 1.40, 0.52]),
        _box("headboard", [0.0, 1.40, 0.32], [2.05, 1.55, 1.05]),
        _box("pillow_left", [0.20, 1.05, 0.52], [0.70, 1.38, 0.66]),
        _box("pillow_right", [1.20, 1.05, 0.52], [1.75, 1.38, 0.66]),
        _box("blanket", [0.10, 0.08, 0.52], [1.95, 0.95, 0.58]),
    ]
    return _furn("furn_bed", pieces, "furniture", CREAM, (0.06, 0.06, 1.98, 1.40, 0.52))


def _nightstand() -> dict:
    pieces = [
        _box("plinth", [0.04, 0.04, 0.00], [0.41, 0.32, 0.05]),
        _box("body", [0.0, 0.0, 0.05], [0.45, 0.38, 0.50]),
        _box("drawer", [0.04, 0.38, 0.22], [0.41, 0.42, 0.42]),
        _box("top", [0.0, 0.0, 0.50], [0.45, 0.40, 0.54]),
    ]
    return _furn("furn_nightstand", pieces, "furniture", WOOD, (0.0, 0.0, 0.45, 0.40, 0.54))


def _wardrobe() -> dict:
    pieces = [
        _box("body", [0.0, 0.0, 0.0], [1.05, 0.52, 1.90]),
        _box("door_left", [0.03, 0.52, 0.06], [0.50, 0.57, 1.84]),
        _box("door_right", [0.54, 0.52, 0.06], [1.02, 0.57, 1.84]),
        _box("handle_left", [0.42, 0.55, 0.90], [0.46, 0.59, 1.10]),
        _box("handle_right", [0.56, 0.55, 0.90], [0.60, 0.59, 1.10]),
    ]
    return _furn("furn_wardrobe", pieces, "furniture", WOOD, None)


def _dining_table() -> dict:
    pieces = [
        _box("top", [0.0, 0.0, 0.72], [1.60, 0.90, 0.76]),
        _box("leg", [0.08, 0.08, 0.0], [0.16, 0.16, 0.72]),
        _box("leg_b", [1.44, 0.74, 0.0], [1.52, 0.82, 0.72]),
    ]
    return _furn("furn_dining_table", pieces, "furniture", WOOD, (0.0, 0.0, 1.60, 0.90, 0.76))


def _office_desk() -> dict:
    pieces = [
        _box("top", [0.0, 0.0, 0.72], [1.40, 0.70, 0.76]),
        _box("leg", [0.06, 0.06, 0.0], [0.12, 0.12, 0.72]),
        _box("leg_b", [1.28, 0.58, 0.0], [1.34, 0.64, 0.72]),
        _box("modesty", [0.12, 0.55, 0.20], [1.28, 0.60, 0.70]),
    ]
    return _furn("furn_office_desk", pieces, "furniture", WOOD, (0.0, 0.0, 1.40, 0.70, 0.76))


def _monitor() -> dict:
    pieces = [
        _box("screen", [0.0, 0.02, 0.12], [0.54, 0.06, 0.42]),
        _box("bezel", [0.0, 0.0, 0.10], [0.54, 0.03, 0.44]),
        _box("neck", [0.24, 0.0, 0.04], [0.30, 0.04, 0.12]),
        _box("foot", [0.12, 0.0, 0.0], [0.42, 0.16, 0.02]),
    ]
    return _furn("furn_monitor", pieces, "furniture", DARK, None)


def _desk_lamp() -> dict:
    pieces = [
        _box("base", [0.0, 0.0, 0.0], [0.14, 0.14, 0.02]),
        _cylinder("stem", 0.012, 0.32, [0.07, 0.07, 0.02]),
        _box("shade", [0.0, 0.0, 0.34], [0.16, 0.16, 0.42]),
    ]
    return _furn("furn_desk_lamp", pieces, "furniture", CREAM, None)


def _floor_lamp() -> dict:
    pieces = [
        _cylinder("pole", 0.02, 1.40, [0.14, 0.14, 0.0]),
        _box("base", [0.0, 0.0, 0.0], [0.28, 0.28, 0.03]),
        _box("shade", [0.02, 0.02, 1.40], [0.26, 0.26, 1.58]),
    ]
    return _furn("furn_floor_lamp", pieces, "furniture", CREAM, None)


def _mouse() -> dict:
    pieces = [
        _box("body", [0.0, 0.0, 0.0], [0.06, 0.11, 0.03]),
        _box("button", [0.005, 0.05, 0.03], [0.055, 0.10, 0.038]),
    ]
    return _furn("furn_mouse", pieces, "context", DARK, None)


BUILDERS = {
    "furn_base_cabinet": _base_cabinet,
    "furn_upper_cabinet": _upper_cabinet,
    "furn_sink": _sink,
    "furn_cooktop": _cooktop,
    "furn_oven": _oven,
    "furn_microwave": _microwave,
    "furn_fridge": _fridge,
    "furn_island": _island,
    "furn_sideboard": _sideboard,
    "furn_sofa": _sofa,
    "furn_armchair": _armchair,
    "furn_coffee": _coffee,
    "furn_media": _media,
    "furn_tv": _tv,
    "furn_bookcase": _bookcase,
    "furn_bed": _bed,
    "furn_nightstand": _nightstand,
    "furn_wardrobe": _wardrobe,
    "furn_dining_table": _dining_table,
    "furn_office_desk": _office_desk,
    "furn_monitor": _monitor,
    "furn_desk_lamp": _desk_lamp,
    "furn_floor_lamp": _floor_lamp,
    "furn_mouse": _mouse,
}


def build_asset(asset_id: str) -> dict:
    if asset_id not in BUILDERS:
        raise KeyError(asset_id)
    return BUILDERS[asset_id]()
