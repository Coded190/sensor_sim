"""Room specification schema and the interior-origin coordinate convention.

Dimensions are interior clear sizes between finished faces. The southwest
interior floor corner is the origin. Slabs and walls grow outward from that box.
"""

from __future__ import annotations

CONVENTION = {
    "up": "Z",
    "units": "m",
    "handedness": "right",
    "x": "east",
    "y": "north",
    "z": "up",
    "origin": "southwest_interior_floor_corner",
    "finished_floor_z": 0.0,
    "interior_corner_m": [0.0, 0.0, 0.0],
    "dimension_meaning": "interior_clear_distance_between_finished_faces",
}
WALL_TANGENT = {"south": "x", "north": "x", "east": "y", "west": "y"}
WALLS = ("south", "north", "east", "west")
SCHEMA_VERSION = "room_spec_v1"
MIN_PIER_M = 0.15
REFERENCE_SHIFT_NOTE = (
    "build_room places the exterior floor corner at the origin and the finished floor at z = floor_thickness. "
    "Room specs place the interior floor corner at the origin and the finished floor at z = 0. "
    "Component AABBs match after adding [wall_thickness, wall_thickness, floor_thickness] to the new-room bounds."
)


class RoomSpecError(ValueError):
    """A room specification is missing a field or disagrees with the coordinate convention."""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoomSpecError(f"reason=not_a_number field={name}")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise RoomSpecError(f"reason=non_finite field={name}")
    return number


def exterior_bounds(width: float, depth: float, height: float, wall: float, floor_t: float, ceiling_t: float) -> dict:
    return {
        "minimum_m": [-wall, -wall, -floor_t],
        "maximum_m": [width + wall, depth + wall, height + ceiling_t],
    }


def wall_faces(width: float, depth: float, height: float, wall: float) -> dict:
    return {
        "south": {"interior_m": 0.0, "exterior_m": -wall, "normal": [0.0, -1.0, 0.0]},
        "north": {"interior_m": depth, "exterior_m": depth + wall, "normal": [0.0, 1.0, 0.0]},
        "west": {"interior_m": 0.0, "exterior_m": -wall, "normal": [-1.0, 0.0, 0.0]},
        "east": {"interior_m": width, "exterior_m": width + wall, "normal": [1.0, 0.0, 0.0]},
        "floor": {"interior_m": 0.0, "normal": [0.0, 0.0, 1.0]},
        "ceiling": {"interior_m": height, "normal": [0.0, 0.0, -1.0]},
    }


def make_specification(
    *,
    room_id: str,
    width: float,
    depth: float,
    height: float,
    openings: list[dict],
    seed: int,
    wall: float = 0.12,
    floor_t: float = 0.05,
    ceiling_t: float = 0.05,
) -> dict:
    """Build a specification and reject it when the openings do not fit."""
    spec = {
        "schema_version": SCHEMA_VERSION,
        "convention": dict(CONVENTION),
        "room_id": room_id,
        "version": "1.0.0",
        "seed": int(seed),
        "interior_width_m": float(width),
        "interior_depth_m": float(depth),
        "interior_height_m": float(height),
        "wall_thickness_m": float(wall),
        "floor_thickness_m": float(floor_t),
        "ceiling_thickness_m": float(ceiling_t),
        "exterior_bounds_m": exterior_bounds(width, depth, height, wall, floor_t, ceiling_t),
        "faces": wall_faces(width, depth, height, wall),
        "openings": openings,
        "corner_owner": "north_and_south_walls_include_returns",
        "frame_m": {"depth_m": 0.02, "width_m": 0.04},
        "materials": {
            "floor": {"type": "diffuse", "reflectance": [0.55, 0.52, 0.46]},
            "ceiling": {"type": "diffuse", "reflectance": [0.90, 0.90, 0.88]},
            "wall": {"type": "diffuse", "reflectance": [0.82, 0.82, 0.78]},
        },
        "backdrop": "none",
        "lighting_id": "ceiling_v1",
    }
    return validate_specification(spec)


def validate_specification(spec: dict) -> dict:
    required = {
        "schema_version",
        "convention",
        "room_id",
        "version",
        "seed",
        "interior_width_m",
        "interior_depth_m",
        "interior_height_m",
        "wall_thickness_m",
        "floor_thickness_m",
        "ceiling_thickness_m",
        "exterior_bounds_m",
        "faces",
        "openings",
        "corner_owner",
        "frame_m",
        "materials",
        "backdrop",
        "lighting_id",
    }
    missing = required - set(spec)
    if missing:
        raise RoomSpecError(f"reason=missing_fields fields={sorted(missing)}")
    if spec["schema_version"] != SCHEMA_VERSION:
        raise RoomSpecError("reason=schema_version")
    if spec["convention"].get("origin") != CONVENTION["origin"] or spec["convention"].get("finished_floor_z") != 0.0:
        raise RoomSpecError("reason=convention")
    if spec["corner_owner"] != "north_and_south_walls_include_returns":
        raise RoomSpecError("reason=corner_owner")
    width = _number(spec["interior_width_m"], "interior_width_m")
    depth = _number(spec["interior_depth_m"], "interior_depth_m")
    height = _number(spec["interior_height_m"], "interior_height_m")
    wall = _number(spec["wall_thickness_m"], "wall_thickness_m")
    floor_t = _number(spec["floor_thickness_m"], "floor_thickness_m")
    ceiling_t = _number(spec["ceiling_thickness_m"], "ceiling_thickness_m")
    if min(width, depth, height, wall, floor_t, ceiling_t) <= 0.0:
        raise RoomSpecError("reason=non_positive_dimension")
    expected = exterior_bounds(width, depth, height, wall, floor_t, ceiling_t)
    _match_bounds(spec["exterior_bounds_m"], expected)
    _match_faces(spec["faces"], wall_faces(width, depth, height, wall))
    openings = [_opening(item, width, depth, height) for item in spec["openings"]]
    _reject_overlap(openings)
    checked = dict(spec)
    checked["openings"] = openings
    return checked


def diversity_specifications(compact: tuple[float, float, float]) -> dict:
    """Three rectangular profiles. Openings differ by wall, not only by scale."""
    width, depth, height = compact
    compact_spec = make_specification(
        room_id="room_compact",
        width=width,
        depth=depth,
        height=height,
        seed=17301,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.25, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.80, "width_m": 1.00, "height_m": 0.90, "sill_m": 0.90},
        ],
    )
    standard = make_specification(
        room_id="room_standard",
        width=4.20,
        depth=3.80,
        height=2.70,
        seed=17303,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.40, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.60, "width_m": 1.40, "height_m": 1.10, "sill_m": 0.90},
            {"id": "window_east", "kind": "window", "wall": "east", "offset_m": 0.50, "width_m": 0.80, "height_m": 1.00, "sill_m": 1.00},
        ],
    )
    spacious = make_specification(
        room_id="room_spacious",
        width=5.50,
        depth=4.80,
        height=3.00,
        seed=17305,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.50, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.80, "width_m": 1.80, "height_m": 1.20, "sill_m": 0.80},
            {"id": "window_west", "kind": "window", "wall": "west", "offset_m": 0.70, "width_m": 1.00, "height_m": 1.20, "sill_m": 0.90},
        ],
    )
    return {"room_compact": compact_spec, "room_standard": standard, "room_spacious": spacious}


def reference_specification() -> dict:
    """The packaged shell, expressed in the interior-origin convention."""
    return make_specification(
        room_id="room_reference",
        width=4.0,
        depth=3.2,
        height=2.5,
        seed=17011,
        openings=[
            {"id": "door_south", "kind": "door", "wall": "south", "offset_m": 0.30, "width_m": 0.90, "height_m": 2.10, "sill_m": 0.0},
            {"id": "window_north", "kind": "window", "wall": "north", "offset_m": 0.50, "width_m": 1.20, "height_m": 1.00, "sill_m": 0.90},
        ],
    )


def _opening(item: dict, width: float, depth: float, height: float) -> dict:
    required = {"id", "kind", "wall", "offset_m", "width_m", "height_m", "sill_m"}
    if required - set(item):
        raise RoomSpecError(f"reason=opening_fields id={item.get('id')}")
    if item["kind"] not in {"door", "window"} or item["wall"] not in WALLS:
        raise RoomSpecError(f"reason=opening_kind id={item['id']}")
    offset = _number(item["offset_m"], "offset_m")
    hole_w = _number(item["width_m"], "width_m")
    hole_h = _number(item["height_m"], "height_m")
    sill = _number(item["sill_m"], "sill_m")
    if item["kind"] == "door" and sill != 0.0:
        raise RoomSpecError(f"reason=door_sill id={item['id']}")
    if item["kind"] == "window" and sill <= 0.0:
        raise RoomSpecError(f"reason=window_sill id={item['id']}")
    tangent_length = width if item["wall"] in {"south", "north"} else depth
    if offset < MIN_PIER_M or offset + hole_w > tangent_length - MIN_PIER_M:
        raise RoomSpecError(f"reason=opening_pier id={item['id']}")
    head = sill + hole_h
    if hole_h <= 0.0 or head >= height:
        raise RoomSpecError(f"reason=opening_height id={item['id']}")
    record = dict(item)
    record["tangent"] = WALL_TANGENT[item["wall"]]
    record["clear_aperture_m"] = _aperture(item["wall"], offset, hole_w, sill, hole_h, width, depth)
    record["height_above_floor_m"] = sill
    return record


def _aperture(wall: str, offset: float, hole_w: float, sill: float, hole_h: float, width: float, depth: float) -> dict:
    z0, z1 = sill, sill + hole_h
    if wall == "south":
        return {"minimum_m": [offset, 0.0, z0], "maximum_m": [offset + hole_w, 0.0, z1]}
    if wall == "north":
        return {"minimum_m": [offset, depth, z0], "maximum_m": [offset + hole_w, depth, z1]}
    if wall == "west":
        return {"minimum_m": [0.0, offset, z0], "maximum_m": [0.0, offset + hole_w, z1]}
    return {"minimum_m": [width, offset, z0], "maximum_m": [width, offset + hole_w, z1]}


def _reject_overlap(openings: list[dict]) -> None:
    for index, left in enumerate(openings):
        for right in openings[index + 1 :]:
            if left["wall"] != right["wall"]:
                continue
            gap = left["offset_m"] + left["width_m"] - right["offset_m"]
            other = right["offset_m"] + right["width_m"] - left["offset_m"]
            if gap > 0.0 and other > 0.0:
                z_gap = (left["sill_m"] + left["height_m"]) - right["sill_m"]
                z_other = (right["sill_m"] + right["height_m"]) - left["sill_m"]
                if z_gap > 0.0 and z_other > 0.0:
                    raise RoomSpecError(f"reason=opening_overlap ids={left['id']},{right['id']}")


def _match_bounds(actual: dict, expected: dict) -> None:
    for key in ("minimum_m", "maximum_m"):
        if key not in actual or len(actual[key]) != 3:
            raise RoomSpecError(f"reason=exterior_bounds field={key}")
        for got, want in zip(actual[key], expected[key], strict=True):
            if abs(float(got) - float(want)) > 1e-9:
                raise RoomSpecError("reason=exterior_bounds")


def _match_faces(actual: dict, expected: dict) -> None:
    for name, face in expected.items():
        got = actual.get(name)
        if not isinstance(got, dict):
            raise RoomSpecError(f"reason=face name={name}")
        for key, value in face.items():
            if isinstance(value, list):
                if [float(item) for item in got.get(key, [])] != value:
                    raise RoomSpecError(f"reason=face name={name}")
            elif abs(float(got.get(key, 99.0)) - float(value)) > 1e-9:
                raise RoomSpecError(f"reason=face name={name}")
