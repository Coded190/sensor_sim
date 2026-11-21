"""Room rules. Positions are interior metres. South is y=0, east is +x, up is +z."""

from __future__ import annotations

from sensor_sim.scene_gen.composition_schema import relation


def _door() -> dict:
    return relation("keep_clear_of", "layout", "door_south", tolerance=0.15, hard=True, reference="world", validator="door_clearance")


def living(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    instances = [
        {"id": "sofa", "category": "sofa", "position": [width * 0.5, 0.9, 0.0], "yaw": 0.0, "forward": [0, 1, 0], "up": [0, 0, 1], "size": [1.6, 0.8, 0.8]},
        {"id": "television", "category": "television", "position": [width * 0.5, depth - 0.4, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "screen_normal": [0, -1, 0], "size": [1.1, 0.15, 0.7]},
        {"id": "table", "category": "table", "position": [width * 0.5, depth * 0.48, 0.0], "yaw": 0.0, "forward": [0, 1, 0], "up": [0, 0, 1], "size": [0.9, 0.5, 0.4]},
    ]
    rules = [
        relation("faces_toward", "sofa", "television", tolerance=0.85, hard=True, reference="semantic", validator="facing_dot"),
        relation("screen_faces", "television", "sofa", tolerance=0.85, hard=True, reference="semantic", validator="screen_dot"),
        relation("between", "table", "sofa", tolerance=0.4, hard=True, reference="world", validator="between_pair"),
        relation("supported_on", "sofa", "floor", tolerance=0.01, hard=True, reference="world", validator="on_floor"),
        _door(),
    ]
    return instances, rules


def office(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    instances = [
        {"id": "desk", "category": "desk", "position": [width * 0.55, depth - 0.7, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [1.2, 0.6, 0.75]},
        {"id": "monitor", "category": "monitor", "position": [width * 0.55, depth - 0.55, 0.75], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "screen_normal": [0, -1, 0], "screen_up": [0, 0, 1], "size": [0.54, 0.2, 0.4]},
        {"id": "chair", "category": "chair", "position": [width * 0.55, depth - 1.45, 0.0], "yaw": 0.0, "forward": [0, 1, 0], "up": [0, 0, 1], "size": [0.5, 0.5, 0.9]},
    ]
    rules = [
        relation("screen_faces", "monitor", "chair", tolerance=0.85, hard=True, reference="semantic", validator="screen_dot"),
        relation("faces_toward", "chair", "desk", tolerance=0.85, hard=True, reference="semantic", validator="facing_dot"),
        relation("aligned_with", "monitor", "world_up", tolerance=0.95, hard=True, reference="semantic", validator="screen_up"),
        relation("aligned_with", "chair", "world_up", tolerance=0.95, hard=True, reference="semantic", validator="seat_up"),
        relation("supported_on", "monitor", "desk", tolerance=0.02, hard=True, reference="world", validator="on_surface"),
        _door(),
    ]
    return instances, rules


def kitchen(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    instances = [
        {"id": "fridge", "category": "refrigerator", "position": [0.5, depth - 0.45, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.7, 0.7, 1.8]},
        {"id": "microwave", "category": "microwave", "position": [1.6, depth - 0.4, 0.9], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.5, 0.35, 0.3]},
        {"id": "cabinet", "category": "cabinet", "position": [width - 0.5, depth * 0.5, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.6, 0.5, 0.9]},
    ]
    rules = [
        relation("front_clearance", "fridge", "room", tolerance=0.6, hard=True, reference="semantic", validator="front_clearance"),
        relation("faces_toward", "microwave", "room", tolerance=0.85, hard=True, reference="semantic", validator="faces_open_room"),
        relation("supported_on", "fridge", "floor", tolerance=0.01, hard=True, reference="world", validator="on_floor"),
        _door(),
    ]
    return instances, rules


def bedroom(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    instances = [
        {"id": "bed", "category": "bed", "position": [width * 0.55, depth - 1.1, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [1.6, 2.0, 0.5]},
        {"id": "nightstand", "category": "nightstand", "position": [width * 0.55 - 1.1, depth - 0.7, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.4, 0.4, 0.5]},
        {"id": "wardrobe", "category": "wardrobe", "position": [0.45, depth * 0.45, 0.0], "yaw": 1.5708, "forward": [1, 0, 0], "up": [0, 0, 1], "size": [0.6, 1.2, 1.9]},
    ]
    rules = [
        relation("supported_on", "bed", "floor", tolerance=0.01, hard=True, reference="world", validator="on_floor"),
        relation("adjacent_to", "nightstand", "bed", tolerance=0.4, hard=True, reference="world", validator="near"),
        _door(),
    ]
    return instances, rules


def dining(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    table = [width * 0.5, depth * 0.5, 0.0]
    instances = [
        {"id": "table", "category": "table", "position": table, "yaw": 0.0, "forward": [0, 1, 0], "up": [0, 0, 1], "size": [1.4, 0.8, 0.75]},
        {"id": "chair_south", "category": "chair", "position": [table[0], table[1] - 0.7, 0.0], "yaw": 0.0, "forward": [0, 1, 0], "up": [0, 0, 1], "size": [0.45, 0.45, 0.9]},
        {"id": "chair_north", "category": "chair", "position": [table[0], table[1] + 0.7, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.45, 0.45, 0.9]},
    ]
    rules = [
        relation("faces_toward", "chair_south", "table", tolerance=0.85, hard=True, reference="semantic", validator="facing_dot"),
        relation("faces_toward", "chair_north", "table", tolerance=0.85, hard=True, reference="semantic", validator="facing_dot"),
        relation("supported_on", "table", "floor", tolerance=0.01, hard=True, reference="world", validator="on_floor"),
        _door(),
    ]
    return instances, rules


def workshop(width: float, depth: float) -> tuple[list[dict], list[dict]]:
    instances = [
        {"id": "bench", "category": "desk", "position": [width * 0.45, depth - 0.6, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [1.6, 0.7, 0.9]},
        {"id": "toolbox", "category": "toolbox", "position": [width * 0.25, depth - 0.55, 0.9], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.4, 0.25, 0.25]},
        {"id": "organizer", "category": "shelf", "position": [width - 0.4, depth * 0.55, 0.0], "yaw": 0.0, "forward": [0, -1, 0], "up": [0, 0, 1], "size": [0.4, 0.8, 1.5]},
    ]
    rules = [
        relation("supported_on", "toolbox", "bench", tolerance=0.02, hard=True, reference="world", validator="on_surface"),
        relation("grouped_with", "organizer", "bench", tolerance=2.5, hard=False, reference="world", validator="near"),
        _door(),
    ]
    return instances, rules


BUILDERS = {
    "living": living,
    "office": office,
    "kitchen": kitchen,
    "bedroom": bedroom,
    "dining": dining,
    "workshop": workshop,
}
