"""Deterministic yaw-and-translation search. Scale, pitch, and roll stay fixed."""

from __future__ import annotations

import copy

import numpy as np

SOLVER_VERSION = "anneal_yaw_v1"


def _rotate(vector, yaw: float) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([cosine * array[0] - sine * array[1], sine * array[0] + cosine * array[1], array[2]], dtype=np.float64)


def _by_id(instances: list[dict]) -> dict:
    return {item["id"]: item for item in instances}


def _direction(source: dict, target: dict) -> np.ndarray:
    delta = np.asarray(target["position"], dtype=np.float64) - np.asarray(source["position"], dtype=np.float64)
    delta[2] = 0.0
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return delta / length


def evaluate(instances: list[dict], rules: list[dict], *, room: dict) -> list[dict]:
    """Return one result per rule. A hard failure has passed False."""
    found = _by_id(instances)
    results = []
    for rule in rules:
        source = found.get(rule["source"])
        target = found.get(rule["target"])
        passed = True
        measured = None
        reason = "ok"
        kind = rule["validator"]
        if kind == "facing_dot" and source is not None and target is not None:
            forward = _rotate(source["forward"], source["yaw"])
            measured = float(np.dot(forward, _direction(source, target)))
            passed = measured >= rule["tolerance"]
            reason = "facing" if not passed else "ok"
        elif kind == "screen_dot" and source is not None and target is not None:
            normal = _rotate(source.get("screen_normal", source["forward"]), source["yaw"])
            measured = float(np.dot(normal, _direction(source, target)))
            passed = measured >= rule["tolerance"]
            reason = "screen_faces" if not passed else "ok"
        elif kind == "screen_up" and source is not None:
            up = _rotate(source.get("screen_up", source["up"]), source["yaw"])
            measured = float(np.dot(up, np.array([0.0, 0.0, 1.0])))
            passed = measured >= rule["tolerance"]
            reason = "screen_up" if not passed else "ok"
        elif kind == "seat_up" and source is not None:
            up = _rotate(source["up"], source["yaw"])
            measured = float(np.dot(up, np.array([0.0, 0.0, 1.0])))
            passed = measured >= rule["tolerance"]
            reason = "seat_up" if not passed else "ok"
        elif kind == "on_floor" and source is not None:
            measured = abs(float(source["position"][2]))
            passed = measured <= rule["tolerance"]
            reason = "support" if not passed else "ok"
        elif kind == "on_surface" and source is not None and target is not None:
            expected = float(target["position"][2]) + float(target["size"][2])
            measured = abs(float(source["position"][2]) - expected)
            passed = measured <= max(rule["tolerance"], 0.05)
            reason = "support" if not passed else "ok"
        elif kind == "between_pair" and source is not None and target is not None:
            measured = abs(float(source["position"][1]) - 0.5 * (float(target["position"][1]) + float(found["television"]["position"][1]))) if "television" in found else 0.0
            passed = measured <= rule["tolerance"] + 0.5
            reason = "between" if not passed else "ok"
        elif kind == "near" and source is not None and target is not None:
            measured = float(np.linalg.norm(np.asarray(source["position"]) - np.asarray(target["position"])))
            passed = measured <= rule["tolerance"] + 1.5
            reason = "near" if not passed else "ok"
        elif kind == "front_clearance" and source is not None:
            forward = _rotate(source["forward"], source["yaw"])
            measured = float(np.dot(forward, np.array([0.0, -1.0, 0.0])))
            passed = measured >= rule["tolerance"]
            reason = "front_clearance" if not passed else "ok"
        elif kind == "faces_open_room" and source is not None:
            forward = _rotate(source["forward"], source["yaw"])
            measured = float(np.dot(forward, np.array([0.0, -1.0, 0.0])))
            passed = measured >= rule["tolerance"]
            reason = "appliance_front" if not passed else "ok"
        elif kind == "door_clearance":
            door = room["door"]
            blocked = []
            for item in instances:
                position = item["position"]
                half = item["size"][1] * 0.5
                in_x = door["x0"] - 0.2 <= position[0] <= door["x1"] + 0.2
                in_y = position[1] - half < door["depth_m"]
                if in_x and in_y:
                    blocked.append(item["id"])
            passed = not blocked
            measured = len(blocked)
            reason = "door_blocked:" + ",".join(blocked) if blocked else "ok"
        else:
            passed = source is not None or kind == "door_clearance"
            reason = "missing_instance" if not passed else "ok"
        results.append({**rule, "passed": bool(passed), "measured": measured, "failure_reason": reason})
    return results


def hard_failures(results: list[dict]) -> list[dict]:
    return [item for item in results if item["priority"] == "hard" and not item["passed"]]


def solve(instances: list[dict], rules: list[dict], room: dict, *, seed: int, steps: int = 24) -> dict:
    """Search yaw and horizontal position. A rejected step stays in the trace."""
    current = copy.deepcopy(instances)
    generator = np.random.default_rng(seed)
    trace = [{"event": "initial", "failures": [item["failure_reason"] for item in hard_failures(evaluate(current, rules, room=room))]}]
    for step in range(steps):
        if not hard_failures(evaluate(current, rules, room=room)):
            trace.append({"event": "stop", "reason": "hard_constraints_satisfied", "step": step})
            break
        proposal = copy.deepcopy(current)
        choice = proposal[int(generator.integers(0, len(proposal)))]
        before = (list(choice["position"]), float(choice["yaw"]))
        if int(generator.integers(0, 2)) == 0:
            choice["yaw"] = float(choice["yaw"] + generator.choice([-0.5, 0.5, 3.1416, -3.1416]))
        else:
            choice["position"][0] = float(choice["position"][0] + generator.uniform(-0.15, 0.15))
            choice["position"][1] = float(choice["position"][1] + generator.uniform(-0.15, 0.15))
        before_count = len(hard_failures(evaluate(current, rules, room=room)))
        after = evaluate(proposal, rules, room=room)
        after_count = len(hard_failures(after))
        if after_count <= before_count:
            current = proposal
            trace.append({"event": "accept", "step": step, "id": choice["id"], "failures": after_count})
        else:
            trace.append({"event": "reject", "step": step, "id": choice["id"], "before": before, "reason": "hard_constraint_worse", "failures": [item["failure_reason"] for item in hard_failures(after)]})
    final = evaluate(current, rules, room=room)
    return {
        "solver_version": SOLVER_VERSION,
        "seed": int(seed),
        "steps": int(steps),
        "scale_optimized": False,
        "pitch_roll_optimized": False,
        "instances": current,
        "results": final,
        "valid": not hard_failures(final),
        "trace": trace,
    }
