"""Three scripted tasks and the predicates that score them.

Success is a binary episode result.

- Slot match: the carried solid is within 5 mm and 15 degrees of that family's fixed slot.
- Tip up: a fallen object is grasped and lifted. A 2 cm rise while carried removes the
  tilt present at the grasp, so the last pose is within 15 degrees of upright.
- Lift: the object stays grasped and at least 2 cm above its rest height for the last
  5 frames, which is 1 second at 5 Hz.
"""

from __future__ import annotations

import numpy as np

from sensor_sim.policy.arm import MAX_TRANSLATION, ArmEnv, _wrap

SLOTS = {
    "ellipsoid": (0.04, 0.045),
    "rounded_box": (0.04, 0.0),
    "bumpy": (0.0, 0.05),
    "capsule": (-0.04, 0.045),
    "torus": (-0.04, 0.0),
    "cylinder": (0.015, -0.015),
    "cone": (-0.015, 0.05),
    "sphere": (0.045, 0.02),
}
POSITION_TOLERANCE_M = 0.005
YAW_TOLERANCE_RAD = np.deg2rad(15.0)
TILT_TOLERANCE_RAD = np.deg2rad(15.0)
LIFT_M = 0.02
HOLD_FRAMES = 5
EPISODE_FRAMES = 40


def success(task: str, family: str, radius: float, poses: list[dict]) -> bool:
    """True when the episode meets that task's predicate. ``poses`` includes the start pose."""
    if len(poses) < 2:
        return False
    if task == "slot":
        last = poses[-1]
        target = np.array([SLOTS[family][0], SLOTS[family][1], radius])
        position_error = float(np.linalg.norm(last["position"] - target))
        yaw_error = abs(float((last["yaw"] + np.pi) % (2.0 * np.pi) - np.pi))
        return last["attached"] and position_error <= POSITION_TOLERANCE_M and yaw_error <= YAW_TOLERANCE_RAD
    if task == "tip":
        return poses[-1]["attached"] and float(poses[-1]["tilt"]) <= TILT_TOLERANCE_RAD
    if task == "lift":
        tail = poses[-HOLD_FRAMES:]
        if len(tail) < HOLD_FRAMES:
            return False
        return all(pose["attached"] and float(pose["position"][2]) >= radius + LIFT_M for pose in tail)
    raise ValueError(f"unknown task {task}")


def _clip_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm > MAX_TRANSLATION:
        delta = delta * (MAX_TRANSLATION / norm)
    return delta


def scripted_episode(task: str, family: str, xy: tuple[float, float], yaw: float, radius: float) -> dict:
    """One expert episode. The returned actions are what :meth:`ArmEnv.step` consumed."""
    tilt = 1.2 if task == "tip" else 0.0
    env = ArmEnv()
    env.reset(xy, yaw if task != "slot" else yaw, tilt, radius)
    start = np.array([xy[0], xy[1], radius])
    grasp = start.copy()
    above = start + np.array([0.0, 0.0, 0.05])
    if task == "slot":
        goal = np.array([SLOTS[family][0], SLOTS[family][1], radius])
        goal_yaw = 0.0
    elif task == "tip":
        goal = start + np.array([0.0, 0.0, 0.04])
        goal_yaw = yaw
    else:
        goal = start + np.array([0.0, 0.0, 0.03])
        goal_yaw = yaw
    stages = (
        (above, yaw, 1.0),
        (grasp, yaw, 1.0),
        (grasp, yaw, 0.0),
        (goal, goal_yaw, 0.0),
        (goal, goal_yaw, 0.0),
    )
    actions = []
    poses = [env.object_pose()]
    frames = []
    stage = 0
    while len(actions) < EPISODE_FRAMES:
        target, target_yaw, grip = stages[min(stage, len(stages) - 1)]
        position, ee_yaw = env.end_effector()
        delta = _clip_delta(position, target)
        yaw_delta = float(np.clip(_wrap(target_yaw - ee_yaw), -0.2, 0.2))
        action = np.array([delta[0], delta[1], delta[2], yaw_delta, grip], dtype=np.float64)
        env.step(action)
        actions.append(action)
        pose = env.object_pose()
        poses.append(pose)
        ee_position, _ee_yaw = env.end_effector()
        frames.append(
            {
                "action": action.copy(),
                "state": env.state().copy(),
                "object": pose,
                "segments": env.link_segments(),
                "ee": ee_position.copy(),
            }
        )
        if float(np.linalg.norm(ee_position - target)) < 0.012:
            stage += 1
    flag = success(task, family, radius, poses)
    return {
        "actions": np.stack(actions),
        "poses": poses[1:],
        "frames": frames,
        "success": flag,
        "radius": radius,
        "family": family,
        "task": task,
        "xy": xy,
        "yaw": yaw,
        "tilt": tilt,
    }
