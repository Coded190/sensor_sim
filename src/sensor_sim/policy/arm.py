"""Scaled 7-DoF Franka-pattern arm in the gray room.

The Cornell room is 0.6 m on a side, so the published Franka link lengths do
not fit. Joint axes follow the Panda order (Z, Y, Z, Y, Z, Y, Z). Lengths are
scaled so the reach is about 0.25 m. State is the seven joint positions plus
the gripper width. The action is a delta of the end effector in x, y, z, and
yaw, plus the absolute gripper width.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Parent offset along +Z, then a hinge. Lengths sum to a folded reach inside the room.
LINKS = (
    (0.055, (0.0, 0.0, 1.0)),
    (0.080, (0.0, 1.0, 0.0)),
    (0.070, (0.0, 0.0, 1.0)),
    (0.080, (0.0, -1.0, 0.0)),
    (0.050, (0.0, 0.0, 1.0)),
    (0.040, (0.0, 1.0, 0.0)),
    (0.035, (0.0, 0.0, 1.0)),
)
BASE_POS = (0.0, -0.16, 0.0)
HOME_Q = np.array([0.0, 1.0, 0.0, -1.4, 0.0, 1.2, 0.4], dtype=np.float64)
MAX_TRANSLATION = 0.02
MAX_YAW = 0.2
GRASP_DISTANCE = 0.035
GRASP_CLOSED = 0.25


def _xml() -> str:
    bodies = []
    indent = "    "
    for index, (length, axis) in enumerate(LINKS):
        name = index + 1
        bodies.append(
            f'{indent}<body name="link{name}" pos="0 0 {length if index else 0.0:.4f}">'
            f'<joint name="j{name}" type="hinge" axis="{axis[0]} {axis[1]} {axis[2]}" range="-2.8 2.8"/>'
            f'<geom name="arm-{index}" type="capsule" fromto="0 0 0 0 0 {length:.4f}" size="0.016"/>'
        )
        indent += "  "
    bodies.append(f'{indent}<body name="ee_body" pos="0 0 0.03"><site name="ee" pos="0 0 0" size="0.01"/>'
                  f'<geom name="finger-a" type="capsule" fromto="0 -0.02 0 0 -0.02 0.02" size="0.006"/>'
                  f'<geom name="finger-b" type="capsule" fromto="0 0.02 0 0 0.02 0.02" size="0.006"/>'
                  f"</body>")
    closer = "</body>" * len(LINKS)
    return f"""
<mujoco model="scaled_franka">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="{BASE_POS[0]} {BASE_POS[1]} {BASE_POS[2]}">
      <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.02"/>
      {"".join(bodies)}
      {closer}
    </body>
    <body name="object" pos="0 0 0.04" mocap="true">
      <geom name="object" type="sphere" size="0.04"/>
      <site name="object_site" pos="0 0 0" size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


def quat_from_yaw_tilt(yaw: float, tilt: float) -> np.ndarray:
    """wxyz quaternion. Tilt is a rotation about X, then yaw about Z."""
    half_tilt = 0.5 * float(tilt)
    half_yaw = 0.5 * float(yaw)
    tilt_q = np.array([np.cos(half_tilt), np.sin(half_tilt), 0.0, 0.0])
    yaw_q = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    return _quat_mul(yaw_q, tilt_q)


def tilt_from_quat(quat: np.ndarray) -> float:
    """Angle between the object's local +Z and the world +Z, in radians."""
    rotated = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    cosine = float(np.clip(rotated[2], -1.0, 1.0))
    return float(np.arccos(cosine))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[1:]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * quat[0] * cross + 2.0 * np.cross(xyz, cross)


def carried_tilt(tilt_at_grasp: float, carried_rise: float) -> float:
    """Tilt left after ``carried_rise`` meters of upward carry. Two centimeters clears it."""
    if float(tilt_at_grasp) <= 1e-6:
        return 0.0
    remaining = 1.0 - float(carried_rise) / LIFT_CLEARS_TILT
    return float(tilt_at_grasp) * max(0.0, remaining)


LIFT_CLEARS_TILT = 0.02


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ArmEnv:
    """Kinematic Franka-pattern arm. The object is a mocap body that attaches when the gripper closes."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_xml())
        self.data = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self.gripper = 1.0
        self.attached = False
        self.grasp_offset = np.zeros(3)
        self.yaw_offset = 0.0
        self.tilt_at_grasp = 0.0
        self.carried_rise = 0.0
        self.object_yaw = 0.0
        self.object_tilt = 0.0
        self.rest_z = 0.04

    def reset(self, xy: tuple[float, float], yaw: float, tilt: float, radius: float) -> None:
        self.data.qpos[:] = HOME_Q
        self.data.qvel[:] = 0.0
        self.gripper = 1.0
        self.attached = False
        self.carried_rise = 0.0
        self.object_yaw = float(yaw)
        self.object_tilt = float(tilt)
        self.rest_z = float(radius)
        self.model.geom_size[self.object_geom][0] = float(radius)
        position = np.array([xy[0], xy[1], self.rest_z])
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = quat_from_yaw_tilt(self.object_yaw, self.object_tilt)
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.data.qpos[:7], dtype=np.float64), [self.gripper]])

    def object_pose(self) -> dict:
        position = np.asarray(self.data.mocap_pos[0], dtype=np.float64).copy()
        return {
            "position": position,
            "yaw": float(self.object_yaw),
            "tilt": float(self.object_tilt),
            "attached": bool(self.attached),
        }

    def end_effector(self) -> tuple[np.ndarray, float]:
        mujoco.mj_forward(self.model, self.data)
        position = np.asarray(self.data.site_xpos[self.ee_id], dtype=np.float64).copy()
        # Site x-axis yaw in the world. The site rotation columns are site_xmat.
        matrix = np.asarray(self.data.site_xmat[self.ee_id], dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        return position, yaw

    def link_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """World-space endpoints of each arm capsule, for the Mitsuba occluder."""
        mujoco.mj_forward(self.model, self.data)
        segments = []
        for index in range(7):
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"arm-{index}")
            center = np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
            matrix = np.asarray(self.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
            half = float(self.model.geom_size[geom][1])
            axis = matrix[:, 2]
            segments.append((center - axis * half, center + axis * half))
        return segments

    def step(self, action: np.ndarray) -> None:
        """Apply one delta. ``action`` is dx, dy, dz, dyaw, gripper width."""
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (5,):
            raise ValueError(f"action must be 5 values, got {command.shape}")
        position, yaw = self.end_effector()
        delta = np.clip(command[:3], -MAX_TRANSLATION, MAX_TRANSLATION)
        delta_yaw = float(np.clip(_wrap(command[3]), -MAX_YAW, MAX_YAW))
        self._ik(position + delta, yaw + delta_yaw)
        risen = float(self.end_effector()[0][2] - position[2])
        self.gripper = float(np.clip(command[4], 0.0, 1.0))
        self._update_grasp(risen)
        mujoco.mj_forward(self.model, self.data)

    def _ik(self, target_position: np.ndarray, target_yaw: float, iterations: int = 25) -> None:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            position, yaw = self.end_effector()
            error = np.concatenate([target_position - position, [_wrap(target_yaw - yaw)]])
            if float(np.linalg.norm(error[:3])) < 1e-4 and abs(error[3]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
            jacobian = np.vstack([jacp[:, :7], jacr[2:3, :7]])
            gram = jacobian @ jacobian.T + 1e-3 * np.eye(4)
            delta_q = jacobian.T @ np.linalg.solve(gram, error)
            self.data.qpos[:7] = np.clip(self.data.qpos[:7] + 0.6 * delta_q, -2.8, 2.8)

    def _update_grasp(self, rise: float) -> None:
        position, yaw = self.end_effector()
        object_position = np.asarray(self.data.mocap_pos[0], dtype=np.float64)
        if self.gripper >= GRASP_CLOSED:
            self.attached = False
            return
        # The end effector is a point. Attachment is when it reaches 1.5 cm of the
        # surface, which is radius + 1.5 cm from the object center.
        limit = max(GRASP_DISTANCE, self.rest_z + 0.015)
        if not self.attached and float(np.linalg.norm(position - object_position)) <= limit:
            self.attached = True
            self.grasp_offset = object_position - position
            self.yaw_offset = _wrap(self.object_yaw - yaw)
            self.tilt_at_grasp = float(self.object_tilt)
            self.carried_rise = 0.0
        if self.attached:
            self.data.mocap_pos[0] = position + self.grasp_offset
            self.object_yaw = _wrap(yaw + self.yaw_offset)
            # Tilt follows the rise the end effector actually achieved. A commanded
            # delta that inverse kinematics does not carry out leaves the object fallen.
            if rise > 0.0:
                self.carried_rise += rise
            self.object_tilt = carried_tilt(self.tilt_at_grasp, self.carried_rise)
            self.data.mocap_quat[0] = quat_from_yaw_tilt(self.object_yaw, self.object_tilt)
