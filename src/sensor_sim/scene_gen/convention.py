"""Stokes frame used by every dataset sample.

The wording matches ``sensor_sim.stokes`` and ``sensor_sim.film.polar``.
Mitsuba's Stokes integrator reports the vector in the sensor frame, with
the reference axis along the camera's horizontal direction.
"""

from __future__ import annotations

STOKES_FRAME = {
    "reference_frame": "Mitsuba sensor frame, aligned with the camera horizontal axis",
    "horizontal_axis": "camera +X, which is image left for this project's look_at cameras",
    "vertical_axis": "camera +Y, which is image up",
    "s1_sign": "S1 is horizontal versus vertical linear polarization",
    "s2_sign": "S2 is +45 degree versus -45 degree linear polarization",
    "s3_handedness": "S3 is right versus left circular polarization",
    "analyzer_angles_rad": [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345],
    "analyzer_formula": "I(theta) = 0.5 * (S0 + S1 cos(2 theta) + S2 sin(2 theta))",
    "aolp_range_rad": "(-pi/2, pi/2]",
    "aolp_formula": "0.5 * atan2(S2, S1), stored as 0 where DoLP < 0.05 or S0 <= 1e-4",
    "spectral_copies": (
        "Each Stokes component is three spectral copies integrated to RGB. "
        "Averaging them discards per-copy differences. The average is not a "
        "hyperspectral cube and not one wavelength."
    ),
    "depth": (
        "On the prototype sphere, |position - camera_origin| exceeds the depth AOV "
        "by the near clip, 0.01 m, to within 0.4 mm. The residual of that difference "
        "is the obliquity of the near plane: the ray meets it at near_clip / "
        "dot(ray, camera_forward). Depth is not camera-space Z. The published "
        "Mitsuba note that depth starts at the pinhole does not match this render."
    ),
    "shape_index": (
        "shape_index 0 is the background. Other integers are Mitsuba shape ids, "
        "not the order of scene.shapes(). A pixel is one object only when the "
        "value is within 1e-3 of an integer. The sample records the integer read "
        "at each object's probe pixel."
    ),
    "primary_normal": (
        "The SfP target stays the camera-space shading normal. On this triangle "
        "sphere the shading and geometric AOVs match. Both match the radius from "
        "the sphere center through the position-AOV hit to a median under 1 degree "
        "on pixels facing the camera. They do not match the continuous analytic "
        "sphere: that surface sits outside the triangle chords."
    ),
    "camera_normal": (
        "Camera-space normals are world shading or geometric normals rotated by "
        "the camera matrix and flipped so they point toward the camera. Toward "
        "the camera is negative camera Z, because camera +Z points into the scene."
    ),
}

SHAPE_INDEX_TOL = 1e-3
NEAR_CLIP = 0.01
S0_MIN = 1e-4
DOP_EXCESS = 1.05
DOLP_KEEP = 0.2
