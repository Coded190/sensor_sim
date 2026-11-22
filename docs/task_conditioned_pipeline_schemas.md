# Task-scene schemas

These documents sit beside `asset_package_v1`. They do not add keys to `asset.json`.

## asset_extension_v1

File: `src/sensor_sim/scene_gen/asset_extension.py`. The sidecar is stored outside the package directory because `verify_package` rejects an unlisted file. It names `asset_id`, `package_version`, and `package_sha256`. Any mismatch is rejected.

The package hash remains the existing digest of canonical `asset.json` plus the listed file bytes. The sidecar and validation reports are not inputs to that hash.

Frames are `asset_root`, `base_contact`, `support_surface`, or a name starting with `contact_`. Each frame has an origin in meters, axes, a rigid transform into the asset frame, and a provenance class: `measured`, `sourced`, `estimated`, or `default`. The table support frame is the tabletop, not the mesh bounding-box origin. A mesh axis is not a semantic front.

Render and collision geometry are separate relative paths. A collision entry names its method: `convex_mesh`, `convex_decomposition`, `primitive_proxy`, or `other`. The same path cannot be both roles. Mass and inertia are omitted from the first qualified assets. A grasp candidate is allowed only with `qualification=unqualified_geometry`.

## scene_manifest_v1

File: `src/sensor_sim/scene_gen/scene_manifest.py`.

- World and asset frames are right-handed, Z-up, meters.
- A transform is a 4 by 4 matrix stored as four rows and applied to column vectors.
- `T_A_B` maps coordinates in frame B into frame A.
- `T_W_I = T_W_A @ T_A_I`.
- The last row is `[0, 0, 0, 1]`.
- A rigid transform must be orthonormal with determinant about +1. Reflections, scale, and shear are rejected.

Conversions that the tests check with known points:

- Blender's own world and Mitsuba's world match this frame, so those matrices are copied.
- Infinigen's exported OBJ is Y-up. `(x, y, z)` becomes `(x, -z, y)`.
- USD's default is Y-up. Canonical `(x, y, z)` becomes USD `(x, z, -y)`.
- The pinhole camera is +X right, +Y down, +Z forward. Mitsuba 3.9.1 on this machine looks along +Z with +Y up. The camera-to-world matrix is multiplied on the right by `diag(-1, -1, 1)`, a 180 degree roll about the forward axis. This was checked against `ScalarTransform4f.look_at`.

Package paths are relative to a declared root. Absolute paths and `..` are rejected. Adapter reports are separate files keyed by the manifest SHA-256. They are not written into the manifest.

A light records `radiometry.mode` as `relative` or `calibrated`. `relative` is not a calibrated radiometric value.

## task_scene_request_v1

File: `src/sensor_sim/scene_gen/task_scene_request.py`.

The request names a task type, a support and objects by category or asset id, count ranges, an optional size, a target region in the `support_surface` frame, camera and light profiles, and five seeds: `selection`, `layout`, `appearance`, `camera`, and `lighting`.

`domain_seed` is `SHA-256("{master}:{domain}")` interpreted as an integer modulo `2**31 - 1`. It does not call Python's `hash`. The first executed task is `pick_and_place` with exactly one support and one object. `attached_to`, `aligned_with`, `facing`, and `reachable_from` fail closed. `reachable_from` stays a robot check.
