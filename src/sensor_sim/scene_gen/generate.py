"""Build one sample: Blender export, XML patch, Stokes EXR, ground truth."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from sensor_sim import __version__
from sensor_sim.film import ROOT
from sensor_sim.film.geometry import write_obj
from sensor_sim.film.images import write_exr
from sensor_sim.film.sfp_math import world_normals_to_camera, camera_matrix
from sensor_sim.scene_gen import GENERATOR_VERSION, SCHEMA_VERSION
from sensor_sim.scene_gen.assets import build_asset
from sensor_sim.scene_gen.blender_export import addon_commit, export_scene
from sensor_sim.scene_gen.checks import analytic_sphere, bsdf_eta, compare_camera, sphere_geometry, stokes_report
from sensor_sim.scene_gen.export_audit import require_logical_shapes
from sensor_sim.scene_gen.identity import indices_for_shape, paint_ids
from sensor_sim.scene_gen.convention import STOKES_FRAME
from sensor_sim.scene_gen.dataset import GROUND_TRUTH, publish, record_failure, sha256
from sensor_sim.scene_gen.dataset_contract import PRODUCTION_DATASET_ID, refuse_production_generation
from sensor_sim.scene_gen.materials import (
    PRIMARY_BSDFS,
    REFERENCE_RESPONSE,
    parameters_match_reference,
    response_class,
    sample_primary,
)
from sensor_sim.scene_gen.placement import place_objects
from sensor_sim.scene_gen.render_sample import (
    choose_variant,
    polarization_mask,
    render_aovs,
    render_stokes,
    stack_stokes,
    write_preview,
)
from sensor_sim.scene_gen.xml_patch import patch_xml

logger = logging.getLogger("sensor_sim.scene_gen")

PROTOTYPE_CAMERA = {
    "position": [0.0, -0.55, 0.32],
    "target": [0.0, 0.0, 0.18],
    "up": [0.0, 0.0, 1.0],
    "fov_deg": 55.0,
    "near_clip": 0.01,
    "far_clip": 100.0,
}
PROTOTYPE_LIGHT = {
    "position": [0.0, -0.02, 0.598],
    "target": [0.0, 0.0, 0.051],
    "up": [0.0, 1.0, 0.0],
    "size": [0.15, 0.15],
    "radiance": 18.0,
}


def child_seeds(seed: int) -> tuple[int, int]:
    """Independent geometry and material seeds from one scene seed."""
    geometry = int(np.random.SeedSequence([int(seed), 1]).generate_state(1)[0])
    material = int(np.random.SeedSequence([int(seed), 2]).generate_state(1)[0])
    return geometry, material


def _blender_version() -> str:
    from sensor_sim.film.blender_build import find_blender

    binary = find_blender()
    if binary is None:
        return "missing"
    text = subprocess.check_output([str(binary), "--version"], text=True)
    return text.splitlines()[0].strip()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_tree_hash() -> str | None:
    """Fingerprint of a dirty tree. Clean checkouts return None."""
    if not _git_dirty():
        return None
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        diff = subprocess.check_output(["git", "diff"], cwd=ROOT)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    extra = []
    for line in untracked.decode().splitlines():
        if line.startswith(("src/sensor_sim/scene_gen/", "configs/scenes/", "assets/blender/", "tests/test_scene_gen", "tests/test_qualification")):
            path = ROOT / line
            if path.is_file():
                extra.append(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b" " + line.encode())
    return hashlib.sha256(status + b"\n" + diff + b"\n" + b"\n".join(extra)).hexdigest()


def _git_dirty() -> bool:
    try:
        text = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(text.strip())


def _versions() -> dict:
    import drjit as dr
    import mitsuba as mi

    return {"mitsuba": mi.__version__, "drjit": dr.__version__}


def prototype_config(seed: int = 0, resolution=(128, 128), spp: int = 16) -> dict:
    """The deterministic sphere. Center z is radius plus the 1 mm gap."""
    return {
        "mode": "prototype",
        "scene_id": "scene_000000",
        "seed": int(seed),
        "geometry_seed": int(seed),
        "material_seed": int(seed),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "spp": int(spp),
        "max_depth": 8,
        "camera": dict(PROTOTYPE_CAMERA),
        "light": dict(PROTOTYPE_LIGHT),
        "objects": [
            {
                "name": "sphere",
                "object_index": 1,
                "instance_id": "sphere:prototype",
                "family": "sphere",
                "asset_id": "sphere",
                "kind": "uv_sphere",
                "radius": 0.05,
                "location": [0.0, 0.0, 0.051],
                "segments": 192,
                "rings": 96,
                "blender_material": "glass",
                "ior": 1.5,
                "bsdf": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
                "material_response": "keeps",
            },
            {
                "name": "ground",
                "object_index": 2,
                "instance_id": "ground:prototype",
                "family": "plane",
                "asset_id": "plane",
                "kind": "plane",
                "size": 0.6,
                "location": [0.0, 0.0, 0.0],
                "color": [0.6, 0.6, 0.6],
                "blender_material": "diffuse",
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
                "material_response": "does_not",
            },
        ],
    }


def _keeping_metals(profiles: dict, bsdf_name: str) -> tuple[str, ...] | None:
    """Metals whose measured profile keeps polarization. Other BSDFs ignore this."""
    if bsdf_name not in ("conductor", "roughconductor"):
        return None
    prefix = f"{bsdf_name}_"
    found = []
    for key, row in profiles.get("metals", {}).items():
        if str(key).startswith(prefix) and row.get("response_class") == "keeps":
            found.append(str(key)[len(prefix) :])
    if not found:
        preset = profiles.get(bsdf_name, {})
        material = preset.get("bsdf", {}).get("material") if isinstance(preset, dict) else None
        if preset.get("response_class") == "keeps" and material:
            found.append(str(material))
    return tuple(found)


def random_config(spec: dict, seed: int) -> dict:
    """One scene. Geometry and material seeds are independent."""
    geometry_seed, material_seed = child_seeds(seed)
    geo = np.random.default_rng(geometry_seed)
    mat = np.random.default_rng(material_seed)
    low, high = spec.get("num_objects", [1, 1])
    forced_objects = spec.get("forced_objects")
    if forced_objects:
        count = len(forced_objects)
    elif spec.get("forced_count") is not None:
        count = int(spec["forced_count"])
    else:
        count = int(geo.integers(int(low), int(high) + 1))
    if count < 1 or count > 4:
        raise RuntimeError(f"object count {count} is outside the Dataset v1 range 1-4")
    raw_assets = spec.get("assets", ["sphere"])
    assets = list(raw_assets.values()) if isinstance(raw_assets, dict) else list(raw_assets)
    if assets and isinstance(assets[0], dict):
        assets = ["sphere"]
    raw_materials = spec.get("materials", list(PRIMARY_BSDFS))
    if isinstance(raw_materials, dict):
        materials = ["dielectric", "pplastic", "conductor", "roughconductor"]
    else:
        materials = list(raw_materials)
    profiles = spec.get("profiles", {})
    allowed = []
    for name in materials:
        entry = profiles.get(name)
        if entry is None and not spec.get("allow_unprofiled", False):
            raise RuntimeError(f"material {name} has no response profile")
        klass = "untested" if entry is None else entry.get("response_class", "untested")
        if klass != "keeps" and not spec.get("include_nonpolarizing", False):
            continue
        allowed.append(name)
    if not allowed:
        raise RuntimeError("no materials are allowed by the profile and the config")
    camera = spec.get("camera", PROTOTYPE_CAMERA)
    light = spec.get("light", PROTOTYPE_LIGHT)
    meshes = []
    objects = []
    pinned = spec.get("forced_preset")
    if pinned is not None and pinned.get("material") == "Au" and spec.get("dataset_id") == PRODUCTION_DATASET_ID:
        raise RuntimeError("gold is excluded from Dataset v1")
    for index in range(count):
        item_spec = forced_objects[index] if forced_objects else {}
        if item_spec.get("family"):
            asset_id = str(item_spec["family"])
        elif spec.get("forced_family"):
            asset_id = str(spec["forced_family"])
        else:
            asset_id = assets[int(geo.integers(0, len(assets)))]
        if asset_id not in ("sphere", "ellipsoid", "rounded_box", "capsule", "torus", "cylinder", "cone", "bumpy"):
            raise RuntimeError(f"unsupported asset family {asset_id} for this generator")
        mesh, shape = build_asset(asset_id, geo, radius=item_spec.get("radius"))
        if pinned is not None:
            bsdf_name, bsdf = sample_primary(mat, preset=pinned)
        else:
            families = [str(spec["forced_material"])] if spec.get("forced_material") else allowed
            bsdf_name, bsdf = sample_primary(mat, families=families)
        response = profiles.get(bsdf_name, {}).get("response_class", "keeps")
        if not parameters_match_reference(bsdf):
            response = "untested"
        realization = json.dumps({"asset_id": asset_id, "schema": SCHEMA_VERSION, "generator": GENERATOR_VERSION, "shape": shape}, sort_keys=True, default=str)
        asset_instance = f"{asset_id}:{hashlib.sha256(realization.encode()).hexdigest()[:16]}"
        objects.append(
            {
                "name": f"object_{index:02d}",
                "object_index": index + 1,
                "instance_id": asset_instance,
                "object_instance_id": f"object_{index:02d}",
                "asset_instance_id": asset_instance,
                "family": shape["family"],
                "asset_id": asset_id,
                "kind": "obj",
                "shape": shape,
                "bsdf": bsdf,
                "material_response": response,
                "mesh": mesh,
            }
        )
        meshes.append(mesh)
    placed = place_objects(
        meshes,
        geo,
        camera,
        seed=seed,
        gap=float(spec.get("gap", spec.get("gap_m", 0.005))),
        max_attempts=max(int(spec.get("placement_attempts", 80)), 400 if spec.get("require_partial_occlusion") else 0),
        require_image_overlap=bool(spec.get("require_partial_occlusion")),
    )
    for item, pose in zip(objects, placed):
        item["yaw"] = pose["yaw"]
        item["xy"] = pose["xy"]
        item["vertices"] = pose["vertices"]
        item["faces"] = item.pop("mesh")["faces"]
    ground = {
        "name": "ground",
        "object_index": count + 1,
        "instance_id": "ground",
        "family": "plane",
        "asset_id": "plane",
        "kind": "plane",
        "size": 0.6,
        "location": [0.0, 0.0, 0.0],
        "color": [0.6, 0.6, 0.6],
        "blender_material": "diffuse",
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
        "material_response": "does_not",
    }
    return {
        "mode": "random",
        "scene_id": str(spec.get("scene_id") or f"scene_{int(seed):06d}"),
        "seed": int(seed),
        "geometry_seed": geometry_seed,
        "material_seed": material_seed,
        "resolution": [int(value) for value in spec.get("resolution", [128, 128])],
        "spp": int(spec.get("spp", 16)),
        "max_depth": int(spec.get("max_depth", 8)),
        "camera": camera,
        "light": light,
        "objects": objects + [ground],
        "dataset_id": spec.get("dataset_id"),
        "generation_authorized": spec.get("generation_authorized", False),
        "forced_preset": pinned,
        "require_partial_occlusion": bool(spec.get("require_partial_occlusion")),
        "repeated_asset": bool(spec.get("repeated_asset")),
        "require_sample_supervision": bool(spec.get("require_sample_supervision")),
    }


def _blender_objects(config: dict, directory: Path) -> list[dict]:
    specs = []
    for item in config["objects"]:
        if item["kind"] == "obj":
            mesh_path = directory / "source" / f"{item['name']}.obj"
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            write_obj(mesh_path, item["vertices"], item["faces"], smooth=True)
            specs.append(
                {
                    "name": item["name"],
                    "kind": "obj",
                    "filename": str(mesh_path),
                    "color": [0.5, 0.5, 0.5],
                    "blender_material": "diffuse",
                    "share_datablock_with": item.get("share_datablock_with"),
                    "object_location": item.get("object_location"),
                }
            )
        else:
            specs.append({key: value for key, value in item.items() if key != "bsdf"})
    return specs


def _ray_pixels(width: int, height: int) -> list[list[int]]:
    return [[width // 2, height // 2], [0, 0], [max(width // 3, 0), min(height // 2, height - 1)]]


def _replacements(config: dict) -> dict[str, dict]:
    if config["mode"] == "prototype":
        return {}
    return {f"mat-obj_{item['name']}": item["bsdf"] for item in config["objects"] if item["kind"] == "obj"}


def _vertex_normal_error(shape, center: list[float]) -> float:
    """Median angle, in degrees, between vertex normals and the sphere radius."""
    point = np.asarray(center, dtype=np.float64)
    count = int(shape.vertex_count())
    step = max(count // 200, 1)
    angles = []
    for index in range(0, count, step):
        position = np.array(shape.vertex_position(index), dtype=np.float64).reshape(-1)[:3]
        normal = np.array(shape.vertex_normal(index), dtype=np.float64).reshape(-1)[:3]
        radial = position - point
        radial /= np.linalg.norm(radial)
        normal /= np.linalg.norm(normal)
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(normal, radial), -1.0, 1.0)))))
    return float(np.median(angles))


def _surface(scene, shape_id: str) -> dict:
    shape = _shape_by_id(scene, shape_id)
    count = int(shape.vertex_count())
    vertices = np.stack([np.array(shape.vertex_position(i)).reshape(-1)[:3] for i in range(count)])
    faces = np.array(shape.faces_buffer()).reshape(-1, 3)
    return {"vertices": vertices, "faces": faces}


def _shape_by_id(scene, shape_id: str):
    for shape in scene.shapes():
        if shape.id() == shape_id:
            return shape
    known = [shape.id() for shape in scene.shapes()]
    raise RuntimeError(f"exported shapes {known} do not include {shape_id}")


def _pixel(sensor, point: list[float], width: int, height: int) -> tuple[int, int]:
    import mitsuba as mi

    transformed = sensor.world_transform().inverse() @ mi.Point3f(float(point[0]), float(point[1]), float(point[2]))
    ndc = np.array(sensor.projection_transform() @ transformed).reshape(-1)
    column = int(np.floor(float(ndc[0]) * width))
    row = int(np.floor(float(ndc[1]) * height))
    if not (0 <= row < height and 0 <= column < width):
        raise RuntimeError(f"point {point} projects outside the image at {(row, column)}")
    return row, column


def _probe_point(item: dict) -> list[float]:
    if item.get("kind") == "plane":
        # The plane's origin sits under the sphere, so the probe is off to one side.
        return [0.22, 0.12, 0.0]
    if item.get("kind") == "uv_sphere":
        return [float(value) for value in item["location"]]
    if "vertices" in item:
        center = np.asarray(item["vertices"], dtype=np.float64).mean(axis=0)
        return [float(value) for value in center]
    raise RuntimeError(f"{item['name']} has no probe point")


def _cloud(shape) -> np.ndarray:
    count = int(shape.vertex_count())
    step = max(count // 250, 1)
    return np.stack(
        [np.array(shape.vertex_position(i), dtype=np.float64).reshape(-1)[:3] for i in range(0, count, step)]
    )


def label_positions(clouds: dict[str, np.ndarray], position: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    """Name of the nearest object at each pixel, or '' when nothing is close."""
    flat = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    best = np.full(flat.shape[0], np.inf)
    names = np.full(flat.shape[0], "", dtype=object)
    finite = np.isfinite(flat).all(axis=1)
    for name, cloud in clouds.items():
        distance = np.linalg.norm(flat[:, None, :] - cloud[None, :, :], axis=-1).min(axis=1)
        closer = finite & (distance < best)
        best[closer] = distance[closer]
        names[closer] = name
    names[best > float(threshold)] = ""
    return names.reshape(position.shape[:2])


def _shape_map(scene, index: np.ndarray, position: np.ndarray, config: dict) -> list[dict]:
    """Match each mesh to the integer shape_index values on its surface.

    ``scene.shapes()`` order is not the AOV index. A fully hidden object keeps
    an empty index list.
    """
    clouds = {}
    for item in config["objects"]:
        clouds[item["name"]] = _cloud(_shape_by_id(scene, f"mesh-{item['name']}"))
    owners = label_positions(clouds, position)
    whole = np.abs(index - np.rint(index)) < 1e-3
    mapping = []
    for item in config["objects"]:
        owned = (owners == item["name"]) & whole & (index > 0.0)
        if int(owned.sum()) == 0:
            indices: list[int] = []
        else:
            indices = sorted({int(round(value)) for value in index[owned]})
        mapping.append(
            {
                "shape_index": None if not indices else indices[0],
                "shape_indices": indices,
                "shape_id": f"mesh-{item['name']}",
                "object_id": item["name"],
                "object_index": int(item["object_index"]),
                "asset_id": item["asset_id"],
                "instance_id": item.get("instance_id", item["name"]),
                "material_id": item.get("material_id", item["bsdf"]["type"]),
                "visible_pixels": int(owned.sum()),
            }
        )
    return mapping


def _log(config: dict, patched: bool) -> None:
    for item in config["objects"]:
        bsdf = item["bsdf"]
        logger.info(
            "scene_gen scene_id=%s seed=%s geometry_seed=%s material_seed=%s object=%s asset=%s "
            "bsdf=%s int_ior=%s alpha=%s patched_bsdf=%s",
            config["scene_id"],
            config["seed"],
            config["geometry_seed"],
            config["material_seed"],
            item["name"],
            item["asset_id"],
            bsdf["type"],
            bsdf.get("int_ior", ""),
            bsdf.get("alpha", ""),
            patched,
        )


def generate_sample(config: dict, output_root: Path, *, overwrite: bool = False) -> Path:
    """Write one complete sample under ``output_root / scene_id``.

    Publication requires a nonempty ``run_id``. Callers that only build a
    config do not go through this function.
    """
    if config.get("dataset_id") == PRODUCTION_DATASET_ID:
        refuse_production_generation(config)
    if not str(config.get("run_id") or "").strip():
        raise RuntimeError("refusing to publish a sample without a run_id")
    if not str(config.get("config_hash") or "").strip():
        config["config_hash"] = hashlib.sha256(
            json.dumps(_jsonable({key: value for key, value in config.items() if key != "config_hash"}), sort_keys=True).encode()
        ).hexdigest()
    output_root = Path(output_root)
    scene_id = config["scene_id"]
    destination = output_root / scene_id
    if destination.is_dir() and (destination / "metadata.json").is_file() and not overwrite:
        logger.info("scene_gen scene_id=%s status=skip", scene_id)
        return destination
    if overwrite and destination.exists():
        import shutil

        shutil.rmtree(destination)
    staging_root = output_root / ".staging"
    staging = staging_root / scene_id
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _build(config, staging)
        publish(staging, destination)
    except Exception as exc:
        record_failure(output_root / "failures", scene_id, str(exc))
        raise
    logger.info("scene_gen scene_id=%s status=published path=%s", scene_id, destination)
    return destination


def _annotate_visibility(mapping: list[dict], object_ids: np.ndarray, camera_shading: np.ndarray) -> None:
    """Keep ``visible_pixels`` as the ownership count and record the image count."""
    from sensor_sim.scene_gen.dataset import (
        TRAINING_MIN_OBJECT_PIXELS,
        discrete_object_ids,
        normal_valid_mask,
        raw_object_counts,
    )

    labels, fractional = discrete_object_ids(object_ids)
    if int(fractional.sum()):
        raise RuntimeError(f"object-id image has {int(fractional.sum())} fractional pixels")
    known = {int(item["object_index"]) for item in mapping}
    unknown = (labels > 0) & ~np.isin(labels, np.array(sorted(known), dtype=np.int32))
    if int(unknown.sum()):
        raise RuntimeError(f"object-id image has {int(unknown.sum())} unknown non-background pixels")
    counts = raw_object_counts(object_ids)
    valid = normal_valid_mask(camera_shading)
    for item in mapping:
        label = int(item["object_index"])
        assigned = labels == label
        raw_count = int(counts.get(label, 0))
        normal_valid = int(np.count_nonzero(assigned & valid))
        item["object_id_pixel_count"] = raw_count
        item["normal_valid_pixel_count"] = normal_valid
        item["normal_supervised_pixel_count"] = normal_valid if raw_count >= TRAINING_MIN_OBJECT_PIXELS else 0


def _intended_objects(mapping: list[dict]) -> list[dict]:
    return [item for item in mapping if item.get("asset_id") != "plane" and str(item.get("object_id")) != "ground"]


def _enforce_slot_rules(config: dict, mapping: list[dict]) -> None:
    """Retry the same slot when a requested scene is not eligible for supervision."""
    from sensor_sim.scene_gen.dataset import TRAINING_MIN_OBJECT_PIXELS
    from sensor_sim.scene_gen.dataset_contract import PRODUCTION_DATASET_ID

    intended = _intended_objects(mapping)
    production = config.get("dataset_id") == PRODUCTION_DATASET_ID or config.get("require_sample_supervision")
    if production:
        eligible = [
            item
            for item in intended
            if int(item["object_id_pixel_count"]) >= TRAINING_MIN_OBJECT_PIXELS and int(item["normal_supervised_pixel_count"]) >= 1
        ]
        if not eligible:
            raise RuntimeError("sample-level normal supervision failed: no intended object has 80 pixels and a valid normal")
    if config.get("repeated_asset"):
        objects = [item for item in config["objects"] if item.get("kind") == "obj"]
        families = {item.get("asset_id") for item in objects}
        instances = [item.get("asset_instance_id") for item in objects]
        transforms = [
            (round(float(item["xy"][0]), 6), round(float(item["xy"][1]), 6), round(float(item["yaw"]), 6))
            for item in objects
        ]
        if len(objects) < 2 or len(families) != 1 or len(set(instances)) != len(instances) or len(set(transforms)) != len(transforms):
            raise RuntimeError("repeated-asset identity failed")
        visible = [item for item in intended if int(item["object_id_pixel_count"]) >= TRAINING_MIN_OBJECT_PIXELS]
        if len(visible) < 2:
            raise RuntimeError("repeated-asset instances are not both visible")
    if config.get("require_partial_occlusion"):
        visible = [
            item
            for item in intended
            if int(item["object_id_pixel_count"]) >= TRAINING_MIN_OBJECT_PIXELS and int(item["normal_supervised_pixel_count"]) >= 1
        ]
        if len(visible) < 2:
            raise RuntimeError("partial occlusion objects are not both supervised")


def _reject_nonuniform_scale(config: dict) -> None:
    """Dataset v1 yaws about Z. Nonuniform scale is not a supported transform."""
    for item in config["objects"]:
        scale = item.get("scale")
        if scale is None:
            continue
        values = [round(float(value), 6) for value in scale]
        if len(set(values)) > 1:
            raise RuntimeError(f"{item['name']} has nonuniform scale {scale}, which Dataset v1 rejects")


def _build(config: dict, staging: Path) -> None:
    _reject_nonuniform_scale(config)
    width, height = int(config["resolution"][0]), int(config["resolution"][1])
    exported = staging / "scene_exported.xml"
    rendered = staging / "scene.xml"
    params = {
        "xml": str(exported),
        "rays": str(staging / "rays.json"),
        "log": str(staging / "blender.log"),
        "params_json": str(staging / "blender_params.json"),
        "resolution": [width, height],
        "spp": int(config["spp"]),
        "max_depth": int(config["max_depth"]),
        "camera": config["camera"],
        "light": config["light"],
        "objects": _blender_objects(config, staging),
        "ray_pixels": _ray_pixels(width, height),
    }
    export_scene(params)
    replacements = _replacements(config)
    patches = patch_xml(exported, rendered, replacements, int(config["max_depth"]))
    _log(config, any(item["path"].startswith("bsdf.mat-") and item["path"].count(".") == 1 for item in patches))
    variant = choose_variant(config.get("variant", "cuda_ad_spectral_polarized"))
    import mitsuba as mi

    mi.set_variant(variant)
    scene = mi.load_file(str(rendered))
    require_logical_shapes(scene, [item["name"] for item in config["objects"]])
    rays = json.loads((staging / "rays.json").read_text())
    camera_report = compare_camera(scene.sensors()[0], config["camera"], rays, width, height)
    sphere = next((item for item in config["objects"] if item["name"] == "sphere"), None) if config.get("mode") == "prototype" else None
    if sphere is not None:
        shape = _shape_by_id(scene, "mesh-sphere")
        sphere_index = [candidate.id() for candidate in scene.shapes()].index("mesh-sphere")
        sphere_geometry(scene, sphere_index, sphere["location"], sphere["radius"])
        eta = bsdf_eta(scene, sphere_index)
        if abs(eta - 1.5) > 1e-5:
            raise RuntimeError(f"sphere dielectric eta is {eta}, expected 1.5")
        if shape.bsdf().class_name() != "SmoothDielectric":
            raise RuntimeError("sphere BSDF is not SmoothDielectric")
        vertex_error = _vertex_normal_error(shape, sphere["location"])
        if vertex_error > 2.0:
            raise RuntimeError(f"sphere vertex normals are {vertex_error:.3f} degrees from the radius")
    stokes_scene, channels, names, bitmap = render_stokes(rendered, variant, int(config["spp"]), int(config["seed"]))
    bitmap.write(str(staging / "polarization.exr"))
    s0 = stack_stokes(channels, "S0")
    s1 = stack_stokes(channels, "S1")
    s2 = stack_stokes(channels, "S2")
    s3 = stack_stokes(channels, "S3")
    report = stokes_report(s0, s1, s2, s3)
    diagnostic = {key: value for key, value in report.items() if key not in ("maps", "analyzers")}
    (staging / "stokes_diagnostics.json").write_text(json.dumps(_jsonable(diagnostic), indent=2) + "\n")
    if int(config["spp"]) >= 64 and report["frac_dop_gt_1_05"] > 0.0:
        raise RuntimeError(f"frac DoP>1.05 is {report['frac_dop_gt_1_05']}")
    write_preview(staging / "preview.png", report["maps"]["s0"])
    _aov_scene, aovs, _aov_names = render_aovs(rendered, variant)
    geo = np.stack([aovs["geo.X"], aovs["geo.Y"], aovs["geo.Z"]], axis=-1)
    shading = np.stack([aovs["nn.X"], aovs["nn.Y"], aovs["nn.Z"]], axis=-1)
    depth = np.asarray(aovs["depth.T"], dtype=np.float64)
    position = np.stack([aovs["pos.X"], aovs["pos.Y"], aovs["pos.Z"]], axis=-1)
    index = np.asarray(aovs["idx.I"], dtype=np.float64)
    matrix = camera_matrix(stokes_scene.sensors()[0])
    truth = staging / "ground_truth"
    truth.mkdir()
    write_exr(truth / "normal_geo_world.exr", geo, ["X", "Y", "Z"])
    write_exr(truth / "normal_geo_camera.exr", world_normals_to_camera(geo, matrix), ["X", "Y", "Z"])
    write_exr(truth / "normal_shading_world.exr", shading, ["X", "Y", "Z"])
    camera_shading = world_normals_to_camera(shading, matrix)
    write_exr(truth / "normal_shading_camera.exr", camera_shading, ["X", "Y", "Z"])
    write_exr(truth / "depth.exr", depth, ["depth"])
    write_exr(truth / "position.exr", position, ["X", "Y", "Z"])
    mapping = _shape_map(scene, index, position, config)
    sensor = scene.sensors()[0]
    height, width = index.shape
    surfaces = {item["shape_id"]: _surface(scene, item["shape_id"]) for item in mapping}
    for item in mapping:
        others = [mesh for key, mesh in surfaces.items() if key != item["shape_id"]]
        item["shape_indices"] = indices_for_shape(
            _shape_by_id(scene, item["shape_id"]),
            index,
            position,
            sensor,
            width,
            height,
            others,
        )
    by_name = {item["object_id"]: item for item in mapping}
    material_of = {}
    for item in config["objects"]:
        key = json.dumps(item["bsdf"], sort_keys=True)
        material_of.setdefault(key, len(material_of) + 1)
        item["material_id"] = material_of[key]
    for item in mapping:
        item["material_id"] = int(next(obj["material_id"] for obj in config["objects"] if obj["name"] == item["object_id"]))
    object_ids, material_ids = paint_ids(index, mapping)
    _annotate_visibility(mapping, object_ids, camera_shading)
    if config.get("require_partial_occlusion"):
        from sensor_sim.scene_gen.dataset import normal_valid_mask
        from sensor_sim.scene_gen.occlusion import partial_occlusion_report

        occlusion = partial_occlusion_report(
            object_ids,
            config["objects"],
            config["camera"],
            width,
            height,
            normal_valid=normal_valid_mask(camera_shading),
        )
        config["occlusion_report"] = occlusion
        if not occlusion["pass"]:
            raise RuntimeError(f"partial occlusion criteria not met {occlusion}")
    _enforce_slot_rules(config, mapping)
    write_exr(truth / "object_ids.exr", object_ids, ["object_id"])
    write_exr(truth / "material_ids.exr", material_ids, ["material_id"])
    lengths = np.linalg.norm(geo, axis=-1)
    foreground = np.isfinite(geo).all(axis=-1) & (np.abs(lengths - 1.0) < 1e-2) & (depth > 0.0)
    write_exr(truth / "valid_mask.exr", foreground.astype(np.float32), ["valid"])
    analytic = None
    if sphere is not None:
        interior = object_ids == float(by_name["sphere"]["object_index"])
        analytic = analytic_sphere(
            geo,
            depth,
            interior,
            center=np.asarray(sphere["location"], dtype=np.float64),
            radius=float(sphere["radius"]),
            origin=np.asarray(config["camera"]["position"], dtype=np.float64),
            sensor=stokes_scene.sensors()[0],
            near_clip=float(camera_report["near_clip"]),
            width=width,
            height=height,
            position=position,
        )
    versions = _versions()
    mesh_hashes = {path.name: sha256(path) for path in (staging / "meshes").glob("*")}
    artifact_paths = [
        staging / "scene.xml",
        staging / "scene_exported.xml",
        staging / "polarization.exr",
    ]
    artifact_paths.extend(staging / "ground_truth" / f"{name}.exr" for name in GROUND_TRUTH)
    artifact_paths.extend(sorted((staging / "meshes").glob("*")))
    artifact_sha256 = {str(path.relative_to(staging)): sha256(path) for path in artifact_paths if path.is_file()}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema_version": SCHEMA_VERSION,
        "visibility_policy_version": 1,
        "invalid_fractional_pixel_count": 0,
        "unknown_object_id_pixel_count": 0,
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "git_diff_hash": _git_tree_hash(),
        "config_hash": config.get("config_hash"),
        "run_id": config.get("run_id"),
        "dataset_id": config.get("dataset_id"),
        "requested_preset": config.get("forced_preset"),
        "loaded_presets": [item.get("bsdf") for item in config["objects"] if item.get("kind") == "obj"],
        "occlusion_report": config.get("occlusion_report"),
        "sample_slot": config.get("sample_slot"),
        "attempt_index": config.get("attempt_index"),
        "stratum": config.get("stratum"),
        "render_seed": int(config.get("render_seed", config["seed"])),
        "scene_seed": int(config["seed"]),
        "scene_id": config["scene_id"],
        "seed": config["seed"],
        "geometry_seed": config["geometry_seed"],
        "material_seed": config["material_seed"],
        "asset_source_hash": {
            item["name"]: item.get("asset_instance_id", item.get("instance_id")) for item in config["objects"]
        },
        "geometry_generator_version": GENERATOR_VERSION,
        "measurement_schema": "15 float32 channels: RGB, S0.RGB, S1.RGB, S2.RGB, S3.RGB; no alpha",
        "ground_truth_convention": "camera-space shading normal is the training target; depth is Mitsuba distance, not pinhole range",
        "status": "complete",
        "blender_version": _blender_version(),
        "addon_commit": addon_commit(),
        "mitsuba": versions["mitsuba"],
        "drjit": versions["drjit"],
        "variant": variant,
        "device": "cuda" if variant.startswith("cuda") else "cpu",
        "spp": int(config["spp"]),
        "resolution": [width, height],
        "rfilter": "box",
        "max_depth": int(config["max_depth"]),
        "camera": config["camera"],
        "light": config["light"],
        "integrator": "stokes/path",
        "channel_names": names,
        "stokes_frame": STOKES_FRAME,
        "xml_patches": patches,
        "patched_bsdf": any(str(item["path"]).startswith("bsdf.mat-obj_") for item in patches),
        "shape_map": mapping,
        "objects": [
            {key: value for key, value in item.items() if key not in ("vertices", "faces", "mesh")}
            for item in config["objects"]
        ],
        "mesh_sha256": mesh_hashes,
        "artifact_sha256": artifact_sha256,
        "logical_shape_ids": [item["shape_id"] for item in mapping],
        "scene_xml_sha256": sha256(rendered),
        "scene_exported_sha256": sha256(exported),
        "camera_report": camera_report,
        "stokes_report": {key: value for key, value in report.items() if key not in ("maps", "analyzers")},
        "analytic_sphere": analytic,
        "polarization_mask_count": int(np.count_nonzero(polarization_mask(foreground, report["maps"]["s0"]))),
    }
    (staging / "scene_config.json").write_text(json.dumps(_jsonable(config), indent=2) + "\n")
    (staging / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key not in ("vertices", "faces", "mesh")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def material_profile(max_dolp: float, mean_dolp: float, fraction: float, mean_dop: float, expected: str) -> dict:
    observed = response_class(max_dolp)
    return {
        "max_dolp": float(max_dolp),
        "mean_dolp": float(mean_dolp),
        "fraction_above_threshold": float(fraction),
        "mean_dop": float(mean_dop),
        "response_class": observed,
        "expected_class": expected,
        "agrees": observed == expected or expected == "untested",
    }
