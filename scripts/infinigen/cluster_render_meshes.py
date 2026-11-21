"""Write tolerance-bounded render copies of meshes that do not fit on the GPU.

The archived OBJ files stay in place. A copy is written only when clustering
reduces the face count and the measured vertex move is within
max(1 mm, 0.1% of the source diagonal).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensor_sim.scene_gen.geometry_compare import surface_tolerance_m
from sensor_sim.scene_gen.mesh_cluster import cluster_surface

PACKAGE = Path("dataset/qualification/asset_realism/exported/LivingRoom/21/mitsuba")
DEST = PACKAGE / "meshes_render"
REPORTS = Path("dataset/qualification/asset_realism/reports")
FACE_LIMIT = 80000


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices = []
    uvs = []
    faces_v = []
    faces_t = []
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            vertices.append(np.fromstring(line[2:], sep=" "))
        elif line.startswith("vt "):
            uvs.append(np.fromstring(line[3:], sep=" ")[:2])
        elif line.startswith("f "):
            corners = line.split()[1:]
            parsed = []
            for token in corners:
                parts = token.split("/")
                parsed.append((int(parts[0]) - 1, int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None))
            for index in range(1, len(parsed) - 1):
                tri = [parsed[0], parsed[index], parsed[index + 1]]
                faces_v.append([item[0] for item in tri])
                faces_t.append([item[1] if item[1] is not None else -1 for item in tri])
    points = np.vstack(vertices) if vertices else np.zeros((0, 3))
    triangles = np.asarray(faces_v, dtype=np.int64) if faces_v else np.zeros((0, 3), dtype=np.int64)
    if uvs and faces_t:
        return points, triangles, np.asarray(faces_t, dtype=np.int64), np.vstack(uvs)
    return points, triangles, None, None


def _write(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        if uvs is not None:
            for uv in uvs:
                handle.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for face in faces:
                one = face + 1
                handle.write(f"f {one[0]}/{one[0]} {one[1]}/{one[1]} {one[2]}/{one[2]}\n")
        else:
            for face in faces:
                one = face + 1
                handle.write(f"f {one[0]} {one[1]} {one[2]}\n")


def _cluster_uvs(inverse: np.ndarray, faces: np.ndarray, uv_index: np.ndarray, uv_values: np.ndarray, output_vertices: int) -> np.ndarray:
    summed = np.zeros((output_vertices, 2), dtype=np.float64)
    count = np.zeros(output_vertices, dtype=np.float64)
    corners = inverse[faces].reshape(-1)
    texels = uv_index.reshape(-1)
    valid = texels >= 0
    np.add.at(summed, corners[valid], uv_values[texels[valid]])
    np.add.at(count, corners[valid], 1.0)
    missing = count == 0
    count[missing] = 1.0
    averaged = summed / count[:, None]
    averaged[missing] = 0.0
    return averaged


def main() -> None:
    manifest = json.loads((PACKAGE / "scene_manifest_qualified.json").read_text(encoding="utf-8"))
    graph = json.loads(Path("dataset/qualification/asset_realism/exported/LivingRoom/21/blend_graph.json").read_text(encoding="utf-8"))
    secondary = next(item for item in graph["objects"] if item["name"] == "camera_0_1")
    seen: dict[str, dict] = {}
    rows = []
    for instance in manifest["instances"]:
        for region in instance["regions"]:
            source = region["mesh"]
            if source in seen:
                region["mesh"] = seen[source]["mesh"]
                continue
            path = PACKAGE / source
            points, triangles, uv_index, uv_values = _load(path)
            if len(triangles) < FACE_LIMIT:
                seen[source] = {"mesh": source, "clustered": False}
                continue
            tolerance = surface_tolerance_m(points)
            cell = float(tolerance / np.sqrt(3.0))
            report = cluster_surface(points, triangles, float(cell))
            if report["max_move_m"] > tolerance + 1e-9:
                rows.append({"mesh": source, "clustered": False, "reason": "move_above_tolerance", **{key: report[key] for key in ("max_move_m", "input_faces", "output_faces")}})
                seen[source] = {"mesh": source, "clustered": False}
                continue
            uvs = None
            if uv_index is not None:
                # inverse is recovered by querying the welded mean. Recompute from cluster_surface via positions.
                # cluster_surface does not return inverse. Average UVs by repeating the quantization.
                shift = points.min(axis=0)
                quantized = np.floor((points - shift) / cell).astype(np.int64)
                span = quantized.max(axis=0) + 1
                code = quantized[:, 0] + span[0] * (quantized[:, 1] + span[1] * quantized[:, 2])
                _unique, inverse = np.unique(code, return_inverse=True)
                uvs = _cluster_uvs(inverse, triangles, uv_index, uv_values, report["output_vertices"])
            destination = DEST / Path(source).name
            _write(destination, report["vertices"], report["faces"], uvs)
            relative = destination.relative_to(PACKAGE).as_posix()
            record = {
                "mesh": source,
                "render_mesh": relative,
                "clustered": True,
                "tolerance_m": tolerance,
                "max_move_m": report["max_move_m"],
                "mean_move_m": report["mean_move_m"],
                "bound_m": report["bound_m"],
                "input_faces": report["input_faces"],
                "output_faces": report["output_faces"],
                "input_vertices": report["input_vertices"],
                "output_vertices": report["output_vertices"],
                "cell_m": report["cell_m"],
            }
            rows.append(record)
            seen[source] = {"mesh": relative, "clustered": True}
            region["mesh"] = relative
            print(
                f"event=clustered faces={report['input_faces']}->{report['output_faces']} move_m={report['max_move_m']:.6f} tol_m={tolerance:.6f} mesh={Path(source).name}",
                flush=True,
            )
    for instance in manifest["instances"]:
        for region in instance["regions"]:
            if region["mesh"] in seen:
                region["mesh"] = seen[region["mesh"]]["mesh"]
    manifest["environment"]["radiance"] = 1.2
    manifest["environment"]["reason"] = "labeled_replacement_for_unconverted_world_shader"
    origin = [float(secondary["matrix_world"][row][3]) for row in range(3)]
    forward = np.array([-secondary["matrix_world"][row][2] for row in range(3)], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    manifest["secondary_camera"] = dict(manifest["camera"])
    manifest["secondary_camera"].update({
        "id": "camera_0_1",
        "origin": origin,
        "target": (np.asarray(origin) + forward).tolist(),
        "matrix_world": secondary["matrix_world"],
    })
    manifest["render_mesh_note"] = "meshes_render contains grid-welded copies. The archived OBJ is unchanged. geometry_valid uses the archived mesh."
    (PACKAGE / "scene_manifest_render.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "clustered": sum(1 for row in rows if row.get("clustered")),
        "input_faces": sum(row.get("input_faces", 0) for row in rows),
        "output_faces": sum(row.get("output_faces", 0) for row in rows),
        "max_move_m": max((row.get("max_move_m", 0.0) for row in rows), default=0.0),
        "definition": "Vertices in one grid cell are replaced by their mean. The cell diagonal equals the export tolerance, because a vertex can move by almost that full diagonal.",
    }
    (REPORTS / "livingroom_21_render_mesh.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("event=cluster_done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
