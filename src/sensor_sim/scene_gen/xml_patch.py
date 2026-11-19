"""Turn an add-on XML file into the XML Mitsuba renders.

``scene_exported.xml`` stays the add-on output. ``scene.xml`` is a copy with
the Stokes integrator, a box filter, exterior IOR 1 on dielectrics that
omitted it, and any BSDF the config replaces.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _float_text(value: float) -> str:
    return repr(float(value))


def _add_value(parent: ET.Element, name: str, value) -> None:
    if isinstance(value, dict) and value.get("type") == "rgb":
        channels = " ".join(_float_text(channel) for channel in value["value"])
        ET.SubElement(parent, "rgb", {"name": name, "value": channels})
        return
    if isinstance(value, str):
        ET.SubElement(parent, "string", {"name": name, "value": value})
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ET.SubElement(parent, "float", {"name": name, "value": _float_text(value)})
        return
    raise TypeError(f"cannot write BSDF parameter {name}={value!r}")


def bsdf_element(bsdf: dict, ident: str) -> ET.Element:
    node = ET.Element("bsdf", {"type": str(bsdf["type"]), "id": ident, "name": ident})
    for key, value in bsdf.items():
        if key == "type":
            continue
        _add_value(node, key, value)
    return node


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local(child.tag) == name]


def _replace(parent: ET.Element, old: ET.Element, new: ET.Element) -> None:
    index = list(parent).index(old)
    parent.remove(old)
    parent.insert(index, new)


def patch_xml(source: Path, dest: Path, replacements: dict[str, dict], max_depth: int) -> list[dict]:
    """Write the rendered XML and return the patch records."""
    tree = ET.parse(source)
    root = tree.getroot()
    patches: list[dict] = []

    for shape in root.iter():
        if _local(shape.tag) == "shape" and shape.get("name") and not shape.get("id"):
            shape.set("id", shape.get("name"))
            patches.append({"path": f"shape.{shape.get('name')}.id", "from": None, "to": shape.get("name")})

    integrators = _children(root, "integrator")
    if len(integrators) != 1:
        raise RuntimeError(f"{source} has {len(integrators)} integrators")
    old = integrators[0]
    old_type = old.get("type")
    depth = max_depth
    for child in old.iter():
        if _local(child.tag) == "integer" and child.get("name") == "max_depth":
            depth = int(float(child.get("value")))
            break
    stokes = ET.Element("integrator", {"type": "stokes", "name": old.get("name") or "integrator"})
    path = ET.SubElement(stokes, "integrator", {"type": "path"})
    ET.SubElement(path, "integer", {"name": "max_depth", "value": str(int(max_depth))})
    _replace(root, old, stokes)
    patches.append({"path": "integrator.type", "from": old_type, "to": "stokes", "inner": "path", "max_depth": int(max_depth), "exported_max_depth": depth})

    for film in root.iter():
        if _local(film.tag) != "film":
            continue
        filters = [child for child in list(film) if _local(child.tag) == "rfilter"]
        if not filters:
            ET.SubElement(film, "rfilter", {"type": "box"})
            patches.append({"path": "sensor.film.rfilter", "from": "default", "to": "box"})
        elif filters[0].get("type") != "box":
            previous = filters[0].get("type")
            filters[0].set("type", "box")
            patches.append({"path": "sensor.film.rfilter", "from": previous, "to": "box"})

    for bsdf in list(root.iter()):
        if _local(bsdf.tag) != "bsdf" or bsdf.get("type") != "dielectric":
            continue
        floats = [child for child in list(bsdf) if _local(child.tag) == "float" and child.get("name") == "ext_ior"]
        if floats:
            continue
        ET.SubElement(bsdf, "float", {"name": "ext_ior", "value": "1.0"})
        patches.append({"path": f"bsdf.{bsdf.get('id')}.ext_ior", "from": None, "to": 1.0})

    by_id = {}
    for bsdf in root.iter():
        if _local(bsdf.tag) == "bsdf" and bsdf.get("id"):
            by_id[bsdf.get("id")] = bsdf
    for ident, bsdf in replacements.items():
        current = by_id.get(ident)
        if current is None:
            raise RuntimeError(f"{source} has no bsdf id {ident}")
        parent = None
        for candidate in root.iter():
            if current in list(candidate):
                parent = candidate
                break
        if parent is None:
            raise RuntimeError(f"bsdf {ident} has no parent")
        _replace(parent, current, bsdf_element(bsdf, ident))
        patches.append({"path": f"bsdf.{ident}", "from": current.get("type"), "to": bsdf["type"]})
    _disambiguate_duplicate_bsdfs(root, patches)
    dest.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return patches


def _bsdf_signature(node: ET.Element) -> tuple:
    parts = [node.get("type")]
    for child in list(node):
        parts.append((_local(child.tag), child.get("name"), child.get("value"), child.get("type")))
    return tuple(parts)


def _parent_of(root: ET.Element, node: ET.Element) -> ET.Element | None:
    for candidate in root.iter():
        if node in list(candidate):
            return candidate
    return None


def _disambiguate_duplicate_bsdfs(root: ET.Element, patches: list[dict]) -> None:
    """Mitsuba 3.9.1 merges shapes whose BSDF parameter lists are identical.

    The merged shape has an empty id, so a second object disappears. The
    rewritten plugin expands to the same eta, alpha, and reflectance. The
    second copy of a rough BSDF spells ``alpha`` as ``alpha_u`` and
    ``alpha_v``. A later copy, or a BSDF with no alpha, is wrapped in a mask
    whose opacity is 1.
    """
    seen: dict[tuple, int] = {}
    for node in list(root.iter()):
        if _local(node.tag) != "bsdf" or not node.get("id"):
            continue
        parent = _parent_of(root, node)
        if parent is not None and _local(parent.tag) == "bsdf":
            continue
        signature = _bsdf_signature(node)
        copy = seen.get(signature, 0)
        seen[signature] = copy + 1
        if copy == 0:
            continue
        if copy == 1 and _split_alpha(node):
            patches.append({"path": f"bsdf.{node.get('id')}.duplicate_guard", "from": "alpha", "to": "alpha_u_v"})
            continue
        _wrap_opaque_mask(root, node, copy)
        patches.append({"path": f"bsdf.{node.get('id')}.duplicate_guard", "from": "identical", "to": f"mask_opacity_1_copy_{copy}"})


def _split_alpha(node: ET.Element) -> bool:
    alphas = [child for child in list(node) if _local(child.tag) == "float" and child.get("name") == "alpha"]
    if not alphas:
        return False
    value = alphas[0].get("value")
    node.remove(alphas[0])
    ET.SubElement(node, "float", {"name": "alpha_u", "value": value})
    ET.SubElement(node, "float", {"name": "alpha_v", "value": value})
    return True


def _wrap_opaque_mask(root: ET.Element, node: ET.Element, copy: int = 1) -> None:
    parent = _parent_of(root, node)
    if parent is None:
        return
    index = list(parent).index(node)
    parent.remove(node)
    mask = ET.Element("bsdf", {"type": "mask", "id": node.get("id"), "name": node.get("name") or node.get("id")})
    if copy <= 1:
        ET.SubElement(mask, "float", {"name": "opacity", "value": "1.0"})
    elif copy == 2:
        ET.SubElement(mask, "spectrum", {"name": "opacity", "value": "1"})
    else:
        ET.SubElement(mask, "rgb", {"name": "opacity", "value": "1 1 1"})
    node.attrib.pop("id", None)
    node.attrib.pop("name", None)
    mask.append(node)
    parent.insert(index, mask)
