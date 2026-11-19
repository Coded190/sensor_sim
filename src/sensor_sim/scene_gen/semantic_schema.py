"""Curated semantic recipes. Text does not invent assets or relations."""

from __future__ import annotations

SCHEMA_VERSION = "semantic_recipe_v1"
ROLES = frozenset({"manipulation_target", "tool", "material", "container", "context", "furniture"})
CLUTTER = frozenset({"low", "moderate", "high"})
RELATIONS = frozenset({"supported_on", "inside", "near", "keep_clear_of"})


class RecipeError(ValueError):
    """A semantic recipe or placement is not valid."""


def require_recipe(recipe: dict) -> dict:
    required = {
        "schema_version",
        "scene_type",
        "room_profile",
        "furniture_instances",
        "object_instances",
        "functional_zones",
        "clutter_level",
        "seed",
    }
    missing = required - set(recipe)
    if missing:
        raise RecipeError(f"reason=missing_fields fields={sorted(missing)}")
    if recipe["schema_version"] != SCHEMA_VERSION:
        raise RecipeError("reason=schema_version")
    if recipe["clutter_level"] not in CLUTTER:
        raise RecipeError("reason=clutter_level")
    if not recipe["furniture_instances"]:
        raise RecipeError("reason=furniture_instances")
    seen = set()
    for collection in (recipe["furniture_instances"], recipe["object_instances"]):
        for item in collection:
            if item["instance_id"] in seen:
                raise RecipeError(f"reason=duplicate_instance id={item['instance_id']}")
            seen.add(item["instance_id"])
            if item.get("role") not in ROLES:
                raise RecipeError(f"reason=role id={item['instance_id']}")
    zones = {zone["id"] for zone in recipe["functional_zones"]}
    for item in recipe["object_instances"]:
        if item["zone_id"] not in zones:
            raise RecipeError(f"reason=zone id={item['instance_id']}")
    return recipe
