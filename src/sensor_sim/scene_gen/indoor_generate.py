"""Default indoor generation. Infinigen is the source unless the user names another.

``generate.py`` remains the Dataset v1 generator. ``indoor_cli.py`` remains the
legacy assembler. New indoor requests come here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sensor_sim.scene_gen.infinigen_backend import InfinigenBackendError, installation_status, require_installation, run_generation

logger = logging.getLogger("sensor_sim.indoor_generate")


class IndoorGenerateError(RuntimeError):
    """The request cannot be generated with the selected source."""


def generate(request: dict, *, indoors_root: Path | None = None, indoors_python: Path | None = None) -> dict:
    """Route one indoor request. A missing Infinigen install or generator fails closed.

    ``request`` keys: ``kind`` (``room`` or ``asset``), ``category``, ``seed``,
    ``output``, optional ``source`` (``infinigen`` or ``sensor_sim``), optional ``fast``.
    """
    source = request.get("source", "infinigen")
    if source == "sensor_sim":
        if request.get("explicit_source") is not True:
            raise IndoorGenerateError("reason=source_override_not_explicit")
        logger.info("indoor_generate event=override source=sensor_sim category=%s", request.get("category"))
        return {"source": "sensor_sim", "explicit": True, "status": "not_generated_in_this_milestone", "category": request.get("category")}
    if source != "infinigen":
        raise IndoorGenerateError(f"reason=unknown_source value={source}")
    try:
        status = require_installation(indoors_root, indoors_python)
    except InfinigenBackendError as exc:
        raise IndoorGenerateError(str(exc)) from exc
    category = str(request.get("category") or "")
    if request.get("kind") == "asset" and category not in {"sofa", "chair", "table", "lamp", "cabinet"}:
        raise IndoorGenerateError(f"reason=generator_missing category={category}")
    if request.get("kind") == "room" and category not in {"LivingRoom", "Kitchen", "Bedroom", "DiningRoom", "Bathroom", "Office"}:
        raise IndoorGenerateError(f"reason=generator_missing category={category}")
    if request.get("execute") is not True:
        return {"source": "infinigen_indoors", "status": "routed", "category": category, "commit": status["commit"], "fast": bool(request.get("fast", True))}
    output = Path(request["output"])
    record = run_generation(category, int(request.get("seed", 0)), output, fast=bool(request.get("fast", True)), root=Path(status["root"]), python=Path(status["python"]))
    record["dispatcher"] = "indoor_generate"
    logger.info("indoor_generate event=infinigen category=%s seed=%s fast=%s", category, request.get("seed"), request.get("fast", True))
    return record


def describe_default() -> dict:
    status = installation_status()
    return {"default_source": "infinigen_indoors", "legacy_cli": "sensor_sim.scene_gen.indoor_cli", "installation": status}
