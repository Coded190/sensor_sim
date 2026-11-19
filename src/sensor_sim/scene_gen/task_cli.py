"""Offline commands for task-conditioned scenes.

Generation does not call a remote model. ``replay`` does not retrieve assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_sim.scene_gen.task_replay import generate, replay
from sensor_sim.scene_gen.task_schema import SCHEMA_VERSION, TaskError, load_task


def book_on_table(seed: int = 17011) -> dict:
    return _task("book_on_table", "Place a book on a table", seed, [_obj("book_1", "book", ["rest"]), _obj("table_1", "table", ["support"])], [
        _support("book_1", "table_1", "top")
    ])


def cup_on_shelf(seed: int = 17011) -> dict:
    return _task("cup_on_shelf", "Place a cup on a shelf", seed, [_obj("cup_1", "cup", ["rest"]), _obj("shelf_1", "shelf", ["support"])], [
        _support("cup_1", "shelf_1", "level_0")
    ])


def book_inside_crate(seed: int = 17011) -> dict:
    return _task(
        "book_inside_crate",
        "Place a book inside a crate",
        seed,
        [_obj("book_1", "book", ["rest"]), _obj("crate_1", "crate", ["contain"])],
        [_support("book_1", "crate_1", "interior_bottom"), _inside("book_1", "crate_1")],
    )


def crate_and_book_on_table(seed: int = 17011) -> dict:
    return _task(
        "crate_and_book_on_table",
        "Place a crate and a book on a table without collision",
        seed,
        [_obj("book_1", "book", ["rest"]), _obj("crate_1", "crate", ["rest"]), _obj("table_1", "table", ["support"])],
        [
            _support("book_1", "table_1", "top"),
            _support("crate_1", "table_1", "top"),
            {"type": "keep_clear_of", "instances": ["book_1", "crate_1"], "frame": "world", "priority": "required", "constraints": {"minimum_m": 0.001}},
        ],
    )


def cup_near_book(seed: int = 17011) -> dict:
    return _task(
        "cup_near_book",
        "Place a cup near a book on a table",
        seed,
        [_obj("book_1", "book", ["rest"]), _obj("cup_1", "cup", ["rest"]), _obj("table_1", "table", ["support"])],
        [
            _support("book_1", "table_1", "top"),
            _support("cup_1", "table_1", "top"),
            {
                "type": "near",
                "instances": ["cup_1", "book_1"],
                "frame": "world",
                "priority": "required",
                "constraints": {"minimum_m": 0.001, "maximum_m": 0.30},
            },
        ],
    )


def objects_in_workspace(seed: int = 17011) -> dict:
    task = book_on_table(seed)
    task["task_id"] = "objects_in_workspace"
    task["name"] = "Keep objects inside a declared workspace"
    task["description"] = "The book and table stay inside a world-space workspace. No robot is implied."
    task["workspace_m"] = {"minimum_m": [-0.1, -0.1, -0.1], "maximum_m": [2.0, 2.0, 2.0]}
    return task


def evaluation_tasks(seed: int = 17011) -> list[dict]:
    return [
        book_on_table(seed),
        cup_on_shelf(seed),
        book_inside_crate(seed),
        crate_and_book_on_table(seed),
        cup_near_book(seed),
        objects_in_workspace(seed),
    ]


def reachable_task() -> dict:
    task = book_on_table()
    task["task_id"] = "reachable_without_robot"
    task["constraints"].append(
        {"type": "reachable_from", "instances": ["book_1", "table_1"], "frame": "world", "priority": "required", "constraints": {}}
    )
    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic local task-conditioned scenes")
    commands = parser.add_subparsers(dest="command", required=True)
    generate_command = commands.add_parser("generate")
    generate_command.add_argument("--task", type=Path, required=True)
    generate_command.add_argument("--catalog", type=Path, default=Path("assets/candidates"))
    generate_command.add_argument("--output", type=Path, default=Path("dataset/qualification/task_scenes"))
    generate_command.add_argument("--top-k", type=int, default=10)
    validate_command = commands.add_parser("validate")
    validate_command.add_argument("--task", type=Path, required=True)
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("--bundle", type=Path, required=True)
    replay_command = commands.add_parser("replay")
    replay_command.add_argument("--bundle", type=Path, required=True)
    replay_command.add_argument("--catalog", type=Path, default=Path("assets/candidates"))
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--catalog", type=Path, default=Path("assets/candidates"))
    evaluate_command.add_argument("--output", type=Path, default=Path("dataset/qualification/task_scenes"))
    evaluate_command.add_argument("--seed", type=int, default=17011)
    evaluate_command.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            task = load_task(Path(args.task).read_text(encoding="utf-8"))["task"]
            result = generate(task, args.catalog, args.output, top_k=args.top_k)
            print(json.dumps({"path": result["path"], "valid": result["valid"], "schema_version": SCHEMA_VERSION}))
        elif args.command == "validate":
            load_task(Path(args.task).read_text(encoding="utf-8"))
            print("result=ok")
        elif args.command == "inspect":
            print(Path(args.bundle).joinpath("bundle.json").read_text(encoding="utf-8"))
        elif args.command == "replay":
            result = replay(args.bundle, args.catalog)
            print(json.dumps({"valid": result["valid"]}))
        elif args.command == "evaluate":
            rows = []
            for task in evaluation_tasks(args.seed):
                result = generate(task, args.catalog, args.output, top_k=args.top_k, seed=args.seed)
                rows.append({"task_id": task["task_id"], "valid": result["valid"], "path": result["path"]})
            print(json.dumps(rows))
    except TaskError as exc:
        print(f"result=error detail={exc}")
        return 1
    return 0


def _task(task_id: str, description: str, seed: int, objects: list[dict], constraints: list[dict]) -> dict:
    return {
        "task_id": task_id,
        "name": description,
        "description": description,
        "category": "placement",
        "required_objects": objects,
        "contextual_objects": [],
        "robot": None,
        "workspace_m": None,
        "constraints": constraints,
        "initial_requirements": ["rigid_pose"],
        "goal_requirements": [],
        "seed": int(seed),
    }


def _obj(instance_id: str, query: str, affordances: list[str]) -> dict:
    return {"instance_id": instance_id, "query": query, "affordances": affordances, "role": "required"}


def _support(child: str, parent: str, surface: str) -> dict:
    return {
        "type": "supported_on",
        "instances": [child, parent],
        "frame": "support_surface_local",
        "frame_instance": parent,
        "surface": surface,
        "priority": "required",
        "constraints": {},
    }


def _inside(child: str, parent: str) -> dict:
    return {
        "type": "inside",
        "instances": [child, parent],
        "frame": "container_interior_local",
        "frame_instance": parent,
        "surface": "cavity",
        "priority": "required",
        "constraints": {"margin_m": 0.005},
    }


if __name__ == "__main__":
    raise SystemExit(main())
