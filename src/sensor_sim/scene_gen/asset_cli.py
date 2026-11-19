"""Command line for indoor asset packages.

Candidates and the production library are listed separately. ``register`` publishes
only a package whose external approval record names that package hash.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sensor_sim.scene_gen.asset_package import (
    PackageError,
    format_catalog,
    package_directory,
    parse_strict_json,
    reconcile_library,
    register_package,
    set_candidate_status,
    verify_package,
)
from sensor_sim.scene_gen.asset_validate import validate_package

logger = logging.getLogger("sensor_sim.scene_gen.asset_cli")


def _load_json(path: Path) -> dict:
    parsed = parse_strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise PackageError(f"reason=json_object path={path}")
    return parsed


def _roles(values: list[str] | None) -> dict[str, str]:
    roles: dict[str, str] = {}
    for item in values or []:
        path, separator, role = item.partition("=")
        if separator != "=" or path == "" or role == "":
            raise PackageError(f"reason=role_syntax value={item}")
        roles[path] = role
    return roles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, validate, and register indoor asset packages")
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("package", help="Create a candidate package from a directory of regular files")
    package.add_argument("--manifest", type=Path, required=True)
    package.add_argument("--source", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    package.add_argument("--role", action="append", default=[])

    inspect = commands.add_parser("inspect", help="Print a verified package manifest")
    inspect.add_argument("--package", type=Path, required=True)

    validate = commands.add_parser("validate", help="Validate package schema, paths, and hashes")
    validate.add_argument("--package", type=Path, required=True)

    verify = commands.add_parser("verify", help="Recompute the package hash from file bytes")
    verify.add_argument("--package", type=Path, required=True)

    listing = commands.add_parser("list", help="List the candidate catalog and the production library")
    listing.add_argument("--candidates", type=Path)
    listing.add_argument("--library", type=Path)

    register = commands.add_parser("register", help="Register an admitted package into the production library")
    register.add_argument("--library", type=Path, required=True)
    register.add_argument("--package", type=Path, required=True)
    register.add_argument("--approval", type=Path, required=True)

    reconcile = commands.add_parser("reconcile", help="Report orphan version directories without deleting them")
    reconcile.add_argument("--library", type=Path, required=True)

    mark = commands.add_parser("mark", help="Set technically_validated or renderer_validated on a candidate")
    mark.add_argument("--catalog", type=Path, required=True)
    mark.add_argument("--asset-id", required=True)
    mark.add_argument("--version", required=True)
    mark.add_argument("--status", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "package":
            manifest = package_directory(args.destination, _load_json(args.manifest), args.source, _roles(args.role))
            sys.stdout.write(json.dumps({"package_sha256": manifest["package_sha256"]}, sort_keys=True) + "\n")
        elif args.command == "inspect":
            sys.stdout.write(json.dumps(verify_package(args.package), sort_keys=True) + "\n")
        elif args.command == "validate":
            manifest = validate_package(args.package)
            sys.stdout.write(f"result=ok package_sha256={manifest['package_sha256']}\n")
        elif args.command == "verify":
            manifest = verify_package(args.package)
            sys.stdout.write(f"result=ok package_sha256={manifest['package_sha256']}\n")
        elif args.command == "list":
            sys.stdout.write(format_catalog(args.candidates, args.library))
        elif args.command == "register":
            manifest = register_package(args.library, args.package, _load_json(args.approval))
            sys.stdout.write(f"result=registered package_sha256={manifest['package_sha256']}\n")
        elif args.command == "reconcile":
            orphans = reconcile_library(args.library)
            sys.stdout.write(json.dumps(orphans, sort_keys=True) + "\n")
        elif args.command == "mark":
            entry = set_candidate_status(args.catalog, args.asset_id, args.version, args.status)
            sys.stdout.write(f"result=ok status={entry['status']} package_sha256={entry['package_sha256']}\n")
        else:
            raise PackageError(f"reason=command command={args.command}")
    except PackageError as exc:
        logger.error("asset_cli event=error detail=%s", exc)
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
