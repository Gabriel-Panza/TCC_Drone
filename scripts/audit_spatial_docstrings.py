#!/usr/bin/env python3
"""List public spatial functions without docstrings; read-only audit."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "astar.py",
    "occupancy.py",
    "navigation.py",
    "traversal.py",
    "execution.py",
    "execution_reserve.py",
    "clearance.py",
    "snapshot.py",
)
TRIVIAL_NAMES = {
    "world_to_voxel",
    "voxel_to_world",
    "occupied_voxels",
    "free_voxels",
    "observed_voxels",
    "state",
    "export_map",
    "as_dict",
    "from_dict",
}


def missing_docstrings(path):
    """Return public functions and classes without a direct docstring."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []
    for node in tree.body:
        candidates = [node]
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            candidates.extend(node.body)
        for candidate in candidates:
            if (
                isinstance(
                    candidate,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
                and not candidate.name.startswith("_")
                and candidate.name not in TRIVIAL_NAMES
                and ast.get_docstring(candidate) is None
            ):
                missing.append((candidate.lineno, candidate.name))
    return missing


def main():
    """Print a stable report and fail when public documentation is missing."""

    missing = []
    for name in MODULES:
        path = ROOT / "spatial_mapping" / name
        missing.extend(
            (str(path.relative_to(ROOT)), line, symbol)
            for line, symbol in missing_docstrings(path)
        )
    for path, line, symbol in missing:
        print(f"{path}:{line}: {symbol}")
    if missing:
        raise SystemExit(2)
    print("Docstrings publicas: OK")


if __name__ == "__main__":
    main()
