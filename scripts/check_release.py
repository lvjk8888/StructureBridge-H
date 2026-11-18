"""Compile and import public modules without writing bytecode or running drivers."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys


def main():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    files = sorted(root.rglob("*.py"))
    modules = []
    for path in files:
        if path.is_symlink():
            raise RuntimeError("Symbolic links are forbidden in the release")
        compile(path.read_bytes(), str(path), "exec")
        rel = path.relative_to(root)
        if rel.parts[0] in {"sbh", "experiments"}:
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts:
                name = ".".join(parts)
                importlib.import_module(name)
                modules.append(name)
    print(f"Syntax passed: {len(files)} Python files; imports passed: {len(modules)} modules.")


if __name__ == "__main__":
    main()
