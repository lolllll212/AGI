"""Which layer may import which, read from the source without importing anything."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "agi"

# layer -> the layers it may import from (besides itself)
ALLOWED = {
    "domain": set(),
    "application": {"domain"},
    "eval": {"domain", "application"},
    "training": {"domain", "application", "eval"},
    "infrastructure": {"domain", "application", "eval", "training"},
}


def layer_of(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "agi" and parts[1] in ALLOWED else None


def module_name(path: Path) -> str:
    parts = ("agi",) + path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(path: Path) -> list[tuple[int, str]]:
    """``(line, absolute module)`` for every import in a file, including those inside functions."""
    module = module_name(path)
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            found.append((node.lineno, target))
            if target == "agi":                              # "from agi import infrastructure"
                found += [(node.lineno, f"agi.{alias.name}") for alias in node.names]
    return found


def violations() -> list[str]:
    problems = []
    for path in sorted(ROOT.rglob("*.py")):
        layer = layer_of(module_name(path))
        if layer is None:
            continue
        for line, target in imports(path):
            other = layer_of(target)
            if target == "agi" and layer != "infrastructure":
                problems.append(f"{path.relative_to(ROOT)}:{line} imports the package root, which pulls in every layer")
            elif other and other != layer and other not in ALLOWED[layer]:
                problems.append(f"{path.relative_to(ROOT)}:{line} ({layer}) imports {target} ({other})")
    return problems


if __name__ == "__main__":
    found = violations()
    print("\n".join(found) or "layers are clean")
    raise SystemExit(1 if found else 0)
