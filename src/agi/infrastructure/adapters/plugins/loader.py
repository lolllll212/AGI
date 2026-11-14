"""Load extra tools from a directory of plugins.

A plugin is a folder holding ``plugin.py`` (or a single ``.py`` file) that
defines ``tools()`` returning a list of :class:`~agi.domain.entities.tool.Tool`:

    plugins/
      math_tools/plugin.py
      text_tools.py

Plugin code runs inside the server process with the server's own permissions.
It is *not* sandboxed, so load only directories you would be willing to import.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agi.domain.entities.tool import Tool


class PluginError(RuntimeError):
    """A plugin could not be loaded, or did not provide what a plugin must."""


def discover(directory: str | Path) -> list[tuple[str, Path]]:
    """``(name, file)`` for every plugin in ``directory``, in name order."""
    root = Path(directory)
    if not root.is_dir():
        raise PluginError(f"plugin directory {str(root)!r} does not exist")
    found = []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir() and (entry / "plugin.py").is_file():
            found.append((entry.name, entry / "plugin.py"))
        elif entry.is_file() and entry.suffix == ".py":
            found.append((entry.stem, entry))
    return found


def load_plugin(name: str, file: Path) -> list[Tool]:
    module_name = f"agi_plugin_{name}"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise PluginError(f"plugin {name!r}: cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise PluginError(f"plugin {name!r} failed while importing: {type(exc).__name__}: {exc}") from exc
    factory = getattr(module, "tools", None)
    if not callable(factory):
        raise PluginError(f"plugin {name!r} must define a function tools() that returns a list of Tool objects")
    try:
        tools = list(factory())
    except Exception as exc:
        raise PluginError(f"plugin {name!r}: tools() failed: {type(exc).__name__}: {exc}") from exc
    for tool in tools:
        if not isinstance(tool, Tool):
            raise PluginError(f"plugin {name!r}: tools() returned {type(tool).__name__}, not a Tool")
    return tools


def load_plugins(directory: str | Path) -> list[Tool]:
    """Every tool from every plugin in ``directory``. Two plugins may not define the same tool name."""
    tools: list[Tool] = []
    owner: dict[str, str] = {}
    for name, file in discover(directory):
        for tool in load_plugin(name, file):
            if tool.name in owner:
                raise PluginError(f"plugins {owner[tool.name]!r} and {name!r} both define a tool named {tool.name!r}")
            owner[tool.name] = name
            tools.append(tool)
    return tools
