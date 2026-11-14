"""The layer rules hold, the checker itself catches a breach, and pre-0.5 import paths still work."""
import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import layers  # noqa: E402


def test_no_layer_imports_a_layer_above_it():
    assert layers.violations() == []


def test_every_package_directory_belongs_to_a_layer():
    top = {path.name for path in layers.ROOT.iterdir() if path.is_dir() and not path.name.startswith("_")}
    assert top == set(layers.ALLOWED)
    assert layers.ALLOWED["domain"] == set()


def test_the_checker_reports_a_breach(tmp_path, monkeypatch):
    for layer in layers.ALLOWED:
        (tmp_path / layer).mkdir()
    (tmp_path / "domain" / "bad.py").write_text("from agi.infrastructure.api import server\n")
    (tmp_path / "application" / "lazy.py").write_text("def f():\n    import agi.eval.suite\n")
    (tmp_path / "application" / "root.py").write_text("import agi\n")
    (tmp_path / "application" / "fine.py").write_text("from agi.domain.exceptions import LLMError\nimport json\n")
    monkeypatch.setattr(layers, "ROOT", tmp_path)
    problems = layers.violations()
    assert len(problems) == 3
    assert any("domain/bad.py:1 (domain) imports agi.infrastructure.api" in p.replace("\\", "/") for p in problems)
    assert any("lazy.py:2 (application) imports agi.eval.suite" in p for p in problems)
    assert any("root.py:1 imports the package root" in p for p in problems)


@pytest.mark.parametrize("old, name, new", [
    ("agi.core.engine", "CognitiveEngine", "agi.application.cortex.engine"),
    ("agi.core.llm", "ScriptedLLM", "agi.infrastructure.adapters.llm.scripted"),
    ("agi.core.sandbox", "Sandbox", "agi.infrastructure.adapters.sandbox.process"),
    ("agi.tools.base", "Tool", "agi.domain.entities.tool"),
    ("agi.memory.graph", "KnowledgeGraph", "agi.infrastructure.adapters.persistence.knowledge_graph"),
    ("agi.api.service", "AgentService", "agi.application.interfaces.service"),
    ("agi.api.http", "AgentServer", "agi.infrastructure.api.server"),
    ("agi.evaluation.dataset", "export_jsonl", "agi.training.dataset"),
])
def test_import_paths_from_before_0_5_still_work_and_say_where_to_go(old, name, new):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match="import path from before 0.5"):
        module = importlib.import_module(old)
    assert getattr(module, name) is getattr(importlib.import_module(new), name)


def test_the_new_paths_import_without_any_warning():
    code = ("import warnings; warnings.simplefilter('error'); import agi; "
            "from agi.application.cortex.engine import CognitiveEngine; "
            "from agi.infrastructure.api.server import AgentServer; print(agi.__version__)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "0.6.0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.import_module("agi.domain.ports")
