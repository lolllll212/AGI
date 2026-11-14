from __future__ import annotations

from pathlib import Path

from agi.application.interfaces.service import AgentService
from agi.eval.suite import write_fixtures

from .models import sales_model


def sales_service(directory: Path, **options) -> AgentService:
    """A service over the benchmark fixtures in ``directory/ws``, answered by :func:`sales_model`."""
    return AgentService.create(sales_model, workspace=write_fixtures(directory / "ws"), **options)
