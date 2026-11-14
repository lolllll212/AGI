"""Settings for ``python -m agi serve``, read from a file.

    python -m agi serve --config config/agi.toml

The file may be TOML (Python 3.11 or later) or JSON. Every key is optional and
has the same name and meaning as the command-line flag (``--max-steps`` is
``max_steps``). A flag given on the command line wins over the file. Secrets do
not belong in this file: the model key and the API token are read from the
environment variables named by ``api_key_env`` and ``token_env``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from agi.domain.value_objects.records import Model


class ConfigError(ValueError):
    """The settings file is missing, unreadable or wrong."""


class Settings(Model):
    # the model
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    demo: bool = False
    # what the agent may touch
    workspace: str | None = None
    read_only: bool = False
    network: bool = False
    sandbox: Literal["process", "wasm", "docker", "auto"] = "process"
    plugins: str | None = None
    # how it thinks
    max_steps: int = Field(default=12, ge=1, le=100)
    context_tokens: int = Field(default=6000, ge=0)
    # where it keeps things
    state: str | None = None
    memory: str | None = None
    # the server
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=0, le=65535)
    token_env: str = "AGI_API_TOKEN"
    token_file: str | None = None
    keys_file: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None
    rate_limit: float | None = Field(default=None, gt=0)
    max_connections: int = Field(default=64, ge=1)
    ui: bool = True
    metrics: bool = True
    allow_framing: bool = False
    # observability
    traces: str | None = None
    otlp_endpoint: str | None = None
    access_log: str | None = None


def load_settings(path: str | Path) -> dict:
    """The settings a file gives, validated. Only the keys present in the file are returned."""
    file = Path(path)
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read {str(file)!r}: {exc.strerror or exc}") from None
    try:
        if file.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            try:
                import tomllib
            except ImportError:                             # Python 3.10
                raise ConfigError("TOML settings need Python 3.11 or later; use a .json file instead") from None
            data = tomllib.loads(raw.decode("utf-8"))
    except ConfigError:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{str(file)!r} is not valid {'JSON' if file.suffix.lower() == '.json' else 'TOML'}: "
                          f"{exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{str(file)!r} must contain a table of settings")
    flat: dict = {}
    for key, value in data.items():                         # allow [server], [model] ... sections for readability
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    try:
        settings = Settings.model_validate(flat)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'file'}: {e['msg']}" for e in exc.errors()[:8])
        raise ConfigError(f"{str(file)!r}: {problems}") from None
    return settings.model_dump(include=set(flat))
