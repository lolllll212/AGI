"""``python -m agi``: see :mod:`agi.infrastructure.cli.main`."""
from agi.infrastructure.cli.main import COMMANDS, DEMOS, main

__all__ = ["COMMANDS", "DEMOS", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
