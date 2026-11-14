"""The HTTP(S) server: :mod:`.server` listens, :mod:`.routes` answers, :mod:`.static` holds the web console."""
from .schemas import HTTPError, Response
from .server import AgentServer, serve

__all__ = ["AgentServer", "HTTPError", "Response", "serve"]
