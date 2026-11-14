"""TLS for the HTTP API."""
from __future__ import annotations

import ssl


def server_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    """A server-side context (TLS 1.2 or later), or ``None`` when no certificate is given."""
    if keyfile and not certfile:
        raise ValueError("keyfile was given without certfile")
    if not certfile:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile, keyfile)
    return context
