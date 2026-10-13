"""HTTP fetching for caller-supplied URIs."""

from __future__ import annotations

__all__ = ["fetch_uri"]

import logging
from urllib.parse import urlparse

import httpx

_logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; everalgo; +https://evermind.ai)"


async def fetch_uri(uri: str, *, timeout: float = 30.0) -> tuple[bytes, str]:
    """Fetch an ``http`` or ``https`` URI, following redirects.

    Any other scheme is refused, so a caller-supplied URI can never read the local filesystem (``file://``) or reach
    another protocol. Private, loopback and link-local addresses are not blocked: a caller that passes untrusted URIs
    must resolve and filter them itself, or fetch through an egress proxy.

    Args:
        uri: The URI to fetch.
        timeout: Timeout of the request, in seconds.

    Returns:
        The response body and its media type, lower-cased and without parameters (``text/html``); the media type is
        empty when the upstream sends no ``Content-Type``.

    Raises:
        ValueError: If ``uri`` is empty or its scheme is not ``http`` or ``https``.
        httpx.HTTPError: If the request fails or the upstream answers with an error status; nothing is retried.
    """
    if not uri:
        raise ValueError("fetch_uri: empty uri")
    scheme = urlparse(uri).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"fetch_uri: only http and https are supported, got scheme {scheme!r}")
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": _USER_AGENT}) as client:
        response = await client.get(uri)
    response.raise_for_status()
    media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    _logger.debug(
        "Fetched %s (status=%d, bytes=%d, type=%s)", uri, response.status_code, len(response.content), media_type
    )
    return response.content, media_type
