"""Small bounded HTTP transport for an explicit OpenAI-compatible endpoint."""
from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


MAX_HTTP_RESPONSE_BYTES = 512 * 1024


class TransportError(RuntimeError):
    pass


class TransportTimeout(TransportError):
    pass


class TransportRedirect(TransportError):
    pass


class ResponseTooLarge(TransportError):
    pass


@dataclass(frozen=True)
class TransportRequest:
    url: str
    headers: dict[str, str]
    body: bytes
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    total_timeout: float = 30.0
    max_response_bytes: int = MAX_HTTP_RESPONSE_BYTES


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes


class StoryDirectorTransport:
    def send(self, request: TransportRequest) -> TransportResponse:
        raise NotImplementedError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPTransport(StoryDirectorTransport):
    def send(self, request: TransportRequest) -> TransportResponse:
        started = time.monotonic()
        opener = urllib.request.build_opener(_NoRedirect())
        outbound = urllib.request.Request(request.url, data=request.body, headers=request.headers, method="POST")
        try:
            with opener.open(outbound, timeout=min(request.connect_timeout, request.total_timeout)) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > request.max_response_bytes:
                    raise ResponseTooLarge("model response exceeds the configured byte limit")
                try:
                    response.fp.raw._sock.settimeout(request.read_timeout)
                except (AttributeError, OSError):
                    pass
                chunks, total = [], 0
                while True:
                    if time.monotonic() - started > request.total_timeout:
                        raise TransportTimeout("model request exceeded the total timeout")
                    chunk = response.read(min(65536, request.max_response_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > request.max_response_bytes:
                        raise ResponseTooLarge("model response exceeds the configured byte limit")
                    chunks.append(chunk)
                return TransportResponse(response.status, b"".join(chunks))
        except urllib.error.HTTPError as error:
            if 300 <= error.code < 400:
                raise TransportRedirect(f"model endpoint returned redirect HTTP {error.code}") from None
            raise TransportError(f"model endpoint returned HTTP {error.code}") from None
        except (TimeoutError, socket.timeout):
            raise TransportTimeout("model request timed out") from None
        except urllib.error.URLError:
            raise TransportError("model endpoint connection failed") from None


def validate_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("LLM base URL is required")
    if len(value) > 2048 or any(ord(character) < 33 for character in value):
        raise ValueError("LLM base URL contains whitespace/control characters or is too long")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM base URL must use http/https and include an explicit host")
    if parsed.username or parsed.password:
        raise ValueError("LLM base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base URL must not contain query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def responses_endpoint(base_url: str) -> str:
    return validate_base_url(base_url) + "/responses"
