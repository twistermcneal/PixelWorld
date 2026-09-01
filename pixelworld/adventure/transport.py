"""Small bounded HTTP transport for explicitly selected OpenAI-compatible endpoints."""
from __future__ import annotations

import http.client
import queue
import socket
import threading
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

class TransportHTTPError(TransportError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"model endpoint returned HTTP {status}")

@dataclass(frozen=True)
class TransportRequest:
    url: str
    headers: dict[str, str]
    body: bytes = b""
    method: str = "POST"
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

class HTTPTransport(StoryDirectorTransport):
    """Use public http.client/socket APIs plus an outer hard total deadline."""

    def send(self, request: TransportRequest) -> TransportResponse:
        result = queue.Queue(maxsize=1)
        connection_holder = []
        lock = threading.Lock()

        def perform():
            connection = None
            try:
                parsed = urlsplit(request.url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise TransportError("model endpoint URL is invalid")
                connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
                connection = connection_class(parsed.hostname, port=parsed.port, timeout=request.connect_timeout)
                with lock:
                    connection_holder.append(connection)
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                connection.request(request.method, path, body=request.body or None, headers=request.headers)
                if connection.sock is not None:
                    connection.sock.settimeout(request.read_timeout)
                response = connection.getresponse()
                if 300 <= response.status < 400:
                    raise TransportRedirect(f"model endpoint returned redirect HTTP {response.status}")
                if not 200 <= response.status < 300:
                    raise TransportHTTPError(response.status)
                length = response.getheader("Content-Length")
                if length:
                    try:
                        if int(length) > request.max_response_bytes:
                            raise ResponseTooLarge("model response exceeds the configured byte limit")
                    except ValueError:
                        pass
                chunks, total = [], 0
                while True:
                    chunk = response.read1(min(65536, request.max_response_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > request.max_response_bytes:
                        raise ResponseTooLarge("model response exceeds the configured byte limit")
                    chunks.append(chunk)
                result.put(TransportResponse(response.status, b"".join(chunks)))
            except TransportError as error:
                result.put(error)
            except (TimeoutError, socket.timeout):
                result.put(TransportTimeout("model request timed out"))
            except (OSError, http.client.HTTPException, ValueError):
                result.put(TransportError("model endpoint connection failed"))
            finally:
                if connection is not None:
                    connection.close()

        threading.Thread(target=perform, name="pixelworld-llm-http", daemon=True).start()
        try:
            outcome = result.get(timeout=request.total_timeout)
        except queue.Empty:
            with lock:
                if connection_holder:
                    connection_holder[0].close()
            raise TransportTimeout("model request exceeded the total timeout") from None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

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

def protocol_endpoint(base_url: str, protocol: str) -> str:
    suffixes = {"responses-v1": "/responses", "chat-completions-json-schema": "/chat/completions"}
    if protocol not in suffixes:
        raise ValueError("LLM protocol must be responses-v1 or chat-completions-json-schema")
    return validate_base_url(base_url) + suffixes[protocol]

def models_endpoint(base_url: str) -> str:
    return validate_base_url(base_url) + "/models"

def responses_endpoint(base_url: str) -> str:
    return protocol_endpoint(base_url, "responses-v1")
