"""Non-generating compatibility preflight for explicitly configured LLM protocols."""
from __future__ import annotations

import json

from .director import OpenAICompatibleConfig, _extract_output_text, build_provider_request_body, decode_single_json_object
from .structured_schema import minimal_provider_probe_wire, wire_to_adventure_spec
from .transport import StoryDirectorTransport, TransportError, TransportHTTPError, TransportRequest, models_endpoint, protocol_endpoint


def check_story_director(config: OpenAICompatibleConfig, transport: StoryDirectorTransport) -> dict:
    config = config.validate()
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    report = {
        "protocol": config.protocol,
        "runtime_configuration": config.runtime_configuration(),
        "base_url_valid": {"ok": True, "detail": "valid http/https base URL"},
        "model_configured": {"ok": True, "detail": "explicit model identifier configured"},
        "endpoint_reachable": {"ok": False, "detail": "not checked"},
        "model_present": {"ok": None, "detail": "model listing not supported or not checked"},
        "protocol_endpoint_present": {"ok": False, "detail": "not checked"},
        "structured_output_schema_accepted": {"ok": False, "detail": "not checked"},
    }
    request_options = {"connect_timeout": config.connect_timeout, "read_timeout": config.read_timeout, "total_timeout": config.total_timeout, "max_response_bytes": config.max_response_bytes}
    try:
        models = transport.send(TransportRequest(url=models_endpoint(config.base_url), headers=headers, method="GET", **request_options))
        report["endpoint_reachable"] = {"ok": True, "detail": "HTTP endpoint responded"}
        envelope = _json_envelope(models.body, "models")
        identifiers = [item.get("id") for item in envelope.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)] if isinstance(envelope.get("data"), list) else []
        present = config.model in identifiers
        report["model_present"] = {"ok": present, "detail": "configured model found" if present else "configured model not found"}
    except TransportHTTPError as error:
        report["endpoint_reachable"] = {"ok": True, "detail": f"HTTP endpoint responded with {error.status}"}
        if error.status not in {404, 405}:
            report["model_present"] = {"ok": False, "detail": f"model listing returned HTTP {error.status}"}
    except (TransportError, ValueError) as error:
        report["model_present"] = {"ok": False, "detail": _safe_detail(error, config.api_key)}

    probe = json.dumps(minimal_provider_probe_wire(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    prompt = "Schema compatibility probe only. Return exactly this bounded WireSpec object: " + probe
    body = build_provider_request_body(config, prompt)
    try:
        response = transport.send(TransportRequest(
            url=protocol_endpoint(config.base_url, config.protocol), headers=headers,
            body=json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            method="POST", **request_options,
        ))
        report["endpoint_reachable"] = {"ok": True, "detail": "HTTP endpoint responded"}
        report["protocol_endpoint_present"] = {"ok": True, "detail": "configured protocol endpoint responded"}
        wire_to_adventure_spec(decode_single_json_object(_extract_output_text(response, config.protocol)))
        report["structured_output_schema_accepted"] = {"ok": True, "detail": "provider accepted the WireSpec v1 strict schema"}
    except TransportHTTPError as error:
        report["endpoint_reachable"] = {"ok": True, "detail": f"HTTP endpoint responded with {error.status}"}
        report["protocol_endpoint_present"] = {"ok": error.status != 404, "detail": f"configured protocol endpoint returned HTTP {error.status}"}
        report["structured_output_schema_accepted"] = {"ok": False, "detail": f"schema probe returned HTTP {error.status}"}
    except TransportError as error:
        report["endpoint_reachable"] = {"ok": False, "detail": _safe_detail(error, config.api_key)}
        report["protocol_endpoint_present"] = {"ok": False, "detail": _safe_detail(error, config.api_key)}
        report["structured_output_schema_accepted"] = {"ok": False, "detail": _safe_detail(error, config.api_key)}
    except ValueError as error:
        report["protocol_endpoint_present"] = {"ok": True, "detail": "configured protocol endpoint responded"}
        report["structured_output_schema_accepted"] = {"ok": False, "detail": _safe_detail(error, config.api_key)}

    required = ("base_url_valid", "model_configured", "endpoint_reachable", "protocol_endpoint_present", "structured_output_schema_accepted")
    model_ok = report["model_present"]["ok"] is not False
    report["ok"] = all(report[item]["ok"] is True for item in required) and model_ok
    return report


def _json_envelope(body: bytes, label: str) -> dict:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{label} endpoint returned invalid JSON") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} endpoint response must be an object")
    return value


def _safe_detail(error: Exception, secret: str = "") -> str:
    if isinstance(error, TransportError):
        text = str(error)
    else:
        text = " ".join(str(error).replace("\r", " ").replace("\n", " ").split())[:240]
    return text.replace(secret, "[redacted]") if secret else text
