"""Provider-neutral story directors and explicitly selected deterministic fixtures."""
from __future__ import annotations

import json
import hashlib
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

from .compiler import compile_adventure
from .models import SCHEMA_VERSION
from .structured_schema import adventure_spec_json_schema, build_system_prompt
from .transport import HTTPTransport, StoryDirectorTransport, TransportRequest, TransportResponse, responses_endpoint, validate_base_url
from .validation import require_valid_game

MAX_MODEL_OUTPUT_BYTES = 128 * 1024
MAX_JSON_DEPTH = 20
MAX_JSON_NODES = 10_000
MAX_REPAIR_ERRORS = 8
MAX_REPAIR_ERROR_LENGTH = 240
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class StoryDirector(ABC):
    source: str

    @abstractmethod
    def create_spec(self, prompt: str) -> dict:
        raise NotImplementedError

    def provenance(self, compile_digest: str) -> dict | None:
        del compile_digest
        return None


class JsonStoryDirector(StoryDirector):
    source = "json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create_spec(self, prompt: str) -> dict:
        del prompt
        return json.loads(self.path.read_text(encoding="utf-8"))


class FixtureStoryDirector(StoryDirector):
    """Returns a named fixture; free prompt text never selects fixture content."""

    def __init__(self, fixture: str = "golden_lab"):
        if fixture not in FIXTURES:
            raise ValueError(f"unknown fixture {fixture!r}; choose one of {', '.join(sorted(FIXTURES))}")
        self.fixture = fixture
        self.source = f"fixture:{fixture}"

    def create_spec(self, prompt: str) -> dict:
        del prompt
        return deepcopy(FIXTURES[self.fixture])


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    total_timeout: float = 30.0
    max_response_bytes: int = 512 * 1024
    max_output_tokens: int = 12_000

    def validate(self):
        base_url = validate_base_url(self.base_url)
        if not isinstance(self.api_key, str) or not self.api_key or "\r" in self.api_key or "\n" in self.api_key:
            raise ValueError("LLM API key is required")
        if not isinstance(self.model, str) or not MODEL_PATTERN.fullmatch(self.model):
            raise ValueError("LLM model is required and must be an explicit model identifier")
        for name in ("connect_timeout", "read_timeout", "total_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if self.total_timeout < self.connect_timeout:
            raise ValueError("total_timeout must be at least connect_timeout")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or not 1024 <= self.max_response_bytes <= 2 * 1024 * 1024:
            raise ValueError("max_response_bytes must be an integer from 1024 to 2097152")
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) or not 1 <= self.max_output_tokens <= 50_000:
            raise ValueError("max_output_tokens must be an integer from 1 to 50000")
        return OpenAICompatibleConfig(base_url, self.api_key, self.model, float(self.connect_timeout), float(self.read_timeout), float(self.total_timeout), self.max_response_bytes, self.max_output_tokens)


class OpenAICompatibleStoryDirector(StoryDirector):
    source = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig, transport: StoryDirectorTransport | None = None):
        self.config = config.validate()
        self.transport = transport or HTTPTransport()
        self._prompt_hash = None
        self._response_hashes = []
        self._attempt_statuses = []
        self._completed_at = None

    def create_spec(self, prompt: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("story prompt must be a non-empty string")
        if len(prompt) > 4000:
            raise ValueError("story prompt exceeds the maximum length of 4000")
        self._prompt_hash = _sha256(prompt.encode("utf-8"))
        self._response_hashes = []
        self._attempt_statuses = []
        previous = None
        errors = None
        for attempt in (1, 2):
            body = self._request_body(prompt, previous, errors)
            response = self.transport.send(TransportRequest(
                url=responses_endpoint(self.config.base_url),
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
                body=json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
                connect_timeout=self.config.connect_timeout, read_timeout=self.config.read_timeout, total_timeout=self.config.total_timeout, max_response_bytes=self.config.max_response_bytes,
            ))
            raw = _extract_output_text(response)
            if len(raw.encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
                raise ValueError("model output exceeds the AdventureSpec response limit")
            self._response_hashes.append(_sha256(raw.encode("utf-8")))
            try:
                spec = decode_single_json_object(raw)
                game = compile_adventure(spec)
                require_valid_game(game)
                self._attempt_statuses.append({"attempt": attempt, "valid": True, "errors": []})
                self._completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                return spec
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                errors = _bounded_errors(error)
                self._attempt_statuses.append({"attempt": attempt, "valid": False, "errors": errors})
                previous = raw
                if attempt == 2:
                    raise ValueError("model failed to produce a valid, compilable and solvable AdventureSpec after 2 attempts: " + "; ".join(errors)) from None
        raise AssertionError("unreachable")

    def _request_body(self, prompt, previous, errors):
        user_text = prompt if previous is None else json.dumps({"task": "Return one complete corrected AdventureSpec JSON object.", "previous_response": previous, "validation_errors": errors}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "model": self.config.model,
            "instructions": build_system_prompt(),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
            "text": {"format": {"type": "json_schema", "name": "pixelworld_adventure_spec_0_6_3", "strict": True, "schema": adventure_spec_json_schema()}},
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
            "stream": False,
        }

    def provenance(self, compile_digest: str) -> dict | None:
        if not self._attempt_statuses or not self._attempt_statuses[-1]["valid"]:
            return None
        git_commit, git_dirty = _git_identity()
        return {
            "schema_version": SCHEMA_VERSION,
            "director_type": self.source,
            "provider_protocol": "openai-responses-v1",
            "model": self.config.model,
            "base_url": validate_base_url(self.config.base_url),
            "prompt_sha256": self._prompt_hash,
            "response_sha256": list(self._response_hashes),
            "attempt_count": len(self._attempt_statuses),
            "attempt_validation": deepcopy(self._attempt_statuses),
            "compile_digest": compile_digest,
            "created_at_utc": self._completed_at,
            "python_version": platform.python_version(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }


def decode_single_json_object(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("model output must be text containing exactly one JSON object")
    stripped = raw.strip()
    if not stripped or stripped.startswith("```"):
        raise ValueError("model output must be one plain JSON object without markdown")
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value}")))
    try:
        value, end = decoder.raw_decode(stripped)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"model output is not strict JSON: {str(error)[:160]}") from None
    if end != len(stripped):
        raise ValueError("model output contains trailing text or multiple JSON documents")
    if not isinstance(value, dict):
        raise ValueError("model output must contain exactly one JSON object")
    _validate_json_shape(value)
    return value


def _validate_json_shape(value):
    nodes = 0
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError("model JSON exceeds the node limit")
        if depth > MAX_JSON_DEPTH:
            raise ValueError("model JSON exceeds the nesting-depth limit")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _extract_output_text(response: TransportResponse) -> str:
    if not isinstance(response, TransportResponse) or response.status < 200 or response.status >= 300:
        raise ValueError("model transport returned a non-success response")
    try:
        envelope = json.loads(response.body.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("model endpoint returned an invalid JSON response envelope") from None
    if not isinstance(envelope, dict):
        raise ValueError("model endpoint response envelope must be an object")
    if isinstance(envelope.get("output_text"), str):
        return envelope["output_text"]
    texts = []
    for output in envelope.get("output", []):
        if isinstance(output, dict):
            for content in output.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    texts.append(content["text"])
    if len(texts) != 1:
        raise ValueError("model endpoint must return exactly one output_text item")
    return texts[0]


def _bounded_errors(error):
    text = " ".join(str(error).replace("\r", " ").replace("\n", " ").split())
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return [(part[:MAX_REPAIR_ERROR_LENGTH] or "validation failed") for part in parts[:MAX_REPAIR_ERRORS]] or ["validation failed"]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=2, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _condition(op: str, path: str, value):
    return {"op": op, "path": path, "value": value}


def _effect(op: str, path: str, value):
    return {"op": op, "path": path, "value": value}


def _object(identifier, name, cls, role, zone, description, *, portable=False, required=True, portal=None, state=None):
    initial = deepcopy(state or {})
    if portable:
        initial = {"taken": False, **initial}
    return {"id": identifier, "name": name, "class": cls, "role": role, "preferred_zone": zone, "location_id": "chronochemical_lab", "description": description, "portable": portable, "required": required, "portal_destination": portal, "initial_state": initial}


GOLDEN_ADVENTURE_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "title": "Professor Knallberts chronochemisches Labor",
    "premise": "Professor Knallberts Zeitmaschine überhitzt. Der Spieler muss ein Kühlmittel herstellen und in die Maschine einsetzen.",
    "tone": "farbenfroh, verrückt und optimistisch",
    "visual_theme": "mad_scientist_lab",
    "player": {"id": "player", "name": "Nova", "location_id": "chronochemical_lab", "start_position": [15, 60]},
    "characters": [{"id": "professor_knallbert", "name": "Professor Knallbert", "archetype": "mad_scientist", "role": "npc", "preferred_zone": "right_npc", "location_id": "chronochemical_lab", "description": "Ein zerzauster Chronochemiker mit leuchtender Schutzbrille.", "default_talk_text": "Zwei Reagenzien, eine Flasche – dann ab damit in die Maschine!", "initial_state": {}}],
    "locations": [{"id": "chronochemical_lab", "name": "Professor Knallberts chronochemisches Labor", "description": "Ein dunkles Neonlabor voller Roboterarme, Zahnräder und Funken.", "theme": "mad_scientist_lab", "size": [128, 72], "mood": "neon_sparks"}],
    "objects": [
        _object("time_machine", "Zeitmaschine", "time_machine", "machine", "center_machine", "Eine überhitzte Maschine mit türkisfarbenem Kern.", state={"cooled": False}),
        _object("control_console", "Bedienpult", "control_console", "console", "left_console", "Blinkende Hebel melden kritische Chronotemperatur."),
        _object("coolant_red", "Rotes Kühlreagenz", "chemical_bottle", "ingredient", "left_reagent", "Ein rotes, zähflüssiges Chronoreagenz.", portable=True),
        _object("coolant_blue", "Blaues Kühlreagenz", "chemical_bottle", "ingredient", "left_reagent", "Ein blaues, eisig funkelndes Chronoreagenz.", portable=True),
        _object("catalyst_green", "Grüner Katalysator", "chemical_bottle", "ingredient", "right_reagent", "Ein optionaler grüner Katalysator mit Warnetikett.", required=False),
        _object("mixing_flask", "Leere Mischflasche", "mixing_flask", "container", "right_container", "Eine druckfeste Flasche für zwei Reagenzien.", portable=True, state={"contents": "empty"}),
        _object("time_portal", "Zeitportal", "time_portal", "exit", "right_exit", "Der Ausgang durch die Zeit; momentan noch inaktiv.", portal="ending", state={"active": False}),
        _object("robot_arm_left", "Roboterarm", "robot_arm", "scenery", "left_wall", "Ein mechanischer Arm sortiert Funken nach Größe.", required=False),
        _object("wall_gears", "Zahnräder", "gear", "scenery", "upper_right", "Ein unmögliches Getriebe läuft rückwärts.", required=False),
    ],
    "inventory_items": [
        {"id": "coolant_red", "name": "Rotes Kühlreagenz", "description": "Rote Komponente des Kühlmittels."},
        {"id": "coolant_blue", "name": "Blaues Kühlreagenz", "description": "Blaue Komponente des Kühlmittels."},
        {"id": "mixing_flask", "name": "Mischflasche", "description": "Kann beide Reagenzien aufnehmen."},
        {"id": "mixed_coolant", "name": "Chronokühlmittel", "description": "Fertig gemischtes violettes Kühlmittel."},
    ],
    "objectives": [{"id": "cool_time_machine", "description": "Kühlmittel mischen und in die Zeitmaschine einsetzen.", "required": True, "initial_state": {"completed": False}}],
    "puzzles": [{"id": "coolant_puzzle", "description": "Zwei Reagenzien und die Flasche einsammeln, mischen und anwenden.", "objective_ids": ["cool_time_machine"], "interaction_ids": ["take_red", "take_blue", "take_flask", "mix_coolant", "cool_machine"]}],
    "interactions": [
        {"id": "take_red", "verb": "take", "target_id": "coolant_red", "item_ids": [], "conditions": [_condition("equals", "objects.coolant_red.taken", False)], "effects": [_effect("set", "objects.coolant_red.taken", True), _effect("inventory_add", "inventory", "coolant_red")], "text": "Das rote Reagenz landet sicher im Inventar.", "animation_hint": "pickup"},
        {"id": "take_blue", "verb": "take", "target_id": "coolant_blue", "item_ids": [], "conditions": [_condition("equals", "objects.coolant_blue.taken", False)], "effects": [_effect("set", "objects.coolant_blue.taken", True), _effect("inventory_add", "inventory", "coolant_blue")], "text": "Das blaue Reagenz ist erstaunlich kalt.", "animation_hint": "pickup"},
        {"id": "take_flask", "verb": "take", "target_id": "mixing_flask", "item_ids": [], "conditions": [_condition("equals", "objects.mixing_flask.taken", False)], "effects": [_effect("set", "objects.mixing_flask.taken", True), _effect("inventory_add", "inventory", "mixing_flask")], "text": "Die Mischflasche ist bereit.", "animation_hint": "pickup"},
        {"id": "mix_coolant", "verb": "combine", "target_id": "mixing_flask", "item_ids": ["coolant_blue", "coolant_red"], "conditions": [_condition("inventory_contains", "inventory", "coolant_red"), _condition("inventory_contains", "inventory", "coolant_blue"), _condition("inventory_contains", "inventory", "mixing_flask"), _condition("equals", "objects.mixing_flask.contents", "empty")], "effects": [_effect("inventory_remove", "inventory", "coolant_red"), _effect("inventory_remove", "inventory", "coolant_blue"), _effect("inventory_remove", "inventory", "mixing_flask"), _effect("inventory_add", "inventory", "mixed_coolant"), _effect("set", "objects.mixing_flask.contents", "mixed_coolant")], "text": "Die Mischung leuchtet violett: Chronokühlmittel!", "animation_hint": "mix"},
        {"id": "cool_machine", "verb": "use", "target_id": "time_machine", "item_ids": ["mixed_coolant"], "conditions": [_condition("inventory_contains", "inventory", "mixed_coolant"), _condition("equals", "objects.time_machine.cooled", False)], "effects": [_effect("inventory_remove", "inventory", "mixed_coolant"), _effect("set", "objects.time_machine.cooled", True), _effect("set", "objects.time_portal.active", True), _effect("set", "objectives.cool_time_machine.completed", True)], "text": "Die Zeitmaschine kühlt ab und das Portal stabilisiert sich!", "animation_hint": "machine_cool"},
    ],
    "ending_conditions": [{"id": "portal_stable", "description": "Die Zeitmaschine ist gekühlt und das Portal aktiv.", "conditions": [_condition("equals", "objects.time_machine.cooled", True), _condition("equals", "objects.time_portal.active", True), _condition("equals", "objectives.cool_time_machine.completed", True)]}],
    "flags": [],
}


def _pirate_object(identifier, name, cls, role, zone, description, *, portable=False, portal=None, state=None):
    value = _object(identifier, name, cls, role, zone, description, portable=portable, portal=portal, state=state)
    value["location_id"] = "moonlit_harbor"
    return value


PIRATE_HARBOR_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "title": "Käptin Kupferzahns Mondscheinhafen",
    "premise": "Der Spieler muss einen Messingschlüssel finden, die Hafenkiste öffnen und damit den Fluchtsteg freigeben.",
    "tone": "abenteuerlich und augenzwinkernd",
    "visual_theme": "pirate_harbor",
    "player": {"id": "deckhand", "name": "Mara", "location_id": "moonlit_harbor", "start_position": [14, 61]},
    "characters": [{"id": "captain_coppertooth", "name": "Käptin Kupferzahn", "archetype": "pirate", "role": "npc", "preferred_zone": "harbor_npc", "location_id": "moonlit_harbor", "description": "Eine Hafenpiratin mit kupfernem Zahn und tadellosem Hut.", "default_talk_text": "Der Schlüssel liegt beim Fass. Die Kiste steuert den Fluchtsteg!", "initial_state": {}}],
    "locations": [{"id": "moonlit_harbor", "name": "Mondscheinhafen", "description": "Ein knarrender Pier zwischen Schiff, Kisten und dunklem Wasser.", "theme": "pirate_harbor", "size": [128, 72], "mood": "sunset"}],
    "objects": [
        _pirate_object("old_brass_key", "Alter Messingschlüssel", "brass_key", "item", "dock_left", "Ein kleiner Schlüssel mit Ankersymbol.", portable=True),
        _pirate_object("captains_lockbox", "Kapitänskiste", "locked_chest", "container", "dock_center", "Eine schwere Kiste mit einem Messingschloss.", state={"unlocked": False}),
        _pirate_object("escape_gangplank", "Fluchtsteg", "harbor_exit", "exit", "dock_exit", "Ein hochgezogener Steg zum wartenden Boot.", portal="ending", state={"open": False}),
        _pirate_object("cargo_barrel", "Rumfass", "barrel", "scenery", "dock_left_wall", "Ein Fass, das verdächtig nüchtern riecht.", state={}),
    ],
    "inventory_items": [{"id": "old_brass_key", "name": "Alter Messingschlüssel", "description": "Passt vermutlich in ein altes Hafenschloss."}],
    "objectives": [{"id": "open_escape_route", "description": "Kiste öffnen und Fluchtsteg freigeben.", "required": True, "initial_state": {"completed": False}}],
    "puzzles": [{"id": "harbor_key_puzzle", "description": "Schlüssel finden und die Steuerkiste öffnen.", "objective_ids": ["open_escape_route"], "interaction_ids": ["take_harbor_key", "unlock_harbor_box"]}],
    "interactions": [
        {"id": "take_harbor_key", "verb": "take", "target_id": "old_brass_key", "item_ids": [], "conditions": [_condition("equals", "objects.old_brass_key.taken", False)], "effects": [_effect("set", "objects.old_brass_key.taken", True), _effect("inventory_add", "inventory", "old_brass_key")], "text": "Der Messingschlüssel verschwindet in Maras Tasche.", "animation_hint": "pickup"},
        {"id": "unlock_harbor_box", "verb": "use", "target_id": "captains_lockbox", "item_ids": ["old_brass_key"], "conditions": [_condition("inventory_contains", "inventory", "old_brass_key"), _condition("equals", "objects.captains_lockbox.unlocked", False)], "effects": [_effect("inventory_remove", "inventory", "old_brass_key"), _effect("set", "objects.captains_lockbox.unlocked", True), _effect("set", "objects.escape_gangplank.open", True), _effect("set", "objectives.open_escape_route.completed", True)], "text": "Das Schloss springt auf und der Fluchtsteg fällt herunter.", "animation_hint": "unlock"},
    ],
    "ending_conditions": [{"id": "gangplank_ready", "description": "Der Fluchtsteg ist offen.", "conditions": [_condition("equals", "objects.escape_gangplank.open", True), _condition("equals", "objectives.open_escape_route.completed", True)]}],
    "flags": [],
}

FIXTURES = {"golden_lab": GOLDEN_ADVENTURE_SPEC, "pirate_harbor": PIRATE_HARBOR_SPEC}
