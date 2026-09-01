"""Strict, bounded and versioned data contracts for untrusted adventure data."""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "0.6.3"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VERBS = {"move_to", "look_at", "talk_to", "take", "use", "combine"}
DECLARATIVE_VERBS = VERBS - {"move_to"}
CONDITION_OPS = {"equals", "inventory_contains", "inventory_missing"}
EFFECT_OPS = {"set", "inventory_add", "inventory_remove"}
LIMITS = {"locations": 1, "characters": 2, "objects": 16, "inventory_items": 16, "interactions": 16, "objectives": 8, "puzzles": 8, "ending_conditions": 4, "flags": 16, "polygon_points": 16, "interaction_conditions": 16, "interaction_effects": 16, "references": 16, "state_fields": 16}
TEXT_LIMITS = {"title": 160, "name": 120, "tone": 160, "description": 600, "premise": 800, "text": 600, "dialogue": 600}
MAX_COORDINATE = 4096.0
MAX_STATE_STRING = 160
MAX_STATE_NUMBER = 1_000_000_000


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: Any, path: str, maximum: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{path} exceeds the maximum of {maximum} items")
    return value


def _keys(value: dict[str, Any], required: set[str], path: str) -> None:
    missing, unknown = sorted(required - value.keys()), sorted(value.keys() - required)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _text(value: Any, path: str, maximum: int = TEXT_LIMITS["description"]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{path} exceeds the maximum length of {maximum}")
    return value


def _optional_text(value: Any, path: str, maximum: int = TEXT_LIMITS["description"]) -> str | None:
    return None if value is None else _text(value, path, maximum)


def _id(value: Any, path: str) -> str:
    text = _text(value, path, 64)
    if not ID_PATTERN.fullmatch(text):
        raise ValueError(f"{path} must be a stable ASCII identifier")
    return text


def _number(value: Any, path: str, minimum: float = -MAX_COORDINATE, maximum: float = MAX_COORDINATE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if not minimum <= result <= maximum:
        raise ValueError(f"{path} must be between {minimum:g} and {maximum:g}")
    return result


def json_scalar_type(value: Any, path: str = "value") -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        if len(value) > MAX_STATE_STRING:
            raise ValueError(f"{path} exceeds the maximum state string length of {MAX_STATE_STRING}")
        return "string"
    if isinstance(value, int):
        if abs(value) > MAX_STATE_NUMBER:
            raise ValueError(f"{path} exceeds the numeric state limit of {MAX_STATE_NUMBER}")
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > MAX_STATE_NUMBER:
            raise ValueError(f"{path} must be finite and within +/-{MAX_STATE_NUMBER}")
        return "number"
    raise ValueError(f"{path} must be a JSON scalar")


def value_matches_type(value: Any, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str) and len(value) <= MAX_STATE_STRING
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= MAX_STATE_NUMBER
    if expected == "number":
        return isinstance(value, float) and math.isfinite(value) and abs(value) <= MAX_STATE_NUMBER
    return False


def _point(value: Any, path: str) -> tuple[float, float]:
    point = _list(value, path, 2)
    if len(point) != 2:
        raise ValueError(f"{path} must contain exactly two numbers")
    return _number(point[0], f"{path}[0]"), _number(point[1], f"{path}[1]")


def _unique_ids(items: list[Any], namespace: str) -> set[str]:
    seen: set[str] = set()
    for index, raw in enumerate(items):
        identifier = _id(_object(raw, f"{namespace}[{index}]").get("id"), f"{namespace}[{index}].id")
        if identifier in seen:
            raise ValueError(f"duplicate id {identifier!r} in {namespace}")
        seen.add(identifier)
    return seen


def _reference(value: Any, choices: set[str], path: str) -> str:
    identifier = _id(value, path)
    if identifier not in choices:
        raise ValueError(f"{path} references unknown id {identifier!r}")
    return identifier


def _state_object(value: Any, path: str) -> dict[str, Any]:
    result = _object(value, path)
    if len(result) > LIMITS["state_fields"]:
        raise ValueError(f"{path} exceeds the maximum of {LIMITS['state_fields']} fields")
    for key, item in result.items():
        _id(key, f"{path} field")
        json_scalar_type(item, f"{path}.{key}")
    return result


def _operation(raw: Any, path: str, allowed: set[str]) -> dict:
    value = _object(raw, path)
    _keys(value, {"op", "path", "value"}, path)
    if value["op"] not in allowed:
        raise ValueError(f"{path}.op is not allowed: {value['op']!r}")
    _text(value["path"], f"{path}.path", 196)
    return value


def _state_path_types(lists: dict[str, list]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for namespace in ("objects", "objectives"):
        for item in lists[namespace]:
            for field, value in item["initial_state"].items():
                paths[f"{namespace}.{item['id']}.{field}"] = json_scalar_type(value)
    for flag in lists["flags"]:
        paths[f"flags.{flag['id']}"] = flag["type"]
    return paths


def _validate_operation(operation: dict, path: str, state_types: dict[str, str], inventory_ids: set[str]) -> None:
    op, state_path, value = operation["op"], operation["path"], operation["value"]
    if op.startswith("inventory_"):
        if state_path != "inventory":
            raise ValueError(f"{path}.path must be 'inventory' for {op}")
        if value not in inventory_ids:
            raise ValueError(f"{path}.value references unknown inventory id {value!r}")
        return
    if state_path not in state_types:
        raise ValueError(f"{path}.path references unknown state field {state_path!r}")
    expected = state_types[state_path]
    if not value_matches_type(value, expected):
        raise ValueError(f"{path}.value must have exact JSON type {expected}")


def validate_adventure_spec(spec: Any, ontology: Any | None = None) -> dict[str, Any]:
    root = _object(spec, "AdventureSpec")
    required = {"schema_version", "title", "premise", "tone", "visual_theme", "player", "characters", "locations", "objects", "inventory_items", "objectives", "puzzles", "interactions", "ending_conditions", "flags"}
    _keys(root, required, "AdventureSpec")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported AdventureSpec schema_version {root['schema_version']!r}")
    _text(root["title"], "AdventureSpec.title", TEXT_LIMITS["title"])
    _text(root["premise"], "AdventureSpec.premise", TEXT_LIMITS["premise"])
    _text(root["tone"], "AdventureSpec.tone", TEXT_LIMITS["tone"])
    _id(root["visual_theme"], "AdventureSpec.visual_theme")
    list_names = ("characters", "locations", "objects", "inventory_items", "objectives", "puzzles", "interactions", "ending_conditions", "flags")
    lists = {name: _list(root[name], name, LIMITS[name]) for name in list_names}
    if len(lists["locations"]) != 1:
        raise ValueError("Phase 1 requires exactly 1 location")
    ids = {name: _unique_ids(items, name) for name, items in lists.items()}
    if ids["objects"] & ids["characters"]:
        raise ValueError("object and character IDs must not collide")
    global_entities = ids["objects"] | ids["characters"]

    location = lists["locations"][0]
    _keys(location, {"id", "name", "description", "theme", "size", "mood"}, "locations[0]")
    _text(location["name"], "locations[0].name", TEXT_LIMITS["name"])
    _text(location["description"], "locations[0].description")
    _id(location["theme"], "locations[0].theme")
    _id(location["mood"], "locations[0].mood")
    size = _list(location["size"], "locations[0].size", 2)
    if len(size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 512 for n in size):
        raise ValueError("locations[0].size must contain two integers from 1 to 512")

    player = _object(root["player"], "player")
    _keys(player, {"id", "name", "location_id", "start_position"}, "player")
    _id(player["id"], "player.id")
    _text(player["name"], "player.name", TEXT_LIMITS["name"])
    _reference(player["location_id"], ids["locations"], "player.location_id")
    px, py = _point(player["start_position"], "player.start_position")
    if not 0 <= px <= size[0] or not 0 <= py <= size[1]:
        raise ValueError("player.start_position lies outside the room")

    for index, item in enumerate(lists["characters"]):
        path = f"characters[{index}]"
        _keys(item, {"id", "name", "archetype", "role", "preferred_zone", "location_id", "description", "default_talk_text", "initial_state"}, path)
        _text(item["name"], f"{path}.name", TEXT_LIMITS["name"])
        for key in ("archetype", "role", "preferred_zone"):
            _id(item[key], f"{path}.{key}")
        _reference(item["location_id"], ids["locations"], f"{path}.location_id")
        _text(item["description"], f"{path}.description")
        _text(item["default_talk_text"], f"{path}.default_talk_text", TEXT_LIMITS["dialogue"])
        _state_object(item["initial_state"], f"{path}.initial_state")

    for index, item in enumerate(lists["objects"]):
        path = f"objects[{index}]"
        _keys(item, {"id", "name", "class", "role", "preferred_zone", "location_id", "description", "portable", "required", "portal_destination", "initial_state"}, path)
        _text(item["name"], f"{path}.name", TEXT_LIMITS["name"])
        for key in ("class", "role", "preferred_zone"):
            _id(item[key], f"{path}.{key}")
        _reference(item["location_id"], ids["locations"], f"{path}.location_id")
        _text(item["description"], f"{path}.description")
        if not isinstance(item["portable"], bool) or not isinstance(item["required"], bool):
            raise ValueError(f"{path}.portable and required must be booleans")
        _optional_text(item["portal_destination"], f"{path}.portal_destination", 120)
        state = _state_object(item["initial_state"], f"{path}.initial_state")
        if item["portable"] and state.get("taken") is not False:
            raise ValueError(f"{path}.initial_state.taken must be false for portable objects")
        if item["role"] == "exit" and not item["portal_destination"]:
            raise ValueError(f"{path}.portal_destination is required for exit objects")
        if item["role"] != "exit" and item["portal_destination"] is not None:
            raise ValueError(f"{path}.portal_destination is only allowed for exit objects")

    for index, item in enumerate(lists["inventory_items"]):
        path = f"inventory_items[{index}]"
        _keys(item, {"id", "name", "description"}, path)
        _text(item["name"], f"{path}.name", TEXT_LIMITS["name"])
        _text(item["description"], f"{path}.description")
    portable_ids = {item["id"] for item in lists["objects"] if item["portable"]}
    if ids["inventory_items"] & ids["objects"] != portable_ids:
        raise ValueError("inventory/object ID overlap must equal exactly the portable object IDs")

    for index, item in enumerate(lists["objectives"]):
        path = f"objectives[{index}]"
        _keys(item, {"id", "description", "required", "initial_state"}, path)
        _text(item["description"], f"{path}.description")
        if not isinstance(item["required"], bool):
            raise ValueError(f"{path}.required must be a boolean")
        if _state_object(item["initial_state"], f"{path}.initial_state").get("completed") is not False:
            raise ValueError(f"{path}.initial_state.completed must be false")

    allowed_types = {"boolean", "string", "integer", "number"}
    for index, item in enumerate(lists["flags"]):
        path = f"flags[{index}]"
        _keys(item, {"id", "type", "initial"}, path)
        if item["type"] not in allowed_types:
            raise ValueError(f"{path}.type must be one of {', '.join(sorted(allowed_types))}")
        if not value_matches_type(item["initial"], item["type"]):
            raise ValueError(f"{path}.initial must have exact JSON type {item['type']}")

    state_types, inventory_ids = _state_path_types(lists), ids["inventory_items"]
    for index, item in enumerate(lists["interactions"]):
        path = f"interactions[{index}]"
        _keys(item, {"id", "verb", "target_id", "item_ids", "conditions", "effects", "text", "animation_hint"}, path)
        verb = item["verb"]
        if verb not in DECLARATIVE_VERBS:
            raise ValueError(f"{path}.verb is not declaratively allowed: {verb!r}")
        _reference(item["target_id"], global_entities, f"{path}.target_id")
        item_refs = _list(item["item_ids"], f"{path}.item_ids", 2)
        for ref_index, ref in enumerate(item_refs):
            _reference(ref, inventory_ids, f"{path}.item_ids[{ref_index}]")
        expected_count = {"take": 0, "look_at": 0, "talk_to": 0, "use": 1, "combine": 2}[verb]
        if len(item_refs) != expected_count:
            raise ValueError(f"{path}.item_ids must contain exactly {expected_count} items for {verb}")
        if verb == "combine":
            if len(set(item_refs)) != 2:
                raise ValueError(f"{path}.item_ids must contain two different items")
            target = next((obj for obj in lists["objects"] if obj["id"] == item["target_id"]), None)
            if target is None or not target["portable"] or target["id"] not in inventory_ids:
                raise ValueError(f"{path}.target_id must be a portable inventory container")
        if verb == "take":
            target = next((obj for obj in lists["objects"] if obj["id"] == item["target_id"]), None)
            if target is None or not target["portable"]:
                raise ValueError(f"{path}.target_id must be a portable object")
        if verb == "talk_to" and item["target_id"] not in ids["characters"]:
            raise ValueError(f"{path}.target_id must be a character for talk_to")
        for cond_index, raw in enumerate(_list(item["conditions"], f"{path}.conditions", LIMITS["interaction_conditions"])):
            _validate_operation(_operation(raw, f"{path}.conditions[{cond_index}]", CONDITION_OPS), f"{path}.conditions[{cond_index}]", state_types, inventory_ids)
        for effect_index, raw in enumerate(_list(item["effects"], f"{path}.effects", LIMITS["interaction_effects"])):
            _validate_operation(_operation(raw, f"{path}.effects[{effect_index}]", EFFECT_OPS), f"{path}.effects[{effect_index}]", state_types, inventory_ids)
        _text(item["text"], f"{path}.text", TEXT_LIMITS["text"])
        _id(item["animation_hint"], f"{path}.animation_hint")

    for index, item in enumerate(lists["puzzles"]):
        path = f"puzzles[{index}]"
        _keys(item, {"id", "description", "objective_ids", "interaction_ids"}, path)
        _text(item["description"], f"{path}.description")
        for ref_index, ref in enumerate(_list(item["objective_ids"], f"{path}.objective_ids", LIMITS["references"])):
            _reference(ref, ids["objectives"], f"{path}.objective_ids[{ref_index}]")
        for ref_index, ref in enumerate(_list(item["interaction_ids"], f"{path}.interaction_ids", LIMITS["references"])):
            _reference(ref, ids["interactions"], f"{path}.interaction_ids[{ref_index}]")

    for index, item in enumerate(lists["ending_conditions"]):
        path = f"ending_conditions[{index}]"
        _keys(item, {"id", "description", "conditions"}, path)
        _text(item["description"], f"{path}.description")
        conditions = _list(item["conditions"], f"{path}.conditions", LIMITS["interaction_conditions"])
        if not conditions:
            raise ValueError(f"{path}.conditions must not be empty")
        for cond_index, raw in enumerate(conditions):
            _validate_operation(_operation(raw, f"{path}.conditions[{cond_index}]", CONDITION_OPS), f"{path}.conditions[{cond_index}]", state_types, inventory_ids)

    if ontology is not None:
        ontology.validate_spec(root)
    return deepcopy(root)


def validate_point(value: Any, path: str = "point") -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must contain exactly two numbers")
    return _number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]")


def _orientation(a, b, c) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if abs(cross) <= 1e-9 else (1 if cross > 0 else -1)


def _on_segment(point, start, end) -> bool:
    return _orientation(start, end, point) == 0 and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])


def _segments_intersect(a, b, c, d) -> bool:
    values = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if values[0] != values[1] and values[2] != values[3]:
        return True
    return any(value == 0 and _on_segment(point, start, end) for value, point, start, end in ((values[0], c, a, b), (values[1], d, a, b), (values[2], a, c, d), (values[3], b, c, d)))


def validate_polygon(value: Any, path: str = "polygon", *, convex: bool = False) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")
    if len(value) > LIMITS["polygon_points"]:
        raise ValueError(f"{path} exceeds the maximum of {LIMITS['polygon_points']} points")
    points = [validate_point(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(points) < 3:
        raise ValueError(f"{path} must contain at least three points")
    if len(set(points)) != len(points):
        raise ValueError(f"{path} contains duplicate points")
    for left in range(len(points)):
        left_next = (left + 1) % len(points)
        for right in range(left + 1, len(points)):
            right_next = (right + 1) % len(points)
            if left_next == right or right_next == left:
                continue
            if _segments_intersect(points[left], points[left_next], points[right], points[right_next]):
                raise ValueError(f"{path} is self-intersecting")
    crosses = []
    for index in range(len(points)):
        a, b, c = points[index - 2], points[index - 1], points[index]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            crosses.append(cross)
    if not crosses:
        raise ValueError(f"{path} has zero area")
    if convex and any(value * crosses[0] < 0 for value in crosses[1:]):
        raise ValueError(f"{path} must be convex")
    return [[x, y] for x, y in points]
