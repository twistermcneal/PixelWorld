"""Strict, versioned data contracts for PixelWorld adventures."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "0.6.3"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VERBS = {"move_to", "look_at", "talk_to", "take", "use", "combine"}
CONDITION_OPS = {"equals", "inventory_contains", "inventory_missing"}
EFFECT_OPS = {"set", "inventory_add", "inventory_remove"}


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _keys(value: dict[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _id(value: Any, path: str) -> str:
    text = _text(value, path)
    if not ID_PATTERN.fullmatch(text):
        raise ValueError(f"{path} must be a stable ASCII identifier")
    return text


def _unique_ids(items: list[Any], namespace: str) -> set[str]:
    seen: set[str] = set()
    for index, raw in enumerate(items):
        item = _object(raw, f"{namespace}[{index}]")
        identifier = _id(item.get("id"), f"{namespace}[{index}].id")
        if identifier in seen:
            raise ValueError(f"duplicate id {identifier!r} in {namespace}")
        seen.add(identifier)
    return seen


def _validate_condition(raw: Any, path: str) -> None:
    value = _object(raw, path)
    _keys(value, {"op", "path", "value"}, path)
    if value["op"] not in CONDITION_OPS:
        raise ValueError(f"{path}.op is not allowed: {value['op']!r}")
    _text(value["path"], f"{path}.path")


def _validate_effect(raw: Any, path: str) -> None:
    value = _object(raw, path)
    _keys(value, {"op", "path", "value"}, path)
    if value["op"] not in EFFECT_OPS:
        raise ValueError(f"{path}.op is not allowed: {value['op']!r}")
    _text(value["path"], f"{path}.path")


def _validate_operation_path(operation: dict, path: str, object_ids: set[str], objective_ids: set[str], inventory_ids: set[str]) -> None:
    op, state_path, value = operation["op"], operation["path"], operation["value"]
    if op.startswith("inventory_"):
        if state_path != "inventory":
            raise ValueError(f"{path}.path must be 'inventory' for {op}")
        if value not in inventory_ids:
            raise ValueError(f"{path}.value references unknown inventory id {value!r}")
        return
    parts = state_path.split(".")
    if any(not ID_PATTERN.fullmatch(part) for part in parts):
        raise ValueError(f"{path}.path must be a typed state path")
    namespace = parts[0]
    if namespace == "flags" and len(parts) == 2:
        return
    if len(parts) != 3:
        raise ValueError(f"{path}.path must be a typed state path")
    identifier = parts[1]
    if namespace == "objects" and identifier in object_ids:
        return
    if namespace == "objectives" and identifier in objective_ids:
        return
    raise ValueError(f"{path}.path references an unknown state namespace or id")


def validate_adventure_spec(spec: Any, ontology: Any | None = None) -> dict[str, Any]:
    """Validate and return an isolated copy of an AdventureSpec.

    The validator deliberately rejects every unknown field. Conditions and
    effects are data-only operations; no expression evaluator is involved.
    """

    root = _object(spec, "AdventureSpec")
    required = {
        "schema_version", "title", "premise", "tone", "visual_theme", "player",
        "characters", "locations", "objects", "inventory_items", "objectives",
        "puzzles", "interactions", "ending_conditions",
    }
    _keys(root, required, "AdventureSpec")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported AdventureSpec schema_version {root['schema_version']!r}")
    for name in ("title", "premise", "tone", "visual_theme"):
        _text(root[name], f"AdventureSpec.{name}")

    player = _object(root["player"], "player")
    _keys(player, {"id", "name", "location_id", "start_position"}, "player")
    _id(player["id"], "player.id")
    _text(player["name"], "player.name")
    _id(player["location_id"], "player.location_id")
    _point(player["start_position"], "player.start_position")

    lists = {name: _list(root[name], name) for name in (
        "characters", "locations", "objects", "inventory_items", "objectives",
        "puzzles", "interactions", "ending_conditions",
    )}
    ids = {name: _unique_ids(items, name) for name, items in lists.items()}

    for index, item in enumerate(lists["characters"]):
        path = f"characters[{index}]"
        _keys(item, {"id", "name", "archetype", "location_id", "description"}, path)
        for key in ("name", "archetype", "description"):
            _text(item[key], f"{path}.{key}")
        _reference(item["location_id"], ids["locations"], f"{path}.location_id")

    for index, item in enumerate(lists["locations"]):
        path = f"locations[{index}]"
        _keys(item, {"id", "name", "description", "theme", "size", "mood"}, path)
        for key in ("name", "description", "theme", "mood"):
            _text(item[key], f"{path}.{key}")
        size = _list(item["size"], f"{path}.size")
        if len(size) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in size):
            raise ValueError(f"{path}.size must contain two positive integers")

    for index, item in enumerate(lists["objects"]):
        path = f"objects[{index}]"
        _keys(item, {"id", "name", "class", "location_id", "description", "portable", "required"}, path)
        for key in ("name", "class", "description"):
            _text(item[key], f"{path}.{key}")
        _reference(item["location_id"], ids["locations"], f"{path}.location_id")
        if not isinstance(item["portable"], bool) or not isinstance(item["required"], bool):
            raise ValueError(f"{path}.portable and required must be booleans")

    for index, item in enumerate(lists["inventory_items"]):
        path = f"inventory_items[{index}]"
        _keys(item, {"id", "name", "description"}, path)
        _text(item["name"], f"{path}.name")
        _text(item["description"], f"{path}.description")

    for index, item in enumerate(lists["objectives"]):
        path = f"objectives[{index}]"
        _keys(item, {"id", "description", "required"}, path)
        _text(item["description"], f"{path}.description")
        if not isinstance(item["required"], bool):
            raise ValueError(f"{path}.required must be a boolean")

    for index, item in enumerate(lists["puzzles"]):
        path = f"puzzles[{index}]"
        _keys(item, {"id", "description", "objective_ids", "interaction_ids"}, path)
        _text(item["description"], f"{path}.description")
        for ref_index, ref in enumerate(_list(item["objective_ids"], f"{path}.objective_ids")):
            _reference(ref, ids["objectives"], f"{path}.objective_ids[{ref_index}]")
        for ref_index, ref in enumerate(_list(item["interaction_ids"], f"{path}.interaction_ids")):
            _reference(ref, ids["interactions"], f"{path}.interaction_ids[{ref_index}]")

    target_ids = ids["objects"] | ids["characters"]
    inventory_ids = ids["inventory_items"]
    for index, item in enumerate(lists["interactions"]):
        path = f"interactions[{index}]"
        _keys(item, {"id", "verb", "target_id", "item_ids", "conditions", "effects", "text", "animation_hint"}, path)
        if item["verb"] not in VERBS - {"move_to"}:
            raise ValueError(f"{path}.verb is not allowed: {item['verb']!r}")
        _reference(item["target_id"], target_ids, f"{path}.target_id")
        for ref_index, ref in enumerate(_list(item["item_ids"], f"{path}.item_ids")):
            _reference(ref, inventory_ids, f"{path}.item_ids[{ref_index}]")
        for cond_index, cond in enumerate(_list(item["conditions"], f"{path}.conditions")):
            _validate_condition(cond, f"{path}.conditions[{cond_index}]")
            _validate_operation_path(cond, f"{path}.conditions[{cond_index}]", ids["objects"], ids["objectives"], inventory_ids)
        for effect_index, effect in enumerate(_list(item["effects"], f"{path}.effects")):
            _validate_effect(effect, f"{path}.effects[{effect_index}]")
            _validate_operation_path(effect, f"{path}.effects[{effect_index}]", ids["objects"], ids["objectives"], inventory_ids)
        _text(item["text"], f"{path}.text")
        _text(item["animation_hint"], f"{path}.animation_hint")

    for index, item in enumerate(lists["ending_conditions"]):
        path = f"ending_conditions[{index}]"
        _keys(item, {"id", "description", "conditions"}, path)
        _text(item["description"], f"{path}.description")
        conditions = _list(item["conditions"], f"{path}.conditions")
        if not conditions:
            raise ValueError(f"{path}.conditions must not be empty")
        for cond_index, cond in enumerate(conditions):
            _validate_condition(cond, f"{path}.conditions[{cond_index}]")
            _validate_operation_path(cond, f"{path}.conditions[{cond_index}]", ids["objects"], ids["objectives"], inventory_ids)

    _reference(player["location_id"], ids["locations"], "player.location_id")
    if ontology is not None:
        ontology.validate_spec(root)
    return deepcopy(root)


def _reference(value: Any, choices: set[str], path: str) -> str:
    identifier = _id(value, path)
    if identifier not in choices:
        raise ValueError(f"{path} references unknown id {identifier!r}")
    return identifier


def _point(value: Any, path: str) -> tuple[float, float]:
    point = _list(value, path)
    if len(point) != 2 or any(not isinstance(n, (int, float)) or isinstance(n, bool) for n in point):
        raise ValueError(f"{path} must contain two numbers")
    return float(point[0]), float(point[1])


def validate_point(value: Any, path: str = "point") -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        not isinstance(n, (int, float)) or isinstance(n, bool) for n in value
    ):
        raise ValueError(f"{path} must contain two numbers")
    return float(value[0]), float(value[1])


def validate_polygon(value: Any, path: str = "polygon", *, convex: bool = False) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")
    points = [validate_point(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(points) < 3:
        raise ValueError(f"{path} must contain at least three points")
    if len(set(points)) != len(points):
        raise ValueError(f"{path} contains duplicate points")
    cross_values = []
    for index in range(len(points)):
        a, b, c = points[index - 2], points[index - 1], points[index]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            cross_values.append(cross)
    if not cross_values:
        raise ValueError(f"{path} has zero area")
    if convex and any(value * cross_values[0] < 0 for value in cross_values[1:]):
        raise ValueError(f"{path} must be convex")
    return [[x, y] for x, y in points]
