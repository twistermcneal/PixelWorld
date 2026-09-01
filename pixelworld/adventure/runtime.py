"""Deterministic runtime with typed saves and atomic interactions."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, VERBS, validate_point, value_matches_type
from .navigation import point_walkable, shortest_route


@dataclass
class ActionResult:
    success: bool
    message: str
    state_changes: list[dict[str, Any]]
    animation_hint: str
    next_available_actions: list[dict[str, Any]]
    movement_path: list[list[float]] | None = None

    def as_dict(self):
        return {"success": self.success, "message": self.message, "state_changes": deepcopy(self.state_changes), "animation_hint": self.animation_hint, "next_available_actions": deepcopy(self.next_available_actions), "movement_path": deepcopy(self.movement_path)}


class AdventureRuntime:
    def __init__(self, game: dict, state: dict | None = None):
        self.game = deepcopy(game)
        self.scene = self.game["scene_graph"]
        self.rules = self.game["runtime_rules"]
        self.state_schema = self.game["state_schema"]
        self.entity_by_id = {item["id"]: item for item in self.scene["entities"]}
        self.inventory_ids = set(self.state_schema["inventory_ids"])
        self.state = self._validate_state(state or self._new_state())

    def _new_state(self):
        state = deepcopy(self.scene["initial_state"])
        state.update({"schema_version": SCHEMA_VERSION, "game_digest": self.game["compile_digest"], "completed": False})
        state["completed"] = self._ending_reached(state)
        return state

    def _validate_state(self, raw: Any) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("runtime save must be an object")
        required = {"schema_version", "game_digest", "player_position", "inventory", "objects", "objectives", "flags", "completed"}
        missing, unknown = required - raw.keys(), raw.keys() - required
        if missing:
            raise ValueError(f"runtime save is missing fields: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"runtime save has unknown fields: {', '.join(sorted(unknown))}")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError("runtime save has an unsupported schema_version")
        if raw["game_digest"] != self.game["compile_digest"]:
            raise ValueError("runtime save belongs to a different compiled game")
        position = validate_point(raw["player_position"], "runtime save player_position")
        collisions = [item["polygon"] for item in self.scene["collision_polygons"]]
        if not point_walkable(position, self.scene["walkboxes"], collisions):
            raise ValueError("runtime save player_position is not walkable")
        if not isinstance(raw["inventory"], list) or any(not isinstance(item, str) or item not in self.inventory_ids for item in raw["inventory"]):
            raise ValueError("runtime save contains an unknown inventory item")
        if raw["inventory"] != sorted(set(raw["inventory"])):
            raise ValueError("runtime save inventory must be sorted and unique")
        self._validate_namespace(raw["objects"], self.state_schema["objects"], "objects")
        self._validate_namespace(raw["objectives"], self.state_schema["objectives"], "objectives")
        self._validate_flat_namespace(raw["flags"], self.state_schema["flags"], "flags")
        if not isinstance(raw["completed"], bool):
            raise ValueError("runtime save completed must be a boolean")
        calculated = self._ending_reached(raw)
        if raw["completed"] is not calculated:
            raise ValueError("runtime save completed does not match ending conditions")
        return deepcopy(raw)

    def _validate_namespace(self, values, schema, label):
        if not isinstance(values, dict) or set(values) != set(schema):
            raise ValueError(f"runtime save {label} IDs do not match the game")
        for identifier, fields in schema.items():
            item = values[identifier]
            if not isinstance(item, dict) or set(item) != set(fields):
                raise ValueError(f"runtime save {label}.{identifier} fields do not match the game")
            for field, expected in fields.items():
                if not value_matches_type(item[field], expected):
                    raise ValueError(f"runtime save {label}.{identifier}.{field} must have exact JSON type {expected}")

    def _validate_flat_namespace(self, values, schema, label):
        if not isinstance(values, dict) or set(values) != set(schema):
            raise ValueError(f"runtime save {label} IDs do not match the game")
        for identifier, expected in schema.items():
            if not value_matches_type(values[identifier], expected):
                raise ValueError(f"runtime save {label}.{identifier} must have exact JSON type {expected}")

    def save_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if path is not None:
            destination = Path(path)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(destination)
        return text

    @classmethod
    def load_json(cls, game: dict, source: str | Path):
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            candidate = str(source)
            text = candidate if candidate.lstrip().startswith("{") else Path(candidate).read_text(encoding="utf-8")
        return cls(game, json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value {value}"))))

    @property
    def completed(self):
        return self.state["completed"]

    def move_to(self, point) -> ActionResult:
        try:
            route = shortest_route(self.state["player_position"], point, self.scene["walkboxes"], self.scene["navigation_edges"], self.scene["collision_polygons"])
            candidate = deepcopy(self.state)
            before = candidate["player_position"]
            candidate["player_position"] = route[-1]
            candidate = self._validate_state(candidate)
        except (ValueError, TypeError) as error:
            return self._result(False, str(error), [], "none")
        self.state = candidate
        return self._result(True, "Ziel erreicht.", [{"path": "player_position", "before": before, "after": route[-1]}], "walk", route)

    def look_at(self, target_id: str) -> ActionResult:
        entity = self.entity_by_id.get(target_id)
        if entity is None or not self._entity_available(entity):
            return self._result(False, "Dort gibt es nichts zu betrachten.", [], "none")
        return self._result(True, entity["description"], [], "look")

    def talk_to(self, target_id: str) -> ActionResult:
        entity = self.entity_by_id.get(target_id)
        if entity is None or entity["hotspot_role"] != "npc" or not self._entity_available(entity) or not entity["default_talk_text"]:
            return self._result(False, "Damit kann man nicht sprechen.", [], "none")
        return self._result(True, entity["default_talk_text"], [], "talk")

    def take(self, target_id: str) -> ActionResult:
        return self._invoke("take", target_id, [])

    def use(self, item_id: str, target_id: str) -> ActionResult:
        return self._invoke("use", target_id, [item_id])

    def combine(self, first_id: str, second_id: str, container_id: str) -> ActionResult:
        if not all(isinstance(item, str) for item in (first_id, second_id, container_id)):
            return self._result(False, "Ungültige Kombinationsparameter.", [], "none")
        return self._invoke("combine", container_id, sorted([first_id, second_id]))

    def perform(self, action: dict) -> ActionResult:
        if not isinstance(action, dict):
            return self._result(False, "Aktion muss ein Objekt sein.", [], "none")
        verb = action.get("verb")
        if verb not in VERBS:
            return self._result(False, f"Unzulässiges Verb: {verb!r}.", [], "none")
        if verb == "move_to":
            return self.move_to(action.get("point"))
        if verb == "look_at":
            return self.look_at(action.get("target_id"))
        if verb == "talk_to":
            return self.talk_to(action.get("target_id"))
        if verb == "take":
            return self.take(action.get("target_id"))
        if verb == "use":
            return self.use(action.get("item_id"), action.get("target_id"))
        return self.combine(action.get("first_id"), action.get("second_id"), action.get("container_id"))

    def _invoke(self, verb, target_id, item_ids):
        entity = self.entity_by_id.get(target_id)
        target_available = entity and (self._entity_available(entity) or (verb == "combine" and target_id in self.state["inventory"]))
        if not target_available:
            return self._result(False, "Ziel ist nicht verfügbar.", [], "none")
        candidates = [item for item in self.rules["interactions"] if item["verb"] == verb and item["target_id"] == target_id and sorted(item["item_ids"]) == sorted(item_ids)]
        if not candidates:
            return self._result(False, "Diese Kombination funktioniert nicht.", [], "none")
        interaction = min(candidates, key=lambda item: item["id"])
        try:
            if not all(self._condition(condition, self.state) for condition in interaction["conditions"]):
                return self._result(False, "Dafür fehlen noch Voraussetzungen.", [], "none")
            route = [list(self.state["player_position"])] if verb == "combine" and target_id in self.state["inventory"] else shortest_route(self.state["player_position"], entity["walk_to_point"], self.scene["walkboxes"], self.scene["navigation_edges"], self.scene["collision_polygons"])
            candidate = deepcopy(self.state)
            changes = []
            if candidate["player_position"] != route[-1]:
                changes.append({"path": "player_position", "before": candidate["player_position"], "after": route[-1]})
                candidate["player_position"] = route[-1]
            for effect in interaction["effects"]:
                changes.append(self._apply_effect(effect, candidate))
            before_completed = candidate["completed"]
            candidate["completed"] = self._ending_reached(candidate)
            if before_completed != candidate["completed"]:
                changes.append({"path": "completed", "before": before_completed, "after": candidate["completed"]})
            candidate = self._validate_state(candidate)
        except (ValueError, TypeError, KeyError) as error:
            return self._result(False, f"Interaktion abgebrochen: {error}", [], "none")
        self.state = candidate
        return self._result(True, interaction["text"], changes, interaction["animation_hint"], route)

    def _entity_available(self, entity):
        state = self.state["objects"][entity["id"]]
        return entity["visible"] and entity["enabled"] and not state.get("taken", False)

    def _condition(self, condition, state):
        if condition["op"] == "equals":
            return self._read_path(condition["path"], state) == condition["value"] and type(self._read_path(condition["path"], state)) is type(condition["value"])
        inventory = state["inventory"]
        return condition["value"] in inventory if condition["op"] == "inventory_contains" else condition["value"] not in inventory

    def _apply_effect(self, effect, state):
        op, path, value = effect["op"], effect["path"], effect["value"]
        if op == "set":
            before = deepcopy(self._read_path(path, state))
            self._write_path(path, deepcopy(value), state)
            return {"path": path, "before": before, "after": deepcopy(value)}
        if path != "inventory":
            raise ValueError(f"inventory operation requires inventory path, got {path!r}")
        before = list(state["inventory"])
        if op == "inventory_add":
            if value not in self.inventory_ids:
                raise ValueError(f"unknown inventory item {value!r}")
            if value in state["inventory"]:
                raise ValueError(f"inventory item {value!r} already exists")
            state["inventory"].append(value)
            state["inventory"].sort()
        elif op == "inventory_remove":
            if value not in state["inventory"]:
                raise ValueError(f"cannot remove missing inventory item {value!r}")
            state["inventory"].remove(value)
        else:
            raise ValueError(f"unsupported effect operation {op!r}")
        return {"path": path, "before": before, "after": list(state["inventory"])}

    def _read_path(self, path, state):
        value = state
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"state path {path!r} does not exist")
            value = value[part]
        return value

    def _write_path(self, path, value, state):
        parts = path.split(".")
        target = state
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"state path {path!r} does not exist")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"state path {path!r} does not exist")
        target[parts[-1]] = value

    def _ending_reached(self, state):
        return any(all(self._condition(condition, state) for condition in ending["conditions"]) for ending in self.rules["ending_conditions"])

    def available_actions(self):
        actions = []
        for interaction in sorted(self.rules["interactions"], key=lambda item: item["id"]):
            entity = self.entity_by_id.get(interaction["target_id"])
            available = entity and (self._entity_available(entity) or (interaction["verb"] == "combine" and interaction["target_id"] in self.state["inventory"]))
            if available and all(self._condition(condition, self.state) for condition in interaction["conditions"]):
                actions.append({"interaction_id": interaction["id"], "verb": interaction["verb"], "target_id": interaction["target_id"], "item_ids": list(interaction["item_ids"])})
        return actions

    def _result(self, success, message, changes, animation, movement_path=None):
        return ActionResult(success, message, changes, animation, self.available_actions(), movement_path)


def action_from_interaction(interaction: dict) -> dict:
    verb = interaction["verb"]
    if verb == "take":
        return {"verb": verb, "target_id": interaction["target_id"]}
    if verb == "use":
        return {"verb": verb, "item_id": interaction["item_ids"][0], "target_id": interaction["target_id"]}
    if verb == "combine":
        return {"verb": verb, "first_id": interaction["item_ids"][0], "second_id": interaction["item_ids"][1], "container_id": interaction["target_id"]}
    return {"verb": verb, "target_id": interaction["target_id"]}
