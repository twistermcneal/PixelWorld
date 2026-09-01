"""Deterministic, headless adventure runtime with validated save games."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, VERBS
from .navigation import shortest_route


@dataclass
class ActionResult:
    success: bool
    message: str
    state_changes: list[dict[str, Any]]
    animation_hint: str
    next_available_actions: list[dict[str, Any]]
    movement_path: list[list[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "state_changes": deepcopy(self.state_changes),
            "animation_hint": self.animation_hint,
            "next_available_actions": deepcopy(self.next_available_actions),
            "movement_path": deepcopy(self.movement_path),
        }


class AdventureRuntime:
    def __init__(self, game: dict, state: dict | None = None):
        self.game = deepcopy(game)
        self.scene = self.game["scene_graph"]
        self.rules = self.game["runtime_rules"]
        self.entity_by_id = {item["id"]: item for item in self.scene["entities"]}
        self.inventory_ids = {item["id"] for item in self.game["adventure"]["inventory_items"]}
        self.state = self._validate_state(state or self._new_state())

    def _new_state(self):
        state = deepcopy(self.scene["initial_state"])
        state.update({"schema_version": SCHEMA_VERSION, "completed": False})
        return state

    def _validate_state(self, raw: Any) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("runtime save must be an object")
        required = {"schema_version", "player_position", "inventory", "objects", "objectives", "flags", "completed"}
        missing, unknown = required - raw.keys(), raw.keys() - required
        if missing:
            raise ValueError(f"runtime save is missing fields: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"runtime save has unknown fields: {', '.join(sorted(unknown))}")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError("runtime save has an unsupported schema_version")
        position = raw["player_position"]
        if not isinstance(position, list) or len(position) != 2 or any(not isinstance(n, (int, float)) or isinstance(n, bool) for n in position):
            raise ValueError("runtime save player_position must contain two numbers")
        if not isinstance(raw["inventory"], list) or any(item not in self.inventory_ids for item in raw["inventory"]):
            raise ValueError("runtime save contains an unknown inventory item")
        if len(raw["inventory"]) != len(set(raw["inventory"])):
            raise ValueError("runtime save inventory contains duplicates")
        expected_objects = set(self.scene["initial_state"]["objects"])
        if not isinstance(raw["objects"], dict) or set(raw["objects"]) != expected_objects:
            raise ValueError("runtime save object namespace does not match the game")
        expected_objectives = set(self.scene["initial_state"]["objectives"])
        if not isinstance(raw["objectives"], dict) or set(raw["objectives"]) != expected_objectives:
            raise ValueError("runtime save objective namespace does not match the game")
        if not isinstance(raw["flags"], dict) or not isinstance(raw["completed"], bool):
            raise ValueError("runtime save flags/completed have invalid types")
        return deepcopy(raw)

    def save_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.state, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def load_json(cls, game: dict, source: str | Path):
        path = Path(source)
        text = path.read_text(encoding="utf-8") if path.is_file() else str(source)
        return cls(game, json.loads(text))

    @property
    def completed(self):
        return self.state["completed"]

    def move_to(self, point) -> ActionResult:
        try:
            route = shortest_route(
                self.state["player_position"], point, self.scene["walkboxes"],
                self.scene["navigation_edges"], self.scene["collision_polygons"],
            )
        except ValueError as error:
            return self._result(False, str(error), [], "none")
        old = self.state["player_position"]
        self.state["player_position"] = route[-1]
        return self._result(True, "Ziel erreicht.", [{"path": "player_position", "before": old, "after": route[-1]}], "walk", route)

    def look_at(self, target_id: str) -> ActionResult:
        entity = self.entity_by_id.get(target_id)
        if entity is None or not self._entity_available(entity):
            return self._result(False, "Dort gibt es nichts zu betrachten.", [], "none")
        return self._result(True, entity["description"], [], "look")

    def talk_to(self, target_id: str) -> ActionResult:
        entity = self.entity_by_id.get(target_id)
        if entity is None or entity["hotspot_role"] != "npc" or not self._entity_available(entity):
            return self._result(False, "Damit kann man nicht sprechen.", [], "none")
        message = "Knallbert: Zwei Reagenzien, eine Flasche – dann ab damit in die Maschine!"
        return self._result(True, message, [], "talk")

    def take(self, target_id: str) -> ActionResult:
        return self._invoke("take", target_id, [])

    def use(self, item_id: str, target_id: str) -> ActionResult:
        return self._invoke("use", target_id, [item_id])

    def combine(self, first_id: str, second_id: str, container_id: str) -> ActionResult:
        return self._invoke("combine", container_id, sorted([first_id, second_id]))

    def perform(self, action: dict) -> ActionResult:
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

    def _invoke(self, verb: str, target_id: str, item_ids: list[str]) -> ActionResult:
        entity = self.entity_by_id.get(target_id)
        if entity is None or not (self._entity_available(entity) or (verb == "combine" and target_id in self.state["inventory"])):
            return self._result(False, "Ziel ist nicht verfügbar.", [], "none")
        desired = sorted(item_ids)
        candidates = [item for item in self.rules["interactions"] if item["verb"] == verb and item["target_id"] == target_id and sorted(item["item_ids"]) == desired]
        if not candidates:
            return self._result(False, "Diese Kombination funktioniert nicht.", [], "none")
        interaction = sorted(candidates, key=lambda item: item["id"])[0]
        if not all(self._condition(condition) for condition in interaction["conditions"]):
            return self._result(False, "Dafür fehlen noch Voraussetzungen.", [], "none")
        if verb == "combine" and target_id in self.state["inventory"]:
            route = [list(self.state["player_position"])]
        else:
            try:
                route = shortest_route(self.state["player_position"], entity["walk_to_point"], self.scene["walkboxes"], self.scene["navigation_edges"], self.scene["collision_polygons"])
            except ValueError:
                return self._result(False, "Das Ziel ist nicht erreichbar.", [], "none")
        changes = []
        old_position = self.state["player_position"]
        self.state["player_position"] = route[-1]
        if old_position != route[-1]:
            changes.append({"path": "player_position", "before": old_position, "after": route[-1]})
        for effect in interaction["effects"]:
            changes.append(self._apply_effect(effect))
        self.state["completed"] = self._ending_reached()
        if self.state["completed"]:
            changes.append({"path": "completed", "before": False, "after": True})
        return self._result(True, interaction["text"], changes, interaction["animation_hint"], route)

    def _entity_available(self, entity):
        state = self.state["objects"].get(entity["id"], {})
        return entity["visible"] and entity["enabled"] and not state.get("taken", False)

    def _condition(self, condition):
        op, path, expected = condition["op"], condition["path"], condition["value"]
        if op == "equals":
            return self._read_path(path) == expected
        if path != "inventory":
            raise ValueError(f"inventory operation requires inventory path, got {path!r}")
        if op == "inventory_contains":
            return expected in self.state["inventory"]
        if op == "inventory_missing":
            return expected not in self.state["inventory"]
        raise ValueError(f"unsupported condition operation {op!r}")

    def _apply_effect(self, effect):
        op, path, value = effect["op"], effect["path"], effect["value"]
        if op == "set":
            before = deepcopy(self._read_path(path))
            self._write_path(path, deepcopy(value))
            return {"path": path, "before": before, "after": deepcopy(value)}
        if path != "inventory":
            raise ValueError(f"inventory operation requires inventory path, got {path!r}")
        before = list(self.state["inventory"])
        if op == "inventory_add":
            if value not in self.inventory_ids:
                raise ValueError(f"unknown inventory item {value!r}")
            if value not in self.state["inventory"]:
                self.state["inventory"].append(value)
                self.state["inventory"].sort()
        elif op == "inventory_remove":
            if value not in self.state["inventory"]:
                raise ValueError(f"cannot remove missing inventory item {value!r}")
            self.state["inventory"].remove(value)
        else:
            raise ValueError(f"unsupported effect operation {op!r}")
        return {"path": path, "before": before, "after": list(self.state["inventory"])}

    def _read_path(self, path):
        parts = path.split(".")
        if parts[0] not in {"objects", "objectives", "flags"} or len(parts) < 2:
            raise ValueError(f"state path {path!r} is not allowed")
        value = self.state
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                if parts[0] == "flags":
                    return None
                raise ValueError(f"state path {path!r} does not exist")
            value = value[part]
        return value

    def _write_path(self, path, value):
        parts = path.split(".")
        if parts[0] not in {"objects", "objectives", "flags"} or len(parts) < 2:
            raise ValueError(f"state path {path!r} is not allowed")
        target = self.state
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"state path {path!r} does not exist")
            target = target[part]
        if parts[-1] not in target and parts[0] != "flags":
            raise ValueError(f"state path {path!r} does not exist")
        target[parts[-1]] = value

    def _ending_reached(self):
        return any(all(self._condition(condition) for condition in ending["conditions"]) for ending in self.rules["ending_conditions"])

    def available_actions(self):
        actions = []
        for interaction in sorted(self.rules["interactions"], key=lambda item: item["id"]):
            entity = self.entity_by_id.get(interaction["target_id"])
            target_available = entity and (
                self._entity_available(entity)
                or (interaction["verb"] == "combine" and interaction["target_id"] in self.state["inventory"])
            )
            if target_available and all(self._condition(condition) for condition in interaction["conditions"]):
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
