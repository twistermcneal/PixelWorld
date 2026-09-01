"""Provider-neutral story-director interfaces and deterministic fixtures."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

from .models import SCHEMA_VERSION


class StoryDirector(ABC):
    """Produces structured AdventureSpec data, never executable code."""

    @abstractmethod
    def create_spec(self, prompt: str) -> dict:
        raise NotImplementedError


class JsonStoryDirector(StoryDirector):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create_spec(self, prompt: str) -> dict:
        del prompt
        return json.loads(self.path.read_text(encoding="utf-8"))


class FixtureStoryDirector(StoryDirector):
    """Returns a test fixture; it is intentionally not represented as LLM output."""

    def create_spec(self, prompt: str) -> dict:
        del prompt
        return deepcopy(GOLDEN_ADVENTURE_SPEC)


def _condition(op: str, path: str, value):
    return {"op": op, "path": path, "value": value}


def _effect(op: str, path: str, value):
    return {"op": op, "path": path, "value": value}


GOLDEN_ADVENTURE_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "title": "Professor Knallberts chronochemisches Labor",
    "premise": "Professor Knallberts Zeitmaschine überhitzt. Der Spieler muss ein Kühlmittel herstellen und in die Maschine einsetzen.",
    "tone": "farbenfroh, verrückt und optimistisch",
    "visual_theme": "mad_scientist_lab",
    "player": {"id": "player", "name": "Nova", "location_id": "chronochemical_lab", "start_position": [15, 60]},
    "characters": [
        {"id": "professor_knallbert", "name": "Professor Knallbert", "archetype": "mad_scientist", "location_id": "chronochemical_lab", "description": "Ein zerzauster Chronochemiker mit leuchtender Schutzbrille."}
    ],
    "locations": [
        {"id": "chronochemical_lab", "name": "Professor Knallberts chronochemisches Labor", "description": "Ein dunkles Neonlabor voller Roboterarme, Zahnräder und Funken.", "theme": "mad_scientist_lab", "size": [128, 72], "mood": "neon_sparks"}
    ],
    "objects": [
        {"id": "time_machine", "name": "Zeitmaschine", "class": "time_machine", "location_id": "chronochemical_lab", "description": "Eine überhitzte Maschine mit türkisfarbenem Kern.", "portable": False, "required": True},
        {"id": "control_console", "name": "Bedienpult", "class": "control_console", "location_id": "chronochemical_lab", "description": "Blinkende Hebel melden kritische Chronotemperatur.", "portable": False, "required": True},
        {"id": "coolant_red", "name": "Rotes Kühlreagenz", "class": "chemical_bottle", "location_id": "chronochemical_lab", "description": "Ein rotes, zähflüssiges Chronoreagenz.", "portable": True, "required": True},
        {"id": "coolant_blue", "name": "Blaues Kühlreagenz", "class": "chemical_bottle", "location_id": "chronochemical_lab", "description": "Ein blaues, eisig funkelndes Chronoreagenz.", "portable": True, "required": True},
        {"id": "catalyst_green", "name": "Grüner Katalysator", "class": "chemical_bottle", "location_id": "chronochemical_lab", "description": "Ein optionaler grüner Katalysator mit Warnetikett.", "portable": False, "required": False},
        {"id": "mixing_flask", "name": "Leere Mischflasche", "class": "mixing_flask", "location_id": "chronochemical_lab", "description": "Eine druckfeste Flasche für zwei Reagenzien.", "portable": True, "required": True},
        {"id": "time_portal", "name": "Zeitportal", "class": "time_portal", "location_id": "chronochemical_lab", "description": "Der Ausgang durch die Zeit; momentan noch inaktiv.", "portable": False, "required": True},
        {"id": "robot_arm_left", "name": "Roboterarm", "class": "robot_arm", "location_id": "chronochemical_lab", "description": "Ein mechanischer Arm sortiert Funken nach Größe.", "portable": False, "required": False},
        {"id": "wall_gears", "name": "Zahnräder", "class": "gear", "location_id": "chronochemical_lab", "description": "Ein unmögliches Getriebe läuft rückwärts.", "portable": False, "required": False}
    ],
    "inventory_items": [
        {"id": "coolant_red", "name": "Rotes Kühlreagenz", "description": "Rote Komponente des Kühlmittels."},
        {"id": "coolant_blue", "name": "Blaues Kühlreagenz", "description": "Blaue Komponente des Kühlmittels."},
        {"id": "mixing_flask", "name": "Mischflasche", "description": "Kann beide Reagenzien aufnehmen."},
        {"id": "mixed_coolant", "name": "Chronokühlmittel", "description": "Fertig gemischtes violettes Kühlmittel."}
    ],
    "objectives": [
        {"id": "cool_time_machine", "description": "Kühlmittel mischen und in die Zeitmaschine einsetzen.", "required": True}
    ],
    "puzzles": [
        {"id": "coolant_puzzle", "description": "Zwei Reagenzien und die Flasche einsammeln, mischen und anwenden.", "objective_ids": ["cool_time_machine"], "interaction_ids": ["take_red", "take_blue", "take_flask", "mix_coolant", "cool_machine"]}
    ],
    "interactions": [
        {"id": "take_red", "verb": "take", "target_id": "coolant_red", "item_ids": [], "conditions": [_condition("equals", "objects.coolant_red.taken", False)], "effects": [_effect("set", "objects.coolant_red.taken", True), _effect("inventory_add", "inventory", "coolant_red")], "text": "Das rote Reagenz landet sicher im Inventar.", "animation_hint": "pickup"},
        {"id": "take_blue", "verb": "take", "target_id": "coolant_blue", "item_ids": [], "conditions": [_condition("equals", "objects.coolant_blue.taken", False)], "effects": [_effect("set", "objects.coolant_blue.taken", True), _effect("inventory_add", "inventory", "coolant_blue")], "text": "Das blaue Reagenz ist erstaunlich kalt.", "animation_hint": "pickup"},
        {"id": "take_flask", "verb": "take", "target_id": "mixing_flask", "item_ids": [], "conditions": [_condition("equals", "objects.mixing_flask.taken", False)], "effects": [_effect("set", "objects.mixing_flask.taken", True), _effect("inventory_add", "inventory", "mixing_flask")], "text": "Die Mischflasche ist bereit.", "animation_hint": "pickup"},
        {"id": "mix_coolant", "verb": "combine", "target_id": "mixing_flask", "item_ids": ["coolant_blue", "coolant_red"], "conditions": [_condition("inventory_contains", "inventory", "coolant_red"), _condition("inventory_contains", "inventory", "coolant_blue"), _condition("inventory_contains", "inventory", "mixing_flask"), _condition("equals", "objects.mixing_flask.contents", "empty")], "effects": [_effect("inventory_remove", "inventory", "coolant_red"), _effect("inventory_remove", "inventory", "coolant_blue"), _effect("inventory_remove", "inventory", "mixing_flask"), _effect("inventory_add", "inventory", "mixed_coolant"), _effect("set", "objects.mixing_flask.contents", "mixed_coolant")], "text": "Die Mischung leuchtet violett: Chronokühlmittel!", "animation_hint": "mix"},
        {"id": "cool_machine", "verb": "use", "target_id": "time_machine", "item_ids": ["mixed_coolant"], "conditions": [_condition("inventory_contains", "inventory", "mixed_coolant"), _condition("equals", "objects.time_machine.cooled", False)], "effects": [_effect("inventory_remove", "inventory", "mixed_coolant"), _effect("set", "objects.time_machine.cooled", True), _effect("set", "objects.time_portal.active", True), _effect("set", "objectives.cool_time_machine.completed", True)], "text": "Die Zeitmaschine kühlt ab und das Portal stabilisiert sich!", "animation_hint": "machine_cool"}
    ],
    "ending_conditions": [
        {"id": "portal_stable", "description": "Die Zeitmaschine ist gekühlt und das Portal aktiv.", "conditions": [_condition("equals", "objects.time_machine.cooled", True), _condition("equals", "objects.time_portal.active", True), _condition("equals", "objectives.cool_time_machine.completed", True)]}
    ]
}
