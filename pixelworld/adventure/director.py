"""Provider-neutral story directors and explicitly selected deterministic fixtures."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

from .models import SCHEMA_VERSION


class StoryDirector(ABC):
    source: str

    @abstractmethod
    def create_spec(self, prompt: str) -> dict:
        raise NotImplementedError


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
