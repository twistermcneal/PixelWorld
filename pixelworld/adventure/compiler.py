"""Compile validated AdventureSpec data into RoomSpec and Scene Graph data."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .models import SCHEMA_VERSION, validate_adventure_spec
from .ontology import ThemeOntology


LAYOUT = {
    "time_machine": ([64, 35], [50, 19, 28, 35], [64, 57], 30, "machine"),
    "control_console": ([39, 38], [33, 31, 13, 14], [40, 50], 35, "console"),
    "coolant_red": ([17, 34], [14, 28, 6, 12], [18, 49], 42, "ingredient"),
    "coolant_blue": ([27, 34], [24, 28, 6, 12], [28, 49], 42, "ingredient"),
    "catalyst_green": ([108, 34], [105, 28, 6, 12], [107, 49], 42, "ingredient"),
    "mixing_flask": ([90, 38], [87, 31, 7, 13], [90, 50], 42, "container"),
    "time_portal": ([116, 30], [110, 13, 12, 31], [115, 49], 28, "exit"),
    "robot_arm_left": ([8, 26], [5, 13, 15, 24], [14, 48], 25, "scenery"),
    "wall_gears": ([91, 20], [83, 10, 17, 19], [90, 48], 15, "scenery"),
    "professor_knallbert": ([101, 49], [97, 35, 9, 19], [96, 56], 48, "npc"),
}


WALKBOXES = [
    {"id": "left_floor", "polygon": [[4, 42], [48, 42], [48, 68], [4, 68]], "z_min": 40, "z_max": 70},
    {"id": "front_floor", "polygon": [[48, 54], [80, 54], [80, 68], [48, 68]], "z_min": 54, "z_max": 70},
    {"id": "right_floor", "polygon": [[80, 42], [124, 42], [124, 68], [80, 68]], "z_min": 40, "z_max": 70},
]

NAVIGATION_EDGES = [
    {"from": "front_floor", "to": "left_floor", "point": [48, 59]},
    {"from": "front_floor", "to": "right_floor", "point": [80, 59]},
]


def _polygon_from_bbox(bbox, padding=1):
    x, y, width, height = bbox
    return [[x - padding, y - padding], [x + width + padding, y - padding], [x + width + padding, y + height + padding], [x - padding, y + height + padding]]


def compile_adventure(spec: dict, ontology: ThemeOntology | None = None) -> dict:
    ontology = ontology or ThemeOntology()
    spec = validate_adventure_spec(spec, ontology)
    location = spec["locations"][0]
    if location["size"] != [128, 72]:
        raise ValueError("0.6.3 Phase 1 requires the fixed logical room size 128x72")
    required_objects = [item["id"] for item in spec["objects"] if item["required"]]
    required_entities = required_objects + [item["id"] for item in spec["characters"]]
    objective_ids = [item["id"] for item in spec["objectives"]]
    room_spec = {
        "schema_version": SCHEMA_VERSION,
        "room_id": location["id"],
        "name": location["name"],
        "description": location["description"],
        "theme": location["theme"],
        "size": location["size"],
        "mood": location["mood"],
        "required_entities": required_entities,
        "required_exits": ["time_portal"],
        "player_start": spec["player"]["start_position"],
        "objective_ids": objective_ids,
    }
    interactions_by_target = {}
    for interaction in spec["interactions"]:
        interactions_by_target.setdefault(interaction["target_id"], []).append(interaction["id"])
    entities = []
    for source in spec["objects"] + spec["characters"]:
        if source["id"] not in LAYOUT:
            if source.get("required", True):
                raise ValueError(f"required entity {source['id']!r} has no deterministic layout")
            continue
        position, bbox, walk_to, z_layer, role = LAYOUT[source["id"]]
        state = {"taken": False} if source.get("portable") else {}
        if source["id"] == "mixing_flask":
            state["contents"] = "empty"
        if source["id"] == "time_machine":
            state["cooled"] = False
        if source["id"] == "time_portal":
            state["active"] = False
        entities.append({
            "id": source["id"], "class": source.get("class", source.get("archetype")),
            "name": source["name"], "description": source["description"],
            "position": position, "bbox": bbox, "hotspot_polygon": _polygon_from_bbox(bbox),
            "walk_to_point": walk_to, "z_layer": z_layer, "state": state,
            "visible": True, "enabled": True, "interactions": sorted(interactions_by_target.get(source["id"], [])),
            "hotspot_role": role, "required": source.get("required", True),
        })
    collision_polygons = [
        {"id": "machine_collision", "polygon": [[50, 19], [78, 19], [78, 54], [50, 54]]},
        {"id": "left_bench_collision", "polygon": [[4, 25], [34, 25], [34, 42], [4, 42]]},
        {"id": "right_bench_collision", "polygon": [[94, 25], [124, 25], [124, 42], [94, 42]]},
    ]
    scene_graph = {
        "schema_version": SCHEMA_VERSION,
        "room_id": location["id"],
        "size": location["size"],
        "theme": location["theme"],
        "palette": ontology.get(location["theme"])["palette"],
        "background_layers": [
            {"id": "lab_wall", "z_layer": 0, "class": "industrial_wall"},
            {"id": "chemical_shelves", "z_layer": 10, "class": "chemical_shelf"},
            {"id": "neon_machinery", "z_layer": 20, "class": "robot_arm"},
        ],
        "semantic_regions": [
            {"id": "laboratory_floor", "class": "laboratory_floor", "polygon": [[0, 40], [128, 40], [128, 72], [0, 72]]},
            {"id": "machine_zone", "class": "metal_platform", "polygon": [[47, 17], [81, 17], [85, 57], [43, 57]]},
        ],
        "walkboxes": deepcopy(WALKBOXES),
        "navigation_edges": deepcopy(NAVIGATION_EDGES),
        "collision_polygons": collision_polygons,
        "occlusion_polygons": [
            {"id": "foreground_bottles", "polygon": [[0, 61], [20, 56], [29, 72], [0, 72]], "z_layer": 90},
            {"id": "foreground_pipes", "polygon": [[105, 62], [128, 54], [128, 72], [100, 72]], "z_layer": 90},
        ],
        "entities": entities,
        "hotspots": [{"id": f"hotspot_{item['id']}", "entity_id": item["id"], "polygon": item["hotspot_polygon"], "role": item["hotspot_role"], "required": item["required"]} for item in entities],
        "walk_to_points": [{"entity_id": item["id"], "point": item["walk_to_point"]} for item in entities],
        "z_layers": {"background": 0, "architecture": 10, "entities": 30, "player_base": 45, "foreground": 90, "ui": 100},
        "portals": [{"id": "time_portal", "entity_id": "time_portal", "type": "time_portal", "active_path": "objects.time_portal.active", "destination": "ending"}],
        "initial_state": {
            "player_position": spec["player"]["start_position"],
            "inventory": [],
            "objects": {item["id"]: deepcopy(item["state"]) for item in entities},
            "objectives": {item["id"]: {"completed": False} for item in spec["objectives"]},
            "flags": {},
        },
    }
    game = {
        "schema_version": SCHEMA_VERSION,
        "adventure": spec,
        "room": room_spec,
        "scene_graph": scene_graph,
        "runtime_rules": {"verbs": ["move_to", "look_at", "talk_to", "take", "use", "combine"], "interactions": deepcopy(spec["interactions"]), "ending_conditions": deepcopy(spec["ending_conditions"])},
    }
    game["compile_digest"] = compile_digest(game)
    return game


def compile_digest(game: dict) -> str:
    payload = deepcopy(game)
    payload.pop("compile_digest", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
