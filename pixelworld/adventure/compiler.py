"""Compile validated AdventureSpec data with deterministic theme templates."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .models import SCHEMA_VERSION, VERBS, json_scalar_type, validate_adventure_spec
from .ontology import ThemeOntology


COMMON_WALKBOXES = [
    {"id": "left_floor", "polygon": [[4, 42], [48, 42], [48, 68], [4, 68]], "z_min": 40, "z_max": 70},
    {"id": "front_floor", "polygon": [[48, 54], [80, 54], [80, 68], [48, 68]], "z_min": 54, "z_max": 70},
    {"id": "right_floor", "polygon": [[80, 42], [124, 42], [124, 68], [80, 68]], "z_min": 40, "z_max": 70},
]
COMMON_EDGES = [{"id": "left_to_front", "from": "left_floor", "to": "front_floor", "point": [48, 59]}, {"id": "front_to_right", "from": "front_floor", "to": "right_floor", "point": [80, 59]}]


def _slot(position, walk_to, z=40):
    return {"position": position, "walk_to": walk_to, "z_layer": z}


THEME_TEMPLATES = {
    "mad_scientist_lab": {
        "zones": {
            "center_machine": [_slot([64, 36], [64, 57], 30)],
            "left_console": [_slot([39, 38], [40, 50], 35)],
            "left_reagent": [_slot([17, 34], [18, 49], 42), _slot([27, 34], [28, 49], 42)],
            "right_reagent": [_slot([108, 34], [107, 49], 42)],
            "right_container": [_slot([90, 38], [90, 50], 42)],
            "right_exit": [_slot([116, 29], [115, 49], 28)],
            "left_wall": [_slot([12, 25], [14, 48], 25)],
            "upper_right": [_slot([91, 19], [90, 48], 15)],
            "right_npc": [_slot([101, 47], [96, 56], 48)],
        },
        "background_layers": [{"id": "lab_wall", "z_layer": 0, "class": "industrial_wall"}, {"id": "chemical_shelves", "z_layer": 10, "class": "chemical_shelf"}, {"id": "neon_machinery", "z_layer": 20, "class": "robot_arm"}],
        "semantic_regions": [{"id": "laboratory_floor", "class": "laboratory_floor", "polygon": [[0, 40], [128, 40], [128, 72], [0, 72]]}, {"id": "central_platform", "class": "metal_platform", "polygon": [[47, 17], [81, 17], [85, 57], [43, 57]]}],
        "collision_polygons": [{"id": "central_fixture_collision", "polygon": [[50, 19], [78, 19], [78, 54], [50, 54]]}, {"id": "left_bench_collision", "polygon": [[4, 25], [34, 25], [34, 42], [4, 42]]}, {"id": "right_bench_collision", "polygon": [[94, 25], [124, 25], [124, 42], [94, 42]]}],
        "occlusion_polygons": [{"id": "foreground_bottles", "polygon": [[0, 61], [20, 56], [29, 72], [0, 72]], "z_layer": 90}, {"id": "foreground_pipes", "polygon": [[105, 62], [128, 54], [128, 72], [100, 72]], "z_layer": 90}],
        "floor_class": "laboratory_floor",
    },
    "pirate_harbor": {
        "zones": {
            "dock_left": [_slot([20, 38], [20, 49], 42)],
            "dock_center": [_slot([62, 39], [62, 57], 38)],
            "dock_exit": [_slot([115, 30], [114, 49], 30)],
            "dock_left_wall": [_slot([9, 34], [13, 49], 35)],
            "harbor_npc": [_slot([96, 48], [91, 56], 48)],
        },
        "background_layers": [{"id": "moonlit_water", "z_layer": 0, "class": "water"}, {"id": "harbor_pier", "z_layer": 10, "class": "wooden_pier"}, {"id": "distant_ship", "z_layer": 20, "class": "ship"}],
        "semantic_regions": [{"id": "dock_surface", "class": "dock", "polygon": [[0, 40], [128, 40], [128, 72], [0, 72]]}],
        "collision_polygons": [{"id": "harbor_water_gap", "polygon": [[48, 18], [80, 18], [80, 54], [48, 54]]}, {"id": "left_cargo_collision", "polygon": [[4, 25], [34, 25], [34, 42], [4, 42]]}, {"id": "right_cargo_collision", "polygon": [[94, 25], [124, 25], [124, 42], [94, 42]]}],
        "occlusion_polygons": [{"id": "foreground_cargo", "polygon": [[0, 63], [21, 57], [29, 72], [0, 72]], "z_layer": 90}, {"id": "foreground_rope", "polygon": [[107, 63], [128, 57], [128, 72], [102, 72]], "z_layer": 90}],
        "floor_class": "dock",
    },
}

CLASS_SIZES = {"time_machine": [28, 35], "control_console": [13, 14], "chemical_bottle": [6, 12], "mixing_flask": [7, 13], "time_portal": [12, 31], "robot_arm": [15, 24], "gear": [17, 19], "mad_scientist": [9, 19], "brass_key": [6, 5], "locked_chest": [18, 13], "harbor_exit": [12, 28], "barrel": [10, 15], "pirate": [9, 19]}


def _bbox(position, entity_class):
    width, height = CLASS_SIZES.get(entity_class, [8, 12])
    return [round(position[0] - width / 2, 4), round(position[1] - height / 2, 4), width, height]


def _polygon_from_bbox(bbox, padding=1):
    x, y, width, height = bbox
    return [[x - padding, y - padding], [x + width + padding, y - padding], [x + width + padding, y + height + padding], [x - padding, y + height + padding]]


def _state_schema(spec):
    objects = {}
    for item in spec["objects"] + spec["characters"]:
        objects[item["id"]] = {key: json_scalar_type(value) for key, value in sorted(item["initial_state"].items())}
    objectives = {item["id"]: {key: json_scalar_type(value) for key, value in sorted(item["initial_state"].items())} for item in spec["objectives"]}
    return {"objects": objects, "objectives": objectives, "flags": {item["id"]: item["type"] for item in spec["flags"]}, "inventory_ids": sorted(item["id"] for item in spec["inventory_items"]), "player_position": "point", "completed": "boolean"}


def compile_adventure(spec: dict, ontology: ThemeOntology | None = None) -> dict:
    ontology = ontology or ThemeOntology()
    spec = validate_adventure_spec(spec, ontology)
    location = spec["locations"][0]
    if location["size"] != [128, 72]:
        raise ValueError("0.6.3 Phase 1 requires the fixed logical room size 128x72")
    if location["theme"] not in THEME_TEMPLATES:
        raise ValueError(f"theme {location['theme']!r} has no Phase 1 layout template")
    template = THEME_TEMPLATES[location["theme"]]
    sources = sorted(spec["objects"] + spec["characters"], key=lambda item: (item["preferred_zone"], item.get("class", item.get("archetype")), item["role"], item["id"]))
    grouped = {}
    for source in sources:
        zone = source["preferred_zone"]
        if zone not in template["zones"]:
            raise ValueError(f"entity {source['id']!r} requests unknown zone {zone!r} for theme {location['theme']!r}")
        grouped.setdefault(zone, []).append(source)
    interactions_by_target = {}
    for interaction in spec["interactions"]:
        interactions_by_target.setdefault(interaction["target_id"], []).append(interaction["id"])
    entities = []
    for zone in sorted(grouped):
        occupants, slots = grouped[zone], template["zones"][zone]
        if len(occupants) > len(slots):
            raise ValueError(f"layout zone {zone!r} has {len(slots)} slots but {len(occupants)} entities")
        for source, slot in zip(occupants, slots):
            entity_class = source.get("class", source.get("archetype"))
            bbox = _bbox(slot["position"], entity_class)
            entities.append({"id": source["id"], "class": entity_class, "name": source["name"], "description": source["description"], "default_talk_text": source.get("default_talk_text"), "position": slot["position"], "bbox": bbox, "hotspot_polygon": _polygon_from_bbox(bbox), "walk_to_point": slot["walk_to"], "z_layer": slot["z_layer"], "state": deepcopy(source["initial_state"]), "visible": True, "enabled": True, "portable": source.get("portable", False), "interactions": sorted(interactions_by_target.get(source["id"], [])), "hotspot_role": source["role"], "required": source.get("required", True), "preferred_zone": zone})
    entities.sort(key=lambda item: item["id"])
    exits = sorted(item["id"] for item in spec["objects"] if item["role"] == "exit" and item["required"])
    if not exits:
        raise ValueError("Phase 1 requires at least one required exit role")
    required_entities = sorted(item["id"] for item in spec["objects"] if item["required"]) + sorted(item["id"] for item in spec["characters"])
    room_spec = {"schema_version": SCHEMA_VERSION, "room_id": location["id"], "name": location["name"], "description": location["description"], "theme": location["theme"], "size": location["size"], "mood": location["mood"], "required_entities": required_entities, "required_exits": exits, "player_start": spec["player"]["start_position"], "objective_ids": sorted(item["id"] for item in spec["objectives"])}
    portals = [{"id": item["id"], "entity_id": item["id"], "type": item["class"], "destination": item["portal_destination"], "completion_gated": True} for item in sorted(spec["objects"], key=lambda value: value["id"]) if item["role"] == "exit"]
    initial_objects = {item["id"]: deepcopy(item["state"]) for item in entities}
    scene_graph = {
        "schema_version": SCHEMA_VERSION, "room_id": location["id"], "size": location["size"], "theme": location["theme"], "palette": ontology.get(location["theme"])["palette"],
        "background_layers": deepcopy(template["background_layers"]), "semantic_regions": deepcopy(template["semantic_regions"]), "walkboxes": deepcopy(COMMON_WALKBOXES), "navigation_edges": deepcopy(COMMON_EDGES), "collision_polygons": deepcopy(template["collision_polygons"]), "occlusion_polygons": deepcopy(template["occlusion_polygons"]),
        "entities": entities, "hotspots": [{"id": f"hotspot_{item['id']}", "entity_id": item["id"], "polygon": item["hotspot_polygon"], "role": item["hotspot_role"], "required": item["required"]} for item in entities], "walk_to_points": [{"id": f"walk_to_{item['id']}", "entity_id": item["id"], "point": item["walk_to_point"]} for item in entities],
        "z_layers": {"background": 0, "architecture": 10, "entities": 30, "player_base": 45, "foreground": 90, "ui": 100}, "portals": portals,
        "initial_state": {"player_position": spec["player"]["start_position"], "inventory": [], "objects": initial_objects, "objectives": {item["id"]: deepcopy(item["initial_state"]) for item in spec["objectives"]}, "flags": {item["id"]: item["initial"] for item in spec["flags"]}},
    }
    game = {"schema_version": SCHEMA_VERSION, "adventure": spec, "room": room_spec, "scene_graph": scene_graph, "state_schema": _state_schema(spec), "runtime_rules": {"verbs": sorted(VERBS), "interactions": deepcopy(spec["interactions"]), "ending_conditions": deepcopy(spec["ending_conditions"])}}
    game["compile_digest"] = compile_digest(game)
    return game


def compile_digest(game: dict) -> str:
    payload = deepcopy(game)
    payload.pop("compile_digest", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
