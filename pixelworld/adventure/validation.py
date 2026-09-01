"""Cross-layer validation for compiled PixelWorld adventures."""

from __future__ import annotations

from .compiler import compile_digest
from .models import SCHEMA_VERSION, validate_adventure_spec, validate_polygon
from .navigation import point_in_polygon, point_walkable, polygons_overlap, shortest_route
from .ontology import ThemeOntology
from .solver import solve_game


def validate_game(game: dict, *, solve: bool = True, max_states: int = 1000) -> dict:
    errors = []
    checks = {}

    def check(name, callback):
        try:
            detail = callback()
            checks[name] = {"passed": True, "detail": detail or "ok"}
        except (ValueError, TypeError, KeyError) as error:
            checks[name] = {"passed": False, "detail": str(error)}
            errors.append(f"{name}: {error}")

    check("adventure_schema", lambda: validate_adventure_spec(game["adventure"], ThemeOntology()) and "strict schema and references valid")
    check("version_contract", lambda: _require(game.get("schema_version") == SCHEMA_VERSION, "unsupported game schema_version"))
    check("compile_digest", lambda: _require(game.get("compile_digest") == compile_digest(game), "compile digest mismatch"))
    check("room_contract", lambda: _validate_room(game["room"]))
    check("scene_graph", lambda: _validate_scene(game["room"], game["scene_graph"]))
    solver_report = None
    if solve and not errors:
        try:
            solver_report = solve_game(game, max_states=max_states)
            _require(solver_report["solvable"], f"adventure is unsolvable within {max_states} states")
            checks["puzzle_solvable"] = {"passed": True, "detail": f"shortest solution has {solver_report['shortest_solution_length']} steps"}
        except (ValueError, TypeError, KeyError) as error:
            checks["puzzle_solvable"] = {"passed": False, "detail": str(error)}
            errors.append(f"puzzle_solvable: {error}")
    return {"valid": not errors, "schema_version": SCHEMA_VERSION, "checks": checks, "errors": errors, "solver": solver_report}


def require_valid_game(game: dict, *, solve=True, max_states=1000):
    report = validate_game(game, solve=solve, max_states=max_states)
    if not report["valid"]:
        raise ValueError("game validation failed: " + "; ".join(report["errors"]))
    return report


def _require(condition, message):
    if not condition:
        raise ValueError(message)
    return "ok"


def _validate_room(room):
    required = {"schema_version", "room_id", "name", "description", "theme", "size", "mood", "required_entities", "required_exits", "player_start", "objective_ids"}
    _require(set(room) == required, f"RoomSpec fields differ: missing={sorted(required - room.keys())}, unknown={sorted(room.keys() - required)}")
    _require(room["schema_version"] == SCHEMA_VERSION, "unsupported RoomSpec schema_version")
    _require(room["size"] == [128, 72], "Phase 1 room must be 128x72")
    _require(bool(room["required_exits"]), "RoomSpec requires an exit")
    _require(bool(room["objective_ids"]), "RoomSpec requires an objective")
    return "strict RoomSpec valid"


def _validate_scene(room, scene):
    required = {"schema_version", "room_id", "size", "theme", "palette", "background_layers", "semantic_regions", "walkboxes", "navigation_edges", "collision_polygons", "occlusion_polygons", "entities", "hotspots", "walk_to_points", "z_layers", "portals", "initial_state"}
    _require(set(scene) == required, f"Scene Graph fields differ: missing={sorted(required - scene.keys())}, unknown={sorted(scene.keys() - required)}")
    _require(scene["schema_version"] == SCHEMA_VERSION, "unsupported Scene Graph schema_version")
    _require(scene["room_id"] == room["room_id"] and scene["size"] == room["size"], "Scene Graph does not match RoomSpec")
    width, height = scene["size"]
    walkboxes = scene["walkboxes"]
    collisions = scene["collision_polygons"]
    for index, walkbox in enumerate(walkboxes):
        validate_polygon(walkbox["polygon"], f"walkboxes[{index}].polygon", convex=True)
        _polygon_in_world(walkbox["polygon"], width, height, f"walkbox {walkbox['id']!r}")
    for index, collision in enumerate(collisions):
        validate_polygon(collision["polygon"], f"collision_polygons[{index}].polygon")
        _polygon_in_world(collision["polygon"], width, height, f"collision {collision['id']!r}")
        for other in collisions[:index]:
            _require(not polygons_overlap(collision["polygon"], other["polygon"]), f"collision polygons {collision['id']!r} and {other['id']!r} overlap")
    entity_ids = {item["id"] for item in scene["entities"]}
    _require(len(entity_ids) == len(scene["entities"]), "Scene Graph entity ids are not unique")
    missing = set(room["required_entities"]) - entity_ids
    _require(not missing, f"required entities are not placed: {', '.join(sorted(missing))}")
    hotspot_entities = set()
    for hotspot in scene["hotspots"]:
        _require(hotspot["entity_id"] in entity_ids, f"hotspot {hotspot['id']!r} references unknown entity")
        _polygon_in_world(validate_polygon(hotspot["polygon"], f"hotspot {hotspot['id']!r}"), width, height, f"hotspot {hotspot['id']!r}")
        hotspot_entities.add(hotspot["entity_id"])
    start = room["player_start"]
    collision_data = [item["polygon"] for item in collisions]
    _require(point_walkable(start, walkboxes, collision_data), "player start is not walkable")
    reachable = []
    for entity in scene["entities"]:
        point = entity["walk_to_point"]
        _require(point_walkable(point, walkboxes, collision_data), f"walk-to point for {entity['id']!r} is not walkable")
        if entity["required"]:
            _require(entity["id"] in hotspot_entities, f"required entity {entity['id']!r} has no hotspot")
            shortest_route(start, point, walkboxes, scene["navigation_edges"], collisions)
            reachable.append(entity["id"])
    portal_ids = {item["id"] for item in scene["portals"]}
    _require(set(room["required_exits"]) <= portal_ids, "required portal is missing")
    _require(bool(scene["portals"]), "success portal is not defined")
    return f"{len(reachable)} required hotspots and {len(scene['portals'])} portal reachable"


def _polygon_in_world(polygon, width, height, label):
    _require(all(0 <= point[0] <= width and 0 <= point[1] <= height for point in polygon), f"{label} lies outside the world")
