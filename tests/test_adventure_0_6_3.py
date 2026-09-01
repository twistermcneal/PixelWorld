import json
from copy import deepcopy

import pytest

from pixelworld import cli
from pixelworld.adventure.compiler import compile_adventure, compile_digest
from pixelworld.adventure.director import FixtureStoryDirector, JsonStoryDirector
from pixelworld.adventure.export import RUNTIME_TEMPLATE
from pixelworld.adventure.models import validate_adventure_spec, validate_polygon
from pixelworld.adventure.navigation import (
    point_in_polygon,
    point_walkable,
    project_to_walkboxes,
    segment_walkable,
    shortest_route,
)
from pixelworld.adventure.ontology import THEMES, ThemeOntology
from pixelworld.adventure.pipeline import generate_adventure
from pixelworld.adventure.runtime import AdventureRuntime
from pixelworld.adventure.solver import solve_game
from pixelworld.adventure.validation import require_valid_game, validate_game


@pytest.fixture
def spec():
    return FixtureStoryDirector().create_spec("ignored fixture prompt")


@pytest.fixture
def game(spec):
    return compile_adventure(spec)


def test_adventure_spec_schema_and_fixture_source(spec):
    assert validate_adventure_spec(spec, ThemeOntology())["schema_version"] == "0.6.3"
    assert spec["title"] == "Professor Knallberts chronochemisches Labor"
    assert len(spec["locations"]) == 1


def test_unknown_top_level_and_nested_fields_are_rejected(spec):
    invalid = deepcopy(spec)
    invalid["python"] = "print('no')"
    with pytest.raises(ValueError, match="unknown fields: python"):
        validate_adventure_spec(invalid)
    invalid = deepcopy(spec)
    invalid["player"]["script"] = "run"
    with pytest.raises(ValueError, match="unknown fields: script"):
        validate_adventure_spec(invalid)


def test_duplicate_ids_are_rejected(spec):
    invalid = deepcopy(spec)
    invalid["objects"].append(deepcopy(invalid["objects"][0]))
    with pytest.raises(ValueError, match="duplicate id 'time_machine'"):
        validate_adventure_spec(invalid)


def test_invalid_references_are_rejected(spec):
    invalid = deepcopy(spec)
    invalid["characters"][0]["location_id"] = "missing_room"
    with pytest.raises(ValueError, match="unknown id 'missing_room'"):
        validate_adventure_spec(invalid)


def test_unknown_theme_and_incompatible_object_are_rejected(spec):
    invalid = deepcopy(spec)
    invalid["visual_theme"] = "imaginary_fallback"
    invalid["locations"][0]["theme"] = "imaginary_fallback"
    with pytest.raises(ValueError, match="unknown visual theme"):
        validate_adventure_spec(invalid, ThemeOntology())
    invalid = deepcopy(spec)
    invalid["objects"][0]["class"] = "pirate_ship"
    with pytest.raises(ValueError, match="incompatible with theme"):
        validate_adventure_spec(invalid, ThemeOntology())


def test_theme_ontology_has_required_versioned_themes():
    assert set(THEMES) == {"mad_scientist_lab", "pirate_harbor", "forest_ruin", "spaceship", "medieval_village"}
    assert ThemeOntology().as_document()["schema_version"] == "0.6.3"
    assert "time_machine" in ThemeOntology().get("mad_scientist_lab")["object_classes"]


def test_polygon_validation_and_point_in_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert validate_polygon(square, convex=True) == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert point_in_polygon([5, 5], square)
    assert point_in_polygon([0, 5], square)
    assert not point_in_polygon([11, 5], square)
    with pytest.raises(ValueError, match="convex"):
        validate_polygon([[0, 0], [4, 0], [2, 1], [4, 4], [0, 4]], convex=True)


def test_projection_and_tie_break_are_deterministic():
    boxes = [
        {"id": "z_box", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        {"id": "a_box", "polygon": [[4, 0], [6, 0], [6, 2], [4, 2]]},
    ]
    assert project_to_walkboxes([3, 1], boxes) == (2.0, 1.0)
    assert project_to_walkboxes([3, 1], list(reversed(boxes))) == (2.0, 1.0)


def test_shortest_path_is_deterministic_and_avoids_machine(game):
    scene = game["scene_graph"]
    first = shortest_route([15, 60], [115, 49], scene["walkboxes"], scene["navigation_edges"], scene["collision_polygons"])
    second = shortest_route([15, 60], [115, 49], scene["walkboxes"], list(reversed(scene["navigation_edges"])), scene["collision_polygons"])
    assert first == second
    collision_data = [item["polygon"] for item in scene["collision_polygons"]]
    assert all(segment_walkable(a, b, scene["walkboxes"], collision_data) for a, b in zip(first, first[1:]))
    assert all(point_walkable(point, scene["walkboxes"], collision_data) for point in first)


def test_all_required_hotspots_and_portal_are_reachable(game):
    report = require_valid_game(game)
    assert report["valid"]
    assert "required hotspots" in report["checks"]["scene_graph"]["detail"]
    assert report["checks"]["puzzle_solvable"]["passed"]


def test_runtime_inventory_and_invalid_verbs(game):
    runtime = AdventureRuntime(game)
    result = runtime.take("coolant_red")
    assert result.success and "coolant_red" in runtime.state["inventory"]
    assert result.state_changes and result.animation_hint == "pickup"
    assert not runtime.take("coolant_red").success
    assert not runtime.perform({"verb": "execute_python", "target_id": "time_machine"}).success


def test_combine_and_use_puzzle(game):
    runtime = AdventureRuntime(game)
    assert not runtime.combine("coolant_red", "coolant_blue", "mixing_flask").success
    for target in ("coolant_red", "coolant_blue", "mixing_flask"):
        assert runtime.take(target).success
    assert runtime.combine("coolant_red", "coolant_blue", "mixing_flask").success
    assert runtime.state["objects"]["mixing_flask"]["contents"] == "mixed_coolant"
    assert runtime.use("mixed_coolant", "time_machine").success
    assert runtime.completed
    assert runtime.state["objects"]["time_portal"]["active"]


def test_save_load_parity_and_save_schema_rejects_unknown_fields(game, tmp_path):
    runtime = AdventureRuntime(game)
    runtime.take("coolant_blue")
    save = tmp_path / "save.json"
    runtime.save_json(save)
    loaded = AdventureRuntime.load_json(game, save)
    assert loaded.state == runtime.state
    invalid = deepcopy(runtime.state)
    invalid["code"] = "danger"
    with pytest.raises(ValueError, match="unknown fields"):
        AdventureRuntime(game, invalid)


def test_solver_finds_complete_shortest_solution(game):
    result = solve_game(game)
    assert result["solvable"]
    assert result["shortest_solution_length"] == 5
    ids = [step["interaction_id"] for step in result["solution"]]
    assert set(ids[:3]) == {"take_red", "take_blue", "take_flask"}
    assert ids[-2:] == ["mix_coolant", "cool_machine"]
    replay = AdventureRuntime(game)
    for step in result["solution"]:
        assert replay.perform(step["action"]).success
    assert replay.completed


def test_unsolvable_adventure_is_rejected(spec):
    invalid = deepcopy(spec)
    effect = next(item for item in invalid["interactions"] if item["id"] == "cool_machine")["effects"]
    next(item for item in effect if item["path"] == "objectives.cool_time_machine.completed")["value"] = False
    game = compile_adventure(invalid)
    report = validate_game(game)
    assert not report["valid"]
    assert "unsolvable" in report["checks"]["puzzle_solvable"]["detail"]
    with pytest.raises(ValueError, match="unsolvable"):
        require_valid_game(game)


def test_compile_digest_is_deterministic(spec):
    first, second = compile_adventure(spec), compile_adventure(deepcopy(spec))
    assert first["compile_digest"] == second["compile_digest"]
    assert first["compile_digest"] == compile_digest(first)


def test_json_story_director_and_browser_export(spec, tmp_path):
    spec_path = tmp_path / "input.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    result = generate_adventure(JsonStoryDirector(spec_path), "ignored", tmp_path / "game")
    assert result["source"] == "json"
    assert result["valid"] and result["solvable"]
    for name in ("adventure_spec.json", "room_spec.json", "scene_graph.json", "game.json", "validation_report.json", "solution.json", "index.html", "runtime.js", "styles.css", "assets/lab-placeholder.svg"):
        assert (tmp_path / "game" / name).is_file()


def test_browser_runtime_has_no_golden_room_special_case_logic():
    assert "target_id===\"time_machine\"" not in RUNTIME_TEMPLATE
    assert "selected===\"coolant_red\"" not in RUNTIME_TEMPLATE
    assert "interaction.effects.forEach(effect)" in RUNTIME_TEMPLATE
    assert "GAME.runtime_rules" in RUNTIME_TEMPLATE


def test_cli_generation_validation_and_solver(tmp_path, capsys):
    output = tmp_path / "cli-game"
    assert cli.main(["adventure-generate", "--version", "0.6.3", "--director", "fixture", "--output", str(output)]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["valid"] and generated["solution_length"] == 5
    assert cli.main(["adventure-validate", "--spec", str(output / "adventure_spec.json")]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert cli.main(["adventure-solve", "--game", str(output / "game.json")]) == 0
    assert json.loads(capsys.readouterr().out)["solvable"]


def test_existing_version_contracts_remain_available():
    from pixelworld.config import VERSION
    from pixelworld.versions.v0_6_2.config import VERSION as VERSION_062

    assert VERSION == "0.6.1"
    assert VERSION_062 == "0.6.2"

