import inspect
import json
import math
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from pixelworld.adventure.compiler import compile_adventure, compile_digest
from pixelworld.adventure.director import FixtureStoryDirector, StoryDirector
from pixelworld.adventure.export import RUNTIME_CORE
from pixelworld.adventure.models import LIMITS, TEXT_LIMITS, validate_adventure_spec, validate_polygon
from pixelworld.adventure.navigation import point_in_polygon, project_to_walkboxes
from pixelworld.adventure.pipeline import generate_adventure
from pixelworld.adventure.runtime import AdventureRuntime
from pixelworld.adventure.solver import solve_game
from pixelworld.adventure.validation import validate_game


ROOT = Path(__file__).resolve().parents[1]
NODE_RUNNER = ROOT / "tests" / "js" / "adventure_core_replay.js"


class StaticDirector(StoryDirector):
    source = "test:static"

    def __init__(self, spec):
        self.spec = spec

    def create_spec(self, prompt):
        del prompt
        return deepcopy(self.spec)


def fixture_game(name="golden_lab"):
    return compile_adventure(FixtureStoryDirector(name).create_spec("prompt is ignored"))


def state_snapshot(runtime):
    return deepcopy({key: runtime.state[key] for key in ("inventory", "objects", "objectives", "flags", "completed")})


@pytest.mark.parametrize("fixture", ["golden_lab", "pirate_harbor"])
def test_python_and_node_core_replay_have_stepwise_state_parity(fixture, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    output = tmp_path / fixture
    result = generate_adventure(FixtureStoryDirector(fixture), "not used for selection", output)
    game = json.loads((output / "game.json").read_text(encoding="utf-8"))
    solution = json.loads((output / "solution.json").read_text(encoding="utf-8"))
    runtime = AdventureRuntime(game)
    expected = []
    for step in solution["solution"]:
        action_result = runtime.perform(step["action"])
        assert action_result.success
        expected.append(state_snapshot(runtime))
    expected_path = tmp_path / f"{fixture}-expected.json"
    expected_path.write_text(json.dumps(expected, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [node, str(NODE_RUNNER), str(output / "runtime-core.cjs"), str(output / "game.json"), str(output / "solution.json"), str(expected_path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    replay = json.loads(completed.stdout)
    assert replay == {"success": True, "steps": len(solution["solution"]), "completed": True}
    assert result["source"] == f"fixture:{fixture}"


def test_second_fixture_is_deterministic_distinct_and_solvable():
    lab = fixture_game("golden_lab")
    pirate = fixture_game("pirate_harbor")
    second_pirate = fixture_game("pirate_harbor")
    assert pirate["compile_digest"] == second_pirate["compile_digest"]
    assert pirate["compile_digest"] != lab["compile_digest"]
    report = validate_game(pirate)
    assert report["valid"]
    assert [step["interaction_id"] for step in report["solver"]["solution"]] == ["take_harbor_key", "unlock_harbor_box"]


def test_fixture_selection_is_explicit_not_prompt_derived():
    director = FixtureStoryDirector("pirate_harbor")
    first = director.create_spec("Professor Knallbert coolant time machine")
    second = director.create_spec("completely unrelated")
    assert first == second
    assert first["visual_theme"] == "pirate_harbor"
    with pytest.raises(ValueError, match="unknown fixture"):
        FixtureStoryDirector("prompt_guess")


def test_general_compiler_accepts_replaced_entity_ids():
    spec = FixtureStoryDirector().create_spec("")
    machine = next(item for item in spec["objects"] if item["id"] == "time_machine")
    machine["id"] = "chronotron_unit"
    professor = spec["characters"][0]
    professor["id"] = "lab_mentor"
    for interaction in spec["interactions"]:
        if interaction["target_id"] == "time_machine":
            interaction["target_id"] = "chronotron_unit"
        for operation in interaction["conditions"] + interaction["effects"]:
            operation["path"] = operation["path"].replace("objects.time_machine.", "objects.chronotron_unit.")
    for ending in spec["ending_conditions"]:
        for condition in ending["conditions"]:
            condition["path"] = condition["path"].replace("objects.time_machine.", "objects.chronotron_unit.")
    game = compile_adventure(spec)
    assert {"chronotron_unit", "lab_mentor"} <= {item["id"] for item in game["scene_graph"]["entities"]}
    assert validate_game(game)["valid"]


def test_compiler_and_generic_runtimes_have_no_golden_entity_or_dialogue_logic():
    from pixelworld.adventure import compiler, runtime

    compiler_source = inspect.getsource(compiler)
    runtime_source = inspect.getsource(runtime)
    for forbidden in ("coolant_red", "professor_knallbert", "Knallbert:"):
        assert forbidden not in compiler_source
        assert forbidden not in runtime_source
        assert forbidden not in RUNTIME_CORE
    assert 'required_exits": ["time_portal"]' not in compiler_source


@pytest.mark.parametrize(
    ("interaction_id", "mutation", "message"),
    [
        ("take_red", lambda item: item["item_ids"].append("coolant_blue"), "exactly 0"),
        ("cool_machine", lambda item: item.__setitem__("item_ids", []), "exactly 1"),
        ("mix_coolant", lambda item: item.__setitem__("item_ids", ["coolant_red"]), "exactly 2"),
        ("mix_coolant", lambda item: item.__setitem__("item_ids", ["coolant_red", "coolant_red"]), "different items"),
        ("take_red", lambda item: item.__setitem__("verb", "move_to"), "not declaratively allowed"),
    ],
)
def test_verb_item_cardinalities(interaction_id, mutation, message):
    spec = FixtureStoryDirector().create_spec("")
    interaction = next(item for item in spec["interactions"] if item["id"] == interaction_id)
    mutation(interaction)
    with pytest.raises(ValueError, match=message):
        validate_adventure_spec(spec)


def test_talk_and_look_item_cardinality():
    spec = FixtureStoryDirector().create_spec("")
    spec["interactions"].append({"id": "talk_hint", "verb": "talk_to", "target_id": "professor_knallbert", "item_ids": ["coolant_red"], "conditions": [], "effects": [], "text": "Hallo", "animation_hint": "talk"})
    with pytest.raises(ValueError, match="exactly 0"):
        validate_adventure_spec(spec)


def test_count_and_text_limits_are_enforced():
    spec = FixtureStoryDirector().create_spec("")
    spec["objects"] = [deepcopy(spec["objects"][0]) for _ in range(LIMITS["objects"] + 1)]
    with pytest.raises(ValueError, match="maximum of 16"):
        validate_adventure_spec(spec)
    spec = FixtureStoryDirector().create_spec("")
    spec["title"] = "x" * (TEXT_LIMITS["title"] + 1)
    with pytest.raises(ValueError, match="maximum length"):
        validate_adventure_spec(spec)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_spec_numbers_are_rejected(value):
    spec = FixtureStoryDirector().create_spec("")
    spec["player"]["start_position"][0] = value
    with pytest.raises(ValueError, match="finite"):
        validate_adventure_spec(spec)


def test_unknown_and_wrong_typed_state_paths_are_rejected_before_compile():
    spec = FixtureStoryDirector().create_spec("")
    spec["interactions"][0]["effects"][0]["path"] = "objects.coolant_red.unknown"
    with pytest.raises(ValueError, match="unknown state field"):
        compile_adventure(spec)
    spec = FixtureStoryDirector().create_spec("")
    spec["interactions"][0]["effects"][0]["value"] = "false"
    with pytest.raises(ValueError, match="exact JSON type boolean"):
        compile_adventure(spec)
    spec = FixtureStoryDirector().create_spec("")
    spec["flags"].append({"id": "alarm", "type": "integer", "initial": True})
    with pytest.raises(ValueError, match="exact JSON type integer"):
        compile_adventure(spec)


def test_state_schema_is_complete_and_typed():
    game = fixture_game()
    schema = game["state_schema"]
    assert schema["objects"]["time_machine"] == {"cooled": "boolean"}
    assert schema["objects"]["mixing_flask"] == {"contents": "string", "taken": "boolean"}
    assert schema["objectives"]["cool_time_machine"] == {"completed": "boolean"}
    assert schema["flags"] == {}
    assert set(schema["inventory_ids"]) == {"coolant_red", "coolant_blue", "mixing_flask", "mixed_coolant"}


def test_savegame_digest_and_deep_manipulation_are_rejected():
    game = fixture_game()
    runtime = AdventureRuntime(game)
    baseline = deepcopy(runtime.state)
    assert baseline["game_digest"] == game["compile_digest"]
    mutations = []
    extra = deepcopy(baseline); extra["objects"]["time_machine"]["injected"] = True; mutations.append(extra)
    missing = deepcopy(baseline); del missing["objects"]["time_machine"]["cooled"]; mutations.append(missing)
    wrong = deepcopy(baseline); wrong["objects"]["time_machine"]["cooled"] = "false"; mutations.append(wrong)
    flag = deepcopy(baseline); flag["flags"]["unknown"] = False; mutations.append(flag)
    nan = deepcopy(baseline); nan["player_position"][0] = math.nan; mutations.append(nan)
    infinity = deepcopy(baseline); infinity["player_position"][0] = math.inf; mutations.append(infinity)
    collision = deepcopy(baseline); collision["player_position"] = [64, 30]; mutations.append(collision)
    forged = deepcopy(baseline); forged["completed"] = True; mutations.append(forged)
    for manipulated in mutations:
        with pytest.raises(ValueError):
            AdventureRuntime(game, manipulated)
    with pytest.raises(ValueError, match="different compiled game"):
        AdventureRuntime(fixture_game("pirate_harbor"), baseline)


def test_save_load_rejects_non_finite_json_constant(tmp_path):
    game = fixture_game()
    state = deepcopy(AdventureRuntime(game).state)
    state["player_position"][0] = math.nan
    text = json.dumps(state)
    with pytest.raises(ValueError, match="non-finite JSON"):
        AdventureRuntime.load_json(game, text)


def test_late_effect_failure_rolls_back_bit_exactly_in_python_and_js(tmp_path):
    spec = FixtureStoryDirector().create_spec("")
    take = next(item for item in spec["interactions"] if item["id"] == "take_red")
    take["effects"].append({"op": "inventory_remove", "path": "inventory", "value": "coolant_blue"})
    game = compile_adventure(spec)
    runtime = AdventureRuntime(game)
    before = runtime.save_json()
    result = runtime.take("coolant_red")
    assert not result.success
    assert runtime.save_json() == before
    assert "coolant_red" not in runtime.state["inventory"]
    assert runtime.state["objects"]["coolant_red"]["taken"] is False
    node = shutil.which("node")
    if node is not None:
        core = tmp_path / "runtime-core.cjs"
        game_path = tmp_path / "game.json"
        core.write_text(RUNTIME_CORE, encoding="utf-8")
        game_path.write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")
        script = "const{Runtime}=require(process.argv[1]),g=require(process.argv[2]),r=new Runtime(g),b=JSON.stringify(r.state),z=r.perform({verb:'take',target_id:'coolant_red'});process.stdout.write(JSON.stringify({success:z.success,rollback:b===JSON.stringify(r.state)}));"
        completed = subprocess.run([node, "-e", script, str(core), str(game_path)], check=True, capture_output=True, text=True)
        assert json.loads(completed.stdout) == {"success": False, "rollback": True}


def test_invalid_navigation_edge_and_self_intersection_are_rejected():
    with pytest.raises(ValueError, match="self-intersecting"):
        validate_polygon([[0, 0], [4, 4], [0, 4], [4, 0]])
    game = fixture_game()
    game["scene_graph"]["navigation_edges"][0]["to"] = "missing_box"
    game["compile_digest"] = compile_digest(game)
    report = validate_game(game, solve=False)
    assert not report["valid"]
    assert "unknown walkbox" in report["checks"]["scene_graph"]["detail"]


def test_numeric_projection_and_boundary_semantics_match_node(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    boxes = [{"id": "z_left", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}, {"id": "a_right", "polygon": [[10, 0], [12, 0], [12, 2], [10, 2]]}]
    expected = project_to_walkboxes([6, 1], boxes)
    assert expected == (2.0, 1.0)
    assert point_in_polygon([2, 1], boxes[0]["polygon"])
    core = tmp_path / "runtime-core.cjs"
    core.write_text(RUNTIME_CORE, encoding="utf-8")
    script = "const c=require(process.argv[1]),b=JSON.parse(process.argv[2]);process.stdout.write(JSON.stringify({p:c.project([6,1],b),edge:c.pointIn([2,1],b[0].polygon)}));"
    result = subprocess.run([node, "-e", script, str(core), json.dumps(boxes)], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"p": [2, 1], "edge": True}


def test_safe_untrusted_text_stays_only_in_game_json(tmp_path):
    payload = "</script><b>unsafe-looking</b> Δ🚀"
    spec = FixtureStoryDirector("pirate_harbor").create_spec("")
    spec["title"] = payload
    spec["characters"][0]["default_talk_text"] = payload
    spec["interactions"][0]["text"] = payload
    output = tmp_path / "safe-export"
    generate_adventure(StaticDirector(spec), "", output)
    assert payload in (output / "game.json").read_text(encoding="utf-8")
    assert payload not in (output / "index.html").read_text(encoding="utf-8")
    assert payload not in (output / "runtime.js").read_text(encoding="utf-8")
    assert "innerHTML" not in (output / "runtime.js").read_text(encoding="utf-8")


def test_pipeline_does_not_leave_partial_or_overwrite_existing_output(tmp_path):
    invalid = FixtureStoryDirector().create_spec("")
    invalid["visual_theme"] = "unknown_theme"
    target = tmp_path / "failed"
    with pytest.raises(ValueError):
        generate_adventure(StaticDirector(invalid), "", target)
    assert not target.exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="will not be overwritten"):
        generate_adventure(FixtureStoryDirector(), "", existing)
    assert marker.read_text(encoding="utf-8") == "keep"
