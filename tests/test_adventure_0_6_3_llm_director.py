import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from pixelworld.adventure.director import (
    FixtureStoryDirector,
    JsonStoryDirector,
    OpenAICompatibleConfig,
    OpenAICompatibleStoryDirector,
    decode_single_json_object,
)
from pixelworld.adventure.pipeline import generate_adventure
from pixelworld.adventure.runtime import AdventureRuntime
from pixelworld.adventure.structured_schema import PHASE2_THEMES, adventure_spec_json_schema, build_system_prompt
from pixelworld.adventure.transport import ResponseTooLarge, StoryDirectorTransport, TransportError, TransportRedirect, TransportRequest, TransportResponse, TransportTimeout, responses_endpoint, validate_base_url


ROOT = Path(__file__).resolve().parents[1]
NODE_RUNNER = ROOT / "tests" / "js" / "adventure_core_replay.js"
SECRET = "sk-test-NEVER-STORE-THIS"


class FakeTransport(StoryDirectorTransport):
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra transport call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        envelope = json.dumps({"output_text": response}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return TransportResponse(200, envelope)


def condition(op, path, value):
    return {"op": op, "path": path, "value": value}


def effect(op, path, value):
    return {"op": op, "path": path, "value": value}


def synthetic_lab_spec():
    spec = FixtureStoryDirector("golden_lab").create_spec("")
    location = spec["locations"][0]
    location.update({"id": "midnight_workshop", "name": "Lyras Mitternachtswerkstatt", "description": "Ein neues Labor mit einer flackernden Mitternachtsmaschine."})
    spec.update({"title": "Lyras Mitternachtswerkstatt", "premise": "Erfinderin Lyra muss eine Sicherung einsetzen, bevor die Mitternachtsmaschine ausfällt.", "tone": "quirlig und gespannt"})
    spec["player"].update({"id": "helper_milo", "name": "Milo", "location_id": "midnight_workshop"})
    spec["characters"] = [{"id": "inventor_lyra", "name": "Erfinderin Lyra", "archetype": "mad_scientist", "role": "npc", "preferred_zone": "right_npc", "location_id": "midnight_workshop", "description": "Eine junge Erfinderin mit einer Uhr voller Schrauben.", "default_talk_text": "Die Keramiksicherung gehört zuerst ins Phasenpult!", "initial_state": {}}]
    by_class = {item["class"]: item for item in spec["objects"]}
    machine = deepcopy(by_class["time_machine"]); machine.update({"id": "midnight_engine", "name": "Mitternachtsmaschine", "location_id": "midnight_workshop", "initial_state": {"stable": False}})
    console = deepcopy(by_class["control_console"]); console.update({"id": "phase_console", "name": "Phasenpult", "location_id": "midnight_workshop", "initial_state": {"repaired": False}})
    fuse = deepcopy(next(item for item in spec["objects"] if item["class"] == "chemical_bottle")); fuse.update({"id": "ceramic_fuse", "name": "Keramiksicherung", "location_id": "midnight_workshop", "description": "Eine violette Sicherung in einem Glasgehäuse.", "initial_state": {"taken": False}})
    portal = deepcopy(by_class["time_portal"]); portal.update({"id": "roof_rift", "name": "Dachspalt", "location_id": "midnight_workshop", "initial_state": {"active": False}})
    spec["objects"] = [machine, console, fuse, portal]
    spec["inventory_items"] = [{"id": "ceramic_fuse", "name": "Keramiksicherung", "description": "Passt in das Phasenpult."}]
    spec["objectives"] = [{"id": "stabilize_midnight_engine", "description": "Sicherung einsetzen und Maschine stabilisieren.", "required": True, "initial_state": {"completed": False}}]
    spec["interactions"] = [
        {"id": "collect_fuse", "verb": "take", "target_id": "ceramic_fuse", "item_ids": [], "conditions": [condition("equals", "objects.ceramic_fuse.taken", False)], "effects": [effect("set", "objects.ceramic_fuse.taken", True), effect("inventory_add", "inventory", "ceramic_fuse")], "text": "Milo nimmt die Keramiksicherung.", "animation_hint": "pickup"},
        {"id": "repair_phase_console", "verb": "use", "target_id": "phase_console", "item_ids": ["ceramic_fuse"], "conditions": [condition("inventory_contains", "inventory", "ceramic_fuse"), condition("equals", "objects.phase_console.repaired", False)], "effects": [effect("inventory_remove", "inventory", "ceramic_fuse"), effect("set", "objects.phase_console.repaired", True), effect("set", "objects.midnight_engine.stable", True), effect("set", "objects.roof_rift.active", True), effect("set", "objectives.stabilize_midnight_engine.completed", True)], "text": "Das Pult summt, die Maschine läuft stabil und der Dachspalt öffnet sich.", "animation_hint": "repair"},
    ]
    spec["puzzles"] = [{"id": "fuse_puzzle", "description": "Sicherung finden und einsetzen.", "objective_ids": ["stabilize_midnight_engine"], "interaction_ids": ["collect_fuse", "repair_phase_console"]}]
    spec["ending_conditions"] = [{"id": "midnight_safe", "description": "Maschine und Ausgang sind stabil.", "conditions": [condition("equals", "objects.midnight_engine.stable", True), condition("equals", "objects.roof_rift.active", True), condition("equals", "objectives.stabilize_midnight_engine.completed", True)]}]
    return spec


def synthetic_pirate_spec():
    spec = FixtureStoryDirector("pirate_harbor").create_spec("")
    location = spec["locations"][0]
    location.update({"id": "storm_pier", "name": "Sturmpier", "description": "Ein windiger Pier mit einem blockierten Fluchtboot."})
    spec.update({"title": "Die Takelage vom Sturmpier", "premise": "Bootsjunge Tavi muss Haken und Griff am Spill befestigen, bevor der Sturm kommt.", "tone": "stürmisch und heiter"})
    spec["player"].update({"id": "cabin_runner", "name": "Tavi", "location_id": "storm_pier"})
    spec["characters"] = [{"id": "quartermaster_ren", "name": "Quartiermeister Ren", "archetype": "pirate", "role": "npc", "preferred_zone": "harbor_npc", "location_id": "storm_pier", "description": "Ein ruhiger Quartiermeister mit wetterfestem Mantel.", "default_talk_text": "Haken und Griff müssen gemeinsam ans Spill.", "initial_state": {}}]
    base_key = next(item for item in spec["objects"] if item["portable"])
    hook = deepcopy(base_key); hook.update({"id": "iron_hook", "name": "Eisenhaken", "location_id": "storm_pier", "preferred_zone": "dock_left", "initial_state": {"taken": False}})
    handle = deepcopy(base_key); handle.update({"id": "wood_handle", "name": "Holzgriff", "location_id": "storm_pier", "preferred_zone": "dock_left_wall", "initial_state": {"taken": False}})
    capstan = deepcopy(next(item for item in spec["objects"] if item["class"] == "locked_chest")); capstan.update({"id": "jammed_capstan", "name": "Klemmendes Spill", "location_id": "storm_pier", "initial_state": {"rigged": False}})
    exit_obj = deepcopy(next(item for item in spec["objects"] if item["role"] == "exit")); exit_obj.update({"id": "storm_skiff", "name": "Sturmboot", "location_id": "storm_pier", "initial_state": {"ready": False}})
    spec["objects"] = [hook, handle, capstan, exit_obj]
    spec["inventory_items"] = [{"id": "iron_hook", "name": "Eisenhaken", "description": "Teil einer improvisierten Kurbel."}, {"id": "wood_handle", "name": "Holzgriff", "description": "Der zweite Teil der Kurbel."}]
    spec["objectives"] = [{"id": "ready_storm_skiff", "description": "Das Spill reparieren und das Boot bereitmachen.", "required": True, "initial_state": {"completed": False}}]
    spec["interactions"] = [
        {"id": "take_iron_hook", "verb": "take", "target_id": "iron_hook", "item_ids": [], "conditions": [condition("equals", "objects.iron_hook.taken", False)], "effects": [effect("set", "objects.iron_hook.taken", True), effect("inventory_add", "inventory", "iron_hook")], "text": "Tavi steckt den Haken ein.", "animation_hint": "pickup"},
        {"id": "take_wood_handle", "verb": "take", "target_id": "wood_handle", "item_ids": [], "conditions": [condition("equals", "objects.wood_handle.taken", False)], "effects": [effect("set", "objects.wood_handle.taken", True), effect("inventory_add", "inventory", "wood_handle")], "text": "Der Holzgriff kommt ins Inventar.", "animation_hint": "pickup"},
        {"id": "rig_capstan", "verb": "use", "target_id": "jammed_capstan", "item_ids": ["iron_hook"], "conditions": [condition("inventory_contains", "inventory", "iron_hook"), condition("inventory_contains", "inventory", "wood_handle"), condition("equals", "objects.jammed_capstan.rigged", False)], "effects": [effect("inventory_remove", "inventory", "iron_hook"), effect("inventory_remove", "inventory", "wood_handle"), effect("set", "objects.jammed_capstan.rigged", True), effect("set", "objects.storm_skiff.ready", True), effect("set", "objectives.ready_storm_skiff.completed", True)], "text": "Haken und Griff bewegen das Spill; das Sturmboot ist bereit.", "animation_hint": "repair"},
    ]
    spec["puzzles"] = [{"id": "capstan_puzzle", "description": "Zwei Teile sammeln und das Spill reparieren.", "objective_ids": ["ready_storm_skiff"], "interaction_ids": ["take_iron_hook", "take_wood_handle", "rig_capstan"]}]
    spec["ending_conditions"] = [{"id": "skiff_ready", "description": "Das Sturmboot ist startklar.", "conditions": [condition("equals", "objects.storm_skiff.ready", True), condition("equals", "objectives.ready_storm_skiff.completed", True)]}]
    return spec


def raw_spec(spec):
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def director_with(responses, **config_overrides):
    values = {"base_url": "https://llm.example.test/v1", "api_key": SECRET, "model": "example-model-1"}
    values.update(config_overrides)
    transport = FakeTransport(responses)
    return OpenAICompatibleStoryDirector(OpenAICompatibleConfig(**values), transport), transport


def test_machine_schema_prompt_and_request_contract():
    schema = adventure_spec_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["visual_theme"]["enum"] == list(PHASE2_THEMES)
    prompt = build_system_prompt()
    assert "Story Director, not the game engine" in prompt
    assert "Knallbert" not in prompt
    director, transport = director_with([raw_spec(synthetic_lab_spec())])
    director.create_spec("Eine freie deutsche Storyidee")
    request = transport.requests[0]
    body = json.loads(request.body)
    assert request.url == "https://llm.example.test/v1/responses"
    assert body["model"] == "example-model-1"
    assert body["store"] is False and body["stream"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"] == schema
    assert SECRET not in request.body.decode("utf-8")


def test_valid_first_response_has_no_repair_and_provenance(tmp_path):
    spec = synthetic_lab_spec()
    director, transport = director_with([raw_spec(spec)])
    output = tmp_path / "game"
    result = generate_adventure(director, "repair a midnight machine", output)
    assert len(transport.requests) == 1
    provenance = json.loads((output / "director_provenance.json").read_text(encoding="utf-8"))
    assert provenance["attempt_count"] == 1
    assert provenance["attempt_validation"] == [{"attempt": 1, "errors": [], "valid": True}]
    assert provenance["prompt_sha256"] == hashlib.sha256(b"repair a midnight machine").hexdigest()
    assert provenance["response_sha256"] == [hashlib.sha256(raw_spec(spec).encode()).hexdigest()]
    assert provenance["compile_digest"] == result["compile_digest"]
    assert provenance["provider_protocol"] == "openai-responses-v1"


def test_invalid_response_gets_one_bounded_repair():
    valid = raw_spec(synthetic_lab_spec())
    director, transport = director_with(["{}", valid])
    assert director.create_spec("idea")["title"] == "Lyras Mitternachtswerkstatt"
    assert len(transport.requests) == 2
    repair = json.loads(transport.requests[1].body)["input"][0]["content"][0]["text"]
    repair_data = json.loads(repair)
    assert repair_data["previous_response"] == "{}"
    assert 1 <= len(repair_data["validation_errors"]) <= 8
    assert all(len(item) <= 240 for item in repair_data["validation_errors"])


def test_two_invalid_responses_abort_and_leave_no_output(tmp_path):
    director, transport = director_with(["{}", "{}"])
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="after 2 attempts") as failure:
        generate_adventure(director, "idea", target)
    assert len(transport.requests) == 2
    assert not target.exists()
    assert SECRET not in str(failure.value)


@pytest.mark.parametrize("raw", ["```json\n{}\n```", "Here is JSON: {}", "{} trailing", "{}{}", '{"value":NaN}', "[]"])
def test_strict_json_rejects_markdown_prose_multiple_documents_and_non_objects(raw):
    with pytest.raises(ValueError):
        decode_single_json_object(raw)
    director, transport = director_with([raw, raw])
    with pytest.raises(ValueError):
        director.create_spec("idea")
    assert len(transport.requests) == 2


def test_model_json_depth_is_bounded():
    value = {}
    for index in range(24):
        value = {f"level_{index}": value}
    with pytest.raises(ValueError, match="nesting-depth"):
        decode_single_json_object(json.dumps(value))


def forest_without_template_spec():
    return {
        "schema_version": "0.6.3", "title": "Waldtor", "premise": "Ein Wächter öffnet ein Tor.", "tone": "ruhig", "visual_theme": "forest_ruin",
        "player": {"id": "walker", "name": "Ari", "location_id": "grove", "start_position": [15, 60]},
        "characters": [{"id": "moss_guard", "name": "Mooswächter", "archetype": "guardian", "role": "npc", "preferred_zone": "forest_edge", "location_id": "grove", "description": "Ein stiller Wächter.", "default_talk_text": "Berühre den Altar.", "initial_state": {}}],
        "locations": [{"id": "grove", "name": "Hain", "description": "Ein alter Hain.", "theme": "forest_ruin", "size": [128, 72], "mood": "dappled"}],
        "objects": [{"id": "stone_gate", "name": "Steintor", "class": "altar", "role": "exit", "preferred_zone": "ruin_center", "location_id": "grove", "description": "Ein steinernes Tor.", "portable": False, "required": True, "portal_destination": "ending", "initial_state": {"active": False}}],
        "inventory_items": [], "objectives": [{"id": "open_gate", "description": "Tor öffnen.", "required": True, "initial_state": {"completed": False}}],
        "puzzles": [{"id": "altar_puzzle", "description": "Altar berühren.", "objective_ids": ["open_gate"], "interaction_ids": ["touch_altar"]}],
        "interactions": [{"id": "touch_altar", "verb": "look_at", "target_id": "stone_gate", "item_ids": [], "conditions": [condition("equals", "objects.stone_gate.active", False)], "effects": [effect("set", "objects.stone_gate.active", True), effect("set", "objectives.open_gate.completed", True)], "text": "Das Tor öffnet sich.", "animation_hint": "activate"}],
        "ending_conditions": [{"id": "gate_open", "description": "Tor offen.", "conditions": [condition("equals", "objects.stone_gate.active", True)]}], "flags": [],
    }


@pytest.mark.parametrize("spec", [lambda: {**synthetic_lab_spec(), "visual_theme": "unknown_theme"}, forest_without_template_spec])
def test_unknown_or_uncompiled_theme_fails_after_at_most_one_repair(spec):
    raw = raw_spec(spec())
    director, transport = director_with([raw, raw])
    with pytest.raises(ValueError):
        director.create_spec("idea")
    assert len(transport.requests) == 2


def test_unsolvable_puzzle_is_repaired_then_aborted():
    spec = synthetic_lab_spec()
    final = spec["interactions"][-1]["effects"][-1]
    final["value"] = False
    raw = raw_spec(spec)
    director, transport = director_with([raw, raw])
    with pytest.raises(ValueError, match="unsolvable"):
        director.create_spec("idea")
    assert len(transport.requests) == 2


@pytest.mark.parametrize("error", [TransportTimeout("timed out"), TransportError("HTTP 500"), ResponseTooLarge("too large"), TransportRedirect("HTTP 302")])
def test_transport_failures_abort_without_repair_or_secret(error):
    director, transport = director_with([error])
    with pytest.raises(type(error)) as failure:
        director.create_spec("idea")
    assert len(transport.requests) == 1
    assert SECRET not in str(failure.value)


def test_model_output_size_is_bounded():
    director, transport = director_with(["{" + (" " * (128 * 1024)) + "}"])
    with pytest.raises(ValueError, match="response limit"):
        director.create_spec("idea")
    assert len(transport.requests) == 1


@pytest.mark.parametrize("url", ["ftp://host/v1", "https:///v1", "https://user:pass@host/v1", "https://host/v1?q=x", "https://host/v1#x"])
def test_invalid_base_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_missing_model_and_api_key_are_rejected():
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleConfig("https://host/v1", "", "model").validate()
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleConfig("https://host/v1", SECRET, "").validate()


def test_api_key_never_appears_in_exception_or_artifacts(tmp_path):
    director, _ = director_with([raw_spec(synthetic_pirate_spec())])
    output = tmp_path / "pirate"
    generate_adventure(director, "storm story", output)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    assert SECRET not in combined
    provenance = json.loads((output / "director_provenance.json").read_text(encoding="utf-8"))
    assert "api_key" not in json.dumps(provenance).lower()


def test_fixture_and_json_directors_never_construct_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("transport must not be used")

    monkeypatch.setattr("pixelworld.adventure.transport.HTTPTransport.send", forbidden)
    generate_adventure(FixtureStoryDirector("pirate_harbor"), "", tmp_path / "fixture")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(raw_spec(synthetic_lab_spec()), encoding="utf-8")
    generate_adventure(JsonStoryDirector(spec_path), "", tmp_path / "json")
    assert not (tmp_path / "fixture" / "director_provenance.json").exists()
    assert not (tmp_path / "json" / "director_provenance.json").exists()


@pytest.mark.parametrize(("name", "factory", "steps"), [("synthetic_lab", synthetic_lab_spec, 2), ("synthetic_pirate", synthetic_pirate_spec, 3)])
def test_synthetic_model_response_full_python_node_and_browser_export(name, factory, steps, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    director, transport = director_with([raw_spec(factory())])
    output = tmp_path / name
    result = generate_adventure(director, "new story idea", output)
    assert result["valid"] and result["solvable"] and result["solution_length"] == steps
    assert len(transport.requests) == 1
    game = json.loads((output / "game.json").read_text(encoding="utf-8"))
    solution = json.loads((output / "solution.json").read_text(encoding="utf-8"))
    runtime = AdventureRuntime(game)
    expected = []
    for step in solution["solution"]:
        assert runtime.perform(step["action"]).success
        expected.append(deepcopy({key: runtime.state[key] for key in ("inventory", "objects", "objectives", "flags", "completed")}))
    expected_path = tmp_path / f"{name}-expected.json"
    expected_path.write_text(json.dumps(expected, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    replay = subprocess.run([node, str(NODE_RUNNER), str(output / "runtime-core.cjs"), str(output / "game.json"), str(output / "solution.json"), str(expected_path)], check=True, capture_output=True, text=True, encoding="utf-8")
    assert json.loads(replay.stdout) == {"success": True, "steps": steps, "completed": True}
    assert (output / "index.html").is_file() and (output / "runtime.js").is_file()
