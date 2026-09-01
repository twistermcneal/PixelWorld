"""High-level generation pipeline for PixelWorld 0.6.3 adventures."""

from __future__ import annotations

import json
from pathlib import Path

from .compiler import compile_adventure
from .director import StoryDirector
from .export import export_browser
from .ontology import ThemeOntology
from .solver import solve_game
from .validation import require_valid_game


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def generate_adventure(director: StoryDirector, prompt: str, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = director.create_spec(prompt)
    game = compile_adventure(spec, ThemeOntology())
    validation = require_valid_game(game)
    solution = validation["solver"] or solve_game(game)
    write_json(output / "adventure_spec.json", game["adventure"])
    write_json(output / "room_spec.json", game["room"])
    write_json(output / "scene_graph.json", game["scene_graph"])
    write_json(output / "game.json", game)
    write_json(output / "validation_report.json", validation)
    write_json(output / "solution.json", solution)
    browser_files = export_browser(game, output)
    return {
        "output": str(output.resolve()),
        "compile_digest": game["compile_digest"],
        "valid": validation["valid"],
        "solvable": solution["solvable"],
        "solution_length": solution["shortest_solution_length"],
        "files": ["adventure_spec.json", "room_spec.json", "scene_graph.json", "game.json", "validation_report.json", "solution.json"] + browser_files,
        "source": "fixture" if director.__class__.__name__ == "FixtureStoryDirector" else "json",
    }

