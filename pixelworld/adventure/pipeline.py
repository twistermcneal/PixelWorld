"""High-level generation pipeline for PixelWorld 0.6.3 adventures."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .compiler import compile_adventure
from .director import StoryDirector
from .export import export_browser
from .ontology import ThemeOntology
from .solver import solve_game
from .validation import require_valid_game


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generate_adventure(director: StoryDirector, prompt: str, output: str | Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists and will not be overwritten: {output}")
    spec = director.create_spec(prompt)
    game = compile_adventure(spec, ThemeOntology())
    validation = require_valid_game(game)
    solution = validation["solver"] or solve_game(game)
    provenance = director.provenance(game["compile_digest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_json(temporary / "adventure_spec.json", game["adventure"])
        write_json(temporary / "room_spec.json", game["room"])
        write_json(temporary / "scene_graph.json", game["scene_graph"])
        write_json(temporary / "game.json", game)
        write_json(temporary / "validation_report.json", validation)
        write_json(temporary / "solution.json", solution)
        if provenance is not None:
            write_json(temporary / "director_provenance.json", provenance)
        browser_files = export_browser(game, temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output.resolve()),
        "compile_digest": game["compile_digest"],
        "valid": validation["valid"],
        "solvable": solution["solvable"],
        "solution_length": solution["shortest_solution_length"],
        "files": ["adventure_spec.json", "room_spec.json", "scene_graph.json", "game.json", "validation_report.json", "solution.json"] + (["director_provenance.json"] if provenance is not None else []) + browser_files,
        "source": director.source,
    }
