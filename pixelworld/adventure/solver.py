"""Bounded deterministic state-space solver for compiled adventures."""

from __future__ import annotations

import hashlib
import json
from collections import deque

from .runtime import AdventureRuntime, action_from_interaction


def _state_key(state):
    relevant = {key: state[key] for key in ("inventory", "objects", "objectives", "flags", "completed")}
    return json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def solve_game(game: dict, max_states: int = 1000) -> dict:
    if not isinstance(max_states, int) or max_states <= 0:
        raise ValueError("max_states must be a positive integer")
    initial = AdventureRuntime(game)
    queue = deque([(initial.state, [])])
    visited = {_state_key(initial.state)}
    examined = 0
    interactions = sorted(game["runtime_rules"]["interactions"], key=lambda item: item["id"])
    while queue and examined < max_states:
        state, path = queue.popleft()
        examined += 1
        runtime = AdventureRuntime(game, state)
        if runtime.completed:
            return _report(True, path, examined, len(visited), max_states)
        available = {item["interaction_id"] for item in runtime.available_actions()}
        for interaction in interactions:
            if interaction["id"] not in available:
                continue
            candidate = AdventureRuntime(game, state)
            action = action_from_interaction(interaction)
            result = candidate.perform(action)
            if not result.success:
                continue
            step = {"interaction_id": interaction["id"], "action": action, "message": result.message}
            next_path = path + [step]
            if candidate.completed:
                return _report(True, next_path, examined, len(visited) + 1, max_states)
            key = _state_key(candidate.state)
            if key not in visited:
                visited.add(key)
                queue.append((candidate.state, next_path))
    return _report(False, [], examined, len(visited), max_states)


def _report(solvable, path, examined, visited, maximum):
    digest = hashlib.sha256(json.dumps(path, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "solvable": solvable,
        "shortest_solution_length": len(path) if solvable else None,
        "solution": path,
        "states_examined": examined,
        "states_visited": visited,
        "max_states": maximum,
        "solution_digest": digest,
    }

