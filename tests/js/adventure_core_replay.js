"use strict";

const fs = require("fs");
const path = require("path");

const [corePath, gamePath, solutionPath, expectedPath] = process.argv.slice(2);
if (!corePath || !gamePath || !solutionPath || !expectedPath) {
  throw new Error("usage: node adventure_core_replay.js CORE GAME SOLUTION EXPECTED");
}
const { Runtime } = require(path.resolve(corePath));
const read = value => JSON.parse(fs.readFileSync(value, "utf8"));
const game = read(gamePath);
const solution = read(solutionPath);
const expected = read(expectedPath);
const runtime = new Runtime(game);
const snapshots = [];
const canonical = value => Array.isArray(value)
  ? value.map(canonical)
  : value && typeof value === "object"
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]))
    : value;
for (let index = 0; index < solution.solution.length; index += 1) {
  const step = solution.solution[index];
  const result = runtime.perform(step.action);
  if (!result.success) throw new Error(`step ${index} (${step.interaction_id}) failed: ${result.message}`);
  const snapshot = {
    inventory: runtime.state.inventory,
    objects: runtime.state.objects,
    objectives: runtime.state.objectives,
    flags: runtime.state.flags,
    completed: runtime.state.completed,
  };
  if (JSON.stringify(canonical(snapshot)) !== JSON.stringify(canonical(expected[index]))) {
    throw new Error(`state parity failed after step ${index} (${step.interaction_id})`);
  }
  snapshots.push(snapshot);
}
if (!runtime.state.completed) throw new Error("ending state was not reached");
process.stdout.write(JSON.stringify({ success: true, steps: snapshots.length, completed: true }));
