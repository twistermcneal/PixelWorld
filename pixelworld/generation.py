import hashlib

import numpy as np

from .config import (
    ACTIONS,
    BIOMES,
    LANDMARK_CLASSES,
    LANDMARK_SIZES,
    LAYOUT_DIM,
    MAX_SLOTS,
    REGIONS,
    SIZE,
    SLOT_LATENT_DIM,
    TERRAIN_LATENT_DIM,
    TERRAINS,
    TRIGGER_TYPES,
)
from .schema import Landscape


TERRAIN_PALETTE = np.asarray(
    [(25, 85, 145), (224, 198, 125), (58, 125, 65), (130, 92, 55), (105, 105, 110), (225, 232, 238)],
    np.uint8,
)
REGION_PALETTE = np.asarray([(235, 204, 126), (92, 150, 78), (115, 110, 112), (32, 92, 48)], np.uint8)


def world_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def transition_seed(current_seed, slot, trigger_type, story_state=0):
    return world_seed(f"{current_seed}:{slot}:{trigger_type}:{story_state}")


def layout_from_seed(seed):
    return np.random.default_rng(seed).random(LAYOUT_DIM).astype(np.float32)


def terrain_params(prompt, seed):
    p = prompt.lower()
    layout = layout_from_seed(seed)
    biome = next((b for b in BIOMES if b in p), BIOMES[int(layout[0] * 4) % 4])
    return (
        BIOMES.index(biome),
        int(layout[1] * 4) % 4,
        22 + int(layout[2] * 21),
        3 + int(layout[3] * 9),
        int(layout[4] * 6),
        int(layout[5] * 6),
        1 + int(layout[6] * 5),
    )


def render_terrain(params, seed):
    biome_id, orientation, shoreline, beach_width, rockiness, _, _ = map(int, params)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    along = xx if orientation in (0, 2) else yy
    across = yy if orientation in (0, 2) else xx
    if orientation in (2, 1):
        across = SIZE - 1 - across
    phase = (seed % 10007) / 10007 * 2 * np.pi
    coast = shoreline + 2.3 * np.sin(along / 7 + phase) + 1.1 * np.sin(along / 3.3 + phase * 1.7)
    terrain = np.full((SIZE, SIZE), TERRAINS["grass"], np.int64)
    terrain[across < coast] = TERRAINS["water"]
    terrain[(across >= coast) & (across < coast + beach_width)] = TERRAINS["sand"]
    land = across >= coast + beach_width
    base = ["grass", "grass", "dirt", "snow"][biome_id]
    terrain[land] = TERRAINS[base]
    rock_field = np.sin(xx * .37 + phase) + np.cos(yy * .29 - phase) + np.sin((xx + yy) * .17)
    terrain[land & (rock_field > 2.35 - rockiness * .12)] = TERRAINS["rock"]
    return terrain


def render_regions(terrain, params, seed):
    *_, forest_level, _ = map(int, params)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    phase = (seed % 8191) / 8191 * 2 * np.pi
    regions = np.full((SIZE, SIZE), REGIONS.index("open_land"), np.int64)
    regions[terrain == TERRAINS["sand"]] = REGIONS.index("beach")
    regions[terrain == TERRAINS["rock"]] = REGIONS.index("rock_field")
    forest_field = np.sin(xx * .16 + phase) + np.cos(yy * .14 - phase * .7) + np.sin((xx - yy) * .09 + phase)
    forestable = np.isin(terrain, [TERRAINS["grass"], TERRAINS["dirt"], TERRAINS["snow"]])
    regions[forestable & (forest_field > 1.65 - forest_level * .18)] = REGIONS.index("forest")
    return regions


def scatter_vegetation(terrain, regions, params, seed):
    density = int(params[-1])
    vegetation = np.zeros((SIZE, SIZE), np.uint8)
    rng = np.random.default_rng(world_seed(f"vegetation:{seed}"))
    ys, xs = np.where(regions == REGIONS.index("forest"))
    order = rng.permutation(len(xs))
    accepted = []
    target = min(len(xs) // 16, density * 18)
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if all((x - ax) ** 2 + (y - ay) ** 2 >= 16 for ax, ay in accepted):
            accepted.append((x, y))
            vegetation[y, x] = 1
            if len(accepted) >= target:
                break
    return vegetation


def generate_landscape(prompt, seed=None):
    from .placement import resolve_anchor

    seed = world_seed(prompt) if seed is None else int(seed)
    layout = layout_from_seed(seed)
    params = terrain_params(prompt, seed)
    terrain = render_terrain(params, seed)
    regions = render_regions(terrain, params, seed)
    vegetation = scatter_vegetation(terrain, regions, params, seed)
    obj = np.zeros((SIZE, SIZE), np.int64)
    objects = {}
    for slot in range(MAX_SLOTS):
        values = layout[
            TERRAIN_LATENT_DIM
            + slot * SLOT_LATENT_DIM:TERRAIN_LATENT_DIM
            + (slot + 1) * SLOT_LATENT_DIM
        ]
        if not (slot == 0 or values[0] > .34):
            continue
        class_id = 2 if slot == 0 else int(values[1] * 4) % 4
        kind = LANDMARK_CLASSES[class_id]
        w, h = LANDMARK_SIZES[kind]
        region_id = int(values[2] * 4) % 4
        anchor_id = int(values[3] * 16) % 16
        resolved = resolve_anchor(regions, terrain, region_id, anchor_id, w, h, seed, slot, obj)
        if resolved is None:
            continue
        x, y, resolved_region = resolved
        action = ACTIONS[int(values[4] * 3) % 3]
        trigger = TRIGGER_TYPES[int(values[5] * 4) % 4]
        oid = slot + 1
        obj[y:y + h, x:x + w] = oid
        vegetation[y:y + h, x:x + w] = 0
        objects[oid] = {
            "class": kind,
            "bbox": [x, y, w, h],
            "region_id": region_id,
            "resolved_region_id": resolved_region,
            "anchor_id": anchor_id,
            "action": action,
            "trigger_type": trigger,
            "next_seed": transition_seed(seed, slot, trigger),
        }
    rgb = TERRAIN_PALETTE[terrain].copy()
    rgb[vegetation > 0] = (25, 72, 38)
    walkable = np.isin(
        terrain, [TERRAINS["sand"], TERRAINS["grass"], TERRAINS["dirt"], TERRAINS["snow"]]
    ).astype(np.uint8)
    walkable[vegetation > 0] = 0
    return Landscape(
        prompt,
        seed,
        BIOMES[params[0]],
        terrain,
        regions,
        vegetation,
        rgb,
        obj,
        walkable,
        (obj > 0).astype(np.uint8),
        params,
        objects,
    )
