import numpy as np

from pixelworld.config import (
    ACTIONS,
    BIOMES,
    LANDMARK_CLASSES,
    LANDMARK_SIZES,
    MAX_SLOTS,
    TERRAINS,
    TRIGGER_TYPES,
)
from pixelworld.generation import (
    TERRAIN_PALETTE,
    render_regions,
    render_terrain,
    scatter_vegetation,
    terrain_params,
    transition_seed,
    world_seed,
)
from pixelworld.model import prompt_vector
from pixelworld.schema import Landscape

from .config import LAYOUT_DIM, LOCAL_OFFSET_PIXELS, SLOT_LATENT_DIM, TERRAIN_LATENT_DIM
from .placement import rasterize_landmarks, resolve_stable_anchor


def layout_from_seed(seed):
    return np.random.default_rng(seed).random(LAYOUT_DIM).astype(np.float32)


def condition_vector(prompt, seed):
    return np.concatenate([prompt_vector(prompt), layout_from_seed(seed)]).astype(np.float32)


def latent_offset(value):
    return float(2.0 * float(value) - 1.0)


def generate_landscape(prompt, seed=None):
    seed = world_seed(prompt) if seed is None else int(seed)
    layout = layout_from_seed(seed)
    params = terrain_params(prompt, seed)
    terrain = render_terrain(params, seed)
    regions = render_regions(terrain, params, seed)
    vegetation = scatter_vegetation(terrain, regions, params, seed)
    object_map = np.zeros_like(terrain)
    objects = {}
    for slot in range(MAX_SLOTS):
        values = layout[
            TERRAIN_LATENT_DIM
            + slot * SLOT_LATENT_DIM : TERRAIN_LATENT_DIM
            + (slot + 1) * SLOT_LATENT_DIM
        ]
        if not (slot == 0 or values[0] > 0.34):
            continue
        class_id = 2 if slot == 0 else int(values[1] * 4) % 4
        kind = LANDMARK_CLASSES[class_id]
        width, height = LANDMARK_SIZES[kind]
        requested_region = int(values[2] * 4) % 4
        anchor_id = int(values[3] * 16) % 16
        offset_x = latent_offset(values[4])
        offset_y = latent_offset(values[5])
        resolved = resolve_stable_anchor(
            regions,
            terrain,
            requested_region,
            anchor_id,
            width,
            height,
            object_map,
            (offset_x, offset_y),
        )
        if resolved is None:
            continue
        x, y = resolved["x"], resolved["y"]
        action = ACTIONS[int(values[6] * 3) % 3]
        trigger = TRIGGER_TYPES[int(values[7] * 4) % 4]
        oid = slot + 1
        object_map[y : y + height, x : x + width] = oid
        vegetation[y : y + height, x : x + width] = 0
        objects[oid] = {
            "class": kind,
            "bbox": [x, y, width, height],
            "requested_region_id": requested_region,
            "resolved_region_id": resolved["resolved_region_id"],
            "region_id": requested_region,
            "anchor_id": anchor_id,
            "anchor_base_x": resolved["anchor_base_x"],
            "anchor_base_y": resolved["anchor_base_y"],
            "offset_x": offset_x,
            "offset_y": offset_y,
            "desired_x": resolved["desired_x"],
            "desired_y": resolved["desired_y"],
            "projection_distance": resolved["projection_distance"],
            "projection_reason": resolved["projection_reason"],
            "action": action,
            "trigger_type": trigger,
            "next_seed": transition_seed(seed, slot, trigger),
        }
    rgb = TERRAIN_PALETTE[terrain].copy()
    rgb[vegetation > 0] = (25, 72, 38)
    walkable = np.isin(
        terrain,
        [TERRAINS["sand"], TERRAINS["grass"], TERRAINS["dirt"], TERRAINS["snow"]],
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
        object_map,
        walkable,
        (object_map > 0).astype(np.uint8),
        params,
        objects,
    )


def scene_graph_arrays(world):
    regions = np.zeros(MAX_SLOTS, np.int64)
    requested_regions = np.zeros(MAX_SLOTS, np.int64)
    anchors = np.zeros(MAX_SLOTS, np.int64)
    presence = np.zeros(MAX_SLOTS, np.float32)
    classes = np.zeros(MAX_SLOTS, np.int64)
    offsets = np.zeros((MAX_SLOTS, 2), np.float32)
    boxes = np.zeros((MAX_SLOTS, 4), np.int64)
    anchor_bases = np.zeros((MAX_SLOTS, 2), np.float32)
    desired = np.zeros((MAX_SLOTS, 2), np.float32)
    projection_distances = np.zeros(MAX_SLOTS, np.float32)
    projection_reasons = [None] * MAX_SLOTS
    for slot in range(MAX_SLOTS):
        metadata = world.objects.get(slot + 1)
        if metadata is None:
            continue
        presence[slot] = 1
        requested_regions[slot] = metadata["requested_region_id"]
        regions[slot] = metadata["resolved_region_id"]
        anchors[slot] = metadata["anchor_id"]
        classes[slot] = LANDMARK_CLASSES.index(metadata["class"])
        boxes[slot] = metadata["bbox"]
        offsets[slot] = [metadata["offset_x"], metadata["offset_y"]]
        anchor_bases[slot] = [metadata["anchor_base_x"], metadata["anchor_base_y"]]
        desired[slot] = [metadata["desired_x"], metadata["desired_y"]]
        projection_distances[slot] = metadata["projection_distance"]
        projection_reasons[slot] = metadata["projection_reason"]
    return {
        "regions": regions,
        "requested_regions": requested_regions,
        "anchors": anchors,
        "presence": presence,
        "classes": classes,
        "offsets": offsets,
        "boxes": boxes,
        "anchor_bases": anchor_bases,
        "desired": desired,
        "projection_distances": projection_distances,
        "projection_reasons": projection_reasons,
    }


def ground_truth_round_trip(world):
    graph = scene_graph_arrays(world)
    result = rasterize_landmarks(
        world.terrain,
        world.regions,
        graph["regions"],
        graph["anchors"],
        graph["presence"],
        graph["classes"],
        offsets=graph["offsets"],
        offset_radius=LOCAL_OFFSET_PIXELS,
        return_details=True,
    )
    object_map, interaction, boxes, resolved_regions, invalid, details = result
    return {
        "object_map_equal": np.array_equal(object_map, world.object_map),
        "interaction_equal": np.array_equal(interaction, world.interaction),
        "boxes_equal": np.array_equal(boxes, graph["boxes"]),
        "resolved_regions_equal": np.array_equal(
            resolved_regions[graph["presence"] > 0.5],
            graph["regions"][graph["presence"] > 0.5],
        ),
        "invalid": invalid,
        "boxes": boxes,
        "details": details,
    }
