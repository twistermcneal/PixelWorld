import numpy as np

from pixelworld.config import DEFAULT_PROMPT, LANDMARK_SIZES, REGIONS, TERRAINS
from pixelworld.evaluation import vegetation_round_trip
from pixelworld.generation import generate_landscape, world_seed
from pixelworld.placement import anchor_candidates, resolve_anchor


def test_world_seed_is_deterministic():
    assert world_seed("pixelworld") == world_seed("pixelworld")
    assert world_seed("pixelworld") != world_seed("PixelWorld")


def test_world_generation_is_deterministic():
    first = generate_landscape(DEFAULT_PROMPT, 424242)
    second = generate_landscape(DEFAULT_PROMPT, 424242)
    assert first.terrain_params == second.terrain_params
    for field in ("terrain", "regions", "vegetation", "objects", "interaction"):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert first.objects == second.objects


def test_vegetation_round_trip():
    assert vegetation_round_trip()


def test_anchor_candidates_keep_seed_permutation():
    world = generate_landscape(DEFAULT_PROMPT, 424242)
    first = anchor_candidates(world.regions, world.terrain, 1, 5, 4, world.seed, 0)
    second = anchor_candidates(world.regions, world.terrain, 1, 5, 4, world.seed, 0)
    other_slot = anchor_candidates(world.regions, world.terrain, 1, 5, 4, world.seed, 1)
    assert first == second
    assert first != other_slot


def test_anchor_fallback_uses_existing_order():
    terrain = np.full((64, 64), TERRAINS["grass"], dtype=np.int64)
    regions = np.full((64, 64), REGIONS.index("open_land"), dtype=np.int64)
    occupied = np.zeros((64, 64), dtype=np.int64)
    result = resolve_anchor(
        regions,
        terrain,
        REGIONS.index("forest"),
        0,
        *LANDMARK_SIZES["chest"],
        123,
        0,
        occupied,
    )
    assert result is not None
    assert result[2] == REGIONS.index("open_land")


def test_generated_objects_do_not_overlap_or_touch_water():
    for seed in range(100, 120):
        world = generate_landscape(DEFAULT_PROMPT, seed)
        occupied = world.object_map > 0
        assert not np.any(occupied & (world.terrain == TERRAINS["water"]))
        expected_area = sum(
            metadata["bbox"][2] * metadata["bbox"][3]
            for metadata in world.objects.values()
        )
        assert int(occupied.sum()) == expected_area
