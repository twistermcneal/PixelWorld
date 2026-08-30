import numpy as np
import pytest

from pixelworld.config import DEFAULT_PROMPT, TERRAINS
from pixelworld.generation import generate_landscape as generate_baseline
from pixelworld.model import LandscapeNet
from pixelworld.training import seed_everything
from pixelworld.versions.v0_6_2.config import CONDITION_DIM, LAYOUT_DIM, VARIANTS, variant_spec
from pixelworld.versions.v0_6_2.generation import (
    condition_vector,
    generate_landscape,
    ground_truth_round_trip,
    layout_from_seed,
    scene_graph_arrays,
)
from pixelworld.versions.v0_6_2.model import create_model
from pixelworld.versions.v0_6_2.placement import (
    anchor_normalized,
    resolve_stable_anchor,
    sorted_candidates,
)


def test_variant_registry():
    assert VARIANTS == ("A", "B", "C", "D", "E")
    assert not variant_spec("A")["stable_anchors"]
    assert variant_spec("B")["stable_anchors"]
    assert variant_spec("C")["offset_input_detached"]
    assert not variant_spec("D")["offset_input_detached"]
    assert variant_spec("E")["auxiliary_xy"]


def test_variant_a_keeps_frozen_model_and_generator():
    seed_everything(42)
    expected = LandscapeNet()
    seed_everything(42)
    actual = create_model("A")
    assert all(
        np.array_equal(expected.state_dict()[name].numpy(), actual.state_dict()[name].numpy())
        for name in expected.state_dict()
    )
    first = generate_baseline(DEFAULT_PROMPT, 500000)
    second = generate_baseline(DEFAULT_PROMPT, 500000)
    assert np.array_equal(first.object_map, second.object_map)


def test_explicit_eight_latent_layout_and_condition_dimensions():
    assert LAYOUT_DIM == 71
    assert CONDITION_DIM == 81
    assert layout_from_seed(42).shape == (71,)
    assert condition_vector(DEFAULT_PROMPT, 42).shape == (81,)


@pytest.mark.parametrize(
    ("anchor_id", "expected"),
    [(0, (0.125, 0.125)), (3, (0.875, 0.125)), (12, (0.125, 0.875)), (15, (0.875, 0.875))],
)
def test_anchor_id_to_normalized_coordinate(anchor_id, expected):
    assert anchor_normalized(anchor_id) == expected


def test_distance_sorting_has_y_then_x_tie_break():
    candidates = [(2, 1), (1, 2), (0, 1), (1, 0)]
    assert sorted_candidates(candidates, 1, 1) == [(1, 0), (0, 1), (2, 1), (1, 2)]


def test_anchor_ids_have_stable_spatial_meaning():
    terrain = np.full((64, 64), TERRAINS["grass"], np.int64)
    regions = np.ones((64, 64), np.int64)
    occupied = np.zeros((64, 64), np.int64)
    top_left = resolve_stable_anchor(regions, terrain, 1, 0, 5, 4, occupied)
    bottom_right = resolve_stable_anchor(regions, terrain, 1, 15, 5, 4, occupied)
    assert (top_left["x"], top_left["y"]) == (8, 8)
    assert bottom_right["x"] > top_left["x"]
    assert bottom_right["y"] > top_left["y"]


@pytest.mark.parametrize("seed", range(20))
def test_raw_offset_targets_never_clip(seed):
    world = generate_landscape(DEFAULT_PROMPT, seed + 1000)
    offsets = scene_graph_arrays(world)["offsets"]
    presence = scene_graph_arrays(world)["presence"] > 0.5
    assert np.all(offsets[presence] >= -1)
    assert np.all(offsets[presence] <= 1)


def test_slot_latents_have_prescribed_order():
    seed = 1234
    layout = layout_from_seed(seed)
    world = generate_landscape(DEFAULT_PROMPT, seed)
    for slot, metadata in ((oid - 1, item) for oid, item in world.objects.items()):
        values = layout[7 + slot * 8 : 7 + (slot + 1) * 8]
        assert metadata["requested_region_id"] == int(values[2] * 4) % 4
        assert metadata["anchor_id"] == int(values[3] * 16) % 16
        assert metadata["offset_x"] == pytest.approx(2 * values[4] - 1)
        assert metadata["offset_y"] == pytest.approx(2 * values[5] - 1)
        assert metadata["desired_x"] == pytest.approx(
            metadata["anchor_base_x"] + 8 * metadata["offset_x"]
        )
        assert metadata["desired_y"] == pytest.approx(
            metadata["anchor_base_y"] + 8 * metadata["offset_y"]
        )


@pytest.mark.parametrize("seed", range(20))
def test_ground_truth_round_trip_and_safety(seed):
    world = generate_landscape(DEFAULT_PROMPT, seed + 1000)
    result = ground_truth_round_trip(world)
    assert result["object_map_equal"]
    assert result["boxes_equal"]
    assert result["interaction_equal"]
    assert result["invalid"] == 0
    occupied = world.object_map > 0
    assert not np.any(occupied & (world.terrain == TERRAINS["water"]))
    expected_area = sum(item["bbox"][2] * item["bbox"][3] for item in world.objects.values())
    assert occupied.sum() == expected_area
    repeated = generate_landscape(DEFAULT_PROMPT, seed + 1000)
    assert np.array_equal(world.object_map, repeated.object_map)


def test_variants_b_to_e_share_identical_target_worlds():
    worlds = [generate_landscape(DEFAULT_PROMPT, 500000) for _ in ("B", "C", "D", "E")]
    expected = scene_graph_arrays(worlds[0])
    for world in worlds[1:]:
        assert np.array_equal(world.terrain, worlds[0].terrain)
        assert np.array_equal(world.regions, worlds[0].regions)
        assert np.array_equal(world.object_map, worlds[0].object_map)
        actual = scene_graph_arrays(world)
        for key in ("regions", "requested_regions", "anchors", "presence", "classes", "offsets", "boxes", "anchor_bases", "desired"):
            assert np.array_equal(actual[key], expected[key])
