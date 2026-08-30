import numpy as np

from .config import LANDMARK_CLASSES, LANDMARK_SIZES, MAX_SLOTS, REGIONS, SIZE, TERRAINS
from .generation import world_seed


def anchor_candidates(regions, terrain, region_id, w, h, seed, slot):
    def window_sum(mask):
        integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]

    region_pixels = window_sum(regions == region_id)
    land_pixels = window_sum(terrain != TERRAINS["water"])
    valid = (region_pixels >= int(np.ceil(.70 * w * h))) & (land_pixels == w * h)
    valid[1::2, :] = False
    valid[:, 1::2] = False
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(world_seed(f"anchors:{seed}:{slot}:{region_id}:{w}:{h}"))
    order = rng.permutation(len(xs))
    return [(int(xs[i]), int(ys[i])) for i in order]


def resolve_anchor(regions, terrain, region_id, anchor_id, w, h, seed, slot, occupied):
    for fallback in [
        region_id,
        REGIONS.index("open_land"),
        REGIONS.index("beach"),
        REGIONS.index("forest"),
        REGIONS.index("rock_field"),
    ]:
        candidates = anchor_candidates(regions, terrain, fallback, w, h, seed, slot)
        if not candidates:
            continue
        for step in range(len(candidates)):
            x, y = candidates[(anchor_id + step) % len(candidates)]
            if np.all(occupied[y:y + h, x:x + w] == 0):
                return x, y, fallback
    return None


def rasterize_landmarks(seed, terrain, regions, region_ids, anchors, presence, classes):
    obj = np.zeros((SIZE, SIZE), np.int64)
    boxes = np.zeros((MAX_SLOTS, 4), np.int64)
    for slot in range(MAX_SLOTS):
        if presence[slot] < .5:
            continue
        kind = LANDMARK_CLASSES[int(classes[slot])]
        w, h = LANDMARK_SIZES[kind]
        resolved = resolve_anchor(
            regions, terrain, int(region_ids[slot]), int(anchors[slot]), w, h, seed, slot, obj
        )
        if resolved is None:
            continue
        x, y, _ = resolved
        obj[y:y + h, x:x + w] = slot + 1
        boxes[slot] = [x, y, w, h]
    return obj, (obj > 0).astype(np.uint8), boxes
