import numpy as np

from pixelworld.config import LANDMARK_CLASSES, LANDMARK_SIZES, MAX_SLOTS, REGIONS, SIZE, TERRAINS

from .config import LOCAL_OFFSET_PIXELS


def anchor_normalized(anchor_id: int) -> tuple[float, float]:
    if not 0 <= int(anchor_id) < 16:
        raise ValueError("anchor_id must be between 0 and 15")
    row, column = divmod(int(anchor_id), 4)
    return (column + 0.5) / 4, (row + 0.5) / 4


def valid_candidates(regions, terrain, region_id: int, width: int, height: int):
    def window_sum(mask):
        integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (
            integral[height:, width:]
            - integral[:-height, width:]
            - integral[height:, :-width]
            + integral[:-height, :-width]
        )

    region_pixels = window_sum(regions == region_id)
    land_pixels = window_sum(terrain != TERRAINS["water"])
    valid = (region_pixels >= int(np.ceil(0.70 * width * height))) & (
        land_pixels == width * height
    )
    valid[1::2, :] = False
    valid[:, 1::2] = False
    ys, xs = np.where(valid)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def sorted_candidates(candidates, target_x: float, target_y: float):
    return sorted(
        candidates,
        key=lambda point: (
            (point[0] - target_x) ** 2 + (point[1] - target_y) ** 2,
            point[1],
            point[0],
        ),
    )


def fallback_regions(region_id: int):
    return [
        int(region_id),
        REGIONS.index("open_land"),
        REGIONS.index("beach"),
        REGIONS.index("forest"),
        REGIONS.index("rock_field"),
    ]


def regional_anchor_base(candidates, anchor_id):
    if not candidates:
        raise ValueError("regional_anchor_base requires at least one candidate")
    xs = [point[0] for point in candidates]
    ys = [point[1] for point in candidates]
    u, v = anchor_normalized(anchor_id)
    target_x = min(xs) + u * (max(xs) - min(xs))
    target_y = min(ys) + v * (max(ys) - min(ys))
    return sorted_candidates(candidates, target_x, target_y)[0]


def resolve_stable_anchor(
    regions,
    terrain,
    region_id,
    anchor_id,
    width,
    height,
    occupied,
    offset=(0.0, 0.0),
):
    requested_region = int(region_id)
    requested_has_candidates = False
    for fallback_index, fallback in enumerate(fallback_regions(requested_region)):
        candidates = valid_candidates(regions, terrain, fallback, width, height)
        if fallback_index == 0:
            requested_has_candidates = bool(candidates)
        if not candidates:
            continue
        anchor_x, anchor_y = regional_anchor_base(candidates, anchor_id)
        desired_x = float(anchor_x + LOCAL_OFFSET_PIXELS * float(offset[0]))
        desired_y = float(anchor_y + LOCAL_OFFSET_PIXELS * float(offset[1]))
        ordered = sorted_candidates(candidates, desired_x, desired_y)
        unconstrained = ordered[0]
        for x, y in ordered:
            if np.all(occupied[y : y + height, x : x + width] == 0):
                projection_distance = float(np.hypot(x - desired_x, y - desired_y))
                if fallback != requested_region:
                    reason = (
                        "missing_requested_region"
                        if not requested_has_candidates
                        else "collision_fallback"
                    )
                elif (x, y) != unconstrained:
                    reason = "collision_projection"
                elif projection_distance <= 1e-7:
                    reason = "exact"
                else:
                    reason = "validity_projection"
                return {
                    "x": x,
                    "y": y,
                    "resolved_region_id": fallback,
                    "anchor_base_x": int(anchor_x),
                    "anchor_base_y": int(anchor_y),
                    "desired_x": desired_x,
                    "desired_y": desired_y,
                    "projection_distance": projection_distance,
                    "projection_reason": reason,
                }
    return None


def rasterize_landmarks(
    terrain,
    regions,
    region_ids,
    anchors,
    presence,
    classes,
    offsets=None,
    offset_radius=LOCAL_OFFSET_PIXELS,
    return_details=False,
):
    if offset_radius != LOCAL_OFFSET_PIXELS:
        raise ValueError(f"0.6.2 latent offsets require radius {LOCAL_OFFSET_PIXELS}")
    object_map = np.zeros((SIZE, SIZE), np.int64)
    boxes = np.zeros((MAX_SLOTS, 4), np.int64)
    resolved_regions = np.full(MAX_SLOTS, -1, np.int64)
    details = [None] * MAX_SLOTS
    invalid = 0
    for slot in range(MAX_SLOTS):
        if presence[slot] < 0.5:
            continue
        kind = LANDMARK_CLASSES[int(classes[slot])]
        width, height = LANDMARK_SIZES[kind]
        offset = (0.0, 0.0) if offsets is None else offsets[slot]
        resolved = resolve_stable_anchor(
            regions,
            terrain,
            int(region_ids[slot]),
            int(anchors[slot]),
            width,
            height,
            object_map,
            offset,
        )
        if resolved is None:
            invalid += 1
            continue
        x, y = resolved["x"], resolved["y"]
        object_map[y : y + height, x : x + width] = slot + 1
        boxes[slot] = [x, y, width, height]
        resolved_regions[slot] = resolved["resolved_region_id"]
        details[slot] = resolved
    result = (
        object_map,
        (object_map > 0).astype(np.uint8),
        boxes,
        resolved_regions,
        invalid,
    )
    return result + (details,) if return_details else result
