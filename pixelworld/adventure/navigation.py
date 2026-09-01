"""Deterministic vector walkbox navigation."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from .models import validate_point, validate_polygon


EPSILON = 1e-9


def point_in_polygon(point, polygon, *, include_boundary: bool = True) -> bool:
    x, y = validate_point(point)
    points = [tuple(p) for p in validate_polygon(polygon)]
    inside = False
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= EPSILON and min(x1, x2) - EPSILON <= x <= max(x1, x2) + EPSILON and min(y1, y2) - EPSILON <= y <= max(y1, y2) + EPSILON:
            return include_boundary
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def closest_point_on_segment(point, start, end):
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= EPSILON:
        return float(ax), float(ay)
    amount = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return ax + amount * dx, ay + amount * dy


def project_to_polygon(point, polygon):
    point = validate_point(point)
    points = [tuple(p) for p in validate_polygon(polygon)]
    if point_in_polygon(point, points):
        return point
    candidates = [closest_point_on_segment(point, points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
    return min(candidates, key=lambda candidate: (_distance(point, candidate), candidate[0], candidate[1]))


def project_to_walkboxes(point, walkboxes):
    candidates = []
    for walkbox in sorted(walkboxes, key=lambda item: item["id"]):
        projected = project_to_polygon(point, walkbox["polygon"])
        candidates.append((_distance(point, projected), projected[0], projected[1], walkbox["id"], projected))
    if not candidates:
        raise ValueError("at least one walkbox is required")
    return min(candidates)[-1]


def _distance(left, right):
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _orientation(a, b, c):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= EPSILON:
        return 0
    return 1 if value > 0 else -1


def segments_intersect(a, b, c, d, *, include_touches=True):
    values = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if values[0] != values[1] and values[2] != values[3]:
        return True
    if include_touches and 0 in values:
        for point, start, end in ((c, a, b), (d, a, b), (a, c, d), (b, c, d)):
            if _orientation(start, end, point) == 0 and min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON and min(start[1], end[1]) - EPSILON <= point[1] <= max(start[1], end[1]) + EPSILON:
                return True
    return False


def segment_crosses_polygon(start, end, polygon) -> bool:
    points = [tuple(p) for p in polygon]
    if point_in_polygon(start, points, include_boundary=False) or point_in_polygon(end, points, include_boundary=False):
        return True
    for index in range(len(points)):
        if segments_intersect(start, end, points[index], points[(index + 1) % len(points)], include_touches=False):
            return True
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    return point_in_polygon(midpoint, points, include_boundary=False)


def point_walkable(point, walkboxes, collisions=()) -> bool:
    return any(point_in_polygon(point, item["polygon"]) for item in walkboxes) and not any(
        point_in_polygon(point, polygon, include_boundary=False) for polygon in collisions
    )


def segment_walkable(start, end, walkboxes, collisions=()) -> bool:
    if any(segment_crosses_polygon(start, end, polygon) for polygon in collisions):
        return False
    distance = _distance(start, end)
    samples = max(2, int(math.ceil(distance * 2)))
    return all(point_walkable((start[0] + (end[0] - start[0]) * i / samples, start[1] + (end[1] - start[1]) * i / samples), walkboxes, collisions) for i in range(samples + 1))


def _containing_boxes(point, walkboxes):
    return sorted(item["id"] for item in walkboxes if point_in_polygon(point, item["polygon"]))


def _edge_map(edges):
    result = {}
    for edge in edges:
        left, right = edge["from"], edge["to"]
        connector = tuple(edge["point"])
        result.setdefault(left, []).append((right, connector))
        result.setdefault(right, []).append((left, connector))
    for value in result.values():
        value.sort(key=lambda item: (item[0], item[1]))
    return result


def shortest_route(start, goal, walkboxes, navigation_edges, collision_polygons=()):
    """Find the deterministic shortest route and smooth its polyline."""

    start = project_to_walkboxes(start, walkboxes)
    goal = project_to_walkboxes(goal, walkboxes)
    collisions = [item.get("polygon", item) for item in collision_polygons]
    if segment_walkable(start, goal, walkboxes, collisions):
        return [list(start), list(goal)]
    starts = _containing_boxes(start, walkboxes)
    goals = set(_containing_boxes(goal, walkboxes))
    graph = _edge_map(navigation_edges)
    queue = []
    for box_id in starts:
        heapq.heappush(queue, (0.0, (box_id,), box_id, tuple(start), [tuple(start)]))
    best = {}
    while queue:
        cost, signature, box_id, current, points = heapq.heappop(queue)
        key = (box_id, current)
        if cost > best.get(key, float("inf")) + EPSILON:
            continue
        if box_id in goals and segment_walkable(current, goal, walkboxes, collisions):
            return _smooth(points + [tuple(goal)], walkboxes, collisions)
        for next_id, connector in graph.get(box_id, []):
            if not segment_walkable(current, connector, walkboxes, collisions):
                continue
            next_cost = cost + _distance(current, connector)
            next_key = (next_id, connector)
            next_signature = signature + (next_id,)
            previous = best.get(next_key)
            if previous is None or next_cost < previous - EPSILON:
                best[next_key] = next_cost
                heapq.heappush(queue, (next_cost, next_signature, next_id, connector, points + [connector]))
    raise ValueError(f"no collision-free route from {start!r} to {goal!r}")


def _smooth(points, walkboxes, collisions):
    result = [points[0]]
    index = 0
    while index < len(points) - 1:
        candidate = len(points) - 1
        while candidate > index + 1 and not segment_walkable(points[index], points[candidate], walkboxes, collisions):
            candidate -= 1
        result.append(points[candidate])
        index = candidate
    return [[round(point[0], 4), round(point[1], 4)] for point in result]


def polygons_overlap(left, right) -> bool:
    left_points = [tuple(p) for p in left]
    right_points = [tuple(p) for p in right]
    if any(point_in_polygon(point, right_points, include_boundary=False) for point in left_points):
        return True
    if any(point_in_polygon(point, left_points, include_boundary=False) for point in right_points):
        return True
    return any(
        segments_intersect(left_points[i], left_points[(i + 1) % len(left_points)], right_points[j], right_points[(j + 1) % len(right_points)], include_touches=False)
        for i in range(len(left_points)) for j in range(len(right_points))
    )

