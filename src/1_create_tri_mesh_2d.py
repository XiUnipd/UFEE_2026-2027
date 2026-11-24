"""Gmsh 2D conforming triangular mesh: four-corner CSV + separate refinement-control CSV.

The boundary consists of four strictly convex corners in CW/CCW order; actual line
segments define the domain, including rotated rectangles.
The XY Euclidean distance to the nearest control point defines the radial size field;
a CSV containing only the X,Y header selects a uniform background size.

Target sizes need not equal actual edge lengths. Defaults are defined below;
equal refined and background sizes specify a uniform target size.

Output: MSH 2.2 ASCII

Run: python python_code/src/1_create_tri_mesh_2d.py
Dependencies: gmsh, numpy, scipy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOMAIN_CORNERS_FILE = PROJECT_DIR / "data" / "boundary.csv"
CONTROL_POINTS_FILE = PROJECT_DIR / "data" / "density_points_myCase.csv"
OUTPUT_FILE = PROJECT_DIR / "out" / "mesh_2d_tri.msh"
BACKGROUND_SIZE = 600.0
REFINED_SIZE = 200.0
REFINEMENT_RADIUS = 200.0
TRANSITION_RADIUS = 4000.0
MAX_TRIANGLES = 4_000_000
MSH_VERSION = 2.2
MESH_ATTEMPTS = (6, 5, 1)
SEED_SPACING_FACTOR = 6.0
CONVEXITY_TOLERANCE = 1e-9
_ALGORITHM_NAMES = {1: "MeshAdapt", 5: "Delaunay", 6: "Frontal-Delaunay"}


@dataclass(frozen=True)
class TriangleMeshStats:
    node_count: int
    triangle_count: int
    area: float
    perimeter: float
    min_area: float
    min_edge_length: float
    max_edge_length: float
    min_corner_angle: float
    max_corner_angle: float
    min_quality: float
    max_quality: float
    boundary_loop: tuple[int, ...]  # 0-based, CCW, starting at the first corner; no repeated closing node.
    boundary_side_ids: tuple[int, ...]  # 0-based sides of the normalized CCW domain, aligned with directed loop edges.


@dataclass(frozen=True)
class MeshSummary:
    output: Path
    report: Path
    domain_corners: tuple[tuple[float, float], ...]
    control_point_count: int
    seed_point_count: int
    stats: TriangleMeshStats
    gmsh_algorithm: str
    attempt_failures: tuple[str, ...]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def geometry_tolerance(corners: np.ndarray) -> float:
    """Account for domain scale and global-coordinate ULPs; compute geometry in local coordinates."""
    scale = float(np.linalg.norm(np.ptp(corners, axis=0)))
    magnitude = float(np.max(np.abs(corners)))
    tolerance = max(1e-11 * scale, 32 * abs(float(np.spacing(magnitude))))
    if not math.isfinite(scale) or scale <= 0 or tolerance >= scale * 1e-3:
        raise ValueError("The domain is degenerate, or coordinates are too large relative to the domain size for sufficient floating-point precision")
    return tolerance


def polygon_area(corners: np.ndarray) -> float:
    p = corners - corners[0]
    return float(.5 * np.sum(_cross(p, np.roll(p, -1, axis=0))))


def polygon_perimeter(corners: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1).sum())


def validate_domain_corners(corners: np.ndarray) -> np.ndarray:
    """Do not sort corners into a convex hull; only reverse CW to CCW, preserving the first corner."""
    c = np.asarray(corners, dtype=np.float64)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        raise ValueError("The boundary must contain exactly four finite X,Y corners (no NaN/Inf)")
    tol = geometry_tolerance(c)
    distances = np.linalg.norm(c[:, None] - c[None, :], axis=2)
    if np.any(distances[np.triu_indices(4, 1)] <= tol):
        raise ValueError("The boundary contains duplicate corners or corners coincident within numerical tolerance")
    for i, j in ((0, 2), (1, 3)):
        a, b, d, e = c[i], c[(i + 1) % 4], c[j], c[(j + 1) % 4]
        ab, de = b - a, e - d
        u = np.array([_cross(ab, d - a), _cross(ab, e - a)]) / np.linalg.norm(ab)
        v = np.array([_cross(de, a - d), _cross(de, b - d)]) / np.linalg.norm(de)
        boxes_overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(d, e)) <=
                               np.minimum(np.maximum(a, b), np.maximum(d, e)) + tol)
        if boxes_overlap and u.min() <= tol and u.max() >= -tol and v.min() <= tol and v.max() >= -tol:
            raise ValueError("The quadrilateral boundary self-intersects or nonadjacent edges touch; provide corners in actual boundary order")
    area = polygon_area(c)
    if abs(area) <= tol * polygon_perimeter(c):
        raise ValueError("The quadrilateral has zero area or is numerically degenerate")
    if area < 0:
        c = c[[0, 3, 2, 1]]
    edges = np.roll(c, -1, axis=0) - c
    lengths = np.linalg.norm(edges, axis=1)
    turn = _cross(edges, np.roll(edges, -1, axis=0))
    threshold = np.maximum(CONVEXITY_TOLERANCE * lengths * np.roll(lengths, -1),
                           tol * (lengths + np.roll(lengths, -1)))
    if np.any(turn <= threshold):
        raise ValueError("The quadrilateral must be strictly convex: concave, collinear, or numerically near-degenerate corners detected")
    return c.copy()


def _read_xy_csv(path: Path | str) -> np.ndarray:
    points = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [item.strip().upper() for item in next(reader, [])]
        if header.count("X") != 1 or header.count("Y") != 1:
            raise ValueError(f"{path}: CSV must contain exactly one X column and one Y column (a header is required)")
        columns = [header.index("X"), header.index("Y")]
        for row in reader:
            if not any(item.strip() for item in row):
                continue
            try:
                point = [float(row[column]) for column in columns]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}: line {reader.line_num} has invalid coordinates") from exc
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"{path}: coordinates on line {reader.line_num} contain NaN/Inf")
            points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def load_domain_corners(path: Path | str) -> np.ndarray:
    """Load four corners in CCW order without repairing invalid rows or corners."""
    try:
        return validate_domain_corners(_read_xy_csv(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_control_points(path: Path | str) -> np.ndarray:
    """Return shape (0,2) for a valid empty CSV; missing or invalid files still raise errors."""
    return np.unique(_read_xy_csv(path), axis=0)


def points_in_domain(points: np.ndarray, corners: np.ndarray, tolerance: float | None = None) -> np.ndarray:
    tol = geometry_tolerance(corners) if tolerance is None else tolerance
    edges = np.roll(corners, -1, axis=0) - corners
    signed_distance = _cross(edges[None], points[:, None, :] - corners[None]) / np.linalg.norm(edges, axis=1)
    return np.all(signed_distance >= -tol, axis=1)


def boundary_projections(points: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distances to the four finite segments, parameters clipped to [0,1], and projections."""
    edges = np.roll(corners, -1, axis=0) - corners
    offset = points[:, None, :] - corners[None]
    t = np.clip(np.sum(offset * edges[None], axis=2) / np.sum(edges * edges, axis=1), 0., 1.)
    projections = corners[None] + t[..., None] * edges[None]
    return np.linalg.norm(points[:, None] - projections, axis=2), t, projections


def _snap_to_boundary(xyz: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Snap only to actual corners/edges within tolerance, corners first; do not clip to X/Y extrema."""
    tol = geometry_tolerance(corners)
    if not np.isfinite(xyz).all() or np.any(np.abs(xyz[:, 2]) > tol):
        raise RuntimeError("Gmsh nodes contain nonfinite coordinates or deviate from the Z=0 plane")
    result = xyz.copy()
    result[:, 2] = 0.
    distance = np.linalg.norm(result[:, None, :2] - corners[None], axis=2)
    nearest = distance.argmin(axis=1)
    at_corner = distance[np.arange(len(result)), nearest] <= tol
    result[at_corner, :2] = corners[nearest[at_corner]]
    d, _, projections = boundary_projections(result[:, :2], corners)
    side = d.argmin(axis=1)
    at_edge = (~at_corner) & (d[np.arange(len(result)), side] <= tol)
    result[at_edge, :2] = projections[np.flatnonzero(at_edge), side[at_edge]]
    return result


def validate_sizes(background_size: float, refined_size: float,
                   refinement_radius: float, transition_radius: float) -> None:
    for name, value in (("background_size", background_size), ("refined_size", refined_size),
                        ("transition_radius", transition_radius), ("refinement_radius", refinement_radius)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(value) or value < 0 or (name != "refinement_radius" and value == 0):
            raise ValueError(f"{name} must be a finite {'nonnegative' if name == 'refinement_radius' else 'positive'} number")
    if refined_size > background_size:
        raise ValueError("refined_size must not exceed background_size")
    if transition_radius <= refinement_radius:
        raise ValueError("transition_radius must be greater than refinement_radius")


def _validate_triangle_arrays(xyz: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz, triangles = np.asarray(xyz, dtype=np.float64), np.asarray(triangles)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise RuntimeError("Nodes must form a nonempty, finite N-by-3 array")
    if np.any(xyz[:, 2] != 0.):
        raise RuntimeError("Nodes with Z != 0 detected")
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise RuntimeError("All elements must be three-node triangles with integer connectivity")
    if triangles.min() < 0 or triangles.max() >= len(xyz):
        raise RuntimeError("Triangles reference nonexistent nodes")
    if len(np.unique(triangles)) != len(xyz):
        raise RuntimeError("Isolated nodes detected")
    keys = np.sort(triangles, axis=1)
    if np.any(np.diff(keys, axis=1) == 0):
        raise RuntimeError("Triangles contain repeated nodes")
    if len(np.unique(keys, axis=0)) != len(triangles):
        raise RuntimeError("Duplicate elements detected")
    return xyz, triangles.astype(np.int64, copy=False)


def validate_triangle_mesh(xyz: np.ndarray, triangles: np.ndarray, corners: np.ndarray) -> TriangleMeshStats:
    """Ensure single coverage using positive triangles, interior-edge cancellation, and one true outer loop.

    Triangle winding numbers are nonnegative. After interior directed edges cancel,
    their sum equals the winding number of the actual domain boundary: 1 inside,
    0 outside. This rejects holes/overlaps with nonzero area, not just area mismatches.
    Open edges must cover the actual boundary segment by segment, rejecting interior
    cracks and T-junction hanging nodes.
    """
    corners = validate_domain_corners(corners)
    tol = geometry_tolerance(corners)
    xyz, triangles = _validate_triangle_arrays(xyz, triangles)
    inside = points_in_domain(xyz[:, :2], corners)
    if not inside.all():
        raise RuntimeError(f"Node lies outside the actual quadrilateral domain: {xyz[~inside][0, :2].tolist()}")
    if np.any(cKDTree(xyz[:, :2]).query(xyz[:, :2], k=2)[0][:, 1] <= tol):
        raise RuntimeError("Duplicate nodes or nodes coincident within numerical tolerance detected")
    p = xyz[triangles, :2]
    p = p - p[:, :1]
    edges = np.roll(p, -1, axis=1) - p
    lengths = np.linalg.norm(edges, axis=2)
    following = np.roll(edges, -1, axis=1)
    cross = _cross(edges, following)
    threshold = np.maximum(CONVEXITY_TOLERANCE * lengths * np.roll(lengths, -1, axis=1),
                           tol * (lengths + np.roll(lengths, -1, axis=1)))
    if not np.isfinite(cross).all() or np.any(lengths <= tol) or np.any(cross <= threshold):
        raise RuntimeError("Triangles are not counterclockwise, have zero area, or are numerically near-degenerate")
    areas = .5 * cross[:, 0]
    start = triangles.ravel()
    end = np.roll(triangles, -1, axis=1).ravel()
    edge_keys = np.minimum(start, end) * len(xyz) + np.maximum(start, end)
    unique_edges, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise RuntimeError("Non-manifold edge: an edge is shared by three or more elements")
    direction = np.bincount(inverse, weights=np.where(start < end, 1., -1.))
    if np.any((counts == 2) & (direction != 0)):
        raise RuntimeError("Interior edge is not shared by two triangles in opposite directions: folding or overlap detected")
    opened = counts[inverse] == 1
    a_ids, b_ids = start[opened], end[opened]
    if not len(a_ids):
        raise RuntimeError("No outer boundary found")
    boundary_ids = np.unique(np.r_[a_ids, b_ids])
    if (len(a_ids) != len(boundary_ids) or len(np.unique(a_ids)) != len(a_ids)
            or len(np.unique(b_ids)) != len(b_ids)):
        raise RuntimeError("Outer boundary branches or is not a closed loop")
    successor = dict(zip(a_ids.tolist(), b_ids.tolist()))
    current, visited, loop = int(a_ids[0]), set(), []
    while current not in visited:
        visited.add(current)
        loop.append(current)
        if current not in successor:
            raise RuntimeError("Outer boundary is not closed")
        current = successor[current]
    if current != a_ids[0] or len(visited) != len(boundary_ids):
        raise RuntimeError("Outer boundary must be a single closed loop: holes or disconnected components detected")
    a, b = xyz[a_ids, :2], xyz[b_ids, :2]
    da, ta, _ = boundary_projections(a, corners)
    db, tb, _ = boundary_projections(b, corners)
    side_match = (da <= tol) & (db <= tol)
    if np.any(side_match.sum(axis=1) != 1):
        raise RuntimeError("Open edge does not lie on exactly one actual boundary segment: hanging node, crack, or corner-crossing boundary edge")
    side = side_match.argmax(axis=1)
    domain_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    for j in range(4):
        indices = np.flatnonzero(side == j)
        if not len(indices):
            raise RuntimeError(f"CCW boundary side {j + 1} is not covered")
        lo, hi = ta[indices, j], tb[indices, j]
        t_tol = tol / domain_lengths[j]
        if np.any(hi - lo <= t_tol):
            raise RuntimeError("Outer boundary has an incorrect orientation or degenerate edges")
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
        if (abs(lo[0]) > t_tol or abs(hi[-1] - 1) > t_tol
                or np.any(np.abs(lo[1:] - hi[:-1]) > 2 * t_tol)):
            raise RuntimeError(f"CCW boundary side {j + 1} has coverage gaps or overlaps")
    for corner in corners:
        if not np.any(np.linalg.norm(xyz[boundary_ids, :2] - corner, axis=1) <= tol):
            raise RuntimeError("An input corner is not an outer-boundary mesh node")
    graph = coo_matrix((np.ones(len(unique_edges), dtype=np.int8),
                        (unique_edges // len(xyz), unique_edges % len(xyz))), shape=(len(xyz), len(xyz)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise RuntimeError("The mesh is disconnected")
    if len(xyz) - len(unique_edges) + len(triangles) != 1:
        raise RuntimeError("Mesh topology is not a single disk without holes (Euler check failed)")
    area, perimeter = float(areas.sum()), float(np.linalg.norm(b - a, axis=1).sum())
    expected_area, expected_perimeter = polygon_area(corners), polygon_perimeter(corners)
    if not math.isclose(area, expected_area, rel_tol=1e-9, abs_tol=8 * tol * expected_perimeter):
        raise RuntimeError("Total triangle area does not match the input domain area")
    if not math.isclose(perimeter, expected_perimeter, rel_tol=1e-9, abs_tol=8 * tol * len(a)):
        raise RuntimeError("Total outer-boundary length does not match the input domain perimeter")
    angles = np.degrees(np.arctan2(cross, -np.sum(edges * following, axis=2)))
    quality = np.clip(4 * math.sqrt(3) * areas / np.sum(lengths ** 2, axis=1), 0., 1.)
    first = int(np.argmin(np.linalg.norm(xyz[loop, :2] - corners[0], axis=1)))
    loop = loop[first:] + loop[:first]
    side_by_start = dict(zip(a_ids.tolist(), side.tolist()))
    return TriangleMeshStats(
        len(xyz), len(triangles), area, perimeter, float(areas.min()),
        float(lengths.min()), float(lengths.max()), float(angles.min()), float(angles.max()),
        float(quality.min()), float(quality.max()), tuple(loop), tuple(side_by_start[i] for i in loop))


def radial_target_size(distance: np.ndarray, *, background_size: float, refined_size: float,
                       refinement_radius: float, transition_radius: float) -> np.ndarray:
    return np.interp(distance, [refinement_radius, transition_radius], [refined_size, background_size])


def select_seed_points(controls: np.ndarray, corners: np.ndarray, spacing: float) -> np.ndarray:
    """Embed only sparse interior seeds to capture the local size field; all controls still define the field."""
    margin = boundary_projections(controls, corners)[0].min(axis=1)
    chosen, cells = [], {}
    for index in np.flatnonzero(margin > spacing):
        point = controls[index]
        cx, cy = (int(v) for v in np.floor((point - corners[0]) / spacing))
        near = [j for i in (-1, 0, 1) for k in (-1, 0, 1) for j in cells.get((cx + i, cy + k), [])]
        if all(np.linalg.norm(point - controls[j]) >= spacing for j in near):
            chosen.append(int(index))
            cells.setdefault((cx, cy), []).append(int(index))
    return np.asarray(chosen, dtype=np.int64)


def _extract_surface_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Check all 2D elements; do not ignore other types, elements on other surfaces, or auxiliary-node references."""
    import gmsh

    types, _, connectivity = gmsh.model.mesh.getElements(2)
    if not len(types) or any(int(t) != 2 for t in types):
        raise RuntimeError(f"All 2D elements must be type 2; actual types: {list(types)}")
    blocks = [np.asarray(block, dtype=np.int64) for block in connectivity]
    if not blocks or any(block.size % 3 for block in blocks) or not any(block.size for block in blocks):
        raise RuntimeError("2D triangle connectivity is empty or has an invalid format")
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    order = np.argsort(tags)
    tags = tags[order]
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[order]
    if len(np.unique(tags)) != len(tags):
        raise RuntimeError("Gmsh returned duplicate node tags")
    connectivity_flat = np.concatenate(blocks)
    used = np.unique(connectivity_flat)
    positions = np.searchsorted(tags, used)
    if np.any(positions >= len(tags)) or not np.array_equal(tags[positions], used):
        raise RuntimeError("2D elements reference nonexistent nodes")
    triangles = np.searchsorted(used, connectivity_flat).reshape(-1, 3).astype(np.int64)
    return coordinates[positions].copy(), triangles


def _generate_triangle_attempt(corners: np.ndarray, controls: np.ndarray, seeds: np.ndarray, *,
                               background_size: float, refined_size: float, refinement_radius: float,
                               transition_radius: float, algorithm: int) -> tuple[np.ndarray, np.ndarray]:
    import gmsh

    gmsh.model.add("radial_triangle_mesh")
    try:
        origin = corners[0]
        geo = gmsh.model.geo
        tags = [geo.addPoint(float(x), float(y), 0.) for x, y in corners - origin]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        corner_tags = dict(zip(map(tuple, corners), tags))
        control_tags = []
        for point in controls:
            tag = corner_tags.get(tuple(point))
            if tag is None:
                tag = geo.addPoint(float(point[0] - origin[0]), float(point[1] - origin[1]), 0.)
            control_tags.append(tag)
        geo.synchronize()
        if len(seeds):
            gmsh.model.mesh.embed(0, [control_tags[i] for i in seeds], 2, surface)
        field = gmsh.model.mesh.field
        if control_tags:
            distance = field.add("Distance")
            field.setNumbers(distance, "PointsList", control_tags)
            size_field = field.add("Threshold")
            for key, value in (("InField", distance), ("SizeMin", refined_size), ("SizeMax", background_size),
                               ("DistMin", refinement_radius), ("DistMax", transition_radius),
                               ("Sigmoid", 0), ("StopAtDistMax", 0)):
                field.setNumber(size_field, key, value)
        else:
            size_field = field.add("MathEval")
            field.setString(size_field, "F", f"{background_size:.17g}")
        field.setAsBackgroundMesh(size_field)
        for key, value in (("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                           ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeMin", 0),
                           ("Mesh.MeshSizeMax", 1e22), ("Mesh.MeshSizeFactor", 1),
                           ("Mesh.Algorithm", algorithm), ("Mesh.RecombineAll", 0),
                           ("Mesh.SubdivisionAlgorithm", 0), ("Mesh.ElementOrder", 1)):
            gmsh.option.setNumber(key, value)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:  # Gmsh's generation API raises generic Exception instances.
            raise RuntimeError(f"Gmsh triangulation failed: {exc}") from exc
        xyz, triangles = _extract_surface_mesh()
        xyz[:, :2] += origin
        xyz = _snap_to_boundary(xyz, corners)
        p = xyz[triangles, :2] - xyz[triangles[:, :1], :2]
        negative = _cross(p[:, 1], p[:, 2]) < 0
        triangles[negative] = triangles[negative][:, [0, 2, 1]]
        return xyz, triangles
    finally:
        gmsh.model.remove()


def read_msh22_triangles(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Strictly read this generator's MSH subset; do not ignore any elements or extra content."""
    with Path(path).open(encoding="ascii") as stream:
        def expect(text: str) -> None:
            if stream.readline().strip() != text:
                raise RuntimeError(f"Output format validation failed; expected {text}")

        def count() -> int:
            value = int(stream.readline())
            if value <= 0:
                raise RuntimeError("MSH node/element counts must be positive")
            return value

        expect("$MeshFormat")
        expect("2.2 0 8")
        expect("$EndMeshFormat")
        expect("$PhysicalNames")
        expect("1")
        expect('2 1 "Domain2D"')
        expect("$EndPhysicalNames")
        expect("$Nodes")
        xyz = np.empty((count(), 3), dtype=np.float64)
        for i in range(len(xyz)):
            parts = stream.readline().split()
            if len(parts) != 4 or int(parts[0]) != i + 1:
                raise RuntimeError("Invalid MSH node numbering or format")
            xyz[i] = [float(v) for v in parts[1:]]
        expect("$EndNodes")
        expect("$Elements")
        triangles = np.empty((count(), 3), dtype=np.int64)
        for i in range(len(triangles)):
            parts = [int(v) for v in stream.readline().split()]
            if len(parts) != 8 or parts[:5] != [i + 1, 2, 2, 1, 1]:
                raise RuntimeError("MSH contains elements other than type 2 or has an invalid element format")
            triangles[i] = np.array(parts[5:]) - 1
        expect("$EndElements")
        if stream.read().strip():
            raise RuntimeError("MSH contains unexpected extra content")
    return _validate_triangle_arrays(xyz, triangles)


def write_msh22_triangles(output: Path | str, xyz: np.ndarray, triangles: np.ndarray,
                          corners: np.ndarray, *, report: dict | None = None) -> None:
    """Atomically commit the mesh after read-back validation; commit any report first, using a hash to detect interrupted updates."""
    validate_triangle_mesh(xyz, triangles, corners)
    output = Path(output)
    xyz, triangles = _validate_triangle_arrays(xyz, triangles)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary, temporary_report = None, None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n", dir=output.parent,
                                         prefix=f".{output.stem}.", suffix=".msh", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n2 1 "Domain2D"\n$EndPhysicalNames\n$Nodes\n')
            stream.write(f"{len(xyz)}\n")
            for i, (x, y, z) in enumerate(xyz, 1):
                stream.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
            stream.write(f"$EndNodes\n$Elements\n{len(triangles)}\n")
            for i, t in enumerate(triangles + 1, 1):
                stream.write(f"{i} 2 2 1 1 {t[0]} {t[1]} {t[2]}\n")
            stream.write("$EndElements\n")
            stream.flush()
            os.fsync(stream.fileno())
        saved_xyz, saved_triangles = read_msh22_triangles(temporary)
        if not np.array_equal(saved_xyz, xyz) or not np.array_equal(saved_triangles, triangles):
            raise RuntimeError("MSH coordinates or connectivity changed after writing")
        validate_triangle_mesh(saved_xyz, saved_triangles, corners)
        if report is not None:
            digest = hashlib.sha256()
            with temporary.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            content = json.dumps({**report, "msh_sha256": digest.hexdigest()},
                                 ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                                             prefix=f".{output.stem}.", suffix=".json", delete=False) as stream:
                temporary_report = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_report, output.with_suffix(".report.json"))
        # Commit the main mesh last so report write/replace failures cannot overwrite an existing mesh.
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)


def _mesh_report(summary: MeshSummary, input_corners: np.ndarray, sizes: dict) -> dict:
    import gmsh

    stats = summary.stats
    ccw_input = polygon_area(input_corners) > 0
    side_mapping = (1, 2, 3, 4) if ccw_input else (4, 3, 2, 1)
    loop = [i + 1 for i in stats.boundary_loop]
    return {
        "schema_version": 1,
        "output": str(summary.output),
        "gmsh_version": gmsh.__version__,
        "format": "MSH 2.2 ASCII; Domain2D; type 2 only",
        "array_contract": {"coordinates": "float64 (N,3), Z=0; original XY",
                           "connectivity": "integer (M,3), 0-based in memory; CCW from +Z",
                           "file_ids": "continuous 1-based node and element IDs"},
        "domain_corners_input": input_corners.tolist(),
        "domain_corners_ccw": summary.domain_corners,
        "domain_area": polygon_area(input_corners) * (1 if ccw_input else -1),
        "domain_perimeter": polygon_perimeter(input_corners),
        "control_point_count": summary.control_point_count,
        "seed_point_count": summary.seed_point_count,
        "target_sizes": sizes,
        "gmsh_algorithm": summary.gmsh_algorithm,
        "attempt_failures": summary.attempt_failures,
        "mesh": {key: value for key, value in asdict(stats).items() if not key.startswith("boundary_")},
        "boundary": {
            "node_id_base": 1,
            "input_side_id_base": 1,
            "loop_node_ids": loop,
            "directed_edges": [[a, b] for a, b in zip(loop, loop[1:] + loop[:1])],
            "input_side_ids": [side_mapping[i] for i in stats.boundary_side_ids],
            "follows_input_direction": ccw_input,
            "input_sides": [{"side_id": i + 1, "start_xy": input_corners[i].tolist(),
                             "end_xy": input_corners[(i + 1) % 4].tolist()} for i in range(4)],
        },
        "definitions": {
            "boundary": "CCW loop starts at input corner 1; last node is not repeated; arrays align by edge",
            "size_field": "nearest-control XY Euclidean distance; constant inner/outer sizes, linear transition; "
                          "empty controls use background_size everywhere; actual edges need not equal targets",
            "seeds": "interior control subset, boundary distance > 6*refined_size and mutual spacing >= "
                     "6*refined_size; helps capture local refinement; other controls need not be mesh nodes",
            "quality": "q=4*sqrt(3)*A/(a*a+b*b+c*c); equilateral=1, degeneracy approaches 0",
        },
        "downstream": "2_2d_ext_3d.py reads this MSH/report pair and generates layered conforming Tet4; "
                      "4_find_DirBC.py accepts its paired 3D outputs for boundary nodes. "
                      "Later Hex8-only facies/BC/well stages are NOT compatible; independent 3D "
                      "quality/discretization checks remain necessary.",
    }


def create_mesh(domain_corners_file: Path | str = DOMAIN_CORNERS_FILE,
                control_points_file: Path | str = CONTROL_POINTS_FILE,
                output_file: Path | str | None = None, *, background_size: float = BACKGROUND_SIZE,
                refined_size: float = REFINED_SIZE, refinement_radius: float = REFINEMENT_RADIUS,
                transition_radius: float = TRANSITION_RADIUS, max_triangles: int = MAX_TRIANGLES,
                verbose: bool = True) -> MeshSummary:
    """Generate, strictly validate, and save the triangular mesh and boundary report without modifying external active Gmsh sessions."""
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        if exc.name != "gmsh":
            raise
        raise RuntimeError("Generating a 2D mesh requires gmsh; reading/validating an existing 2D mesh does not") from exc

    validate_sizes(background_size, refined_size, refinement_radius, transition_radius)
    if isinstance(max_triangles, bool) or not isinstance(max_triangles, int) or max_triangles < 1:
        raise ValueError("max-triangles must be a positive integer")
    output = Path(OUTPUT_FILE if output_file is None else output_file).resolve()
    if output.suffix.lower() != ".msh":
        raise ValueError("The output path must end in .msh")
    inputs = {Path(domain_corners_file).resolve(), Path(control_points_file).resolve()}
    if {output, output.with_suffix(".report.json").resolve()} & inputs:
        raise ValueError("The mesh or accompanying report must not overwrite input files")
    input_corners = _read_xy_csv(domain_corners_file)
    try:
        corners = validate_domain_corners(input_corners)
    except ValueError as exc:
        raise ValueError(f"{domain_corners_file}: {exc}") from exc
    controls = load_control_points(control_points_file)
    outside = ~points_in_domain(controls, corners)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} refinement points lie outside the actual quadrilateral boundary; first coordinate: {controls[outside][0].tolist()}")
    if refined_size <= 100 * geometry_tolerance(corners):
        raise ValueError("refined_size is too small relative to global-coordinate precision")
    if gmsh.isInitialized():
        raise RuntimeError("A Gmsh session is already active; finalize it first to avoid modifying other models")
    sizes = dict(background_size=float(background_size), refined_size=float(refined_size),
                 refinement_radius=float(refinement_radius), transition_radius=float(transition_radius))
    seeds = select_seed_points(controls, corners, SEED_SPACING_FACTOR * refined_size)
    try:
        gmsh.initialize([], readConfigFiles=False)
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.option.setNumber("General.NumThreads", 1)
        failures = []
        for algorithm in MESH_ATTEMPTS:
            name = _ALGORITHM_NAMES[algorithm]
            try:
                xyz, triangles = _generate_triangle_attempt(corners, controls, seeds, algorithm=algorithm, **sizes)
                stats = validate_triangle_mesh(xyz, triangles, corners)
            except RuntimeError as exc:
                failures.append(f"{name}: {exc}")
                if verbose:
                    print(f"Rejected attempt: {failures[-1]}")
                continue
            if stats.triangle_count > max_triangles:
                raise ValueError(f"Actual triangle count {stats.triangle_count:,} exceeds max-triangles={max_triangles:,}")
            summary = MeshSummary(
                output, output.with_suffix(".report.json"),
                tuple(tuple(float(v) for v in p) for p in corners),
                len(controls), len(seeds), stats, name, tuple(failures))
            write_msh22_triangles(output, xyz, triangles, corners, report=_mesh_report(summary, input_corners, sizes))
            return summary
        raise RuntimeError("All Gmsh triangulation attempts failed validation; nothing was exported: " + "; ".join(failures))
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--domain-corners", type=Path, default=DOMAIN_CORNERS_FILE, help="X,Y CSV with four corners in boundary order")
    parser.add_argument("--control-points", type=Path, default=CONTROL_POINTS_FILE, help="X,Y refinement-point CSV; a header-only file is allowed")
    parser.add_argument("--output", type=Path, default=None, help="Default: out/mesh_2d_tri.msh, with a matching .report.json")
    parser.add_argument("--background-size", type=float, default=BACKGROUND_SIZE)
    parser.add_argument("--refined-size", type=float, default=REFINED_SIZE)
    parser.add_argument("--refinement-radius", type=float, default=REFINEMENT_RADIUS)
    parser.add_argument("--transition-radius", type=float, default=TRANSITION_RADIUS)
    parser.add_argument("--max-triangles", type=int, default=MAX_TRIANGLES, help="Maximum actual triangle count")
    parser.add_argument("--quiet", action="store_true", help="Suppress Gmsh progress; still print final statistics and failed attempts")
    args = parser.parse_args(argv)
    try:
        result = create_mesh(args.domain_corners, args.control_points, args.output,
                             background_size=args.background_size, refined_size=args.refined_size,
                             refinement_radius=args.refinement_radius, transition_radius=args.transition_radius,
                             max_triangles=args.max_triangles, verbose=not args.quiet)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    stats = result.stats
    import gmsh

    print("\n2D TRIANGULAR MESH REPORT")
    print(f"Gmsh API: {gmsh.__version__}; MSH 2.2 ASCII; type 2 only; Z=0; CCW from +Z")
    print(f"Domain corners (CCW): {result.domain_corners}")
    print(f"Control points: {result.control_point_count}; embedded sparse seeds: {result.seed_point_count}")
    print(f"Target sizes: {args.refined_size:g} .. {args.background_size:g}; refinement radius: "
          f"{args.refinement_radius:g}; transition ends: {args.transition_radius:g}")
    print(f"Gmsh: {result.gmsh_algorithm}; rejected attempts: {len(result.attempt_failures)}")
    for failure in result.attempt_failures:
        print(f"  {failure}")
    print(f"Nodes: {stats.node_count}; triangles: {stats.triangle_count}")
    print(f"Area: {stats.area:.12g}; perimeter: {stats.perimeter:.12g}; min triangle area: {stats.min_area:.8g}")
    print(f"Actual edge length: {stats.min_edge_length:.8g} .. {stats.max_edge_length:.8g}")
    print(f"Interior angles: {stats.min_corner_angle:.6g} .. {stats.max_corner_angle:.6g} degrees")
    print(f"q=4*sqrt(3)*A/sum(edge^2): {stats.min_quality:.8g} .. {stats.max_quality:.8g}; equilateral=1")
    print("Target sizes guide the size field; actual edge lengths need not equal the targets.")
    print("2_2d_ext_3d.py consumes this MSH/report pair for conforming Tet4. Later Hex8-only stages are "
          "NOT compatible; 2D acceptance does not guarantee 3D quality or discretization suitability.")
    print(f"Validated output: {result.output}")
    print(f"Boundary/quality report (1-based MSH IDs, verify msh_sha256): {result.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
