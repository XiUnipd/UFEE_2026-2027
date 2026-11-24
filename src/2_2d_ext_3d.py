"""Upstream triangle MSH + companion JSON + top/bottom XYZ -> conforming Tet4 MSH and VTK.

Output contains only type 4 / Domain3D, with contiguous 1-based node/element IDs;
Also write a same-stem ASCII legacy VTK 2.0 file containing only VTK_TETRA=10,
with 0-based connectivity and the same row order as MSH.

Requires numpy and scipy, not gmsh; verify both mesh hashes before paired use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import QhullError, cKDTree


def _load_upstream():
	spec = importlib.util.spec_from_file_location(
		"_tetra_upstream_tri_mesh", Path(__file__).with_name("1_create_tri_mesh_2d.py"))
	if spec is None or spec.loader is None:
		raise RuntimeError("Cannot load the upstream triangular mesh module")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


_tri = _load_upstream()

# ======================== Editable parameters ========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
MESH_2D_FILE = PROJECT_DIR / "out" / "mesh_2d_tri.msh"
TOP_XYZ_FILE = PROJECT_DIR / "data" / "dem.xyz"
BOTTOM_XYZ_FILE = PROJECT_DIR / "data" / "bottom.xyz"
OUTPUT_FILE = PROJECT_DIR / "out" / "mesh_3d_tet.msh"

DZ = 20.0                 # Maximum layer thickness (m); shared layer count, equal thickness within each column
MAX_TETRAHEDRA = 4_000_000  # Maximum final tetrahedron count, not prism count
EXTRAPOLATION_MODE = "nearest"              # "error": reject exterior nodes; "nearest": distance-limited nearest-neighbour fill
TOP_MAX_EXTRAPOLATION_DISTANCE = 100     # Top fill distance limit (XY units); nearest mode requires a finite positive value
BOTTOM_MAX_EXTRAPOLATION_DISTANCE = 100  # Bottom fill distance limit (XY units); nearest mode requires a finite positive value
WRITE_CHUNK_SIZE = 100_000
# Shared layer count is ceil(maximum column thickness / DZ); thinner columns have proportionally thinner layers.
# ================================================================


def file_sha256(path: Path | str) -> str:
	digest = hashlib.sha256()
	with Path(path).open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _check_gmsh_session() -> None:
	gmsh = sys.modules.get("gmsh")
	if gmsh is not None and gmsh.isInitialized():
		raise RuntimeError("An active Gmsh session already exists; end it first")


def _reject_json_constant(value):
	raise ValueError(f"JSON does not allow non-finite values: {value}")


def _finite_json_float(value):
	number = float(value)
	if not math.isfinite(number):
		raise ValueError(f"JSON number exceeds the finite floating-point range: {value}")
	return number


def _match_metadata(actual, expected, label: str) -> None:
	"""Strictly validate ID/boolean types; coordinates may be JSON integers or floats."""
	if isinstance(expected, dict):
		if not isinstance(actual, dict):
			raise ValueError(f"{label} must be a JSON object")
		for key, value in expected.items():
			if key not in actual:
				raise ValueError(f"{label} is missing {key}")
			_match_metadata(actual[key], value, f"{label}.{key}")
	elif isinstance(expected, (list, tuple)):
		if not isinstance(actual, list) or len(actual) != len(expected):
			raise ValueError(f"{label} has an invalid length or type")
		for i, (a, e) in enumerate(zip(actual, expected)):
			_match_metadata(a, e, f"{label}[{i}]")
	elif isinstance(expected, float):
		if type(actual) not in (int, float) or not math.isfinite(actual) or actual != expected:
			raise ValueError(f"{label} does not match the actual mesh")
	elif type(actual) is not type(expected) or actual != expected:
		raise ValueError(f"{label} does not match the actual mesh")


@dataclass(frozen=True)
class PlanarMesh:
	path: Path
	report_path: Path
	msh_sha256: str
	report_sha256: str
	xyz: np.ndarray
	triangles: np.ndarray
	corners_input: np.ndarray
	corners_ccw: np.ndarray
	boundary_loop: np.ndarray  # 0-based; preserve the original upstream node/element row order
	input_side_ids: np.ndarray  # Original CSV side IDs, 1-based

	@property
	def xy(self) -> np.ndarray:
		return self.xyz[:, :2]


def read_planar_mesh(path: Path | str, report_path: Path | str | None = None) -> PlanarMesh:
	"""Reuse strict upstream reading/validation; hashes do not replace geometry/boundary checks."""
	_check_gmsh_session()
	path = Path(path).resolve()
	report_path = Path(report_path).resolve() if report_path is not None else path.with_suffix(".report.json")
	report_bytes = report_path.read_bytes()
	report = json.loads(report_bytes.decode("utf-8"), parse_constant=_reject_json_constant,
						parse_float=_finite_json_float)
	_match_metadata(report, {"schema_version": 1}, "report")
	digest = file_sha256(path)
	_match_metadata(report, {"msh_sha256": digest}, "report")
	xyz, triangles = _tri.read_msh22_triangles(path)
	if file_sha256(path) != digest:
		raise ValueError("The upstream MSH was modified while being read")
	if "domain_corners_input" not in report:
		raise ValueError("report is missing domain_corners_input")
	corners_input = np.asarray(report["domain_corners_input"], dtype=np.float64)
	corners_ccw = _tri.validate_domain_corners(corners_input)
	stats = _tri.validate_triangle_mesh(xyz, triangles, corners_ccw)
	ccw = _tri.polygon_area(corners_input) > 0
	mapping = (1, 2, 3, 4) if ccw else (4, 3, 2, 1)
	loop = [i + 1 for i in stats.boundary_loop]
	side_ids = [mapping[i] for i in stats.boundary_side_ids]
	expected_boundary = {
		"node_id_base": 1, "input_side_id_base": 1, "loop_node_ids": loop,
		"directed_edges": [[a, b] for a, b in zip(loop, loop[1:] + loop[:1])],
		"input_side_ids": side_ids, "follows_input_direction": bool(ccw),
		"input_sides": [{"side_id": i + 1, "start_xy": corners_input[i].tolist(),
						 "end_xy": corners_input[(i + 1) % 4].tolist()} for i in range(4)],
	}
	_match_metadata(report, {
		"domain_corners_input": corners_input.tolist(), "domain_corners_ccw": corners_ccw.tolist(),
		"domain_area": abs(float(_tri.polygon_area(corners_input))),
		"domain_perimeter": float(_tri.polygon_perimeter(corners_input)),
		"mesh": {"node_count": len(xyz), "triangle_count": len(triangles)},
		"boundary": expected_boundary,
	}, "report")
	return PlanarMesh(path, report_path, digest, hashlib.sha256(report_bytes).hexdigest(),
					  xyz, triangles, corners_input, corners_ccw, np.asarray(stats.boundary_loop),
					  np.asarray(side_ids, dtype=np.int64))


EXTRAPOLATION_MODES = ("error", "nearest")


def _xy_text(point: np.ndarray) -> str:
	return "[" + ", ".join(f"{value:.12g}" for value in np.asarray(point, dtype=float)) + "]"


def validate_extrapolation(mode: str, max_distance: float | None,
						   label: str = "max-extrapolation-distance") -> float | None:
	"""Validate extrapolation mode and fill distance; nearest requires a finite positive limit."""
	if not isinstance(mode, str) or mode not in EXTRAPOLATION_MODES:
		raise ValueError(f"extrapolation-mode must be one of {' / '.join(EXTRAPOLATION_MODES)}; got {mode!r}")
	if max_distance is not None:
		if (isinstance(max_distance, (bool, np.bool_))
				or not isinstance(max_distance, (int, float, np.integer, np.floating))
				or not math.isfinite(max_distance) or max_distance <= 0):
			raise ValueError(f"{label} must be a finite positive number (in XY units); got {max_distance!r}")
		max_distance = float(max_distance)
	if mode == "nearest" and max_distance is None:
		raise ValueError(f"nearest mode requires an explicit {label} (finite and positive, in XY units); unlimited extrapolation is not allowed")
	return max_distance


@dataclass(frozen=True)
class SurfaceSample:
	"""Surface values at 2D mesh nodes and nearest-neighbour fill records."""
	path: Path
	heights: np.ndarray
	extrapolated: np.ndarray  # bool: True marks a node filled by nearest neighbour
	distances: np.ndarray     # XY distance to the nearest original sample for filled nodes; NaN for interpolated nodes
	max_allowed_distance: float | None = None
	sha256: str | None = None

	@property
	def node_count(self) -> int:
		return len(self.heights)

	@property
	def extrapolated_count(self) -> int:
		return int(np.count_nonzero(self.extrapolated))

	@property
	def interpolated_count(self) -> int:
		return self.node_count - self.extrapolated_count

	@property
	def extrapolated_ratio(self) -> float:
		return self.extrapolated_count / self.node_count if self.node_count else 0.0

	@property
	def max_distance(self) -> float:
		return float(np.nanmax(self.distances)) if self.extrapolated_count else 0.0


def sample_surface(path: Path | str, xy: np.ndarray, *, mode: str = "error",
				   max_distance: float | None = None,
				   label: str = "max-extrapolation-distance") -> SurfaceSample:
	"""Use bilinear interpolation for regular XYZ grids, linear triangulation for scattered data.

	Preserve all finite interpolated values. For nodes outside the interpolation domain
	(the regular grid's bounding rectangle or the scattered samples' convex hull),
	error mode raises; nearest mode uses the nearest original sample's Z within the distance limit.
	Non-finite values inside the interpolation domain are errors in every mode.
	"""
	max_distance = validate_extrapolation(mode, max_distance, label)
	path = Path(path)
	digest = file_sha256(path)
	xy = np.asarray(xy, dtype=float)
	if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
		raise ValueError("Interpolation coordinates must be a finite N-by-2 XY array")
	data = np.loadtxt(path, dtype=float, ndmin=2, encoding="utf-8-sig")
	if data.shape[1] != 3 or len(data) < 3 or not np.isfinite(data).all():
		raise ValueError(f"{path}: at least three rows of finite X Y Z values are required")
	if len(np.unique(data[:, :2], axis=0)) != len(data):
		raise ValueError(f"{path}: XYZ contains duplicate XY coordinates")
	xs, xi = np.unique(data[:, 0], return_inverse=True)
	ys, yi = np.unique(data[:, 1], return_inverse=True)
	if len(xs) < 2 or len(ys) < 2:
		raise ValueError(f"{path}: the surface must have a nonzero extent in both X and Y")
	origin = data[:, :2].min(axis=0)  # Translate to reduce roundoff for large projected coordinates; distances are unchanged
	local = xy - origin
	heights = np.full(len(xy), np.nan)
	if len(xs) * len(ys) == len(data):
		grid = np.empty((len(ys), len(xs)))
		grid[yi, xi] = data[:, 2]
		interpolator = RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=np.nan)
		inside = ((xy[:, 0] >= xs[0]) & (xy[:, 0] <= xs[-1])
				  & (xy[:, 1] >= ys[0]) & (xy[:, 1] <= ys[-1]))
		if len(xy):
			heights = np.asarray(interpolator(xy[:, ::-1]), dtype=float).reshape(-1)
	else:
		try:
			interpolator = LinearNDInterpolator(data[:, :2] - origin, data[:, 2])
			# Use the interpolator's Delaunay triangulation and tolerance to test convex-hull membership.
			inside = interpolator.tri.find_simplex(local) >= 0
			if len(xy):
				heights = np.asarray(interpolator(local), dtype=float).reshape(-1)
		except QhullError as exc:
			raise ValueError(f"{path}: scattered samples cannot form a valid 2D surface") from exc
	finite = np.isfinite(heights)
	anomalous = ~finite & inside
	if anomalous.any():
		raise ValueError(
			f"{path}: {int(anomalous.sum())} nodes inside the interpolation domain have non-finite elevations; "
			f"fill will not mask this interpolation error; first XY={_xy_text(xy[anomalous][0])}"
		)
	missing = ~finite  # All remaining missing nodes are outside the domain; preserve finite interpolated values
	distances = np.full(len(xy), np.nan)
	if missing.any():
		if mode == "error":
			raise ValueError(
				f"{path}: {int(missing.sum())} mesh nodes are not covered; extrapolation is prohibited; first XY={_xy_text(xy[missing][0])}"
				" (for controlled fill, use --extrapolation-mode nearest and set the maximum fill distance)"
			)
		assert max_distance is not None  # validate_extrapolation guarantees a distance limit in nearest mode
		tree = cKDTree(data[:, :2] - origin)
		nearest_distance, nearest_index = tree.query(local[missing], k=1)
		nearest_distance = np.asarray(nearest_distance, dtype=float)
		too_far = nearest_distance > max_distance
		if too_far.any():
			far_xy = xy[missing][too_far]
			worst = int(np.argmax(nearest_distance))
			raise ValueError(
				f"{path}: {int(too_far.sum())} nodes outside the interpolation domain have XY distances to the nearest original sample exceeding "
				f"{label}={max_distance:.12g}; maximum distance {nearest_distance[worst]:.12g}"
				f" (XY={_xy_text(xy[missing][worst])}); first over-limit XY={_xy_text(far_xy[0])}. "
				"Check XYZ coverage and coordinate-system consistency (distances use XY units)"
			)
		heights[missing] = data[nearest_index, 2]
		distances[missing] = nearest_distance
	if file_sha256(path) != digest:
		raise ValueError(f"{path}: XYZ was modified during interpolation")
	return SurfaceSample(path, heights, missing, distances, max_distance, digest)


def interpolate_surface(path: Path | str, xy: np.ndarray, *, mode: str = "error",
						max_distance: float | None = None) -> np.ndarray:
	"""Return node elevations; the default error mode prohibits extrapolation."""
	return sample_surface(path, xy, mode=mode, max_distance=max_distance).heights


def uniform_fractions(thickness: float, dz: float = DZ, max_layers: int = 100_000) -> np.ndarray:
	"""Choose a shared layer count from maximum column thickness; return uniform top-to-bottom fractions 0..1."""
	for name, value in (("thickness", thickness), ("dz", dz)):
		if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
				or not 0 < value <= sys.float_info.max or float(value) == 0):
			raise ValueError(f"{name} must be a finite positive number representable as float64")
	if isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 1:
		raise ValueError("Layer/tetrahedron count limit is too small; at least one layer is required (three tetrahedra per triangle)")
	ratio = float(thickness) / float(dz)
	if not math.isfinite(ratio) or ratio > max_layers:
		raise ValueError("Estimated layer or tetrahedron count exceeds the limit; increase layer thickness or max-elements")
	return np.linspace(0., 1., max(1, math.ceil(ratio)) + 1)


_TET_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
_TET_EDGES = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))


def tetrahedron_metrics(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""D=det([p1-p0,p2-p0,p3-p0]) using column vectors, consistent with Gmsh Tet4 positive orientation."""
	points = np.asarray(points, dtype=np.float64)
	if points.ndim != 3 or points.shape[1:] != (4, 3) or not len(points) or not np.isfinite(points).all():
		raise ValueError("Tet4 coordinates must be a nonempty finite (T,4,3) array")
	e = points[:, 1:] - points[:, :1]
	cofactors = np.stack((np.cross(e[:, 1], e[:, 2]), np.cross(e[:, 2], e[:, 0]),
						 np.cross(e[:, 0], e[:, 1])), axis=1)
	determinant = np.einsum("ij,ij->i", e[:, 0], cofactors[:, 0])
	# Bound first-order determinant error using per-axis coordinate ULPs, not large XY scales that penalize thin Z layers.
	ulp = np.max(np.abs(np.spacing(np.abs(points))), axis=1)
	roundoff = 8 * np.sum(np.abs(cofactors) * ulp[:, None, :], axis=(1, 2))
	relative = 1e-12 * np.prod(np.linalg.norm(e, axis=2), axis=1)
	if not np.isfinite(determinant).all() or np.any(determinant <= np.maximum(roundoff, relative)):
		raise ValueError("Tetrahedra have non-positive volume/Jacobian or numerically near-degenerate geometry")
	lengths = np.linalg.norm(points[:, _TET_EDGES[:, 0]] - points[:, _TET_EDGES[:, 1]], axis=2)
	volume = determinant / 6
	quality = 12 * np.power(3 * volume, 2 / 3) / np.sum(lengths ** 2, axis=1)
	return determinant, lengths, np.clip(quality, 0., 1.)


def split_triangle_prisms(triangles: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
	"""Split by global 2D node order; return three 0-based Tet4 per column in upper+lower."""
	triangles = np.asarray(triangles)
	upper, lower = np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)
	if (upper.ndim != 2 or upper.shape[1] != 3 or upper.shape != lower.shape
			or not np.isfinite(upper).all() or not np.isfinite(lower).all()
			or not np.array_equal(upper[:, :2], lower[:, :2]) or np.any(upper[:, 2] <= lower[:, 2])):
		raise ValueError("Upper/lower layers must have identical finite XY and upper.z > lower.z at every node")
	if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
			or not np.issubdtype(triangles.dtype, np.integer)
			or triangles.min() < 0 or triangles.max() >= len(upper)):
		raise ValueError("Invalid triangle connectivity")
	a, b, c = np.sort(triangles, axis=1).T
	n = len(upper)
	tets = np.stack((np.column_stack((a, b, c, c + n)),
					np.column_stack((a, b, b + n, c + n)),
					np.column_stack((a, a + n, b + n, c + n))), axis=1)
	p = upper[np.column_stack((a, b, c)), :2]
	area_sign = _tri._cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
	# Layers run downward; template D signs are -S,+S,-S. Orient solely by this combinatorial rule.
	swap = (area_sign[:, None] * np.array([-1, 1, -1])) < 0
	tets[swap] = tets[swap][:, [0, 2, 1, 3]]
	tets = tets.reshape(-1, 4).astype(np.int64)
	tetrahedron_metrics(np.vstack((upper, lower))[tets])
	return tets


@dataclass
class LayeredTetraGeometry:
	"""Shared-layer geometry recoverable downstream from written meshes without XYZ interpolation records."""

	source: PlanarMesh
	top: np.ndarray
	bottom: np.ndarray
	fractions: np.ndarray
	parameters: dict

	@property
	def layer_count(self) -> int:
		return len(self.fractions) - 1

	@property
	def node_count(self) -> int:
		return len(self.source.xy) * len(self.fractions)

	@property
	def tetrahedron_count(self) -> int:
		return 3 * len(self.source.triangles) * self.layer_count

	def coordinates(self, layer: int) -> np.ndarray:
		if not 0 <= layer <= self.layer_count:
			raise IndexError("Node layer index out of range")
		if layer == 0:
			z = self.top
		elif layer == self.layer_count:
			z = self.bottom
		else:
			z = self.top - (self.top - self.bottom) * self.fractions[layer]
		return np.column_stack((self.source.xy, z))

	def tetrahedra(self, layer: int) -> np.ndarray:
		if not 0 <= layer < self.layer_count:
			raise IndexError("Tetrahedron layer-segment index out of range")
		tets = split_triangle_prisms(self.source.triangles, self.coordinates(layer), self.coordinates(layer + 1))
		return tets + layer * len(self.source.xy)

	def column_volumes(self) -> np.ndarray:
		p = self.source.xy[self.source.triangles]
		areas = .5 * _tri._cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
		return areas * (self.top - self.bottom)[self.source.triangles].mean(axis=1)

	def expected_volume(self) -> float:
		return float(self.column_volumes().sum())


@dataclass
class LayeredTetraMesh(LayeredTetraGeometry):
	top_sample: SurfaceSample
	bottom_sample: SurfaceSample


def _crossing_message(xy: np.ndarray, top: np.ndarray, bottom: np.ndarray,
					  top_sample: SurfaceSample, bottom_sample: SurfaceSample) -> str:
	thickness = top - bottom
	bad = np.flatnonzero(~np.isfinite(thickness) | (thickness <= 0))
	first = int(bad[0])
	worst = int(bad[np.nanargmin(thickness[bad])]) if np.isfinite(thickness[bad]).any() else first
	involved = int(np.count_nonzero(top_sample.extrapolated[bad] | bottom_sample.extrapolated[bad]))
	return (
		f"Top/bottom surfaces intersect, are inverted, or have zero thickness: all nodes must satisfy top > bottom; {len(bad)}/{len(xy)} nodes violate this, "
		f"including {involved} involving nearest-neighbour fill; first XY={_xy_text(xy[first])}, "
		f"top={top[first]:.12g}, bottom={bottom[first]:.12g}; "
		f"minimum thickness {thickness[worst]:.12g} at XY={_xy_text(xy[worst])}. "
		"The script does not automatically swap or correct elevations; check top/bottom XYZ or reduce the maximum fill distance"
	)


def build_layered_mesh(mesh_2d: Path | str = MESH_2D_FILE, top_xyz: Path | str = TOP_XYZ_FILE,
					   bottom_xyz: Path | str = BOTTOM_XYZ_FILE, *, dz: float = DZ,
					   max_elements: int = MAX_TETRAHEDRA, extrapolation_mode: str = EXTRAPOLATION_MODE,
					   top_max_distance: float | None = TOP_MAX_EXTRAPOLATION_DISTANCE,
					   bottom_max_distance: float | None = BOTTOM_MAX_EXTRAPOLATION_DISTANCE,
					   mesh_2d_report: Path | str | None = None) -> LayeredTetraMesh:
	if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements <= 0:
		raise ValueError("max-elements must be a positive integer")
	top_label, bottom_label = "top-max-extrapolation-distance", "bottom-max-extrapolation-distance"
	top_max_distance = validate_extrapolation(extrapolation_mode, top_max_distance, top_label)
	bottom_max_distance = validate_extrapolation(extrapolation_mode, bottom_max_distance, bottom_label)
	source = read_planar_mesh(mesh_2d, mesh_2d_report)
	xy = source.xy
	max_layers = min(100_000, max_elements // (3 * len(source.triangles)))
	# Reject before interpolation/generation if even one layer exceeds the budget.
	uniform_fractions(dz, dz, max_layers)
	top_sample = sample_surface(top_xyz, xy, mode=extrapolation_mode,
								max_distance=top_max_distance, label=top_label)
	bottom_sample = sample_surface(bottom_xyz, xy, mode=extrapolation_mode,
								   max_distance=bottom_max_distance, label=bottom_label)
	top = top_sample.heights
	bottom = bottom_sample.heights
	thickness = top - bottom
	if not np.isfinite(thickness).all() or np.any(thickness <= 0):
		raise ValueError(_crossing_message(xy, top, bottom, top_sample, bottom_sample))
	fractions = uniform_fractions(float(thickness.max()), dz, max_layers)
	parameters = dict(dz=float(dz), max_elements=max_elements, extrapolation_mode=extrapolation_mode)
	mesh = LayeredTetraMesh(source, top, bottom, fractions, parameters, top_sample, bottom_sample)
	_validate_layering(mesh)
	return mesh


def _validate_layering(mesh: LayeredTetraGeometry) -> None:
	source = mesh.source
	stats = _tri.validate_triangle_mesh(source.xyz, source.triangles, source.corners_ccw)
	np_loop = np.asarray(stats.boundary_loop)
	mapping = np.array([1, 2, 3, 4]) if _tri.polygon_area(source.corners_input) > 0 else np.array([4, 3, 2, 1])
	if (not np.array_equal(source.boundary_loop, np_loop)
			or not np.array_equal(source.input_side_ids, mapping[list(stats.boundary_side_ids)])):
		raise ValueError("In-memory source boundary mapping is inconsistent")
	n = len(source.xyz)
	if (mesh.top.shape != (n,) or mesh.bottom.shape != (n,) or not np.isfinite(mesh.top).all()
			or not np.isfinite(mesh.bottom).all() or np.any(mesh.top <= mesh.bottom)):
		raise ValueError("Invalid top/bottom surfaces or non-positive thickness")
	f = mesh.fractions
	if (f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all() or f[0] != 0 or f[-1] != 1
			or np.any(np.diff(f) <= 0)):
		raise ValueError("Layer fractions must increase strictly from 0 to 1")
	expected = uniform_fractions(float(np.max(mesh.top - mesh.bottom)), mesh.parameters["dz"])
	if f.shape != expected.shape or not np.allclose(f, expected, rtol=0, atol=8 * np.finfo(float).eps):
		raise ValueError("Layer fractions must match the shared uniform layering defined by dz")
	if mesh.tetrahedron_count > mesh.parameters["max_elements"]:
		raise ValueError("Actual tetrahedron count exceeds max-elements")
	previous = mesh.coordinates(0)[:, 2]
	for k in range(mesh.layer_count):
		current = mesh.coordinates(k + 1)[:, 2]
		dz = previous - current
		ulp = np.maximum(np.abs(np.spacing(previous)), np.abs(np.spacing(current)))
		if not np.isfinite(dz).all() or np.any(dz <= 8 * ulp):
			raise ValueError(f"Layer segment {k} has non-positive thickness or is below float64 elevation resolution")
		if np.any(dz > mesh.parameters["dz"] * (1 + 1e-12) + 8 * ulp):
			raise ValueError("Actual layer thickness exceeds dz")
		previous = current


def _orientation(faces: np.ndarray) -> np.ndarray:
	inversions = ((faces[:, 0] > faces[:, 1]).astype(np.int8)
				  + (faces[:, 0] > faces[:, 2]) + (faces[:, 1] > faces[:, 2]))
	return np.where(inversions % 2, -1, 1)


def _face_topology(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
	"""Validate all triangular faces in one slab; return outward open faces, 0-based owners, and internal-face count."""
	if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
		raise ValueError("Duplicate tetrahedra detected")
	faces = tets[:, _TET_FACES].reshape(-1, 3)
	_, first, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
										return_index=True, return_inverse=True, return_counts=True)
	if np.any(counts > 2):
		raise ValueError("Non-manifold face: shared by three or more tetrahedra")
	directions = np.bincount(inverse, weights=_orientation(faces))
	if np.any((counts == 2) & (directions != 0)):
		raise ValueError("Internal faces are not shared by tetrahedra with opposite orientations")
	order = np.argsort(inverse)
	offsets = np.r_[0, np.cumsum(counts[:-1])]
	a = order[offsets[counts == 2]] // 4
	b = order[offsets[counts == 2] + 1] // 4
	graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(tets), len(tets)))
	if connected_components(graph, directed=False, return_labels=False) != 1:
		raise ValueError("Tetrahedral slab has disconnected volume components")
	opened = first[counts == 1]
	return faces[opened], opened // 4, int(np.count_nonzero(counts == 2))


def _expected_slab_surface(mesh: LayeredTetraGeometry, layer: int) -> np.ndarray:
	n = len(mesh.source.xyz)
	top = mesh.source.triangles + layer * n
	bottom = mesh.source.triangles[:, [0, 2, 1]] + (layer + 1) * n
	u = mesh.source.boundary_loop + layer * n
	v = np.roll(mesh.source.boundary_loop, -1) + layer * n
	sides = np.stack((np.column_stack((u, u + n, v + n)),
					  np.column_stack((u, v + n, v))), axis=1)
	reverse = u > v
	sides[reverse, 0] = np.column_stack((u[reverse], u[reverse] + n, v[reverse]))
	sides[reverse, 1] = np.column_stack((v[reverse], u[reverse] + n, v[reverse] + n))
	return np.vstack((top, bottom, sides.reshape(-1, 3)))


def _surface_owners(faces: np.ndarray, owners: np.ndarray, expected: np.ndarray) -> np.ndarray:
	keys, expected_keys = np.sort(faces, axis=1), np.sort(expected, axis=1)
	order = np.lexsort(keys.T[::-1])
	target = np.lexsort(expected_keys.T[::-1])
	if not np.array_equal(keys[order], expected_keys[target]):
		raise ValueError("Open faces do not match expected layer/side faces: internal cracks, gaps, or inconsistent side diagonals")
	if not np.array_equal(_orientation(faces[order]), _orientation(expected[target])):
		raise ValueError("Invalid boundary face orientation; faces must point outward")
	result = np.empty(len(expected), dtype=np.int64)
	result[target] = owners[order]
	return result


def _validate_closed_surface(faces: np.ndarray) -> None:
	edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
	unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
	directions = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
	if np.any(counts != 2) or np.any(directions != 0):
		raise ValueError("Model exterior surface is not closed or has inconsistent orientation")
	order = np.argsort(inverse).reshape(-1, 2) // 3
	graph = coo_matrix((np.ones(len(order), dtype=np.int8), (order[:, 0], order[:, 1])),
					   shape=(len(faces), len(faces)))
	if connected_components(graph, directed=False, return_labels=False) != 1:
		raise ValueError("Model exterior surface is disconnected")
	if len(np.unique(faces)) - len(unique) + len(faces) != 2:
		raise ValueError("Model exterior surface failed the Euler check")


@dataclass(frozen=True)
class TetraValidation:
	stats: dict
	boundary_faces: np.ndarray  # 0-based, outward
	boundary_owners: np.ndarray  # 0-based Tet4 ID
	boundary_groups: np.ndarray  # 0=top, 1=bottom, 2..5=input side 1..4


def validate_tetrahedral_mesh(mesh: LayeredTetraGeometry, xyz: np.ndarray | None = None,
							 tetrahedra: np.ndarray | None = None) -> TetraValidation:
	"""Build face adjacency per slab to limit memory; match inter-layer caps exactly and check exterior closure/connectivity."""
	_validate_layering(mesh)
	n, m = len(mesh.source.xyz), len(mesh.source.triangles)
	per_slab = 3 * m
	if xyz is not None:
		xyz = np.asarray(xyz, dtype=np.float64)
		if xyz.shape != (mesh.node_count, 3) or not np.isfinite(xyz).all():
			raise ValueError("3D node count/shape is invalid or coordinates are non-finite")
		for k in range(mesh.layer_count + 1):
			if not np.array_equal(xyz[k * n:(k + 1) * n], mesh.coordinates(k)):
				raise ValueError("3D nodes deviate from the required original XY / layered elevations")
	if tetrahedra is not None:
		tetrahedra = np.asarray(tetrahedra)
		if (tetrahedra.shape != (mesh.tetrahedron_count, 4)
				or not np.issubdtype(tetrahedra.dtype, np.integer)
				or tetrahedra.min() < 0 or tetrahedra.max() >= mesh.node_count):
			raise ValueError("Invalid tetrahedron connectivity count, integer type, or node indices")
	volume, min_volume, min_jacobian = 0., math.inf, math.inf
	min_edge, max_edge, min_quality, max_quality = math.inf, 0., math.inf, 0.
	internal_faces = 0
	exterior, exterior_owners, exterior_groups = [], [], []
	column_volumes = mesh.column_volumes()
	for k in range(mesh.layer_count):
		tets = mesh.tetrahedra(k) if tetrahedra is None else tetrahedra[k * per_slab:(k + 1) * per_slab]
		local = tets - k * n
		if np.any(local < 0) or np.any(local >= 2 * n):
			raise ValueError("Tetrahedron connectivity spans non-adjacent layers")
		if np.any(np.diff(np.sort(local, axis=1), axis=1) == 0):
			raise ValueError("Tetrahedra contain repeated nodes")
		if len(np.unique(local)) != 2 * n:
			raise ValueError("Layer segment contains isolated nodes")
		parent_nodes = (local % n).reshape(m, 3, 4)
		if not np.all(np.any(parent_nodes[..., None] == mesh.source.triangles[:, None, None, :], axis=-1)):
			raise ValueError("Tetrahedra do not belong to the original triangular columns specified by the numbering rules")
		nodes = (np.vstack((mesh.coordinates(k), mesh.coordinates(k + 1))) if xyz is None
				 else xyz[k * n:(k + 2) * n])
		determinants, lengths, quality = tetrahedron_metrics(nodes[local])
		volumes = determinants / 6
		expected = column_volumes * (mesh.fractions[k + 1] - mesh.fractions[k])
		areas = column_volumes / (mesh.top - mesh.bottom)[mesh.source.triangles].mean(axis=1)
		z_ulp = np.maximum(np.abs(np.spacing(nodes[:n, 2])), np.abs(np.spacing(nodes[n:, 2])))
		rounding_budget = 16 * areas * z_ulp[mesh.source.triangles].mean(axis=1)
		if np.any(np.abs(volumes.reshape(m, 3).sum(axis=1) - expected) > 1e-9 * expected + rounding_budget):
			raise ValueError("Tetrahedron volume sum in a column layer segment does not match the independent analytical volume")
		faces, owners, internals = _face_topology(tets)
		expected_faces = _expected_slab_surface(mesh, k)
		owners = _surface_owners(faces, owners + k * per_slab, expected_faces)
		# Top/bottom caps use the same 2D triangles with opposite orientations; per-slab checks exclude open internal faces between layers.
		internal_faces += internals + (m if k else 0)
		select = np.r_[np.arange(m) if k == 0 else np.empty(0, dtype=int),
					   np.arange(m, 2 * m) if k == mesh.layer_count - 1 else np.empty(0, dtype=int),
					   np.arange(2 * m, len(expected_faces))]
		groups = np.r_[np.zeros(m, dtype=np.int8), np.ones(m, dtype=np.int8),
					   np.repeat(mesh.source.input_side_ids + 1, 2)]
		exterior.append(expected_faces[select])
		exterior_owners.append(owners[select])
		exterior_groups.append(groups[select])
		volume += float(volumes.sum())
		min_volume = min(min_volume, float(volumes.min()))
		min_jacobian = min(min_jacobian, float(determinants.min()))
		min_edge, max_edge = min(min_edge, float(lengths.min())), max(max_edge, float(lengths.max()))
		min_quality, max_quality = min(min_quality, float(quality.min())), max(max_quality, float(quality.max()))
	if not math.isclose(volume, mesh.expected_volume(), rel_tol=1e-9):
		raise ValueError("Total tetrahedron volume does not match the independent analytical volume")
	boundary_faces = np.vstack(exterior)
	_validate_closed_surface(boundary_faces)
	stats = dict(node_count=mesh.node_count, tetrahedron_count=mesh.tetrahedron_count,
				 layer_count=mesh.layer_count, volume=volume, expected_volume=mesh.expected_volume(),
				 min_volume=min_volume, min_jacobian=min_jacobian,
				 min_edge_length=min_edge, max_edge_length=max_edge,
				 min_quality=min_quality, max_quality=max_quality,
				 internal_face_count=internal_faces, boundary_face_count=len(boundary_faces),
				 connected=True, closed_boundary=True, opposite_internal_faces=True)
	return TetraValidation(stats, boundary_faces, np.concatenate(exterior_owners), np.concatenate(exterior_groups))


def read_msh22_tetrahedra(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
	"""Read only the fixed Domain3D/type 4 subset; reject other dimensions, elements, or extra content."""
	with Path(path).open(encoding="ascii") as stream:
		def expect(value):
			if stream.readline().strip() != value:
				raise ValueError(f"Invalid MSH format; expected {value}")

		def count():
			value = int(stream.readline())
			if value < 1:
				raise ValueError("MSH node/element counts must be positive")
			return value

		expect("$MeshFormat")
		expect("2.2 0 8")
		expect("$EndMeshFormat")
		expect("$PhysicalNames")
		expect("1")
		expect('3 1 "Domain3D"')
		expect("$EndPhysicalNames")
		expect("$Nodes")
		xyz = np.empty((count(), 3), dtype=np.float64)
		for i in range(len(xyz)):
			parts = stream.readline().split()
			if len(parts) != 4 or int(parts[0]) != i + 1:
				raise ValueError("Invalid MSH node format/numbering")
			xyz[i] = [float(v) for v in parts[1:]]
		expect("$EndNodes")
		expect("$Elements")
		tets = np.empty((count(), 4), dtype=np.int64)
		for i in range(len(tets)):
			parts = [int(v) for v in stream.readline().split()]
			if len(parts) != 9 or parts[:5] != [i + 1, 4, 2, 1, 1]:
				raise ValueError("MSH must contain only type 4 / Domain3D")
			tets[i] = np.array(parts[5:]) - 1
		expect("$EndElements")
		if stream.read().strip():
			raise ValueError("MSH contains unexpected extra content")
	if (not np.isfinite(xyz).all() or tets.min() < 0 or tets.max() >= len(xyz)
			or np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0)):
		raise ValueError("MSH contains invalid coordinates or node connectivity")
	return xyz, tets


def _write_vtk20_ascii(stream, xyz: np.ndarray, tetrahedra: np.ndarray) -> None:
	"""Write legacy CELLS format with per-cell integer lithology fixed at 1."""
	stream.write("# vtk DataFile Version 2.0\nConforming tetrahedral mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
	stream.write(f"POINTS {len(xyz)} double\n")
	for start in range(0, len(xyz), WRITE_CHUNK_SIZE):
		np.savetxt(stream, xyz[start:start + WRITE_CHUNK_SIZE], fmt="%.17g")
	count = len(tetrahedra)
	stream.write(f"CELLS {count} {5 * count}\n")
	for start in range(0, count, WRITE_CHUNK_SIZE):
		cells = tetrahedra[start:start + WRITE_CHUNK_SIZE]
		np.savetxt(stream, np.column_stack((np.full(len(cells), 4, dtype=np.int64), cells)), fmt="%d")
	stream.write(f"CELL_TYPES {count}\n")
	for start in range(0, count, WRITE_CHUNK_SIZE):
		stream.write("10\n" * min(WRITE_CHUNK_SIZE, count - start))
	stream.write(f"CELL_DATA {count}\nSCALARS lithology int 1\nLOOKUP_TABLE default\n")
	for start in range(0, count, WRITE_CHUNK_SIZE):
		stream.write("1\n" * min(WRITE_CHUNK_SIZE, count - start))


def read_vtk20_tetrahedra(path: Path | str, *, require_lithology: bool = False) -> tuple[np.ndarray, np.ndarray]:
	"""Strictly read Tet4 with fixed lithology=1; accept legacy attribute-free VTK by default."""
	with Path(path).open(encoding="ascii") as stream:
		def expect(value):
			if stream.readline().strip() != value:
				raise ValueError(f"Invalid VTK 2.0 ASCII format; expected {value}")

		expect("# vtk DataFile Version 2.0")
		expect("Conforming tetrahedral mesh")
		expect("ASCII")
		expect("DATASET UNSTRUCTURED_GRID")
		parts = stream.readline().split()
		if len(parts) != 3 or parts[0] != "POINTS" or parts[2] != "double" or int(parts[1]) < 1:
			raise ValueError("Invalid VTK POINTS format/count")
		xyz = np.empty((int(parts[1]), 3), dtype=np.float64)
		for i in range(len(xyz)):
			parts = stream.readline().split()
			if len(parts) != 3:
				raise ValueError("VTK nodes must have three coordinates")
			xyz[i] = [float(v) for v in parts]
		parts = stream.readline().split()
		if (len(parts) != 3 or parts[0] != "CELLS" or int(parts[1]) < 1
				or int(parts[2]) != 5 * int(parts[1])):
			raise ValueError("VTK CELLS must use the legacy four-node cell format")
		tets = np.empty((int(parts[1]), 4), dtype=np.int64)
		for i in range(len(tets)):
			parts = [int(v) for v in stream.readline().split()]
			if len(parts) != 5 or parts[0] != 4:
				raise ValueError("VTK may contain only four-node tetrahedra")
			tets[i] = parts[1:]
		expect(f"CELL_TYPES {len(tets)}")
		for _ in range(len(tets)):
			expect("10")
		header = stream.readline()
		while header and not header.strip():
			header = stream.readline()
		if header:
			if header.strip() != f"CELL_DATA {len(tets)}":
				raise ValueError("Invalid VTK CELL_DATA format or count does not match tetrahedra")
			expect("SCALARS lithology int 1")
			expect("LOOKUP_TABLE default")
			for _ in range(len(tets)):
				expect("1")
		elif require_lithology:
			raise ValueError("VTK is missing CELL_DATA/lithology")
		if stream.read().strip():
			raise ValueError("VTK contains unexpected extra content")
	if (not np.isfinite(xyz).all() or tets.min() < 0 or tets.max() >= len(xyz)
			or np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0)):
		raise ValueError("VTK contains invalid coordinates or node connectivity")
	return xyz, tets


def _sample_report(sample: SurfaceSample) -> dict:
	return dict(path=str(sample.path.resolve()), sha256=sample.sha256,
				interpolated_count=sample.interpolated_count, extrapolated_count=sample.extrapolated_count,
				extrapolated_ratio=sample.extrapolated_ratio, max_fill_distance=sample.max_distance,
				max_allowed_distance=sample.max_allowed_distance)


def _output_report(mesh: LayeredTetraMesh, result: TetraValidation, output: Path,
				   digest: str, vtk_digest: str) -> dict:
	n, m = len(mesh.source.xyz), len(mesh.source.triangles)
	thickness = mesh.top - mesh.bottom
	first_dz = mesh.coordinates(0)[:, 2] - mesh.coordinates(1)[:, 2]
	last_dz = mesh.coordinates(mesh.layer_count - 1)[:, 2] - mesh.coordinates(mesh.layer_count)[:, 2]
	return {
		"schema_version": 2, "output": str(output), "msh_sha256": digest,
		"format": "MSH 2.2 ASCII; Domain3D; type 4 only",
		"vtk_output": str(output.with_suffix(".vtk")), "vtk_sha256": vtk_digest,
		"vtk_format": "legacy VTK 2.0 ASCII; UNSTRUCTURED_GRID; VTK_TETRA=10 only",
		"source": {"mesh_2d": str(mesh.source.path), "msh_sha256": mesh.source.msh_sha256,
				   "mesh_2d_report": str(mesh.source.report_path), "report_sha256": mesh.source.report_sha256,
				   "node_count": n, "triangle_count": m},
		"parameters": mesh.parameters, "fractions": mesh.fractions.tolist(),
		"thickness": {"column_range": [float(thickness.min()), float(thickness.max())],
					  "actual_first_layer_range": [float(first_dz.min()), float(first_dz.max())],
					  "actual_last_layer_range": [float(last_dz.min()), float(last_dz.max())]},
		"surface_samples": {"top": _sample_report(mesh.top_sample), "bottom": _sample_report(mesh.bottom_sample)},
		"numbering": {
			"id_base": 1, "layer_index_base": 0, "source_ids": "source row index + 1, unchanged",
			"node": "id=k*N+i+1; source_node_id=(id-1)%N+1; layer=(id-1)//N",
			"tetrahedron": "id=k*(3*M)+3*t+s+1; source_triangle_id=((id-1)//3)%M+1; "
						   "layer_segment=(id-1)//(3*M); child=(id-1)%3 (0,1,2)",
			"vtk": "point/cell row order identical to MSH; 0-based VTK index = MSH ID - 1; "
				   "boundary report IDs remain 1-based",
			"split": "sort source node IDs a<b<c; sets (a0,b0,c0,c1), (a0,b0,b1,c1), "
					 "(a0,a1,b1,c1); orientation fixed by XY winding and downward layer direction",
			"side_diagonal": "upper smaller-source-ID to lower larger-source-ID",
		},
		"validation": result.stats,
		"boundary": {
			"node_id_base": 1, "owner_id_base": 1, "orientation": "outward",
			"group_names": {"0": "top", "1": "bottom", "2": "input_side_1",
							"3": "input_side_2", "4": "input_side_3", "5": "input_side_4"},
			"face_node_ids": (result.boundary_faces + 1).tolist(),
			"owner_tetrahedron_ids": (result.boundary_owners + 1).tolist(),
			"group_ids": result.boundary_groups.tolist(),
			"input_corners": mesh.source.corners_input.tolist(),
		},
		"definitions": {
			"jacobian": "D=det(columns(p1-p0,p2-p0,p3-p0)); volume=D/6; positive, no absolute value",
			"quality": "q=12*(3V)^(2/3)/sum(six edge lengths squared); regular=1, degeneracy approaches 0",
			"volume_reference": "sum(A_triangle*mean(final vertex thicknesses)); piecewise-linear surfaces",
			"null": "no distance limit in error extrapolation mode",
			"atomicity": "report then VTK then MSH committed; three files are not one transaction; "
						 "verify msh_sha256 and vtk_sha256 before paired use",
		},
		"downstream": "3_assign_facies_3d.py accepts Tet4 MSH alone for facies assignment. "
					  "4_find_DirBC.py accepts paired Tet4 outputs for boundary nodes. "
					  "5_find_DirBC_cells.py, 6_find_NeumBC_cells.py, "
					  "7_assign_dirBC.py and other Hex8-specific stages are NOT migrated. "
					  "Positive Tet4 volume/quality does not imply Delaunay or TPFA suitability.",
	}


def _protect_output(mesh: LayeredTetraMesh, output: Path) -> None:
	if output.suffix.lower() != ".msh":
		raise ValueError("Output must be a .msh file (MSH 2.2)")
	inputs = [mesh.source.path, mesh.source.report_path, mesh.top_sample.path, mesh.bottom_sample.path]
	for target in (output, output.with_suffix(".vtk"), output.with_suffix(".report.json")):
		for source in inputs:
			if target.resolve() == source.resolve() or (target.exists() and source.exists() and target.samefile(source)):
				raise ValueError("Mesh or companion report outputs must not overwrite input files")


@contextmanager
def _staged_file(output: Path, suffix: str, temporary: list[Path], *, encoding: str = "ascii"):
	with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="\n", dir=output.parent,
									 prefix=f".{output.stem}.", suffix=suffix, delete=False) as stream:
		path = Path(stream.name)
		temporary.append(path)
		yield path, stream
		stream.flush()
		os.fsync(stream.fileno())


def write_mesh(mesh: LayeredTetraMesh, output_file: Path | str = OUTPUT_FILE, verbose: bool = True) -> dict:
	"""Write MSH, ASCII legacy VTK 2.0, and a companion report for one mesh; commit the main MSH last."""
	output = Path(output_file).resolve()
	_protect_output(mesh, output)
	_check_gmsh_session()
	for path, digest in ((mesh.source.path, mesh.source.msh_sha256),
						 (mesh.source.report_path, mesh.source.report_sha256),
						 (mesh.top_sample.path, mesh.top_sample.sha256),
						 (mesh.bottom_sample.path, mesh.bottom_sample.sha256)):
		if digest is None or file_sha256(path) != digest:
			raise ValueError(f"{path}: input was modified between construction and writing, or its source hash is missing")
	result = validate_tetrahedral_mesh(mesh)
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = []
	try:
		with _staged_file(output, ".msh", temporary) as (temporary_msh, stream):
			stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n3 1 "Domain3D"\n$EndPhysicalNames\n$Nodes\n')
			stream.write(f"{mesh.node_count}\n")
			n = len(mesh.source.xyz)
			for k in range(mesh.layer_count + 1):
				for i, (x, y, z) in enumerate(mesh.coordinates(k), k * n + 1):
					stream.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
			stream.write(f"$EndNodes\n$Elements\n{mesh.tetrahedron_count}\n")
			per_slab = 3 * len(mesh.source.triangles)
			for k in range(mesh.layer_count):
				for i, t in enumerate(mesh.tetrahedra(k) + 1, k * per_slab + 1):
					stream.write(f"{i} 4 2 1 1 {t[0]} {t[1]} {t[2]} {t[3]}\n")
			stream.write("$EndElements\n")
		xyz, tets = read_msh22_tetrahedra(temporary_msh)
		for k in range(mesh.layer_count):
			if not np.array_equal(tets[k * per_slab:(k + 1) * per_slab], mesh.tetrahedra(k)):
				raise ValueError("MSH connectivity changed after writing")
		# Repeat exact node comparisons and all geometry/topology checks on the read-back data.
		validate_tetrahedral_mesh(mesh, xyz, tets)
		with _staged_file(output, ".vtk", temporary) as (temporary_vtk, stream):
			_write_vtk20_ascii(stream, xyz, tets)
		vtk_xyz, vtk_tets = read_vtk20_tetrahedra(temporary_vtk, require_lithology=True)
		if not np.array_equal(vtk_xyz, xyz) or not np.array_equal(vtk_tets, tets):
			raise ValueError("VTK coordinates or connectivity do not match MSH after writing")
		del vtk_xyz, vtk_tets
		report = _output_report(mesh, result, output, file_sha256(temporary_msh), file_sha256(temporary_vtk))
		content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
		with _staged_file(output, ".json", temporary, encoding="utf-8") as (temporary_report, stream):
			stream.write(content)
		os.replace(temporary_report, output.with_suffix(".report.json"))
		os.replace(temporary_vtk, output.with_suffix(".vtk"))
		os.replace(temporary_msh, output)
		if verbose:
			print(f"Validated Tet4 outputs: {output}, {output.with_suffix('.vtk')}", flush=True)
		return {"output": output, "vtk_output": output.with_suffix(".vtk"),
				"report": output.with_suffix(".report.json"), **result.stats}
	finally:
		for path in temporary:
			path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
									 allow_abbrev=False)
	parser.add_argument("--mesh-2d", type=Path, default=MESH_2D_FILE)
	parser.add_argument("--mesh-2d-report", type=Path, default=None, help="Defaults to the .report.json with the same stem as the 2D MSH")
	parser.add_argument("--top", type=Path, default=TOP_XYZ_FILE)
	parser.add_argument("--bottom", type=Path, default=BOTTOM_XYZ_FILE)
	parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="MSH path; also generates .vtk and .report.json files with the same stem")
	parser.add_argument("--dz", type=float, default=DZ, help="Maximum layer thickness; shared layer count, equal thickness within each column")
	parser.add_argument("--max-elements", type=int, default=MAX_TETRAHEDRA, help="Maximum final tetrahedron count")
	parser.add_argument("--extrapolation-mode", choices=EXTRAPOLATION_MODES, default=EXTRAPOLATION_MODE,
						help="error: reject exterior nodes; nearest (default): fill only exterior nodes by nearest neighbour; requires top/bottom distance limits")
	parser.add_argument("--top-max-extrapolation-distance", type=float,
						default=TOP_MAX_EXTRAPOLATION_DISTANCE,
						help="Top: maximum Euclidean distance from target XY to the nearest original sample XY (in XY units)")
	parser.add_argument("--bottom-max-extrapolation-distance", type=float,
						default=BOTTOM_MAX_EXTRAPOLATION_DISTANCE,
						help="Bottom: maximum Euclidean distance from target XY to the nearest original sample XY (in XY units)")
	parser.add_argument("--quiet", action="store_true")
	args = parser.parse_args(argv)
	try:
		mesh = build_layered_mesh(args.mesh_2d, args.top, args.bottom, dz=args.dz,
								  max_elements=args.max_elements,
								  extrapolation_mode=args.extrapolation_mode,
								  top_max_distance=args.top_max_extrapolation_distance,
								  bottom_max_distance=args.bottom_max_extrapolation_distance,
								  mesh_2d_report=args.mesh_2d_report)
		if not args.quiet:
			print(f"Generating {mesh.layer_count} layer segments, {mesh.node_count:,} nodes, "
				  f"{mesh.tetrahedron_count:,} tetrahedra...", flush=True)
		result = write_mesh(mesh, args.output, not args.quiet)
	except (OSError, ValueError, RuntimeError) as exc:
		parser.exit(2, f"error: {exc}\n")
	thickness = mesh.top - mesh.bottom
	first_dz = mesh.coordinates(0)[:, 2] - mesh.coordinates(1)[:, 2]
	last_dz = mesh.coordinates(mesh.layer_count - 1)[:, 2] - mesh.coordinates(mesh.layer_count)[:, 2]
	print("\n3D TETRAHEDRAL MESH REPORT")
	print(f"Output: {result['output']} (MSH 2.2 ASCII, type 4 only, Domain3D)")
	print(f"VTK output: {result['vtk_output']} (legacy VTK 2.0 ASCII, VTK_TETRA=10 only)")
	print(f"2D nodes: {len(mesh.source.xy)}; 2D triangles: {len(mesh.source.triangles)}")
	print(f"Layer segments: {mesh.layer_count}; 3D nodes: {mesh.node_count}; tetrahedra: {mesh.tetrahedron_count}")
	print(f"Extrapolation mode: {mesh.parameters['extrapolation_mode']}")
	any_filled = False
	for name, sample in (("Top", mesh.top_sample), ("Bottom", mesh.bottom_sample)):
		any_filled |= sample.extrapolated_count > 0
		limit = "n/a" if sample.max_allowed_distance is None else f"{sample.max_allowed_distance:.12g}"
		print(
			f"{name} surface ({sample.path.name}): interpolated nodes {sample.interpolated_count}/{sample.node_count}; "
			f"nearest-filled nodes {sample.extrapolated_count} ({sample.extrapolated_ratio:.4%}); "
			f"max fill distance {sample.max_distance:.8g} (limit {limit}, XY units)"
		)
	if any_filled:
		print("WARNING: nearest-neighbour fill is piecewise constant and may create elevation steps "
			  "along the XYZ coverage edge.")
	print(f"Bottom range: {mesh.bottom.min():.12g} .. {mesh.bottom.max():.12g}")
	print(f"Thickness range: {thickness.min():.8g} .. {thickness.max():.8g}")
	print(f"Requested dz: {args.dz:g}")
	print(f"Actual first-layer dz: {first_dz.min():.8g} .. {first_dz.max():.8g}")
	print(f"Actual last-layer dz: {last_dz.min():.8g} .. {last_dz.max():.8g}")
	print(f"Total volume: {result['volume']:.12g}; minimum tetrahedron volume: {result['min_volume']:.8g}")
	print(f"Minimum determinant/Jacobian: {result['min_jacobian']:.8g}; "
		  f"q range: {result['min_quality']:.8g} .. {result['max_quality']:.8g}")
	print("Shared uniform layers: equal dz within each column; thinner columns have proportionally smaller dz.")
	print(f"Boundary owners, numbering and hash: {result['report']}")
	print("3_assign_facies_3d.py accepts Tet4 MSH alone. Other Hex8-only BC/well stages are NOT migrated; "
		  "positive volume does not imply TPFA suitability.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
