"""Assign lithology from one geological simulation realization to upstream pure Tet4 MSH.

Output is legacy VTK 2.0 ASCII.

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile

import gmsh
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _load_upstream():
	spec = importlib.util.spec_from_file_location(
		"_facies_upstream_tetra_mesh", Path(__file__).with_name("2_2d_ext_3d.py"))
	if spec is None or spec.loader is None:
		raise RuntimeError("Cannot load the upstream tetrahedral mesh module")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


_tet = _load_upstream()

# ======================== Editable parameters ========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
MESH_FILE = PROJECT_DIR / "out" / "mesh_3d_tet.msh"
EXPORT_BOUNDARIES = True    # CLI appends side/bottom Tri3 by default; False restores pure Tet4
MESH_REPORT = None          # None uses the .report.json with the same stem as the input MSH
COORDINATES_FILE = PROJECT_DIR / "data" / "Tec_FaciesZone-Indic.plt"
OUT_FILE = PROJECT_DIR / "data" / "tsimFaci.out"  # Reread this one file on every run
XYZ_COLUMNS = (0, 1, 2)    # Zero-based; default XYZ in the first three columns; use (1, 2, 3) with a leading ID
COVERAGE_MODE = "auto"      # Only determines fallback eligibility for empty cells; inside samples always vote.
							# "auto": detect XYZ bounds for complete grids, valid-node masks for clipped grids, 3D convex hulls for other samples;
							# "lattice": require all bracketing XYZ sample nodes at the cell center (equivalent to center bounds checks for complete grids);
							# "convex_hull": require all cell vertices inside the 3D hull; assume its interior is valid, without detecting concavities or holes
DOMAIN_XY_FILE = None       # Optional GeoJSON Polygon/MultiPolygon with holes; only restricts the valid domain
FALLBACK_DZ = None          # Optional positive number; None uses the column's median adjacent Z spacing
MAX_SUPPORT_GRID_POINTS = 8_000_000  # Coverage-count prefix-table limit; above it, query actual columns without expanding the full grid
OUTPUT_FILE = PROJECT_DIR / "out" / "mesh_3d_with_facies.vtk"
OUTPUT_MODE = "simple"      # "simple": only CELL_DATA/lithology; "debug": retain all existing attributes
UNDEFINED_FACIES = 9         # User-specified lithology for cells outside simulation coverage
CHUNK_SIZE = 4096
MAX_CANDIDATES = 2_000_000   # Maximum candidates expanded at once, to bound memory use
COORDINATE_TOLERANCE = 1e-8  # Input coordinate units; absorbs roundoff only, not a half-voxel coverage extension
CONTAINMENT_BATCH_SIZE = 65_536
# ================================================================
VTK_VERSION = "2.0"
VTK_ENCODING = "ASCII"
METHOD_NAMES = {0: "unsupported or excluded -> undefined", 1: "inside-tetrahedron mode",
				2: "empty tetrahedron -> nearest-column vertical mode",
				3: "empty tetrahedron -> nearest point"}


def read_coordinates(path: Path | str, *, xyz_columns: tuple[int, int, int] = XYZ_COLUMNS
					 ) -> tuple[np.ndarray, tuple[int, int, int]]:
	"""Skip the first three lines and read XYZ in original row order; infer grid dimensions from coordinates."""
	if (len(xyz_columns) != 3 or any(isinstance(c, (bool, np.bool_))
			or not isinstance(c, (int, np.integer)) or c < 0 for c in xyz_columns)
			or len(set(xyz_columns)) != 3):
		raise ValueError("XYZ_COLUMNS must contain three distinct nonnegative integer column indices (zero-based)")
	with Path(path).open(encoding="utf-8-sig") as stream:
		points = np.loadtxt(stream, skiprows=3, usecols=xyz_columns, dtype=float, ndmin=2, comments=None)
	if not len(points) or points.shape[1] != 3 or not np.isfinite(points).all():
		raise ValueError(f"{path}: nonempty finite XYZ coordinates are required")
	nx, ny, nz = (len(np.unique(points[:, axis])) for axis in range(3))
	return points, (nx, ny, nz)


def read_realization(path: Path | str, expected_rows: int) -> np.ndarray:
	"""Skip the first two lines; read one lithology column in original row order and check the coordinate row count."""
	with Path(path).open(encoding="utf-8-sig") as stream:
		values = np.loadtxt(stream, skiprows=2, dtype=float, ndmin=2, comments=None)
	if len(values) != expected_rows:
		raise ValueError(f"{path}: OUT and PLT data row counts differ (OUT: {len(values)}, PLT: {expected_rows})")
	if values.shape[1] != 1:
		raise ValueError(f"{path}: OUT must provide a single column of lithology codes, one per row")
	if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
			or np.any(values == 0) or np.any(np.abs(values) > np.iinfo(np.int32).max)):
		raise ValueError(f"{path}: lithology codes must be nonzero finite integers within the int32 range")
	return np.abs(values[:, 0]).astype(np.int32)


@dataclass
class XYDomain:
	"""Horizontal GeoJSON domain; conservatively check cell XY boxes without crossing concavities or holes."""
	polygons: list[list[np.ndarray]]

	@classmethod
	def read(cls, path: Path | str):
		data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
		polygons = []

		def collect(item):
			if not isinstance(item, dict):
				raise ValueError("The valid domain requires GeoJSON Polygon/MultiPolygon")
			kind = item.get("type")
			if kind == "Feature":
				collect(item.get("geometry"))
			elif kind == "FeatureCollection":
				for feature in item.get("features", []):
					collect(feature)
			elif kind in ("Polygon", "MultiPolygon"):
				groups = [item.get("coordinates")] if kind == "Polygon" else item.get("coordinates", [])
				for group in groups:
					if not group:
						raise ValueError("Valid-domain polygons must not be empty")
					rings = []
					for ring in group:
						values = np.asarray(ring, dtype=float)
						if (values.ndim != 2 or values.shape[1] < 2 or len(values) < 3
								or not np.isfinite(values).all()):
							raise ValueError("Valid-domain rings require at least three finite XY vertices")
						values = values[:, :2]
						if not np.array_equal(values[0], values[-1]):
							values = np.vstack((values, values[0]))
						if len(np.unique(values, axis=0)) < 3:
							raise ValueError("Valid-domain rings require at least three distinct XY vertices")
						rings.append(values)
					polygons.append(rings)
			else:
				raise ValueError("The valid domain supports only GeoJSON Polygon/MultiPolygon and their Feature collections")

		collect(data)
		if not polygons:
			raise ValueError("The valid domain contains no polygons")
		return cls(polygons)

	@staticmethod
	def ring_contains(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
		inside, boundary = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
		for a, b in zip(ring[:-1], ring[1:]):
			delta = b - a
			length2 = float(delta @ delta)
			if length2 == 0:
				continue
			fraction = np.clip((points - a) @ delta / length2, 0., 1.)
			boundary |= np.linalg.norm(points - (a + fraction[:, None] * delta), axis=1) <= COORDINATE_TOLERANCE
			cross = (a[1] > points[:, 1]) != (b[1] > points[:, 1])
			if delta[1] != 0:
				inside ^= cross & (points[:, 0] < a[0] + (points[:, 1] - a[1]) * delta[0] / delta[1])
		return inside | boundary

	def covers_boxes(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
		corners = np.stack((lower, np.column_stack((upper[:, 0], lower[:, 1])),
			upper, np.column_stack((lower[:, 0], upper[:, 1]))), axis=1)
		flat = corners.reshape(-1, 2)
		included = np.zeros(len(flat), dtype=bool)
		for rings in self.polygons:
			valid = self.ring_contains(flat, rings[0])
			for hole in rings[1:]:
				valid &= ~self.ring_contains(flat, hole)
			included |= valid
		covered = included.reshape(-1, 4).all(axis=1)
		# Interior vertices may still span holes/concavities; conservatively reject any boundary entering the box interior.
		starts = np.vstack([ring[:-1] for rings in self.polygons for ring in rings])
		ends = np.vstack([ring[1:] for rings in self.polygons for ring in rings])
		delta = ends - starts
		for i in np.flatnonzero(covered):
			lo, hi = lower[i] + COORDINATE_TOLERANCE, upper[i] - COORDINATE_TOLERANCE
			if np.any(lo >= hi):
				continue
			first, last = np.zeros(len(starts)), np.ones(len(starts))
			possible = np.ones(len(starts), dtype=bool)
			for axis in range(2):
				moving = delta[:, axis] != 0
				possible &= moving | ((starts[:, axis] >= lo[axis]) & (starts[:, axis] <= hi[axis]))
				denominator = np.where(moving, delta[:, axis], 1.)
				a, b = (lo[axis] - starts[:, axis]) / denominator, (hi[axis] - starts[:, axis]) / denominator
				first = np.maximum(first, np.where(moving, np.minimum(a, b), 0.))
				last = np.minimum(last, np.where(moving, np.maximum(a, b), 1.))
			if np.any(possible & (first <= last)):
				covered[i] = False
		return covered


@dataclass
class SourceGrid:
	points: np.ndarray
	facies: np.ndarray
	axes: tuple[np.ndarray, np.ndarray, np.ndarray]
	row_grid: np.ndarray | None  # [z,y,x] -> original row index only for complete Cartesian grids; no missing-value sentinel
	codes: np.ndarray
	source_type: str = "regular"
	coverage_mode: str = "bounds"
	domain: XYDomain | None = None
	fallback_dz: float | None = None
	column_xy: np.ndarray | None = None
	point_columns: np.ndarray | None = None
	column_rows: np.ndarray | None = None
	column_offsets: np.ndarray | None = None
	column_dz: np.ndarray | None = None
	column_tree: cKDTree | None = None
	point_tree: cKDTree | None = None
	support_prefix: np.ndarray | None = None
	hull_equations: np.ndarray | None = None
	hull_origin: np.ndarray | None = None

	@classmethod
	def build(cls, points: np.ndarray, facies: np.ndarray, grid: tuple[int, int, int] | None = None,
			  *, coverage_mode: str = COVERAGE_MODE, domain: XYDomain | None = None,
			  fallback_dz: float | None = FALLBACK_DZ):
		points, facies = np.asarray(points, dtype=float), np.asarray(facies)
		if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
			raise ValueError("Coordinates must be a nonempty finite N-by-3 array")
		if facies.shape != (len(points),):
			raise ValueError("Coordinate and lithology record counts differ")
		if facies.dtype.kind not in "iu" or np.any(facies <= 0) or np.any(facies > np.iinfo(np.int32).max):
			raise ValueError("Lithology codes must be positive int32 values")
		if coverage_mode not in ("auto", "lattice", "convex_hull"):
			raise ValueError("COVERAGE_MODE must be auto, lattice, or convex_hull")
		if fallback_dz is not None and (isinstance(fallback_dz, (bool, np.bool_))
				or not isinstance(fallback_dz, (int, float, np.integer, np.floating))
				or not np.isfinite(fallback_dz) or fallback_dz <= 0):
			raise ValueError("FALLBACK_DZ must be a finite positive number or None")
		order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
		ordered = points[order]
		if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
			raise ValueError("PLT coordinates contain duplicate XYZ points; duplicate votes or overwriting the original row mapping are not allowed")
		axes, indices = [], []
		for axis in range(3):
			values, inverse = np.unique(points[:, axis], return_inverse=True)
			axes.append(values)
			indices.append(inverse)
		dimensions = tuple(len(axis) for axis in axes)
		if grid is not None and tuple(grid) != dimensions:
			raise ValueError("Supplied grid axis dimensions do not match the actual coordinates")
		uniform = [len(a) < 3 or np.allclose(np.diff(a), a[1] - a[0], rtol=1e-6, atol=1e-10) for a in axes]
		source = cls(points, facies, tuple(axes), None, np.unique(facies),
					 domain=domain, fallback_dz=fallback_dz)
		ix, iy, iz = indices
		if math.prod(dimensions) == len(points):
			rows = np.empty(len(points), dtype=np.int64)
			rows[(iz * dimensions[1] + iy) * dimensions[0] + ix] = np.arange(len(points))
			source.row_grid = rows.reshape(dimensions[::-1])
			source.source_type = "regular" if all(uniform) else "rectilinear"
		else:
			starts = np.r_[0, np.flatnonzero(np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)) + 1]
			source.column_xy = ordered[starts, :2]
			source.column_rows, source.column_offsets = order, np.r_[starts, len(points)]
			counts = np.diff(source.column_offsets)
			source.point_columns = np.empty(len(points), dtype=np.int64)
			source.point_columns[order] = np.repeat(np.arange(len(starts)), counts)
			source.column_dz = np.array([np.median(np.diff(ordered[a:b, 2])) if b - a > 1 else math.inf
				for a, b in zip(source.column_offsets[:-1], source.column_offsets[1:])])
			source.column_tree = cKDTree(source.column_xy)
			source.point_tree = cKDTree(points)
			columnar = bool(np.any(counts >= 2))
			# Allow clipped columns with one remaining sample; four XY positions are the minimum for a horizontal sampling quadrilateral.
			lattice = columnar and len(starts) >= 4 and all(uniform[:2])
			source.source_type = "columnar" if lattice or np.all(counts >= 2) else "scattered"
			source.coverage_mode = ("lattice" if lattice else "convex_hull") if coverage_mode == "auto" else coverage_mode
			if source.coverage_mode == "lattice" and math.prod(dimensions) <= MAX_SUPPORT_GRID_POINTS:
				missing = np.ones(dimensions[::-1], dtype=np.int32)
				missing[iz, iy, ix] = 0
				prefix = np.pad(missing, ((1, 0), (1, 0), (1, 0)))
				for axis in range(3):
					np.cumsum(prefix, axis=axis, dtype=np.int32, out=prefix)
				source.support_prefix = prefix
		if coverage_mode == "convex_hull":
			source.coverage_mode = "convex_hull"
		if source.coverage_mode == "convex_hull":
			source.hull_origin = points.min(axis=0)
			try:
				source.hull_equations = ConvexHull(points - source.hull_origin).equations
			except QhullError:
				# Coplanar/collinear samples cannot cover nonzero-volume cells, but nearest-point queries remain valid.
				source.hull_equations = np.empty((0, 4))
		return source

	@property
	def dz(self) -> float:
		if self.fallback_dz is not None:
			return float(self.fallback_dz)
		if self.row_grid is not None:
			return float(np.median(np.diff(self.axes[2]))) if len(self.axes[2]) > 1 else math.inf
		finite = self.column_dz[np.isfinite(self.column_dz)]
		return float(np.median(finite)) if len(finite) else math.inf

	def vertical_spacings(self, rows: np.ndarray) -> np.ndarray:
		if self.row_grid is not None or self.fallback_dz is not None:
			return np.full(len(rows), self.dz)
		return self.column_dz[self.point_columns[rows]]

	def nearest(self, targets: np.ndarray) -> np.ndarray:
		"""Exact 3D nearest neighbour; retain the regular-grid algorithm and break ties by smallest original PLT row index."""
		targets = np.asarray(targets, dtype=float)
		if not len(targets):
			return np.empty(0, dtype=np.int64)
		if self.row_grid is None:
			if len(self.points) == 1:
				return np.zeros(len(targets), dtype=np.int64)
			distance, neighbors = self.point_tree.query(targets, k=2, eps=0)
			best = neighbors[:, 0].copy()
			# The tree does not enforce row-index tie-breaking; gather possible ties and compare original physical coordinates exactly.
			tied = np.isclose(distance[:, 0], distance[:, 1], rtol=32 * np.finfo(float).eps, atol=0.)
			for i in np.flatnonzero(tied):
				radius = np.nextafter(distance[i, 0] * (1 + 32 * np.finfo(float).eps), np.inf)
				ids = np.asarray(self.point_tree.query_ball_point(targets[i], radius, eps=0), dtype=np.int64)
				d2 = np.sum((self.points[ids] - targets[i]) ** 2, axis=1)
				best[i] = ids[d2 == d2.min()].min()
			return best
		choices = []
		for axis, values in enumerate(self.axes):
			high = np.searchsorted(values, targets[:, axis])
			choices.append((np.clip(high - 1, 0, len(values) - 1), np.clip(high, 0, len(values) - 1)))
		best = np.full(len(targets), len(self.points), dtype=np.int64)
		best_distance = np.full(len(targets), np.inf)
		for pick in itertools.product((0, 1), repeat=3):
			ix, iy, iz = (choices[axis][pick[axis]] for axis in range(3))
			rows = self.row_grid[iz, iy, ix]
			distance = np.sum((self.points[rows] - targets) ** 2, axis=1)
			better = (distance < best_distance) | ((distance == best_distance) & (rows < best))
			best[better], best_distance[better] = rows[better], distance[better]
		return best

	def _column_sections(self, lower: np.ndarray, upper: np.ndarray, maximum: int):
		"""Enumerate actual XY columns and their Z windows; bound XY candidates before actual 3D candidates."""
		centers = (lower[:, :2] + upper[:, :2]) / 2
		radii = np.max((upper[:, :2] - lower[:, :2]) / 2, axis=1) + COORDINATE_TOLERANCE
		counts = self.column_tree.query_ball_point(centers, radii, p=np.inf, eps=0, return_length=True)
		shared_z = bool(np.all(np.diff(self.column_offsets) == len(self.axes[2])))
		for a, b in bounded_ranges(counts, maximum):
			lists = self.column_tree.query_ball_point(centers[a:b], radii[a:b], p=np.inf, eps=0)
			columns = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=int(counts[a:b].sum()))
			owners = np.repeat(np.arange(a, b), counts[a:b])
			xy = self.column_xy[columns]
			valid = ((xy >= lower[owners, :2] - COORDINATE_TOLERANCE).all(axis=1)
				& (xy <= upper[owners, :2] + COORDINATE_TOLERANCE).all(axis=1))
			columns, owners = columns[valid], owners[valid]
			if shared_z:
				first = self.column_offsets[columns] + np.searchsorted(self.axes[2], lower[owners, 2] - COORDINATE_TOLERANCE)
				last = self.column_offsets[columns] + np.searchsorted(self.axes[2], upper[owners, 2] + COORDINATE_TOLERANCE, side="right")
			else:
				first, last = np.empty(len(columns), dtype=np.int64), np.empty(len(columns), dtype=np.int64)
				for j, (column, owner) in enumerate(zip(columns, owners)):
					start, stop = self.column_offsets[column:column + 2]
					zs = self.points[self.column_rows[start:stop], 2]
					first[j] = start + np.searchsorted(zs, lower[owner, 2] - COORDINATE_TOLERANCE)
					last[j] = start + np.searchsorted(zs, upper[owner, 2] + COORDINATE_TOLERANCE, side="right")
			yield a, b, owners, first, last

	def candidate_batches(self, lower: np.ndarray, upper: np.ndarray, maximum: int):
		if self.row_grid is not None:
			first = np.column_stack([np.searchsorted(a, lower[:, i] - COORDINATE_TOLERANCE)
				for i, a in enumerate(self.axes)])
			last = np.column_stack([np.searchsorted(a, upper[:, i] + COORDINATE_TOLERANCE, side="right")
				for i, a in enumerate(self.axes)])
			widths = last - first
			sizes = np.prod(widths, axis=1)
			for a, b in bounded_ranges(sizes, maximum):
				lengths = sizes[a:b]
				owners = np.repeat(np.arange(a, b), lengths)
				offsets = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
				nx, ny = widths[owners, 0], widths[owners, 1]
				ix = first[owners, 0] + offsets % nx
				iy = first[owners, 1] + offsets // nx % ny
				iz = first[owners, 2] + offsets // (nx * ny)
				yield owners, self.row_grid[iz, iy, ix]
			return
		for a, b, owners, first, last in self._column_sections(lower, upper, maximum):
			lengths = last - first
			sizes = np.zeros(b - a, dtype=np.int64)
			np.add.at(sizes, owners - a, lengths)
			for c, d in bounded_ranges(sizes, maximum):
				selected = (owners >= a + c) & (owners < a + d)
				numbers = lengths[selected]
				rows = np.repeat(owners[selected], numbers)
				offsets = np.arange(int(numbers.sum())) - np.repeat(np.cumsum(numbers) - numbers, numbers)
				ids = self.column_rows[np.repeat(first[selected], numbers) + offsets]
				yield rows, ids

	def column_window_batches(self, nearest: np.ndarray, lower: np.ndarray, upper: np.ndarray, maximum: int):
		if self.row_grid is not None:
			first = np.searchsorted(self.axes[2], lower - COORDINATE_TOLERANCE)
			last = np.searchsorted(self.axes[2], upper + COORDINATE_TOLERANCE, side="right")
			xy = self.points[nearest, :2]
			ix, iy = np.searchsorted(self.axes[0], xy[:, 0]), np.searchsorted(self.axes[1], xy[:, 1])
		else:
			columns = self.point_columns[nearest]
			first, last = np.empty(len(nearest), dtype=np.int64), np.empty(len(nearest), dtype=np.int64)
			for i, column in enumerate(columns):
				a, b = self.column_offsets[column:column + 2]
				zs = self.points[self.column_rows[a:b], 2]
				first[i] = a + np.searchsorted(zs, lower[i] - COORDINATE_TOLERANCE)
				last[i] = a + np.searchsorted(zs, upper[i] + COORDINATE_TOLERANCE, side="right")
		lengths = last - first
		for a, b in bounded_ranges(lengths, maximum):
			numbers = lengths[a:b]
			rows = np.repeat(np.arange(a, b), numbers)
			offsets = np.arange(int(numbers.sum())) - np.repeat(np.cumsum(numbers) - numbers, numbers)
			positions = first[rows] + offsets
			yield rows, (self.row_grid[positions, iy[rows], ix[rows]] if self.row_grid is not None
				else self.column_rows[positions])

	def covers_cells(self, xyz: np.ndarray, maximum: int = MAX_CANDIDATES) -> np.ndarray:
		"""Query whole-cell coverage; supports_fallback checks centers for regular/clipped-grid fallback eligibility."""
		lower, upper = xyz.min(axis=1), xyz.max(axis=1)
		covered = ((lower >= np.array([a[0] for a in self.axes]) - COORDINATE_TOLERANCE).all(axis=1)
			& (upper <= np.array([a[-1] for a in self.axes]) + COORDINATE_TOLERANCE).all(axis=1))
		ids = np.flatnonzero(covered)
		if self.coverage_mode == "lattice" and len(ids):
			# Require all bracketing sample nodes for the entire cell box; handle concavities and holes conservatively.
			first = np.column_stack([np.clip(np.searchsorted(a, lower[ids, i] + COORDINATE_TOLERANCE, side="right") - 1, 0, len(a) - 1)
				for i, a in enumerate(self.axes)])
			last = np.column_stack([np.clip(np.searchsorted(a, upper[ids, i] - COORDINATE_TOLERANCE), 0, len(a) - 1)
				for i, a in enumerate(self.axes)])
			last = np.maximum(first, last)
			if self.support_prefix is not None:
				p = self.support_prefix
				x0, y0, z0 = first.T
				x1, y1, z1 = (last + 1).T
				missing = (p[z1, y1, x1] - p[z0, y1, x1] - p[z1, y0, x1] - p[z1, y1, x0]
					+ p[z0, y0, x1] + p[z0, y1, x0] + p[z1, y0, x0] - p[z0, y0, x0])
				covered[ids] = missing == 0
			else:
				node_low = np.column_stack([a[first[:, i]] for i, a in enumerate(self.axes)])
				node_high = np.column_stack([a[last[:, i]] for i, a in enumerate(self.axes)])
				counts = np.zeros(len(ids), dtype=np.int64)
				for _, _, owners, a, b in self._column_sections(node_low, node_high, maximum):
					np.add.at(counts, owners, b - a)
				expected = np.array([math.prod(map(int, widths)) for widths in last - first + 1], dtype=object)
				covered[ids] = counts == expected
		elif self.coverage_mode == "convex_hull" and len(ids):
			if not len(self.hull_equations):
				covered[ids] = False
			else:
				# Four interior vertices suffice to cover a Tet4 in a convex domain; batch hull faces to bound memory.
				x = xyz[ids] - self.hull_origin
				valid = np.ones(len(ids), dtype=bool)
				for section in range(0, len(self.hull_equations), 64):
					planes = self.hull_equations[section:section + 64]
					valid &= (x @ planes[:, :3].T + planes[:, 3] <= COORDINATE_TOLERANCE).all(axis=(1, 2))
				covered[ids] = valid
		if self.domain is not None:
			ids = np.flatnonzero(covered)
			covered[ids] &= self.domain.covers_boxes(lower[ids, :2], upper[ids, :2])
		return covered

	def supports_fallback(self, xyz: np.ndarray, maximum: int = MAX_CANDIDATES) -> np.ndarray:
		"""Regular/clipped grids require only local XYZ nodes at the center; explicit domains still constrain the entire cell."""
		if self.coverage_mode == "convex_hull":
			# Scattered samples lack a bracketing XYZ lattice; retain the existing 3D convex-hull coverage rule.
			return self.covers_cells(xyz, maximum)
		centers = xyz.mean(axis=1)
		# Reduce queries to centers, reusing tolerances, missing-node counts, and sparse-column queries.
		# A center exactly on an axis sample coordinate needs only that node, not an additional adjacent interval.
		queries = np.broadcast_to(centers[:, None, :], xyz.shape)
		supported = self.covers_cells(queries, maximum)
		if self.domain is not None:
			ids = np.flatnonzero(supported)
			supported[ids] &= self.domain.covers_boxes(xyz[ids].min(axis=1)[:, :2],
													xyz[ids].max(axis=1)[:, :2])
		return supported


@dataclass(frozen=True)
class BoundaryData:
	faces: np.ndarray       # 0-based, outward, side and bottom faces only
	owners: np.ndarray     # 0-based Tet4 row indices
	groups: np.ndarray     # Upstream groups: 1=bottom, 2..5=sides
	mesh_path: Path
	report_path: Path
	mesh_sha256: str
	report_sha256: str
	content_sha256: str

	@property
	def tags(self) -> np.ndarray:
		return np.where(self.groups == 1, 2, 1).astype(np.int32)


@dataclass
class TetMesh:
	points: np.ndarray
	cells: np.ndarray
	node_tags: np.ndarray
	element_tags: np.ndarray
	boundaries: BoundaryData | None = None


def validate_mesh(mesh: TetMesh) -> None:
	if (mesh.points.ndim != 2 or mesh.points.shape[1] != 3 or not len(mesh.points)
			or mesh.points.dtype.kind not in "iuf" or not np.isfinite(mesh.points).all()):
		raise ValueError("Tet4 nodes must be a nonempty finite N-by-3 array")
	if (mesh.cells.ndim != 2 or mesh.cells.shape[1] != 4 or not len(mesh.cells)
			or mesh.cells.dtype.kind not in "iu" or mesh.cells.min() < 0
			or mesh.cells.max() >= len(mesh.points)):
		raise ValueError("Tet4 connectivity must be a nonempty valid integer (T,4) array")
	for tags, count in ((mesh.node_tags, len(mesh.points)), (mesh.element_tags, len(mesh.cells))):
		if (tags.shape != (count,) or tags.dtype.kind not in "iu"
				or not np.array_equal(tags, np.arange(1, count + 1))):
			raise ValueError("Node/element IDs must be contiguous and 1-based in original row order")
	for start in range(0, len(mesh.cells), CHUNK_SIZE):
		stop = min(start + CHUNK_SIZE, len(mesh.cells))
		try:
			_tet.tetrahedron_metrics(mesh.points[mesh.cells[start:stop]])
		except ValueError as exc:
			raise ValueError(f"Tet4 element IDs {start + 1}..{stop}: {exc}") from exc


def _report_item(value, key: str):
	if not isinstance(value, dict) or key not in value:
		raise ValueError(f"Mesh report is missing object field {key}")
	return value[key]


def _report_array(value, shape: tuple[int | None, ...], label: str, *, integer: bool = False) -> np.ndarray:
	array = np.asarray(value)
	if (array.ndim != len(shape) or any(n is not None and n != actual for n, actual in zip(shape, array.shape))
			or array.dtype.kind not in ("iu" if integer else "iuf") or not np.isfinite(array).all()):
		raise ValueError(f"Mesh report {label} has an invalid array shape or numeric type")
	if integer and (int(array.min(initial=0)) < np.iinfo(np.int64).min
					or int(array.max(initial=0)) > np.iinfo(np.int64).max):
		raise ValueError(f"Mesh report {label} exceeds the int64 range")
	array = array.astype(np.int64 if integer else np.float64)
	_tet._match_metadata(value, array.tolist(), label)
	return array


def _boundary_digest(mesh: TetMesh, faces: np.ndarray, owners: np.ndarray, groups: np.ndarray) -> str:
	digest = hashlib.sha256()
	for array in (mesh.points, mesh.cells, faces, owners, groups):
		digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
		digest.update(np.ascontiguousarray(array).tobytes())
	return digest.hexdigest()


def _check_boundary_state(mesh: TetMesh) -> None:
	b = mesh.boundaries
	if b is None:
		return
	if _boundary_digest(mesh, b.faces, b.owners, b.groups) != b.content_sha256:
		raise ValueError("Validated mesh or boundary data was modified")
	for path, expected in ((b.mesh_path, b.mesh_sha256), (b.report_path, b.report_sha256)):
		if _tet.file_sha256(path) != expected:
			raise ValueError(f"Input file was modified during processing: {path}")


def _read_boundaries(mesh: TetMesh, path: Path, report_path: Path, digest: str) -> BoundaryData:
	"""Recover upstream layering from actual MSH and reuse topology validation; JSON is not a trusted face selector."""
	report_bytes = report_path.read_bytes()
	report = json.loads(report_bytes.decode("utf-8"), parse_constant=_tet._reject_json_constant,
						parse_float=_tet._finite_json_float)
	_tet._match_metadata(report, {"schema_version": 2, "msh_sha256": digest}, "report")
	source = _report_item(report, "source")
	n, m = (_report_item(source, key) for key in ("node_count", "triangle_count"))
	if (type(n) is not int or type(m) is not int or not 3 <= n <= len(mesh.points)
			or not 1 <= m <= len(mesh.cells) // 3):
		raise ValueError("Mesh report has invalid source node/triangle counts")
	fractions = _report_array(_report_item(report, "fractions"), (None,), "fractions")
	layers = len(fractions) - 1
	if layers < 1 or len(mesh.points) != n * (layers + 1) or len(mesh.cells) != 3 * m * layers:
		raise ValueError("Mesh counts do not match the reported source counts/layering")
	parameters = _report_item(report, "parameters")
	dz = _report_array(_report_item(parameters, "dz"), (), "parameters.dz")
	if dz <= 0:
		raise ValueError("Mesh report dz must be a finite positive number")
	max_elements = _report_item(parameters, "max_elements")
	if type(max_elements) is not int or max_elements < len(mesh.cells):
		raise ValueError("Mesh report max_elements is invalid or insufficient")

	parent = np.sort((mesh.cells[:3 * m] % n).reshape(m, 12), axis=1)
	first = np.column_stack((np.ones(m, dtype=bool), np.diff(parent, axis=1) != 0))
	if np.any(first.sum(axis=1) != 3):
		raise ValueError("Each group of three child tetrahedra in the first layer segment must belong to one source triangle")
	triangles = parent[first].reshape(m, 3)
	xyz = np.column_stack((mesh.points[:n, :2], np.zeros(n)))
	p = xyz[triangles, :2]
	reverse = _tet._tri._cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]) < 0
	triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
	boundary = _report_item(report, "boundary")
	corners = _report_array(_report_item(boundary, "input_corners"), (4, 2), "input_corners")
	ccw = _tet._tri.validate_domain_corners(corners)
	stats = _tet._tri.validate_triangle_mesh(xyz, triangles, ccw)
	mapping = np.array([1, 2, 3, 4]) if _tet._tri.polygon_area(corners) > 0 else np.array([4, 3, 2, 1])
	planar = _tet.PlanarMesh(path, report_path, "", "", xyz, triangles, corners, ccw,
							np.asarray(stats.boundary_loop), mapping[list(stats.boundary_side_ids)])
	geometry = _tet.LayeredTetraGeometry(planar, mesh.points[:n, 2], mesh.points[-n:, 2],
									   fractions, parameters)
	validated = _tet.validate_tetrahedral_mesh(geometry, xyz=mesh.points, tetrahedra=mesh.cells)
	_tet._match_metadata(boundary, {"node_id_base": 1, "owner_id_base": 1, "orientation": "outward",
		"group_names": {"0": "top", "1": "bottom", "2": "input_side_1",
						"3": "input_side_2", "4": "input_side_3", "5": "input_side_4"}}, "boundary")
	count = len(validated.boundary_faces)
	faces = _report_array(_report_item(boundary, "face_node_ids"), (count, 3), "face_node_ids", integer=True)
	owners = _report_array(_report_item(boundary, "owner_tetrahedron_ids"), (count,),
						   "owner_tetrahedron_ids", integer=True)
	groups = _report_array(_report_item(boundary, "group_ids"), (count,), "group_ids", integer=True)
	if (faces.min() < 1 or faces.max() > len(mesh.points) or owners.min() < 1
			or owners.max() > len(mesh.cells) or groups.min() < 0 or groups.max() > 5):
		raise ValueError("Mesh report boundary node/owner/group IDs are out of range")
	faces, owners = faces - 1, owners - 1
	keys, expected_keys = np.sort(faces, axis=1), np.sort(validated.boundary_faces, axis=1)
	order, expected_order = np.lexsort(keys.T[::-1]), np.lexsort(expected_keys.T[::-1])
	if (not np.array_equal(keys[order], expected_keys[expected_order])
			or not np.array_equal(_tet._orientation(faces[order]),
								  _tet._orientation(validated.boundary_faces[expected_order]))
			or not np.array_equal(owners[order], validated.boundary_owners[expected_order])
			or not np.array_equal(groups[order], validated.boundary_groups[expected_order])):
		raise ValueError("Mesh report boundary faces, outward winding, owners, or groups do not match the actual topology")
	statistics = _report_item(report, "validation")
	for key, expected in validated.stats.items():
		actual = _report_item(statistics, key)
		if isinstance(expected, float):
			value = _report_array(actual, (), f"validation.{key}")
			if not math.isclose(float(value), expected, rel_tol=1e-9):
				raise ValueError(f"Mesh report validation.{key} does not match the actual mesh")
		else:
			_tet._match_metadata(actual, expected, f"validation.{key}")
	selected = validated.boundary_groups != 0
	faces, owners, groups = (a[selected] for a in
							(validated.boundary_faces, validated.boundary_owners, validated.boundary_groups))
	return BoundaryData(faces, owners, groups, path.resolve(), report_path.resolve(), digest,
						hashlib.sha256(report_bytes).hexdigest(), _boundary_digest(mesh, faces, owners, groups))


def read_mesh(path: Path | str, *, export_boundaries: bool = False,
			  mesh_report: Path | str | None = None) -> TetMesh:
	"""Read only upstream MSH; optionally validate a companion report for side/bottom export without reordering nodes or volume cells."""
	if not export_boundaries and mesh_report is not None:
		raise ValueError("mesh_report can only be used with boundary export")
	path = Path(path)
	if path.suffix.lower() != ".msh":
		raise ValueError("Mesh input must be .msh (MSH 2.2 ASCII / type 4 / Domain3D)")
	if not path.is_file():
		raise ValueError(f"MSH does not exist: {path}")
	if gmsh.isInitialized():
		raise RuntimeError("An active Gmsh session already exists; end it first")
	digest = _tet.file_sha256(path) if export_boundaries else ""
	points, cells = _tet.read_msh22_tetrahedra(path)
	mesh = TetMesh(points, cells, np.arange(1, len(points) + 1), np.arange(1, len(cells) + 1))
	validate_mesh(mesh)
	if export_boundaries:
		report_path = Path(mesh_report) if mesh_report is not None else path.with_suffix(".report.json")
		mesh.boundaries = _read_boundaries(mesh, path, report_path, digest)
		_check_boundary_state(mesh)
	return mesh


def vote(source: SourceGrid, rows: np.ndarray, sample_ids: np.ndarray,
		 centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""One vote per sample; break modal ties by the candidate nearest the center, then by original row index."""
	rows = np.asarray(rows, dtype=np.int64)
	sample_ids = np.asarray(sample_ids, dtype=np.int64)
	centers = np.asarray(centers, dtype=float)
	n = len(centers)
	labels = np.zeros(n, dtype=np.int32)
	counts = np.bincount(rows, minlength=n)
	ties = np.zeros(n, dtype=bool)
	if not len(rows):
		return labels, counts, ties
	categories = np.searchsorted(source.codes, source.facies[sample_ids])
	table = np.bincount(rows * len(source.codes) + categories, minlength=n * len(source.codes)).reshape(n, -1)
	maxima = table.max(axis=1)
	ties = ((table == maxima[:, None]).sum(axis=1) > 1) & (counts > 0)
	eligible = table[rows, categories] == maxima[rows]
	candidates, owners = sample_ids[eligible], rows[eligible]
	distance = np.sum((source.points[candidates] - centers[owners]) ** 2, axis=1)
	ranking = np.lexsort((candidates, distance, owners))
	chosen = ranking[np.r_[True, np.diff(owners[ranking]) != 0]]
	labels[owners[chosen]] = source.facies[candidates[chosen]]
	return labels, counts, ties


def bounded_ranges(sizes: np.ndarray, maximum: int):
	"""Size sub-batches by candidates, not just cells; never expand the entire simulation domain for all cells at once."""
	if sizes.max(initial=0) > maximum:
		raise ValueError("A single cell's candidate count exceeds MAX_CANDIDATES; increase the limit or reduce mesh cell size")
	prefix = np.r_[0, np.cumsum(sizes)]
	start = 0
	while start < len(sizes):
		stop = min(len(sizes), int(np.searchsorted(prefix, prefix[start] + maximum, side="right") - 1))
		yield start, stop
		start = stop


def cell_tolerances(xyz: np.ndarray) -> np.ndarray:
	"""Per-axis physical tolerances account for input roundoff and large-coordinate ULPs; shared by search and containment."""
	return np.maximum(COORDINATE_TOLERANCE, 8 * np.abs(np.spacing(xyz)).max(axis=1))


def tet4_contains(xyz: np.ndarray, targets: np.ndarray) -> np.ndarray:
	"""Compute barycentric coordinates for validated Tet4 batches; include faces/edges/vertices within physical tolerance."""
	xyz = np.asarray(xyz, dtype=float)
	targets = np.asarray(targets, dtype=float)
	if (xyz.ndim != 3 or xyz.shape[1:] != (4, 3) or targets.shape != (len(xyz), 3)
			or not np.isfinite(xyz).all() or not np.isfinite(targets).all()):
		raise ValueError("Tet4 containment queries require finite (n,4,3) cell coordinates and (n,3) sample points")
	edges = xyz[:, 1:] - xyz[:, :1]
	try:
		local = np.linalg.solve(edges.transpose(0, 2, 1), (targets - xyz[:, 0])[..., None])[..., 0]
	except np.linalg.LinAlgError as exc:
		raise ValueError("Tet4 containment query encountered a degenerate cell") from exc
	weights = np.column_stack((1 - local.sum(axis=1), local))
	cofactors = np.stack((np.cross(edges[:, 1], edges[:, 2]),
						 np.cross(edges[:, 2], edges[:, 0]), np.cross(edges[:, 0], edges[:, 1])), axis=1)
	determinant = np.einsum("ni,ni->n", edges[:, 0], cofactors[:, 0])
	if np.any(determinant <= 0) or not np.isfinite(determinant).all():
		raise ValueError("Tet4 containment query encountered a non-positive-volume cell")
	gradients = cofactors / determinant[:, None, None]
	gradients = np.concatenate((-gradients.sum(axis=1, keepdims=True), gradients), axis=1)
	tolerance = cell_tolerances(xyz)
	weight_tolerance = np.sum(np.abs(gradients) * tolerance[:, None, :], axis=2)
	weight_tolerance += 32 * np.finfo(float).eps * np.maximum(1., np.abs(weights).sum(axis=1))[:, None]
	return ((weights >= -weight_tolerance).all(axis=1)
			& (targets >= xyz.min(axis=1) - tolerance).all(axis=1)
			& (targets <= xyz.max(axis=1) + tolerance).all(axis=1))


def inside_samples(source: SourceGrid, xyz: np.ndarray, maximum: int):
	"""Use bounding boxes only to find candidates; barycentric coordinates determine actual tetrahedron containment."""
	extra = cell_tolerances(xyz) - COORDINATE_TOLERANCE
	lower, upper = xyz.min(axis=1) - extra, xyz.max(axis=1) + extra
	for rows, sample_ids in source.candidate_batches(lower, upper, maximum):
		if not len(rows):
			yield rows, sample_ids
			continue
		inside_rows, inside_ids = [], []
		for first_candidate in range(0, len(rows), CONTAINMENT_BATCH_SIZE):
			section = slice(first_candidate, first_candidate + CONTAINMENT_BATCH_SIZE)
			owners, candidates = rows[section], sample_ids[section]
			contained = tet4_contains(xyz[owners], source.points[candidates])
			inside_rows.append(owners[contained])
			inside_ids.append(candidates[contained])
		yield np.concatenate(inside_rows), np.concatenate(inside_ids)


@dataclass
class Assignment:
	lithology: np.ndarray
	method: np.ndarray
	inside_count: np.ndarray
	sample_count: np.ndarray
	tied_mode: np.ndarray
	cell_dz: np.ndarray


def assign_empty_cells(source: SourceGrid, xyz: np.ndarray, targets: np.ndarray,
					   result: Assignment, maximum: int) -> None:
	"""Assign locally supported empty cells using the nearest point or the vertical mode in the nearest 3D sample's column."""
	centers = xyz.mean(axis=1)
	nearest = source.nearest(centers)
	result.lithology[targets] = source.facies[nearest]
	result.method[targets] = 3
	result.sample_count[targets] = 1
	thick = result.cell_dz[targets] > source.vertical_spacings(nearest) * (1 + 1e-6)
	positions = np.flatnonzero(thick)
	if len(positions):
		zmin, zmax = xyz[positions, :, 2].min(axis=1), xyz[positions, :, 2].max(axis=1)
		for rows, sample_ids in source.column_window_batches(nearest[thick], zmin, zmax, maximum):
			labels, counts, ties = vote(source, rows, sample_ids, centers[positions])
			found = counts > 0
			ids = targets[positions[found]]
			result.lithology[ids], result.method[ids] = labels[found], 2
			result.sample_count[ids], result.tied_mode[ids] = counts[found], ties[found]


def assign_facies(mesh: TetMesh, source: SourceGrid, *, undefined: int = UNDEFINED_FACIES,
				  chunk_size: int = CHUNK_SIZE, max_candidates: int = MAX_CANDIDATES,
				  verbose: bool = True) -> Assignment:
	if isinstance(undefined, bool) or not isinstance(undefined, int) or not 1 <= undefined <= np.iinfo(np.int32).max:
		raise ValueError("Undefined lithology must be a positive int32 code")
	if undefined in source.codes:
		raise ValueError("Undefined lithology code duplicates an actual OUT lithology; choose another UNDEFINED_FACIES")
	if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
			or isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0):
		raise ValueError("CHUNK_SIZE and MAX_CANDIDATES must be positive integers")
	validate_mesh(mesh)
	n = len(mesh.cells)
	result = Assignment(np.full(n, undefined, dtype=np.int32), np.zeros(n, dtype=np.int32),
						np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
						np.zeros(n, dtype=bool), np.empty(n))
	for start in range(0, n, chunk_size):
		stop = min(start + chunk_size, n)
		xyz = mesh.points[mesh.cells[start:stop]]
		lower, upper = xyz.min(axis=1), xyz.max(axis=1)
		result.cell_dz[start:stop] = upper[:, 2] - lower[:, 2]
		# Check actual inside samples first; vertices beyond sample bounds or the convex hull must not skip voting.
		local = (np.arange(len(xyz)) if source.domain is None
				 else np.flatnonzero(source.domain.covers_boxes(lower[:, :2], upper[:, :2])))
		targets = start + local
		if len(local):
			xyz = xyz[local]
			centers = xyz.mean(axis=1)
			for rows, sample_ids in inside_samples(source, xyz, max_candidates):
				labels, counts, ties = vote(source, rows, sample_ids, centers)
				found = counts > 0
				ids = targets[found]
				result.lithology[ids], result.method[ids] = labels[found], 1
				result.inside_count[ids] = counts[found]
				result.sample_count[ids], result.tied_mode[ids] = counts[found], ties[found]

			empty = np.flatnonzero(result.inside_count[targets] == 0)
			if len(empty):
				empty = empty[source.supports_fallback(xyz[empty], max_candidates)]
			if len(empty):
				assign_empty_cells(source, xyz[empty], targets[empty], result, max_candidates)
		if verbose and (stop == n or start // chunk_size % 25 == 0):
			print(f"Assigned {stop:,}/{n:,} tetrahedra", flush=True)
	return result


def _vtk_fields(mesh: TetMesh, result: Assignment, output_mode: str) -> dict[str, np.ndarray]:
	if output_mode not in ("simple", "debug"):
		raise ValueError("OUTPUT_MODE must be 'simple' or 'debug'")
	fields = {"lithology": result.lithology}
	if output_mode == "debug":
		fields.update({"assignment_method": result.method,
					   "outside_simulation": (result.method == 0).astype(np.int32),
					   "inside_sample_count": result.inside_count, "sample_count": result.sample_count,
					   "tied_mode": result.tied_mode.astype(np.int32), "original_element_tag": mesh.element_tags})
	for values in fields.values():
		if values.shape != (len(mesh.cells),) or values.dtype.kind not in "iu":
			raise ValueError("VTK cell attributes have an invalid length or integer type")
		if values.max(initial=0) > np.iinfo(np.int64).max or values.min(initial=0) < np.iinfo(np.int64).min:
			raise ValueError("VTK attributes exceed the int64 range")
	if np.any(result.lithology <= 0):
		raise ValueError("VTK lithology must contain one positive integer per cell")
	if mesh.boundaries is not None:
		fields = {name: np.concatenate((values, mesh.boundaries.tags if name == "lithology"
										else np.full(len(mesh.boundaries.faces), -1, dtype=np.int64)))
				  for name, values in fields.items()}
	return fields


def _vtk_integer_type(values: np.ndarray) -> str:
	return ("int" if values.min(initial=0) >= np.iinfo(np.int32).min
			and values.max(initial=0) <= np.iinfo(np.int32).max else "long")


def _write_vtk20_ascii(stream, mesh: TetMesh, fields: dict[str, np.ndarray], output_mode: str) -> None:
	faces = mesh.boundaries.faces if mesh.boundaries is not None else np.empty((0, 3), dtype=np.int64)
	total = len(mesh.cells) + len(faces)
	stream.write(
		f"# vtk DataFile Version {VTK_VERSION}\nTetrahedral mesh with facies\n"
		f"{VTK_ENCODING}\nDATASET UNSTRUCTURED_GRID\n"
	)
	stream.write(f"POINTS {len(mesh.points)} double\n")
	np.savetxt(stream, mesh.points, fmt="%.17g")
	stream.write(f"CELLS {total} {5 * len(mesh.cells) + 4 * len(faces)}\n")
	for start in range(0, len(mesh.cells), 100_000):
		cells = mesh.cells[start:start + 100_000]
		np.savetxt(stream, np.column_stack((np.full(len(cells), 4), cells)), fmt="%d")
	for start in range(0, len(faces), 100_000):
		part = faces[start:start + 100_000]
		np.savetxt(stream, np.column_stack((np.full(len(part), 3), part)), fmt="%d")
	stream.write(f"CELL_TYPES {total}\n")
	for start in range(0, len(mesh.cells), 100_000):
		stream.write("10\n" * min(100_000, len(mesh.cells) - start))
	for start in range(0, len(faces), 100_000):
		stream.write("5\n" * min(100_000, len(faces) - start))
	stream.write(f"CELL_DATA {total}\n")
	for name, values in fields.items():
		stream.write(f"SCALARS {name} {_vtk_integer_type(values)} 1\nLOOKUP_TABLE default\n")
		np.savetxt(stream, values, fmt="%d")
	if output_mode == "debug":
		stream.write(f"POINT_DATA {len(mesh.points)}\nSCALARS original_node_tag long 1\nLOOKUP_TABLE default\n")
		np.savetxt(stream, mesh.node_tags, fmt="%d")


def _verify_vtk(path: Path, mesh: TetMesh, fields: dict[str, np.ndarray], output_mode: str) -> None:
	"""Strictly read back this writer's subset and compare values in chunks; do not use the upstream all-1 facies reader."""
	faces = mesh.boundaries.faces if mesh.boundaries is not None else np.empty((0, 3), dtype=np.int64)
	total = len(mesh.cells) + len(faces)
	with path.open(encoding="ascii") as stream:
		def expect(value):
			if stream.readline().rstrip("\n") != value:
				raise ValueError(f"VTK read-back format mismatch; expected {value}")

		def compare(expected, dtype):
			expected = np.asarray(expected)
			if expected.ndim == 1:
				expected = expected[:, None]
			width = expected.shape[1]
			for start in range(0, len(expected), 100_000):
				part = expected[start:start + 100_000]
				converters = {i: int for i in range(width)} if dtype == np.int64 else None
				lines = list(itertools.islice(stream, len(part)))
				if len(lines) != len(part) or any(not line.strip() for line in lines):
					raise ValueError("VTK read-back data rows are missing")
				actual = np.loadtxt(lines, dtype=dtype,
									ndmin=2, comments=None, converters=converters)
				if not np.array_equal(actual, part):
					raise ValueError("VTK read-back coordinates, connectivity, or attributes do not match the data to be written")

		expect(f"# vtk DataFile Version {VTK_VERSION}")
		expect("Tetrahedral mesh with facies")
		expect(VTK_ENCODING)
		expect("DATASET UNSTRUCTURED_GRID")
		expect(f"POINTS {len(mesh.points)} double")
		compare(mesh.points, float)
		expect(f"CELLS {total} {5 * len(mesh.cells) + 4 * len(faces)}")
		for start in range(0, len(mesh.cells), 100_000):
			cells = mesh.cells[start:start + 100_000]
			compare(np.column_stack((np.full(len(cells), 4), cells)), np.int64)
		for start in range(0, len(faces), 100_000):
			part = faces[start:start + 100_000]
			compare(np.column_stack((np.full(len(part), 3), part)), np.int64)
		expect(f"CELL_TYPES {total}")
		compare(np.full(len(mesh.cells), 10), np.int64)
		compare(np.full(len(faces), 5), np.int64)
		expect(f"CELL_DATA {total}")
		for name, values in fields.items():
			expect(f"SCALARS {name} {_vtk_integer_type(values)} 1")
			expect("LOOKUP_TABLE default")
			compare(values, np.int64)
		if output_mode == "debug":
			expect(f"POINT_DATA {len(mesh.points)}")
			expect("SCALARS original_node_tag long 1")
			expect("LOOKUP_TABLE default")
			compare(mesh.node_tags, np.int64)
		if stream.read().strip():
			raise ValueError("VTK read-back contains unexpected extra content")


def _same_file(a: Path, b: Path) -> bool:
	return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def protect_outputs(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
	for i, target in enumerate(outputs):
		if any(_same_file(target, source) for source in inputs + outputs[:i]):
			raise ValueError("Outputs must not overwrite input files or refer to the same file as another output")


def write_vtk(path: Path | str, mesh: TetMesh, result: Assignment, *, output_mode: str = OUTPUT_MODE,
			  report: str | None = None, input_paths: tuple[Path, ...] = ()) -> None:
	"""Stage and validate VTK and an optional text report; commit only after all preparation succeeds."""
	path = Path(path)
	if path.suffix.lower() != ".vtk":
		raise ValueError("Output must end in .vtk")
	report_path = path.with_suffix(".txt")
	outputs = (path, report_path) if report is not None else (path,)
	if mesh.boundaries is not None:
		input_paths += (mesh.boundaries.mesh_path, mesh.boundaries.report_path)
	protect_outputs(outputs, input_paths)
	validate_mesh(mesh)
	_check_boundary_state(mesh)
	fields = _vtk_fields(mesh, result, output_mode)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporaries = []
	try:
		with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n", dir=path.parent,
										 prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
			temporary = Path(stream.name)
			temporaries.append(temporary)
			_write_vtk20_ascii(stream, mesh, fields, output_mode)
			stream.flush()
			os.fsync(stream.fileno())
		_verify_vtk(temporary, mesh, fields, output_mode)
		if report is not None:
			with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
											 prefix=f".{report_path.name}.", suffix=".tmp", delete=False) as stream:
				temporary_report = Path(stream.name)
				temporaries.append(temporary_report)
				stream.write(report)
				stream.flush()
				os.fsync(stream.fileno())
			if temporary_report.read_text(encoding="utf-8") != report:
				raise ValueError("Text report read-back does not match")
		protect_outputs(outputs, input_paths)
		_check_boundary_state(mesh)
		if report is not None:
			temporary_report.replace(report_path)
		temporary.replace(path)
	finally:
		for temporary in temporaries:
			temporary.unlink(missing_ok=True)


def make_report(mesh: TetMesh, source: SourceGrid, result: Assignment, undefined: int) -> str:
	lines = ["FACIES ASSIGNMENT REPORT", f"VTK format: legacy {VTK_VERSION} {VTK_ENCODING}",
			 f"Nodes: {len(mesh.points)}; tetrahedra: {len(mesh.cells)}",
			 f"Source samples: {len(source.points)}; source type: {source.source_type}",
			 f"Coverage mode: {source.coverage_mode}; explicit XY domain: {source.domain is not None}",
			 f"Source dz summary (median): {source.dz:g}; fallback dz override: {source.fallback_dz}",
			 f"Mesh vertical extents: {result.cell_dz.min():.8g} .. {result.cell_dz.max():.8g}",
			 f"Cells thicker than summary dz: {int((result.cell_dz > source.dz * (1 + 1e-6)).sum())}",
			 f"Undefined facies: {undefined}"]
	if source.column_offsets is not None:
		lengths = np.diff(source.column_offsets)
		lines += [f"Actual XY columns: {len(lengths)}; samples per column: {lengths.min()} .. {lengths.max()}"]
	for method, name in METHOD_NAMES.items():
		count = int(np.count_nonzero(result.method == method))
		lines.append(f"Method {method} ({name}): {count} ({100 * count / len(mesh.cells):.4f}%)")
	lines.append(f"Tied modes: {int(result.tied_mode.sum())}")
	codes, counts = np.unique(result.lithology, return_counts=True)
	lines += [f"Lithology {code}: {count}" for code, count in zip(codes, counts)]
	lines.append(f"Boundary export: {mesh.boundaries is not None}")
	if mesh.boundaries is not None:
		b = mesh.boundaries
		side_nodes = np.unique(b.faces[b.tags == 1])
		bottom_nodes = np.unique(b.faces[b.tags == 2])
		lines += [f"Mesh report: {b.report_path}", f"Input MSH SHA256: {b.mesh_sha256}",
				  f"Input report SHA256: {b.report_sha256}",
				  "Boundary labels: side=1; bottom=2; top triangles not exported"]
		for tag, name in ((1, "side"), (2, "bottom")):
			faces = b.faces[b.tags == tag]
			lines.append(f"Boundary {name} (tag {tag}): faces={len(faces)}; nodes={len(np.unique(faces))}")
		shared = len(np.intersect1d(side_nodes, bottom_nodes))
		lines += [f"Shared side/bottom nodes: {shared}",
				  f"Bottom-priority side nodes: {len(side_nodes) - shared}",
				  "MATLAB node IDs are 1-based; use setdiff(sideNodesAll, bottomNodes) for bottom priority.",
				  "Boundary labels are stored in the first CELL_DATA scalar; material statistics count Tet4 only."]
	lines += ["", "Interpretation:",
			  "- All actual inside-tetrahedron 3D source points vote equally (NOT a mode of column modes).",
			  "- Face/edge boundary samples are included within numerical tolerance; shared samples may vote in both cells.",
			  "- Ties: nearest eligible candidate to four-vertex centroid, then smallest original PLT row number.",
			  "- Cells with inside samples vote even when their vertices extend beyond source bounds, lattice support or convex hull.",
			  "- No inside point with local source support: nearest 3D sample's column + cell Z window mode for thick cells; otherwise nearest 3D point.",
			  "- Thick-cell threshold uses that column's median adjacent Z spacing, or the explicit fallback dz.",
			  "- Unsupported empty cells and excluded XY domains remain undefined; outside_simulation means method=0.",
			  "- Source and mesh must share CRS, units and elevation datum. No reprojection or Z sign conversion.",
			  "- Negative source codes are converted to absolute values before voting; PLT auxiliary columns are ignored.",
			  "- Regular, clipped/columnar and scattered XYZ inputs are accepted. Voting is not volume-weighted.",
			  "- MSH coordinates, connectivity and original row order are preserved; no upstream VTK is read.",
			  f"- ASCII VTK {VTK_VERSION} contains only finite integer attributes; lithology is CELL_DATA."]
	if source.coverage_mode == "lattice":
		lines.append("- Empty-cell fallback requires all bracketing XYZ lattice nodes at the cell center; missing nodes outside this local stencil do not reject fallback.")
	elif source.coverage_mode == "bounds":
		lines.append("- Complete-grid empty-cell fallback requires the cell center within source XYZ bounds.")
	elif source.coverage_mode == "convex_hull":
		lines.append("- Empty-cell fallback assumes the 3D convex hull is valid between samples; supply an XY domain to exclude concavities/holes.")
	if source.domain is not None:
		lines.append("- The explicit XY domain must conservatively cover the entire cell XY bounding box, including cells with inside samples.")
	lines.append(f"- Containment physical tolerance: max({COORDINATE_TOLERANCE:g}, 8 coordinate ULPs) per axis.")
	return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--mesh", type=Path, default=MESH_FILE,
						help="Upstream MSH 2.2 ASCII / type 4 / Domain3D; companion JSON required by default; upstream VTK is not read")
	parser.add_argument("--export-boundaries", action=argparse.BooleanOptionalAction, default=EXPORT_BOUNDARIES,
						help="Append Tri3 with side tag 1 and bottom tag 2 (enabled by default); --no-export-boundaries outputs pure Tet4")
	parser.add_argument("--mesh-report", type=Path, default=MESH_REPORT,
						help="Report required for boundary export; defaults to the input MSH's same-stem .report.json; cannot be set with boundaries disabled")
	parser.add_argument("--coordinates", type=Path, default=COORDINATES_FILE)
	parser.add_argument("--out-file", type=Path, default=OUT_FILE)
	parser.add_argument("--xyz-columns", type=int, nargs=3, default=XYZ_COLUMNS, metavar=("X", "Y", "Z"),
						help="Zero-based XYZ column indices; default 0 1 2; use 1 2 3 with a leading ID")
	parser.add_argument("--coverage-mode", choices=("auto", "lattice", "convex_hull"), default=COVERAGE_MODE,
						help="Fallback eligibility for empty cells: auto detects coverage; lattice checks bracketing XYZ nodes at the center; convex_hull retains whole-cell hull checks")
	parser.add_argument("--domain-xy", type=Path, default=DOMAIN_XY_FILE,
						help="Optional horizontal GeoJSON Polygon/MultiPolygon domain with concavities and holes; must use the same coordinate system as XYZ")
	parser.add_argument("--fallback-dz", type=float, default=FALLBACK_DZ,
						help="Optional positive override for the thick-cell threshold in vertical fallback; defaults to the column's median adjacent Z spacing")
	parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
	parser.add_argument("--output-mode", choices=("simple", "debug"), default=OUTPUT_MODE,
						help="simple retains only lithology; debug retains all debug attributes (does not affect the text report)")
	parser.add_argument("--undefined-facies", type=int, default=UNDEFINED_FACIES)
	parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
	parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
	parser.add_argument("--quiet", action="store_true")
	args = parser.parse_args(argv)
	try:
		if args.output.suffix.lower() != ".vtk":
			raise ValueError("Output must end in .vtk")
		if not args.export_boundaries and args.mesh_report is not None:
			raise ValueError("--mesh-report can only be used with boundary export")
		report_path = args.output.with_suffix(".txt")
		input_paths = [args.mesh, args.coordinates, args.out_file]
		if args.domain_xy is not None:
			input_paths.append(args.domain_xy)
		if args.export_boundaries:
			input_paths.append(args.mesh_report if args.mesh_report is not None
							   else args.mesh.with_suffix(".report.json"))
		protect_outputs((args.output, report_path), tuple(input_paths))
		domain = XYDomain.read(args.domain_xy) if args.domain_xy is not None else None
		points, grid = read_coordinates(args.coordinates, xyz_columns=tuple(args.xyz_columns))
		facies = read_realization(args.out_file, len(points))  # No caching, random selection, or aggregation of other OUT files
		source = SourceGrid.build(points, facies, grid, coverage_mode=args.coverage_mode,
							  domain=domain, fallback_dz=args.fallback_dz)
		mesh = read_mesh(args.mesh, export_boundaries=args.export_boundaries, mesh_report=args.mesh_report)
		result = assign_facies(mesh, source, undefined=args.undefined_facies, chunk_size=args.chunk_size,
							   max_candidates=args.max_candidates, verbose=not args.quiet)
		report = (f"Mesh: {args.mesh.resolve()}\nCoordinates: {args.coordinates.resolve()}\n"
				  f"XYZ columns (zero-based): {tuple(args.xyz_columns)}\n"
				  f"Domain XY: {args.domain_xy.resolve() if args.domain_xy is not None else 'none'}\n"
				  f"Realization: {args.out_file.resolve()}\n"
				  + make_report(mesh, source, result, args.undefined_facies))
		write_vtk(args.output, mesh, result, output_mode=args.output_mode,
				  report=report, input_paths=tuple(input_paths))
	except (OSError, ValueError, RuntimeError) as exc:
		parser.exit(2, f"error: {exc}\n")
	print(report, end="")
	print(f"VTK: {args.output.resolve()}\nReport: {report_path.resolve()}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
