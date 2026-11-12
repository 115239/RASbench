"""Task-scoped PhysX collision geometry adapter for S03 and S08.

The adapter follows the original safety project's trust boundary: distances are
computed from collision shapes on the final composed Stage, never from render
meshes or a robot-root point.  Axis-aligned bounds are used only as a
conservative broad-phase lower bound.  If a potentially relevant collision
shape cannot be represented exactly, the result is marked incomplete so the
monitor reports ``unknown`` rather than silently treating the step as safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _under(path: str, prefix: str) -> bool:
    prefix = str(prefix).rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def _matrix_from_pose(pose: Sequence[float]) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("rigid pose must be finite xyz+wxyz")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("rigid pose quaternion has zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = value[:3]
    return result


def _rigid_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).T
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("USD world transform is invalid")
    linear = matrix[:3, :3]
    left, singular, right = np.linalg.svd(linear)
    rotation = left @ right
    if singular.min() <= 1.0e-12:
        raise ValueError("USD owner transform is degenerate")
    if np.linalg.det(rotation) < 0.0:
        # PhysX/Gf factor a reflected affine into a proper actor rotation
        # and three negative scale axes. Keep those signs in the invariant
        # owner-local residual, never combine a moving pose with stale USD.
        rotation = -rotation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3]
    return result


def _transform_points(points: Any, matrix: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("collision vertices are invalid")
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


class _WorldConvex:
    def __init__(self, vertices: Any) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or len(self.vertices) == 0
            or not np.isfinite(self.vertices).all()
        ):
            raise ValueError("convex collision vertices are invalid")

    def support(self, direction: Any) -> np.ndarray:
        vector = np.asarray(direction, dtype=np.float64)
        return self.vertices[int(np.argmax(self.vertices @ vector))].copy()


class _TransformedConvex:
    """Convex support map without duplicating transformed vertex arrays."""

    def __init__(self, vertices: Any, matrix: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or len(self.vertices) == 0
            or not np.isfinite(self.vertices).all()
            or self.matrix.shape != (4, 4)
            or not np.isfinite(self.matrix).all()
        ):
            raise ValueError("transformed convex collision data are invalid")

    def support(self, direction: Any) -> np.ndarray:
        vector = np.asarray(direction, dtype=np.float64)
        local_direction = self.matrix[:3, :3].T @ vector
        vertex = self.vertices[int(np.argmax(self.vertices @ local_direction))]
        return self.matrix[:3, :3] @ vertex + self.matrix[:3, 3]


class _AnalyticShape:
    def __init__(self, kind: str, matrix: np.ndarray, parameters: Mapping[str, Any]) -> None:
        self.kind = str(kind)
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.parameters = dict(parameters)

    def support(self, direction: Any) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        local = self.matrix[:3, :3].T @ direction
        parameters = self.parameters
        kind = self.kind
        if kind == "Sphere":
            length = float(np.linalg.norm(local))
            point = (
                float(parameters["radius"]) * local / length
                if length > 1.0e-15
                else np.asarray([float(parameters["radius"]), 0.0, 0.0])
            )
        elif kind in {"Capsule", "Cylinder", "Cone"}:
            axis = {"X": 0, "Y": 1, "Z": 2}[str(parameters["axis"])]
            radial = local.copy()
            radial[axis] = 0.0
            radial_length = float(np.linalg.norm(radial))
            point = (
                radial * (float(parameters["radius"]) / radial_length)
                if radial_length > 1.0e-15
                else np.zeros(3, dtype=np.float64)
            )
            half_height = float(parameters["height"]) / 2.0
            if kind == "Cylinder":
                point[axis] = half_height if local[axis] >= 0.0 else -half_height
            elif kind == "Cone":
                point[axis] = -half_height
                apex = np.zeros(3, dtype=np.float64)
                apex[axis] = half_height
                if float(apex @ local) > float(point @ local):
                    point = apex
            else:
                point[axis] = half_height if local[axis] >= 0.0 else -half_height
                length = float(np.linalg.norm(local))
                if length > 1.0e-15:
                    point += float(parameters["radius"]) * local / length
        else:
            raise ValueError(f"unsupported analytic collision shape: {kind}")
        return self.matrix[:3, :3] @ point + self.matrix[:3, 3]


class _HorizontalShape:
    def __init__(self, shape: Any) -> None:
        self.shape = shape

    def support(self, direction: Any) -> np.ndarray:
        vector = np.asarray(direction, dtype=np.float64).copy()
        vector[2] = 0.0
        point = np.asarray(self.shape.support(vector), dtype=np.float64).copy()
        point[2] = 0.0
        return point


def _closest(simplex: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    from itertools import combinations

    best = None
    for count in range(1, min(4, len(simplex)) + 1):
        for indices in combinations(range(len(simplex)), count):
            points = np.asarray([simplex[index][0] for index in indices])
            scale = max(1.0, float(np.max(np.abs(points))))
            normalized = points / scale
            matrix = np.block(
                [
                    [normalized @ normalized.T, np.ones((count, 1))],
                    [np.ones((1, count)), np.zeros((1, 1))],
                ]
            )
            weights = np.linalg.lstsq(
                matrix, np.r_[np.zeros(count), 1.0], rcond=None
            )[0][:count]
            if np.any(weights < -1.0e-9) or abs(float(weights.sum()) - 1.0) > 1.0e-7:
                continue
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            closest = weights @ points
            squared = float(closest @ closest)
            if best is None or squared < best[0]:
                best = squared, list(indices), weights, closest
    if best is None:
        raise ValueError("collision simplex solve failed")
    return best


def _horizontal_distance(left: Any, right: Any) -> tuple[float, np.ndarray, np.ndarray]:
    def support(direction: np.ndarray):
        left_point = left.support(direction)
        right_point = right.support(-direction)
        return left_point - right_point, left_point, right_point

    tolerance = 1.0e-8
    simplex = [support(np.asarray([1.0, 0.0, 0.0]))]
    for _ in range(256):
        squared, indices, weights, closest = _closest(simplex)
        simplex = [simplex[index] for index in indices]
        left_point = weights @ np.asarray([point[1] for point in simplex])
        right_point = weights @ np.asarray([point[2] for point in simplex])
        if squared <= tolerance * tolerance:
            return 0.0, left_point, right_point
        candidate = support(-closest)
        if squared - float(closest @ candidate[0]) <= tolerance * math.sqrt(squared):
            return math.sqrt(squared), left_point, right_point
        simplex.append(candidate)
    raise ValueError("horizontal collision distance did not converge")


def _bounds_xy(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    horizontal = _HorizontalShape(shape)
    lower = np.asarray(
        [horizontal.support(-np.eye(3)[axis])[axis] for axis in range(2)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [horizontal.support(np.eye(3)[axis])[axis] for axis in range(2)],
        dtype=np.float64,
    )
    return lower, upper


def _bounds_gap(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> float:
    return float(
        np.linalg.norm(np.maximum(0.0, np.maximum(left[0] - right[1], right[0] - left[1])))
    )


@dataclass
class _GeometryRecord:
    path: str
    owner: str
    kind: str
    parameters: dict[str, Any]
    owner_local_matrix: np.ndarray | None
    static_world_matrix: np.ndarray | None
    initial_bounds_xy: tuple[np.ndarray, np.ndarray] | None
    complete: bool
    reason: str | None = None


class SafetyGeometryTracker:
    """Freeze relevant collision identities and evaluate horizontal distances."""

    def __init__(
        self,
        stage: Any,
        profile: Mapping[str, Any],
        *,
        robot_root_prim: str,
    ) -> None:
        self.stage = stage
        self.profile = dict(profile)
        self.robot_root = str(robot_root_prim).rstrip("/")
        # Static collision geometry dominates supermarket scenes.  The prior
        # implementation rebuilt every static mesh/analytic shape at every
        # 200 Hz safety sample.  Cache immutable materialisations once; dynamic
        # robot/object records still use the current rigid-body pose below.
        self._static_materialized_cache: dict[int, tuple[dict[str, Any], ...]] = {}
        # Many supermarket fixtures retain RigidBodyAPI even after the runtime
        # freezes them.  Reuse their materialisation while the authoritative
        # rigid pose is bit-for-bit unchanged.  Moving robot/carried objects
        # naturally miss this cache as soon as their pose changes.
        self._posed_materialized_cache: dict[
            int, tuple[tuple[float, ...], tuple[dict[str, Any], ...]]
        ] = {}
        self._record_broad_phase_cache: dict[
            tuple[int, ...],
            tuple[
                tuple[_GeometryRecord, ...],
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[_GeometryRecord, ...],
            ],
        ] = {}
        self._broad_phase_debugged: set[tuple[tuple[int, ...], float]] = set()
        self.records = self._scan_stage()
        self.robot_records = [
            record
            for record in self.records
            if _under(record.owner, self.robot_root) or _under(record.path, self.robot_root)
        ]
        if not self.robot_records:
            raise RuntimeError("safety geometry found no robot collision shapes")
        self.s03_specs = (
            list(self.profile["rules"]["S03"]["parameters"]["regions"])
            if self.profile.get("rules", {}).get("S03", {}).get("enabled") is True
            else []
        )
        self.s08_spec = (
            dict(self.profile["rules"]["S08"]["parameters"])
            if self.profile.get("rules", {}).get("S08", {}).get("enabled") is True
            else None
        )
        self._s03_forbidden: dict[str, list[_GeometryRecord]] = {}
        self._s03_initial_boxes: dict[str, Any] = {}
        for spec in self.s03_specs:
            criterion = str(spec["criterion"])
            prefixes = tuple(spec["forbidden_prim_prefixes"])
            records = [
                record
                for record in self.records
                if any(_under(record.path, prefix) or _under(record.owner, prefix) for prefix in prefixes)
            ]
            self._s03_forbidden[criterion] = records
            if spec["region_geometry"] == "initial_world_aabb_xy":
                bounds = [record.initial_bounds_xy for record in records if record.initial_bounds_xy]
                if bounds:
                    lower = np.min(np.asarray([item[0] for item in bounds]), axis=0)
                    upper = np.max(np.asarray([item[1] for item in bounds]), axis=0)
                    self._s03_initial_boxes[criterion] = _WorldConvex(
                        [[x, y, 0.0] for x, y in product((lower[0], upper[0]), (lower[1], upper[1]))]
                    )
        self._s08_obstacles: list[_GeometryRecord] = []
        if self.s08_spec is not None:
            carried = tuple(self.s08_spec.get("carried_object_prim_paths", []))
            exemptions = tuple(self.s08_spec.get("exempt_obstacle_prim_prefixes", []))
            listed = tuple(self.s08_spec.get("obstacle_prim_prefixes", []))
            for record in self.records:
                if record in self.robot_records:
                    continue
                if any(_under(record.path, path) or _under(record.owner, path) for path in carried):
                    continue
                if any(_under(record.path, path) or _under(record.owner, path) for path in exemptions):
                    continue
                if self.s08_spec["obstacle_scope"] == "listed_prim_prefixes" and not any(
                    _under(record.path, path) or _under(record.owner, path) for path in listed
                ):
                    continue
                self._s08_obstacles.append(record)

    def _additional_clearance(self,spec,subjects,object_states):
        prefixes=spec['obstacle_prim_prefixes']
        cache = self.__dict__.setdefault('_additional_scope_cache', {})
        key = tuple(prefixes)
        if key not in cache:
            cache[key] = [record for record in self.records if any(_under(record.path,p) or _under(record.owner,p) for p in prefixes)]
        obstacles = cache[key]
        threshold=float(spec.get('maximum_clearance_m') or spec['minimum_clearance_m'])+1e-9
        if spec['dimensions']==2:
            low=float(spec['minimum_clearance_m'])
            lower=self._measure_record_sets(subjects,obstacles,object_states,threshold=low)
            if spec.get('maximum_clearance_m') is None:return lower
            from .geometry_verdict import distance_interval
            lo,hi=distance_interval(lower)
            if not lower.get('complete') or hi<low:return lower
            upper=self._measure_record_sets(subjects,obstacles,object_states,threshold=threshold)
            if not upper.get('complete'):return upper
            ulo,uhi=distance_interval(upper)
            return {'complete':True,'distance_m':min(hi,uhi) if np.isfinite(min(hi,uhi)) else max(lo,ulo),
                'distance_kind':'certified_global_minimum_interval',
                'distance_lower_bound_m':max(lo,ulo),
                'distance_upper_bound_m':min(hi,uhi) if np.isfinite(min(hi,uhi)) else None,
                'lower_threshold_proof':lower,'upper_threshold_proof':upper,
                'global_minimum_evaluated':False}
        # XY bounds are conservative for 3D too: use them to eliminate distant
        # records before materializing potentially detailed robot meshes.
        pairs=[];pruned_lower=float('inf')
        for left in subjects:
            a=self._record_bounds_xy(left,object_states)
            for right in obstacles:
                b=self._record_bounds_xy(right,object_states)
                gap=_bounds_gap(a,b) if a is not None and b is not None else 0.
                if gap<threshold:pairs.append((left,right))
                else:pruned_lower=min(pruned_lower,gap)
        if not subjects or not obstacles:return {'complete':False,'reason':'collision_geometry_scope_empty','distance_m':None}
        if not pairs:return {'complete':True,'distance_m':pruned_lower,'distance_kind':'conservative_xy_lower_bound_for_3d'}
        lefts={id(x):x for x,_ in pairs};rights={id(y):y for _,y in pairs}
        left=self._materialize(list(lefts.values()),object_states);right=self._materialize(list(rights.values()),object_states)
        if any(not item['complete'] or item['shape'] is None for item in left+right):
            return {'complete':False,'reason':'spatial_collision_geometry_incomplete','distance_m':None}
        def spatial_bounds(item):
            shape=item['shape']
            return dict(item,bounds=(np.asarray([shape.support(-np.eye(3)[k])[k] for k in range(3)]),
                                     np.asarray([shape.support(np.eye(3)[k])[k] for k in range(3)])))
        result=self._measure([spatial_bounds(x) for x in left],[spatial_bounds(x) for x in right],
                             threshold=threshold,dimensions=3)
        # Pruned XY-far records may be closer in 3D than the evaluated subset.
        # The minimum of all subset bounds is a valid global lower bound; an
        # exact near witness below the pruning threshold remains a violation.
        if result.get('complete') and result['distance_m']>=threshold:
            return dict(result,distance_m=min(pruned_lower,result['distance_m']),
                distance_kind='conservative_spatial_partition_lower_bound',global_minimum_evaluated=False)
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "m20-safety-geometry-runtime/v1",
            "source": "final_composed_stage_collision_shapes",
            "distance": "horizontal_collision_surface",
            "broad_phase": "world_xy_aabb_lower_bound",
            "broad_phase_is_not_reported_as_exact_distance": True,
            "robot_collision_shape_count": len(self.robot_records),
            "registered_collision_shape_count": len(self.records),
            "static_collision_record_count": sum(
                record.owner_local_matrix is None for record in self.records
            ),
            "dynamic_collision_record_count": sum(
                record.owner_local_matrix is not None for record in self.records
            ),
            "incomplete_collision_shapes": [
                {"path": record.path, "reason": record.reason}
                for record in self.records
                if not record.complete
            ],
            "s03_criteria": [str(spec["criterion"]) for spec in self.s03_specs],
            "s08_obstacle_shape_count": len(self._s08_obstacles),
        }

    def sample(self, object_states: Mapping[str, Any], active_rules=None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.s03_specs and (active_rules is None or "S03" in active_rules):
            result["S03"] = {}
            for spec in self.s03_specs:
                criterion = str(spec["criterion"])
                subject_records = self._subject_records(spec)
                if spec["region_geometry"] in ("underbody_volumes_xyz", "surface_prisms_xyz"):
                    from .spatial_regions import measure_forbidden
                    result['S03'][criterion] = measure_forbidden(self, subject_records,
                        self._s03_forbidden.get(criterion, []), object_states, spec)
                elif spec["region_geometry"] == "initial_world_aabb_xy":
                    region = self._s03_initial_boxes.get(criterion)
                    forbidden = [] if region is None else [("initial_world_aabb_xy", region)]
                    result["S03"][criterion] = self._measure_records_against_shapes(
                        subject_records,
                        forbidden,
                        object_states,
                        threshold=float(spec["minimum_clearance_m"]),
                        static_complete=region is not None,
                    )
                else:
                    result["S03"][criterion] = self._measure_record_sets(
                        subject_records,
                        self._s03_forbidden.get(criterion, []),
                        object_states,
                        threshold=float(spec["minimum_clearance_m"]),
                    )
        if self.s08_spec is not None and (active_rules is None or "S08" in active_rules):
            subjects = self._subject_records(self.s08_spec)
            if self.s08_spec.get('height_overlap_required'):
                from .spatial_regions import measure_height_filtered
                result['S08'] = measure_height_filtered(self, subjects, self._s08_obstacles,
                    object_states, float(self.s08_spec['minimum_clearance_m']))
            else:
                result["S08"] = self._measure_record_sets(subjects, self._s08_obstacles,
                    object_states, threshold=float(self.s08_spec["minimum_clearance_m"]))
            result['S08']['additional_clearances']={
                spec['criterion']:self._additional_clearance(spec,subjects,object_states)
                for spec in self.s08_spec.get('additional_clearances',[])}
        return result

    def _measure_records_against_shapes(
        self,
        left_records: Sequence[_GeometryRecord],
        right_shapes: Sequence[tuple[str, Any]],
        object_states: Mapping[str, Any],
        *,
        threshold: float,
        static_complete: bool,
    ) -> dict[str, Any]:
        left = self._materialize(left_records, object_states)
        right = [
            {
                "path": path,
                "shape": shape,
                "bounds": _bounds_xy(shape),
                "complete": static_complete,
                "reason": None if static_complete else "initial_region_bounds_unavailable",
            }
            for path, shape in right_shapes
        ]
        return self._measure(left, right, threshold=threshold)

    def _convex_record_bound(self,record,object_states):
        """Convex union is a conservative distance bound, never a collider substitute."""
        native_bound=record.parameters.get('collision_bound_source')=='physx_property_query'
        if not record.complete and not native_bound:return None
        matrix=record.static_world_matrix
        if record.owner_local_matrix is not None:
            state=object_states.get(record.owner)
            if not isinstance(state,Mapping):return None
            matrix=_matrix_from_pose(state['pose'])@record.owner_local_matrix
        if native_bound and not record.complete:
            lo,hi=record.parameters['local_bounds_xyz']
            vertices=np.asarray(list(product(*zip(lo,hi))),dtype=np.float64)
            return _TransformedConvex(vertices,matrix)
        if record.kind!='Mesh':
            shapes=self._shapes(record,matrix)
            return shapes[0] if len(shapes)==1 else None
        if 'distance_bound_hull_vertices' not in record.parameters:
            vertices=np.unique(np.concatenate(record.parameters['convex_parts']),axis=0)
            if len(vertices)>16:
                from scipy.spatial import ConvexHull,QhullError
                try:vertices=vertices[ConvexHull(vertices).vertices]
                except QhullError:pass
            record.parameters['distance_bound_hull_vertices']=vertices
        return _TransformedConvex(record.parameters['distance_bound_hull_vertices'],matrix)

    def _subject_records(self, spec):
        # Profiles and the collision inventory are frozen for one Tracker.
        # Resolve carried/subject selectors once, not once per 200 Hz sample.
        key = (tuple(spec['subject_body_names']) if 'subject_body_names' in spec else None,
               bool(spec.get('include_carried_objects')), tuple(spec.get('carried_object_prim_paths', ())))
        cache = self.__dict__.setdefault('_subject_scope_cache', {})
        if key not in cache:
            subjects = list(self.robot_records)
            if key[0] is not None:
                names = set(key[0])
                subjects = [r for r in subjects if r.owner.rsplit('/', 1)[-1] in names]
            if key[1]:
                subjects.extend(r for r in self.records if any(
                    _under(r.path, path) or _under(r.owner, path) for path in key[2]))
            cache[key] = subjects
        return cache[key]

    def _right_bounds_arrays(self, records, states):
        # Only CPU bounds/identity arrays are retained. Dynamic world bounds
        # still consume this exact sample's measured pose. Cache is per episode
        # and bounded; no state or PhysX handle crosses a Tracker lifetime.
        cache = self.__dict__.setdefault('_right_arrays_cache', {})
        key = id(records)
        entry = cache.get(key)
        if entry is None or entry['records'] is not records or entry['count'] != len(records):
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            lo = np.zeros((len(records), 2), dtype=np.float64)
            hi = np.zeros_like(lo)
            dynamic = []
            missing = []
            for index, record in enumerate(records):
                if record.owner_local_matrix is not None:
                    dynamic.append((index, record))
                    continue
                bound = self._record_bounds_xy(record, states)
                if bound is None:
                    missing.append(index)
                else:
                    lo[index], hi[index] = bound
            entry = dict(records=records, count=len(records), lo=lo, hi=hi,
                         dynamic=dynamic, static_missing=missing,
                         owners=np.asarray([r.owner for r in records], dtype=object), owner_rows={})
            # Fixed spatial partitions of STATIC records only. Per-sample
            # dynamic rows are always evaluated explicitly, including missing.
            if len(records) > 256 and not missing:
                dynamic_indices = {index for index, _ in dynamic}
                static = np.asarray([i for i in range(len(records)) if i not in dynamic_indices], dtype=np.int64)
                leaves = []
                def split(indices):
                    if len(indices) <= 128:
                        leaves.append(np.sort(indices)); return
                    centers = (lo[indices] + hi[indices]) * .5
                    axis = int(np.argmax(np.ptp(centers, axis=0)))
                    order = np.argsort(centers[:, axis], kind='stable')
                    middle = len(indices) // 2
                    split(indices[order[:middle]]); split(indices[order[middle:]])
                if len(static): split(static)
                entry['partitions'] = leaves
                entry['partition_lo'] = np.asarray([lo[idx].min(axis=0) for idx in leaves]).reshape(-1, 2)
                entry['partition_hi'] = np.asarray([hi[idx].max(axis=0) for idx in leaves]).reshape(-1, 2)
                entry['dynamic_indices'] = np.asarray([i for i, _ in dynamic], dtype=np.int64)
            cache[key] = entry
        missing = list(entry['static_missing'])
        for index, record in entry['dynamic']:
            bound = self._record_bounds_xy(record, states)
            if bound is None:
                missing.append(index)
            else:
                entry['lo'][index], entry['hi'][index] = bound
        return entry, [records[index].path for index in sorted(missing)]

    @staticmethod
    def _record_candidates(arrays, lo, hi, owner, threshold):
        """Exact old broad-phase results with conservative partition pruning.

        Partition bounds only reject records that cannot be either near or
        the closest far record. Reported distances still come from individual
        record bounds using the original NumPy norm; ties use original indices.
        """
        near = []
        best = float('inf')
        best_index = 0
        def evaluate(indices):
            nonlocal best, best_index
            if not len(indices): return
            gaps = np.linalg.norm(np.maximum(0., np.maximum(
                lo - arrays['hi'][indices], arrays['lo'][indices] - hi)), axis=1)
            gaps[arrays['owners'][indices] == owner] = np.inf
            mask = gaps <= threshold if threshold == 0 else gaps < threshold
            near.extend((float(gaps[k]), int(indices[k])) for k in np.flatnonzero(mask))
            far = np.where(mask, np.inf, gaps)
            k = int(np.argmin(far)); value = float(far[k]); index = int(indices[k])
            if value < best or value == best and index < best_index:
                best, best_index = value, index
        if 'partitions' not in arrays:
            evaluate(np.arange(arrays['count']))
        else:
            evaluate(arrays['dynamic_indices'])
            gaps = np.linalg.norm(np.maximum(0., np.maximum(
                lo - arrays['partition_hi'], arrays['partition_lo'] - hi)), axis=1)
            for index in np.argsort(gaps, kind='stable'):
                # A roundoff margin only visits additional leaves; it never
                # changes returned distances or threshold classification.
                if gaps[index] > max(best, threshold) + 1e-12: break
                evaluate(arrays['partitions'][index])
        return near, best, best_index

    def _measure_record_sets(self,left_records,right_records,object_states,*,threshold):
        if not left_records or not right_records:
            return {'complete':False,'distance_m':None,'reason':'collision_geometry_scope_empty'}
        left_bounds=[self._record_bounds_xy(x,object_states) for x in left_records]
        arrays, right_missing = self._right_bounds_arrays(right_records, object_states)
        missing=[r.path for r,b in zip(left_records,left_bounds) if b is None] + right_missing
        if missing:return {'complete':False,'distance_m':None,'reason':'collision_record_pose_or_bounds_unavailable','missing_prims':missing}
        rl=arrays['lo'];ru=arrays['hi']
        owners=arrays['owners']
        lower=float('inf');witness=None;unknown=[];near=[]
        # Bound one robot record against the scene, never every triangle against
        # every scene record. Memory is O(scene records), not O(triangle pairs).
        for i,(lo,hi) in enumerate(left_bounds):
            candidates, far_distance, j = self._record_candidates(
                arrays, lo, hi, left_records[i].owner, threshold)
            near.extend((gap, i, index) for gap, index in candidates)
            if far_distance < lower:
                lower = far_distance
                witness = {'left_path': left_records[i].path, 'right_path': right_records[j].path}
        for gap,i,j in sorted(near):
            l,r=left_records[i],right_records[j]
            a=self._convex_record_bound(l,object_states);b=self._convex_record_bound(r,object_states)
            if a is not None and b is not None:
                bound,_,_=_horizontal_distance(_HorizontalShape(a),_HorizontalShape(b))
                if bound>0 if threshold==0 else bound>=threshold:
                    if bound<lower:lower=bound;witness={'left_path':l.path,'right_path':r.path}
                    continue
            measured=self._measure(self._materialize([l],object_states),self._materialize([r],object_states),threshold=threshold)
            distance=measured.get('distance_m')
            if measured.get('complete') and distance is not None:
                violates=distance<=0 if threshold==0 else distance<threshold
                if violates and 'lower_bound' not in str(measured.get('distance_kind')):
                    return dict(measured,distance_kind='exact_witness_upper_bound_on_global_minimum',
                                global_minimum_evaluated=False,threshold_verdict='violated')
                if distance<lower:lower=distance;witness=measured
            else:unknown.append(measured)
        if unknown:return {'complete':False,'distance_m':None,'reason':'potentially_relevant_collision_geometry_incomplete','incomplete_pairs':unknown}
        if not np.isfinite(lower):return {'complete':False,'distance_m':None,'reason':'collision_geometry_has_no_distinct_pairs'}
        return dict(witness or {},complete=True,distance_m=lower,distance_kind='conservative_hierarchical_collision_lower_bound',
                    safe_lower_bound_m=lower,threshold_m=float(threshold),global_minimum_evaluated=False)

    def _record_bounds_xy(
        self,
        record: _GeometryRecord,
        object_states: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a conservative current-world bound without expanding parts."""

        if not record.complete and record.parameters.get('collision_bound_source')!='physx_property_query':
            return None
        if record.owner_local_matrix is None:
            return record.initial_bounds_xy
        raw_state = object_states.get(record.owner)
        if not isinstance(raw_state, Mapping):
            return None
        try:
            pose_key=tuple(raw_state['pose'])
            if not hasattr(self,'_record_bounds_pose_cache'):
                self._record_bounds_pose_cache={};self._owner_matrix_cache={}
            cached=self._record_bounds_pose_cache.get(id(record))
            if cached is not None and cached[0]==pose_key:return cached[1]
            owner=self._owner_matrix_cache.get(record.owner)
            if owner is None or owner[0]!=pose_key:
                owner=(pose_key,_matrix_from_pose(pose_key))
                self._owner_matrix_cache[record.owner]=owner
            matrix = owner[1] @ record.owner_local_matrix
            if record.kind == "Mesh":
                local_bounds = record.parameters.get("local_bounds_xyz")
                if local_bounds is None:
                    return None
                lower, upper = (
                    np.asarray(local_bounds[0], dtype=np.float64),
                    np.asarray(local_bounds[1], dtype=np.float64),
                )
                # Exact AABB of a transformed local box, avoiding eight corner
                # transforms and repeated quaternion conversion per mesh.
                center=matrix[:3,:3]@((lower+upper)*.5)+matrix[:3,3]
                half=np.abs(matrix[:3,:3])@((upper-lower)*.5)
                bounds=(center[:2]-half[:2],center[:2]+half[:2])
                self._record_bounds_pose_cache[id(record)]=(pose_key,bounds)
                return bounds
            shapes = self._shapes(record, matrix) if record.complete else []
            if len(shapes) == 1:
                bounds = _bounds_xy(shapes[0])
                self._record_bounds_pose_cache[id(record)] = (pose_key, bounds)
                return bounds
        except (KeyError, TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _measure(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], *, threshold: float, dimensions: int = 2) -> dict[str, Any]:
        if not left or not right:
            return {
                "complete": False,
                "distance_m": None,
                "reason": "collision_geometry_scope_empty",
            }
        if len(left)*len(right)>131072:
            results=[]
            for start in range(0,len(left),256):
                for offset in range(0,len(right),256):
                    result=SafetyGeometryTracker._measure(left[start:start+256],right[offset:offset+256],threshold=threshold,dimensions=dimensions)
                    distance=result.get('distance_m')
                    violation=distance is not None and (distance<=0 if threshold==0 else distance<threshold)
                    if result.get('complete') and violation and 'lower_bound' not in str(result.get('distance_kind')):
                        return dict(result,distance_kind='exact_witness_upper_bound_on_global_minimum',global_minimum_evaluated=False)
                    results.append(result)
            unknown=[r for r in results if not r.get('complete')]
            if unknown:return {'complete':False,'distance_m':None,'reason':'potentially_relevant_collision_geometry_incomplete','incomplete_pairs':unknown}
            result=min(results,key=lambda r:r['distance_m'])
            return dict(result,distance_kind='conservative_partitioned_collision_lower_bound',global_minimum_evaluated=False)
        # Compute the broad phase in NumPy.  A supermarket S08 scope contains
        # roughly 37 robot shapes x 1,400 obstacle shapes; constructing and
        # sorting 50k Python tuples at 200 Hz made simulated startup take many
        # minutes.  The matrix contains the same AABB lower bounds, and the
        # stable argsort below preserves the original candidate ordering.
        left_lower = np.asarray([item["bounds"][0] for item in left], dtype=np.float64)
        left_upper = np.asarray([item["bounds"][1] for item in left], dtype=np.float64)
        right_lower = np.asarray([item["bounds"][0] for item in right], dtype=np.float64)
        right_upper = np.asarray([item["bounds"][1] for item in right], dtype=np.float64)
        separation = np.maximum(
            0.0,
            np.maximum(
                left_lower[:, None, :] - right_upper[None, :, :],
                right_lower[None, :, :] - left_upper[:, None, :],
            ),
        )
        lower_bounds = np.linalg.norm(separation, axis=2)
        left_owners = np.asarray([item.get("owner") for item in left], dtype=object)
        right_owners = np.asarray([item.get("owner") for item in right], dtype=object)
        same_owner = (
            (left_owners[:, None] != None)  # noqa: E711 - elementwise object comparison
            & (left_owners[:, None] == right_owners[None, :])
        )
        lower_bounds[same_owner] = np.inf
        flat_bounds = lower_bounds.reshape(-1)
        finite_indices = np.flatnonzero(np.isfinite(flat_bounds))
        if finite_indices.size == 0:
            return {
                "complete": False,
                "distance_m": None,
                "reason": "collision_geometry_has_no_distinct_pairs",
            }
        order = finite_indices[
            np.argsort(flat_bounds[finite_indices], kind="stable")
        ]
        best = float("inf")
        witness = None
        incomplete_nearby = []
        right_count = len(right)
        for flat_index in order:
            left_index, right_index = divmod(int(flat_index), right_count)
            lower_bound = float(flat_bounds[flat_index])
            left_item = left[left_index]
            right_item = right[right_index]
            if lower_bound >= best:
                break
            lower_bound_proves_safe = (
                lower_bound > 0.0 if threshold == 0.0 else lower_bound >= threshold
            )
            if lower_bound_proves_safe and best == float("inf"):
                return {
                    "complete": True,
                    "distance_m": float(lower_bound),
                    "distance_kind": "conservative_aabb_lower_bound",
                    "safe_lower_bound_m": float(lower_bound),
                    "threshold_m": float(threshold),
                    "left_path": left_item["path"],
                    "right_path": right_item["path"],
                }
            if not left_item["complete"] or not right_item["complete"]:
                incomplete_nearby.append({
                    "left_path": left_item["path"],
                    "right_path": right_item["path"],
                    "lower_bound_m": float(lower_bound),
                    "left_reason": left_item.get("reason"),
                    "right_reason": right_item.get("reason"),
                })
                continue
            distance, left_point, right_point = _horizontal_distance(
                _HorizontalShape(left_item["shape"]) if dimensions==2 else left_item['shape'],
                _HorizontalShape(right_item["shape"]) if dimensions==2 else right_item['shape'],
            )
            if distance < best:
                best = float(distance)
                witness = {
                    "left_path": left_item["path"],
                    "right_path": right_item["path"],
                    "left_point_world": left_point.tolist(),
                    "right_point_world": right_point.tolist(),
                }
            if best <= 0.0:
                break
        proven_violation = math.isfinite(best) and (best <= 0.0 if threshold == 0.0 else best < threshold)
        if incomplete_nearby and not proven_violation:
            return {
                "complete": False,
                "distance_m": None if not math.isfinite(best) else best,
                "reason": "potentially_relevant_collision_geometry_incomplete",
                "incomplete_pairs": incomplete_nearby,
                "exact_witness": witness,
            }
        if not math.isfinite(best) or witness is None:
            return {
                "complete": False,
                "distance_m": None,
                "reason": "exact_collision_distance_unavailable",
                "incomplete_pairs": incomplete_nearby,
            }
        return {
            "complete": True,
            "distance_m": best,
            "distance_kind": "exact_horizontal_collision_surface" if dimensions==2 else "exact_spatial_collision_surface",
            "threshold_m": float(threshold),
            **witness,
        }

    def _triangle_items(self, record, matrix):
        """Reuse dynamic triangle containers; batch the identical support calls."""
        if record.kind != 'Mesh' or record.owner_local_matrix is None or not record.complete:
            return None
        cache = self.__dict__.setdefault('_triangle_item_cache', {})
        key = id(record)
        if key not in cache:
            parts = record.parameters.get('convex_parts', [])
            if len(parts) < 128 or any(np.shape(part) != (3, 3) for part in parts):
                cache[key] = None
            else:
                vertices = np.stack(parts)
                shapes = [_TransformedConvex(part, matrix) for part in parts]
                items = tuple(dict(path=f'{record.path}#part{i}', owner=record.owner,
                    shape=shape, bounds=(np.empty(2), np.empty(2)), complete=True, reason=None)
                    for i, shape in enumerate(shapes))
                cache[key] = (vertices, shapes, items, np.arange(len(parts)))
        batch = cache[key]
        if batch is None:
            return None
        vertices, shapes, items, indices = batch
        if np.shape(matrix) != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('transformed convex collision data are invalid')
        bounds = np.empty((len(items), 4), dtype=np.float64)
        for column, (axis, sign) in enumerate(((0, -1.), (1, -1.), (0, 1.), (1, 1.))):
            direction = np.zeros(3); direction[axis] = sign
            local = matrix[:3, :3].T @ direction
            # Batched matmul preserves each triangle's original 3x3 @ 3
            # operation and first argmax, rather than approximating by boxes.
            choices = np.argmax(np.matmul(vertices, local), axis=1)
            selected = vertices[indices, choices]
            points = np.matmul(matrix[:3, :3], selected[..., None])[..., 0] + matrix[:3, 3]
            bounds[:, column] = points[:, axis]
        for i, (shape, item) in enumerate(zip(shapes, items)):
            shape.matrix = matrix
            item['bounds'][0][:] = bounds[i, :2]
            item['bounds'][1][:] = bounds[i, 2:]
        return items

    def _materialize(
        self,
        records: Sequence[_GeometryRecord],
        object_states: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = []
        for record in records:
            cache_key = id(record)
            pose_key: tuple[float, ...] | None = None
            if record.owner_local_matrix is None:
                cached = self._static_materialized_cache.get(cache_key)
                if cached is not None:
                    result.extend(cached)
                    continue
            matrix = record.static_world_matrix
            if record.owner_local_matrix is not None:
                raw_state = object_states.get(record.owner)
                if not isinstance(raw_state, Mapping):
                    result.append({
                        "path": record.path,
                        "owner": record.owner,
                        "shape": None,
                        "bounds": (
                            np.full(2, -np.inf),
                            np.full(2, np.inf),
                        ),
                        "complete": False,
                        "reason": "rigid_owner_state_unavailable",
                    })
                    continue
                try:
                    raw_pose = np.asarray(raw_state["pose"], dtype=np.float64)
                    matrix = _matrix_from_pose(raw_pose) @ record.owner_local_matrix
                    pose_key = tuple(float(value) for value in raw_pose)
                except (KeyError, TypeError, ValueError) as error:
                    result.append({
                        "path": record.path,
                        "owner": record.owner,
                        "shape": None,
                        "bounds": (
                            np.full(2, -np.inf),
                            np.full(2, np.inf),
                        ),
                        "complete": False,
                        "reason": f"rigid_owner_pose_invalid: {error}",
                    })
                    continue
                cached_pose = self._posed_materialized_cache.get(cache_key)
                if cached_pose is not None and cached_pose[0] == pose_key:
                    result.extend(cached_pose[1])
                    continue
            if matrix is None:
                continue
            try:
                triangle_items = self._triangle_items(record, matrix)
                if triangle_items is not None:
                    result.extend(triangle_items)
                    if record.owner_local_matrix is None:
                        self._static_materialized_cache[cache_key] = tuple(triangle_items)
                    else:
                        self._posed_materialized_cache[cache_key] = (pose_key, triangle_items)
                    continue
                shapes = self._shapes(record, matrix) if record.complete else []
            except (KeyError, TypeError, ValueError) as error:
                shapes = []
                record_reason = f"collision_shape_materialization_failed: {error}"
            else:
                record_reason = record.reason
            if not shapes:
                fallback_bounds=self._record_bounds_xy(record,object_states)
                if fallback_bounds is None:
                    fallback_bounds=(np.full(2,-np.inf),np.full(2,np.inf))
                items = ({
                    "path": record.path,
                    "owner": record.owner,
                    "shape": None,
                    "bounds": fallback_bounds,
                    "complete": False,
                    "reason": record_reason or "collision_shape_unavailable",
                },)
                if record.parameters.get('collision_bound_source') == 'physx_property_query' and record.parameters.get('local_bounds_xyz') is not None:
                    lower, upper = (np.asarray(v, dtype=np.float64) for v in record.parameters['local_bounds_xyz'])
                    if lower.shape == (3,) and upper.shape == (3,) and np.isfinite(lower).all() and np.isfinite(upper).all():
                        vertices = _transform_points(np.asarray(list(product(*zip(lower, upper)))), matrix)
                        items[0]['bounds_xyz'] = (vertices.min(axis=0), vertices.max(axis=0))
                        items[0]['bounds_xyz_certified'] = True
                        items[0]['bound_source'] = 'physx_property_query'
                result.extend(items)
                if record.owner_local_matrix is None:
                    self._static_materialized_cache[cache_key] = items
                elif pose_key is not None:
                    self._posed_materialized_cache[cache_key] = (pose_key, items)
                continue
            items = []
            for index, shape in enumerate(shapes):
                items.append({
                    "path": record.path if len(shapes) == 1 else f"{record.path}#part{index}",
                    "owner": record.owner,
                    "shape": shape,
                    "bounds": _bounds_xy(shape),
                    "complete": True,
                    "reason": None,
                })
            result.extend(items)
            if record.owner_local_matrix is None:
                self._static_materialized_cache[cache_key] = tuple(items)
            elif pose_key is not None:
                self._posed_materialized_cache[cache_key] = (pose_key, tuple(items))
        return result

    @staticmethod
    def _shapes(record: _GeometryRecord, matrix: np.ndarray) -> list[Any]:
        parameters = record.parameters
        if record.kind == "Mesh":
            parts = parameters.get("convex_parts", [])
            return [_TransformedConvex(part, matrix) for part in parts]
        if record.kind == "Cube":
            vertices = np.asarray(list(product((-0.5, 0.5), repeat=3))) * float(parameters["size"])
            return [_WorldConvex(_transform_points(vertices, matrix))]
        if record.kind in {"Sphere", "Capsule", "Cylinder", "Cone"}:
            return [_AnalyticShape(record.kind, matrix, parameters)]
        return []

    def _scan_stage(self) -> list[_GeometryRecord]:
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        records = []
        for prim in Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not None and not bool(enabled):
                continue
            path = str(prim.GetPath())
            owner_prim = prim
            while owner_prim and not owner_prim.IsPseudoRoot() and not owner_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                owner_prim = owner_prim.GetParent()
            owner = str(owner_prim.GetPath()) if owner_prim and not owner_prim.IsPseudoRoot() else path
            world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
            has_rigid_body = bool(
                owner_prim
                and not owner_prim.IsPseudoRoot()
                and owner_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            rigid_body_enabled = (
                UsdPhysics.RigidBodyAPI(owner_prim).GetRigidBodyEnabledAttr().Get()
                if has_rigid_body
                else None
            )
            # Runtime-frozen fixtures keep the API schema but explicitly set
            # rigidBodyEnabled=false.  PhysX cannot move those owners, so their
            # final composed world geometry is static.  Treating API presence
            # alone as motion caused thousands of disabled supermarket assets
            # to be rebuilt on every safety sample.
            dynamic = has_rigid_body and rigid_body_enabled is not False
            if dynamic:
                owner_world = _rigid_matrix(cache.GetLocalToWorldTransform(owner_prim))
                owner_local = np.linalg.inv(owner_world) @ world
                static_world = None
            else:
                owner_local = None
                static_world = world
            kind = prim.GetTypeName()
            parameters: dict[str, Any] = {"source_prim_type": str(kind)}
            complete = True
            reason = None
            try:
                if kind == "Mesh":
                    parameters["convex_parts"] = self._mesh_parts(
                        prim, cache, np.eye(4, dtype=np.float64),
                        evidence=parameters.setdefault("mesh_sources", []),
                    )
                elif kind == "Xform":
                    mode = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                    if mode in {"convexHull", "convexDecomposition", "sdf"}:
                        # Imported robot STL assets author the collision API on
                        # the parent Xform. Its cooked representation owns the
                        # collision; child render triangles are not that hull.
                        parts = self._mesh_parts(
                            prim, cache, np.eye(4, dtype=np.float64),
                            evidence=parameters.setdefault("mesh_sources", []),
                        )
                    else:
                        parts = []
                        for child in Usd.PrimRange(prim):
                            if child == prim or not child.IsA(UsdGeom.Mesh):
                                continue
                            child_world = np.asarray(cache.GetLocalToWorldTransform(child), dtype=np.float64).T
                            child_local = np.linalg.inv(world) @ child_world
                            parts.extend(self._mesh_parts(
                                child, cache, child_local,
                                evidence=parameters.setdefault("mesh_sources", []),
                            ))
                    kind = "Mesh"
                    parameters["convex_parts"] = parts
                elif kind == "Cube":
                    parameters["size"] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                elif kind == "Sphere":
                    parameters["radius"] = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                elif kind in {"Capsule", "Cylinder", "Cone"}:
                    schema = getattr(UsdGeom, kind)(prim)
                    parameters.update(
                        radius=float(schema.GetRadiusAttr().Get()),
                        height=float(schema.GetHeightAttr().Get()),
                        axis=str(schema.GetAxisAttr().Get()),
                    )
                elif kind == "Plane":
                    complete = False
                    reason = "infinite_plane_requires_explicit_exemption"
                else:
                    complete = False
                    reason = f"unsupported_collision_prim_type:{kind}"
                if kind == "Mesh" and not parameters.get("convex_parts"):
                    complete = False
                    reason = "collision_mesh_has_no_exact_parts"
                    # SDF's collision surface is not a convex representation.
                    # Query actual cooked PhysX collider bounds for a safe-only
                    # distance certificate; near cases remain unknown. Never
                    # reinterpret visual triangles/extents as the SDF surface.
                    mode=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                    parameters['collision_approximation']=mode
                    if mode in {'sdf', 'convexHull', 'convexDecomposition'}:
                        reflected_sdf = np.linalg.det(world[:3, :3]) < 0.0
                        native = None if reflected_sdf else self._native_collider_bound(prim, owner_prim)
                        if reflected_sdf:
                            reason = 'reflected_sdf_native_bound_convention_unverified'
                        if native is not None:
                            parameters['local_bounds_xyz']=(np.asarray(native['aabb_local_min']),np.asarray(native['aabb_local_max']))
                            parameters['collision_bound_source']='physx_property_query'
                            parameters['collision_bound_evidence']=native
                            reason=('sdf_exact_surface_unavailable_native_bound_only' if mode=='sdf'
                                    else 'cooked_collision_surface_unavailable_native_bound_only')
                elif kind == "Mesh":
                    part_lowers = np.asarray(
                        [np.min(part, axis=0) for part in parameters["convex_parts"]],
                        dtype=np.float64,
                    )
                    part_uppers = np.asarray(
                        [np.max(part, axis=0) for part in parameters["convex_parts"]],
                        dtype=np.float64,
                    )
                    parameters["local_bounds_xyz"] = (
                        np.min(part_lowers, axis=0),
                        np.max(part_uppers, axis=0),
                    )
            except Exception as error:
                complete = False
                reason = f"collision_geometry_extraction_failed:{type(error).__name__}:{error}"
            try:
                bounds = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lower = np.asarray(bounds.GetMin(), dtype=np.float64)[:2]
                upper = np.asarray(bounds.GetMax(), dtype=np.float64)[:2]
                initial_bounds = (lower, upper) if np.isfinite([*lower, *upper]).all() else None
            except Exception:
                initial_bounds = None
            if not complete:
                initial_bounds=None
                if parameters.get('collision_bound_source')=='physx_property_query':
                    lo,hi=parameters['local_bounds_xyz']
                    center=world[:3,:3]@((lo+hi)*.5)+world[:3,3]
                    half=np.abs(world[:3,:3])@((hi-lo)*.5)
                    initial_bounds=(center[:2]-half[:2],center[:2]+half[:2])
            # USD render extents are not an authoritative bound on cooked
            # PhysX hulls. Derive complete-shape broad bounds from those exact
            # collision parts; retain USD only as an incomplete fallback.
            if complete:
                if kind=='Mesh':
                    lo,hi=parameters['local_bounds_xyz']
                    center=world[:3,:3]@((lo+hi)*.5)+world[:3,3]
                    half=np.abs(world[:3,:3])@((hi-lo)*.5)
                    initial_bounds=(center[:2]-half[:2],center[:2]+half[:2])
                else:
                    prototype=_GeometryRecord(path,owner,kind,parameters,None,world,None,True)
                    exact_shapes=self._shapes(prototype,world)
                    if exact_shapes:
                        bs=[_bounds_xy(shape) for shape in exact_shapes]
                        initial_bounds=(np.min([b[0] for b in bs],axis=0),np.max([b[1] for b in bs],axis=0))
            if len(records)%250==0:
                print('[SAFETY_GEOMETRY_SCAN] records=%d prim=%s'%(len(records),path),flush=True)
            records.append(
                _GeometryRecord(
                    path=path,
                    owner=owner,
                    kind=kind,
                    parameters=parameters,
                    owner_local_matrix=owner_local,
                    static_world_matrix=static_world,
                    initial_bounds_xy=initial_bounds,
                    complete=complete,
                    reason=reason,
                )
            )
        return records

    def _native_collider_bound(self,prim,owner_prim):
        """Cached native collision-local bound, returned only after query completion.

        PhysX property-query bounds are in unscaled collider-local coordinates
        (as tested by the installed NVIDIA PhysxPropertyQueryInterface suite),
        so the ordinary composed collider transform applies exactly once.
        Never pump simulation or substitute render extents if a query is async,
        incomplete, invalid, or lacks this collider's authoritative result.
        """
        from pxr import UsdUtils,PhysicsSchemaTools
        from omni.physx import get_physx_property_query_interface
        from omni.physx.bindings._physx import PhysxPropertyQueryResult
        if not owner_prim or owner_prim.IsPseudoRoot():return None
        if not hasattr(self,'_native_collider_bound_cache'):self._native_collider_bound_cache={}
        owner=str(owner_prim.GetPath());path=str(prim.GetPath())
        if owner not in self._native_collider_bound_cache:
            received={};finished=[]
            def callback(info):
                if info.result!=PhysxPropertyQueryResult.VALID:return
                lo=np.asarray(list(info.aabb_local_min),dtype=np.float64)
                hi=np.asarray(list(info.aabb_local_max),dtype=np.float64)
                if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi):return
                collider=str(PhysicsSchemaTools.intToSdfPath(info.path_id))
                received[collider]={'query_owner':owner,'collider':collider,'result':'VALID',
                    'aabb_local_min':lo.tolist(),'aabb_local_max':hi.tolist(),
                    'query_local_pos':list(info.local_pos),'query_local_rot_xyzw':list(info.local_rot),
                    'coordinate_frame':'unscaled_collider_local','role':'safe_lower_bound_only_not_exact_surface'}
            get_physx_property_query_interface().query_prim(
                stage_id=UsdUtils.StageCache.Get().GetId(prim.GetStage()).ToLongInt(),
                prim_id=PhysicsSchemaTools.sdfPathToInt(owner),timeout_ms=1000,
                collider_fn=callback,finished_fn=lambda:finished.append(True))
            self._native_collider_bound_cache[owner]=received if finished else {}
        return self._native_collider_bound_cache[owner].get(path)

    @staticmethod
    def _mesh_parts(
        prim: Any, cache: Any, extra_matrix: np.ndarray,
        evidence: list[dict[str, Any]] | None = None,
    ) -> list[np.ndarray]:
        """Return exact convex pieces in the collision Prim's local frame."""

        from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics, UsdUtils
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

        mode = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        world_determinant = float(np.linalg.det(
            np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)[:3, :3]
        ))
        reflected = world_determinant < 0.0
        if evidence is not None:
            evidence.append({
                "prim_path": str(prim.GetPath()),
                "source_prim_type": str(prim.GetTypeName()),
                "collision_approximation": mode,
                "world_linear_determinant": world_determinant,
                "extra_matrix": np.asarray(extra_matrix, dtype=np.float64).tolist(),
                "cooked_reflection_undone": reflected and mode in {"convexHull", "convexDecomposition"},
            })
        if mode in {"convexHull", "convexDecomposition"}:
            cooked = []
            # PhysX cooking already bakes global reflection into convex
            # vertices, while excluding scale magnitudes. Undo that baked
            # inversion so the composed USD affine below applies it once.
            # This also covers static mirrored colliders, not just owners.
            def receive(result, convexes):
                if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                    return
                for convex in convexes:
                    vertices = np.asarray(
                        [[float(v.x), float(v.y), float(v.z)] for v in convex.vertices],
                        dtype=np.float64,
                    )
                    if reflected:
                        vertices = -vertices
                    cooked.append(_transform_points(vertices, extra_matrix))

            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=UsdUtils.StageCache.Get().GetId(prim.GetStage()).ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                run_asynchronously=False,
                on_result=receive,
            )
            return cooked

        if mode not in {"none", "", "None"}:
            return []
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
        indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
        parts = []
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            for offset in range(1, count - 1):
                triangle = points[[face[0], face[offset], face[offset + 1]]]
                parts.append(_transform_points(triangle, extra_matrix))
        return parts
