"""Convert parsed RTX GenericModelOutput into the stable LiDAR frame contract."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .runtime_contracts import LidarFrame


class RtxLidarFrameAssembler:
    def __init__(self, parse_object_ids: Callable[[Any], Any], *, maximum_range_m: float = 100.0):
        self.parse_object_ids = parse_object_ids
        self.maximum_range_m = float(maximum_range_m)

    def assemble(
        self,
        gmo: Any,
        stable_id_map: Mapping[int, str] | None,
        *,
        timestamp_s: float,
        sensor_to_world: np.ndarray,
        frame_id: str,
    ) -> LidarFrame:
        azimuth = np.deg2rad(np.asarray(gmo.x, dtype=np.float32))
        elevation = np.deg2rad(np.asarray(gmo.y, dtype=np.float32))
        distance = np.asarray(gmo.z, dtype=np.float32)
        horizontal = distance * np.cos(elevation)
        points = np.column_stack([
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            distance * np.sin(elevation),
        ]).astype(np.float32, copy=False)
        valid = np.all(np.isfinite(points), axis=1) & (distance > 0.05) & (distance < self.maximum_range_m)
        object_ids = None
        if getattr(gmo, "objId", None) is not None:
            object_ids = np.asarray(self.parse_object_ids(gmo.objId), dtype=object)[valid]
        intensity = getattr(gmo, "intensity", None)
        if intensity is not None:
            intensity = np.asarray(intensity)[valid]
        return LidarFrame(
            timestamp_s=timestamp_s,
            points=points[valid],
            object_ids=object_ids,
            stable_id_map=stable_id_map,
            intensity=intensity,
            distance=distance[valid],
            azimuth=azimuth[valid],
            elevation=elevation[valid],
            sensor_to_world=sensor_to_world,
            frame_id=frame_id,
        )
