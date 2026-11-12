"""Reusable RTX LiDAR capture and four-view video composition components."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw
from omni.replicator.core import Writer

try:
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
    EXPERIMENTAL_RTX_AVAILABLE = True
except ModuleNotFoundError:
    EXPERIMENTAL_RTX_AVAILABLE = False
from isaacsim.sensors.rtx import LidarRtx, get_gmo_data


_ACTIVE_STREAM = None


class M20FourViewLidarWriter(Writer):
    def __init__(self):
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]

    def write(self, data):
        if _ACTIVE_STREAM is not None:
            _ACTIVE_STREAM.consume_writer_data(data)


class RtxLidarStream:
    """Own an RTX LiDAR writer and expose robot-frame point-cloud snapshots."""

    def __init__(self, lidar_sensor, output_dir: Path, max_range_m: float = 12.0, backend: str = "experimental"):
        self.sensor = lidar_sensor
        self.output_dir = Path(output_dir)
        self.max_range_m = float(max_range_m)
        self.backend = backend
        self.points = np.empty((0, 3), dtype=np.float32)
        # RTX object IDs are 128-bit values.  NumPy/HDF5 have no portable
        # uint128 dtype, so keep the exact value as a fixed-width hex string.
        self.object_ids = np.empty((0,), dtype="<U32")
        self.stable_id_map = {}
        self.scan_count = 0
        self.total_valid_points = 0
        self.last_timestamp_s = 0.0
        self.prim_path = str(getattr(lidar_sensor, "prim_path", ""))
        self.output_frame = "UNKNOWN"
        self.diagnostics = []
        self.first_points = None
        self.mount_translation = np.zeros(3, dtype=np.float64)
        self.deferred_config = None

    @classmethod
    def create(cls, prim_path: str, output_dir: Path, config: dict):
        global _ACTIVE_STREAM
        translation = np.asarray(config.get("translation_xyz", [0.25, 0.0, 0.35]), dtype=np.float32)
        requested_backend = str(config.get("backend", "isaacsim-5.1")).strip().lower()
        use_experimental = requested_backend == "experimental"
        if use_experimental and not EXPERIMENTAL_RTX_AVAILABLE:
            raise RuntimeError("experimental RTX LiDAR backend requested but unavailable")
        if use_experimental:
            lidar = Lidar.create(
                prim_path,
                config=config.get("model", "Example_Rotary"),
                translations=translation,
                aux_output_level=config.get("aux_output_level", "FULL"),
            )
            sensor = LidarSensor(lidar, annotators=[])
            stream = cls(sensor, output_dir, config.get("max_range_m", 12.0), "experimental")
        else:
            # RTX render products cannot be constructed reliably until the
            # referenced stage has completed its first World.reset().
            stream = cls(None, output_dir, config.get("max_range_m", 12.0), "isaacsim-5.1")
            stream.prim_path = prim_path
            stream.deferred_config = dict(config)
            stream.output_frame = "SENSOR_REQUESTED"
            # Match the Isaac Sim 5.1 lifecycle: author a valid OmniLidar prim
            # and schema before World.reset, then bind LidarRtx afterwards.
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                prim = stage.DefinePrim(prim_path, "OmniLidar")
            if not prim.HasAPI("OmniSensorGenericLidarCoreAPI"):
                prim.ApplyAPI("OmniSensorGenericLidarCoreAPI")
        _ACTIVE_STREAM = stream
        stream.mount_translation = translation.astype(np.float64)
        if use_experimental:
            try:
                rep.WriterRegistry.register(M20FourViewLidarWriter)
            except Exception:
                pass
            sensor.attach_writer("M20FourViewLidarWriter")
        return stream

    def initialize_after_world_reset(self) -> None:
        """Initialize RTX products only after the stage and PhysX are ready."""
        if self.backend != "isaacsim-5.1":
            return
        config = self.deferred_config or {}
        self.sensor = LidarRtx(
            prim_path=self.prim_path,
            name="m20_rtx_lidar",
            translation=np.asarray(config.get("translation_xyz", [0.25, 0.0, 0.35]), dtype=np.float32),
            **{"omni:sensor:Core:outputFrameOfReference": "SENSOR"},
            **{"omni:sensor:Core:auxOutputType": str(config.get("aux_output_level", "FULL"))},
        )
        self.sensor.initialize()
        # Points and objId must be taken from the same GMO buffer.  Pairing a
        # point-cloud annotator with a separate ID buffer is not guaranteed to
        # preserve return ordering.
        self.sensor.attach_annotator("GenericModelOutput")
        self.sensor.attach_annotator("StableIdMap")
        try:
            import omni.usd
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.prim_path)
            self.output_frame = str(prim.GetAttribute("omni:sensor:Core:outputFrameOfReference").Get())
            self.prim_path = str(prim.GetPath())
        except Exception:
            self.output_frame = "SENSOR_REQUESTED"

    def update_mount(self, base_position, base_orientation) -> None:
        """Rigidly place the RTX sensor from the M20 base pose each render step."""
        position = np.asarray(base_position, dtype=np.float64).reshape(3)
        w, x, y, z = np.asarray(base_orientation, dtype=np.float64).reshape(4)
        rotation = np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        sensor_position = position + rotation @ self.mount_translation
        if self.sensor is not None:
            self.sensor.set_world_pose(position=sensor_position, orientation=np.asarray(base_orientation, dtype=np.float64))

    def poll(self) -> None:
        if self.backend != "isaacsim-5.1":
            return
        current = self.sensor.get_current_frame()
        payload = current.get("GenericModelOutput")
        if isinstance(payload, dict):
            payload = payload.get("data")
        if payload is None:
            return
        output = get_gmo_data(payload)
        if not getattr(output, "numElements", 0):
            return
        azimuth = np.deg2rad(np.asarray(output.x, dtype=np.float32))
        elevation = np.deg2rad(np.asarray(output.y, dtype=np.float32))
        distance = np.asarray(output.z, dtype=np.float32)
        horizontal = distance * np.cos(elevation)
        points = np.column_stack((
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            distance * np.sin(elevation),
        ))
        valid = np.all(np.isfinite(points), axis=1) & (distance > 0.08) & (distance <= self.max_range_m)
        if np.any(valid):
            self.points = points[valid].astype(np.float32, copy=False)
            raw_object_ids = getattr(output, "objId", None)
            if raw_object_ids is not None:
                decoded = LidarRtx.get_object_ids(raw_object_ids)
                if len(decoded) == len(points):
                    self.object_ids = np.asarray(
                        [f"{int(value):032x}" for value in decoded], dtype="<U32"
                    )[valid]
                else:
                    self.object_ids = np.empty((0,), dtype="<U32")
            stable_map_payload = current.get("StableIdMap")
            if isinstance(stable_map_payload, dict):
                stable_map_payload = stable_map_payload.get("data")
            if stable_map_payload is not None:
                try:
                    raw = np.asarray(stable_map_payload).tobytes()
                    decoded_map = LidarRtx.decode_stable_id_mapping(raw)
                    self.stable_id_map = {
                        f"{int(object_id):032x}": str(prim_path)
                        for object_id, prim_path in decoded_map.items()
                    }
                except Exception as exc:
                    self.diagnostics.append({"stable_id_map_error": repr(exc)})
            self.scan_count += 1
            self.total_valid_points += int(len(self.points))

    def consume_writer_data(self, data: dict) -> None:
        for render_product in data.get("renderProducts", {}).values():
            payload = render_product.get("GenericModelOutput")
            if isinstance(payload, dict):
                payload = payload.get("data")
            if payload is None:
                continue
            output = parse_generic_model_output_data(payload)
            if not output.numElements:
                continue
            azimuth = np.deg2rad(np.asarray(output.x, dtype=np.float32))
            elevation = np.deg2rad(np.asarray(output.y, dtype=np.float32))
            distance = np.asarray(output.z, dtype=np.float32)
            horizontal = distance * np.cos(elevation)
            points = np.column_stack((
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                distance * np.sin(elevation),
            ))
            valid = np.all(np.isfinite(points), axis=1) & (distance > 0.08) & (distance <= self.max_range_m)
            if np.any(valid):
                self.points = points[valid].astype(np.float32, copy=False)
                raw_object_ids = getattr(output, "objId", None)
                if raw_object_ids is not None:
                    decoded = LidarRtx.get_object_ids(raw_object_ids)
                    if len(decoded) == len(points):
                        self.object_ids = np.asarray(
                            [f"{int(value):032x}" for value in decoded], dtype="<U32"
                        )[valid]
                self.scan_count += 1
                self.total_valid_points += int(len(self.points))

    def frame(self, timestamp_s: float) -> dict:
        self.poll()
        self.last_timestamp_s = float(timestamp_s)
        if self.first_points is None and len(self.points):
            self.first_points = self.points.copy()
        position = [float("nan")] * 3
        orientation = [float("nan")] * 4
        try:
            sensor_position, sensor_orientation = self.sensor.get_world_pose()
            position = np.asarray(sensor_position, dtype=np.float64).reshape(-1)[:3].tolist()
            orientation = np.asarray(sensor_orientation, dtype=np.float64).reshape(-1)[:4].tolist()
        except Exception:
            pass
        sensor_to_world = np.eye(4, dtype=np.float64)
        if np.isfinite(position).all() and np.isfinite(orientation).all():
            w, x, y, z = np.asarray(orientation, dtype=np.float64)
            sensor_to_world[:3, :3] = np.asarray([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ])
            sensor_to_world[:3, 3] = np.asarray(position, dtype=np.float64)
        ranges = np.linalg.norm(self.points, axis=1) if len(self.points) else np.empty(0)
        sample = self.points[:: max(1, len(self.points) // 2048)] if len(self.points) else self.points
        self.diagnostics.append({
            "timestamp_s": self.last_timestamp_s,
            "scan_index": self.scan_count,
            "point_count": int(len(self.points)),
            "object_id_count": int(len(self.object_ids)),
            "nonzero_object_id_count": int(np.count_nonzero(self.object_ids != "0" * 32)),
            "stable_id_map_count": int(len(self.stable_id_map)),
            "sensor_world_position_xyz": position,
            "sensor_world_orientation_wxyz": orientation,
            "range_m_p10": float(np.percentile(ranges, 10)) if len(ranges) else None,
            "range_m_p50": float(np.percentile(ranges, 50)) if len(ranges) else None,
            "range_m_p90": float(np.percentile(ranges, 90)) if len(ranges) else None,
            "sample_sha256": hashlib.sha256(np.asarray(sample, dtype=np.float32).tobytes()).hexdigest(),
        })
        return {
            "points": self.points.copy(),
            **({"object_ids": self.object_ids.copy()} if len(self.object_ids) == len(self.points) else {}),
            **({"stable_id_map": dict(self.stable_id_map)} if self.stable_id_map else {}),
            "sensor_to_world": sensor_to_world,
            "frame_id": "m20_rtx_lidar",
            "timestamp_s": self.last_timestamp_s,
            "scan_index": self.scan_count,
        }

    def render_panel(self, points: np.ndarray | None = None, size=(640, 360)) -> Image.Image:
        cloud = self.points if points is None else np.asarray(points, dtype=np.float32)
        width, height = size
        image = Image.new("RGB", size, (7, 12, 17))
        draw = ImageDraw.Draw(image)
        cx, horizon = width // 2, 110

        def project(x, y, z):
            depth = max(0.35, float(x) + 1.4)
            return int(cx + 215.0 * float(y) / depth), int(horizon + 235.0 * (0.42 * float(x) - float(z)) / depth)

        for x in range(1, 13):
            draw.line((project(x, -6, -0.45), project(x, 6, -0.45)), fill=(22, 48, 58), width=1)
        for y in range(-6, 7):
            draw.line((project(0.15, y, -0.45), project(12, y, -0.45)), fill=(22, 48, 58), width=1)
        if cloud.ndim == 2 and len(cloud):
            keep = (cloud[:, 0] > -0.1) & (cloud[:, 2] > -1.2) & (cloud[:, 2] < 3.5)
            cloud = cloud[keep]
            if len(cloud) > 18000:
                cloud = cloud[:: int(math.ceil(len(cloud) / 18000))]
            ranges = np.linalg.norm(cloud[:, :2], axis=1)
            for point, distance in zip(cloud, ranges):
                px, py = project(*point)
                if 1 <= px < width - 1 and 34 <= py < height - 1:
                    ratio = min(1.0, float(distance) / self.max_range_m)
                    z_ratio = float(np.clip((point[2] + 1.0) / 3.5, 0.0, 1.0))
                    color = (int(55 + 200 * z_ratio), int(225 - 100 * ratio), int(255 - 135 * z_ratio))
                    draw.point((px, py), fill=color)
        origin = project(0.15, 0, -0.3)
        draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(255, 255, 255))
        # A compact ego-centric top view makes range changes easy to see while
        # retaining the 3D perspective as the primary visualization.
        inset = (width - 168, 40, width - 8, 200)
        draw.rectangle(inset, fill=(3, 8, 12), outline=(85, 105, 115))
        ix, iy = (inset[0] + inset[2]) // 2, (inset[1] + inset[3]) // 2
        scale = 6.0
        for radius_m in (4, 8, 12):
            radius_px = int(radius_m * scale)
            draw.ellipse((ix-radius_px, iy-radius_px, ix+radius_px, iy+radius_px), outline=(28, 55, 65))
        if cloud.ndim == 2 and len(cloud):
            top = cloud[:: max(1, len(cloud) // 6000)]
            for x, y, z in top:
                px, py = int(ix - y * scale), int(iy - x * scale)
                if inset[0] < px < inset[2] and inset[1] < py < inset[3]:
                    draw.point((px, py), fill=(80, 210, 245) if z < 0.5 else (245, 140, 150))
        draw.polygon(((ix, iy-6), (ix-4, iy+5), (ix+4, iy+5)), fill="white")
        draw.rectangle((0, 0, width, 32), fill=(0, 0, 0))
        draw.text((12, 9), f"M20-centred RTX LiDAR 3D | scans={self.scan_count} | points={len(self.points):,}", fill="white")
        return image

    def save_final(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        npz = self.output_dir / "lidar_final_scan.npz"
        np.savez_compressed(
            npz,
            points=self.points,
            object_ids=self.object_ids,
            timestamp_s=self.last_timestamp_s,
            scan_count=self.scan_count,
        )
        comparison = self.output_dir / "lidar_first_final_scans.npz"
        np.savez_compressed(
            comparison,
            first_points=np.empty((0, 3), dtype=np.float32) if self.first_points is None else self.first_points,
            final_points=self.points,
        )
        diagnostics_path = self.output_dir / "lidar_frame_diagnostics.json"
        diagnostics_path.write_text(json.dumps(self.diagnostics, indent=2) + "\n", encoding="utf-8")
        stable_map_path = self.output_dir / "lidar_stable_id_map.json"
        stable_map_path.write_text(json.dumps(self.stable_id_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = {
            "schema": "m20-rtx-lidar-summary/v1",
            "coordinate_frame": "M20 sensor frame: X forward, Y left, Z up; metres",
            "backend": self.backend,
            "sensor_prim_path": self.prim_path,
            "output_frame": self.output_frame,
            "scan_count": self.scan_count,
            "final_point_count": int(len(self.points)),
            "final_object_id_count": int(len(self.object_ids)),
            "stable_id_map_count": int(len(self.stable_id_map)),
            "total_valid_points": self.total_valid_points,
            "final_scan": str(npz),
            "first_final_scans": str(comparison),
            "frame_diagnostics": str(diagnostics_path),
            "stable_id_map": str(stable_map_path),
        }
        (self.output_dir / "lidar_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


class FourViewComposer:
    """Scene-independent M20 observation mosaic."""

    LABELS = ("Third-person follow", "M20 input camera", "Piper wrist camera", "M20-centred RTX LiDAR 3D")

    @staticmethod
    def camera_rgb(camera, size=(640, 360)) -> Image.Image:
        value = np.asarray(camera.get_rgb(device="cpu"))
        if value.ndim != 3 or value.shape[2] < 3:
            raise RuntimeError(f"camera returned invalid shape {value.shape}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        return Image.fromarray(value).resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def label(panel: Image.Image, text: str) -> Image.Image:
        panel = panel.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 32), fill=(0, 0, 0))
        draw.text((12, 9), text, fill="white")
        return panel

    def compose(self, follow_camera, input_camera, wrist_camera, lidar_panel: Image.Image) -> Image.Image:
        canvas = Image.new("RGB", (1280, 720))
        canvas.paste(self.label(self.camera_rgb(follow_camera), self.LABELS[0]), (0, 0))
        canvas.paste(self.label(self.camera_rgb(input_camera), self.LABELS[1]), (640, 0))
        canvas.paste(self.label(self.camera_rgb(wrist_camera), self.LABELS[2]), (0, 360))
        canvas.paste(lidar_panel.resize((640, 360), Image.Resampling.LANCZOS), (640, 360))
        return canvas
