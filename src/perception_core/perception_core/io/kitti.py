"""Reader for the KITTI raw dataset (Velodyne LiDAR + 3D tracklet labels).

Turns a KITTI raw drive into :class:`~perception_core.common.types.Frame`
objects so the *same* detect -> track -> predict pipeline used on synthetic and
CARLA data can be benchmarked on real sensor data. LiDAR-only: needs the
``*_sync`` drive, the day's calibration, and the drive's ``tracklet_labels.xml``.

KITTI's Velodyne frame is already right-handed (x forward, y left, z up), which
matches ``perception_core``. Ego motion is recovered from the OXTS GPS/IMU so
tracks and forecasts live in a stable world frame; ground-truth tracklet boxes
are lifted into the same world frame for a fair comparison.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from perception_core.common.geometry import transform_box
from perception_core.common.types import (
    Box3D,
    Detection,
    Frame,
    ObjectClass,
    SensorCalibration,
)

# KITTI object types -> our taxonomy.
_KITTI_LABELS = {
    "Car": "car", "Van": "van", "Truck": "truck",
    "Pedestrian": "pedestrian", "Person (sitting)": "pedestrian",
    "Cyclist": "bicycle", "Tram": "bus", "Misc": "unknown",
}
_EARTH_RADIUS = 6378137.0  # metres (WGS-84 sphere used by the KITTI devkit)


def _read_rt(path: str) -> np.ndarray:
    """Parse a KITTI ``R:``/``T:`` calibration file into a 4x4 matrix."""
    R = T = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("R:"):
                R = np.array([float(v) for v in line[2:].split()]).reshape(3, 3)
            elif line.startswith("T:"):
                T = np.array([float(v) for v in line[2:].split()])
    m = np.eye(4)
    if R is not None:
        m[:3, :3] = R
    if T is not None:
        m[:3, 3] = T
    return m


def _oxts_to_pose(records: List[np.ndarray], scale: float) -> List[np.ndarray]:
    """Convert OXTS (lat, lon, alt, roll, pitch, yaw) records to world<-imu poses."""
    poses, origin_inv = [], None
    for rec in records:
        lat, lon, alt, roll, pitch, yaw = rec[:6]
        mx = scale * lon * np.pi * _EARTH_RADIUS / 180.0
        my = scale * _EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        t = np.array([mx, my, alt])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        pose = np.eye(4)
        pose[:3, :3] = Rz @ Ry @ Rx
        pose[:3, 3] = t
        if origin_inv is None:
            origin_inv = np.linalg.inv(pose)
        poses.append(origin_inv @ pose)  # relative to the first frame
    return poses


# --------------------------------------------------------------------------- #
# Tracklet labels                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Tracklet:
    """One annotated object and its per-frame boxes in the Velodyne frame."""

    label: ObjectClass
    first_frame: int
    boxes: List[Box3D] = field(default_factory=list)  # one per frame, contiguous

    def frame_range(self) -> range:
        return range(self.first_frame, self.first_frame + len(self.boxes))


def parse_tracklets(xml_path: str) -> List[Tracklet]:
    """Parse KITTI ``tracklet_labels.xml`` into :class:`Tracklet` objects.

    KITTI stores each pose as the box's *bottom* centre ``(tx, ty, tz)`` and a
    heading ``rz`` in the Velodyne frame; ``h/w/l`` are the extents. We lift the
    centre to the box middle (``z += h/2``) to match :class:`Box3D`.
    """
    root = ET.parse(xml_path).getroot()
    tracklets_el = root.find("tracklets")
    if tracklets_el is None:  # some exports omit the wrapper
        tracklets_el = root
    out: List[Tracklet] = []
    for item in tracklets_el.findall("item"):
        obj_type = (item.findtext("objectType") or "").strip()
        label = ObjectClass.from_str(_KITTI_LABELS.get(obj_type, "unknown"))
        h = float(item.findtext("h") or 0.0)
        w = float(item.findtext("w") or 0.0)
        l = float(item.findtext("l") or 0.0)
        first = int(item.findtext("first_frame") or 0)
        poses = item.find("poses")
        boxes: List[Box3D] = []
        if poses is not None:
            for pose in poses.findall("item"):
                tx = float(pose.findtext("tx") or 0.0)
                ty = float(pose.findtext("ty") or 0.0)
                tz = float(pose.findtext("tz") or 0.0)
                rz = float(pose.findtext("rz") or 0.0)
                boxes.append(Box3D(x=tx, y=ty, z=tz + h / 2.0, l=l, w=w, h=h, yaw=rz))
        out.append(Tracklet(label=label, first_frame=first, boxes=boxes))
    return out


# --------------------------------------------------------------------------- #
# Calibration                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class KittiCalib:
    velo_from_imu: np.ndarray                  # 4x4: p_velo = velo_from_imu @ p_imu
    lidar_to_cam: Dict[str, np.ndarray] = field(default_factory=dict)  # velo -> rect cam
    intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)    # 3x3
    image_size: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def sensor_calibration(self) -> SensorCalibration:
        return SensorCalibration(
            lidar_to_cam=dict(self.lidar_to_cam),
            intrinsics=dict(self.intrinsics),
            image_size=dict(self.image_size),
        )


def _read_matrix(path: str, key: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    with open(path) as fh:
        for line in fh:
            if line.startswith(key + ":"):
                vals = [float(v) for v in line.split(":", 1)[1].split()]
                return np.array(vals).reshape(shape)
    return None


def load_calib(calib_dir: str) -> KittiCalib:
    """Load the day's ``calib_velo_to_cam`` / ``calib_imu_to_velo`` /
    ``calib_cam_to_cam`` files into extrinsics + a rectified cam2 projection.
    """
    velo_from_imu = _read_rt(os.path.join(calib_dir, "calib_imu_to_velo.txt"))
    velo_to_cam0 = _read_rt(os.path.join(calib_dir, "calib_velo_to_cam.txt"))

    calib = KittiCalib(velo_from_imu=velo_from_imu)
    cam_to_cam = os.path.join(calib_dir, "calib_cam_to_cam.txt")
    if os.path.exists(cam_to_cam):
        R_rect = np.eye(4)
        r = _read_matrix(cam_to_cam, "R_rect_00", (3, 3))
        if r is not None:
            R_rect[:3, :3] = r
        P2 = _read_matrix(cam_to_cam, "P_rect_02", (3, 4))
        size = _read_matrix(cam_to_cam, "S_rect_02", (1, 2))
        if P2 is not None:
            K = P2[:3, :3]
            # Fold the projection-matrix baseline term into the cam2 extrinsic so
            # a plain pinhole K projects correctly: t2 = K^-1 @ P2[:, 3].
            velo_to_rect = R_rect @ velo_to_cam0
            t2 = np.linalg.solve(K, P2[:3, 3])
            cam2 = velo_to_rect.copy()
            cam2[:3, 3] += t2
            calib.lidar_to_cam["cam2"] = cam2
            calib.intrinsics["cam2"] = K
            if size is not None:
                calib.image_size["cam2"] = (int(size[0, 0]), int(size[0, 1]))
    return calib


# --------------------------------------------------------------------------- #
# OXTS ego poses                                                              #
# --------------------------------------------------------------------------- #
def _read_oxts(oxts_dir: str) -> List[np.ndarray]:
    files = sorted(f for f in os.listdir(oxts_dir) if f.endswith(".txt"))
    records = []
    for name in files:
        with open(os.path.join(oxts_dir, name)) as fh:
            records.append(np.fromstring(fh.read(), sep=" "))
    return records


def _read_timestamps(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    ts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            hms = line.split(" ")[1]
            h, m, s = hms.split(":")
            ts.append(int(h) * 3600 + int(m) * 60 + float(s))
    if not ts:
        return None
    arr = np.array(ts)
    return arr - arr[0]


# --------------------------------------------------------------------------- #
# Reader                                                                      #
# --------------------------------------------------------------------------- #
class KittiRawReader:
    """Yield :class:`Frame` objects from a KITTI raw ``*_sync`` drive.

    ``root`` is the directory holding ``<date>/`` (e.g. ``.../2011_09_26``). Each
    frame carries the raw Velodyne cloud, the ``world <- velo`` ego pose (from
    OXTS), the calibration, and ground-truth boxes lifted into the world frame so
    they line up with the pipeline's world-frame tracks. Ground-truth object
    identity (the tracklet index) is stashed in ``Detection.attributes['gt_id']``
    so tracking ID switches can be scored correctly.
    """

    #: KITTI image folders per camera name used elsewhere in the stack.
    _CAMERA_DIRS = {"cam0": "image_00", "cam1": "image_01",
                    "cam2": "image_02", "cam3": "image_03"}

    def __init__(self, root: str, date: str, drive: int, use_oxts: bool = True,
                 load_images: bool = False, camera: str = "cam2") -> None:
        self.date = date
        self.drive = drive
        calib_dir = os.path.join(root, date)
        self.drive_dir = os.path.join(calib_dir, f"{date}_drive_{drive:04d}_sync")

        velo_dir = os.path.join(self.drive_dir, "velodyne_points", "data")
        self._velo_files = sorted(
            os.path.join(velo_dir, f) for f in os.listdir(velo_dir) if f.endswith(".bin")
        )
        self.calib = load_calib(calib_dir)

        # Optional RGB frames (for the camera / camera-LiDAR-fusion detector path).
        self.camera = camera
        self._image_files: List[str] = []
        if load_images:
            img_dir = os.path.join(self.drive_dir, self._CAMERA_DIRS[camera], "data")
            self._image_files = sorted(
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.endswith((".png", ".jpg"))
            )

        # Ego trajectory (world <- velo) from OXTS, else static identity poses.
        n = len(self._velo_files)
        if use_oxts:
            records = _read_oxts(os.path.join(self.drive_dir, "oxts", "data"))
            scale = np.cos(records[0][0] * np.pi / 180.0)
            world_from_imu = _oxts_to_pose(records, scale)
            imu_from_velo = np.linalg.inv(self.calib.velo_from_imu)
            self._poses = [p @ imu_from_velo for p in world_from_imu]
        else:
            self._poses = [np.eye(4) for _ in range(n)]

        self._timestamps = _read_timestamps(
            os.path.join(self.drive_dir, "velodyne_points", "timestamps.txt")
        )

        # Per-frame ground truth, indexed by frame number (Velodyne-frame boxes).
        self._gt: List[List[Tuple[Box3D, ObjectClass, int]]] = [[] for _ in range(n)]
        xml = os.path.join(self.drive_dir, "tracklet_labels.xml")
        if os.path.exists(xml):
            for tid, trk in enumerate(parse_tracklets(xml)):
                for j, box in enumerate(trk.boxes):
                    f = trk.first_frame + j
                    if 0 <= f < n:
                        self._gt[f].append((box, trk.label, tid))

    def __len__(self) -> int:
        return len(self._velo_files)

    def _timestamp(self, index: int) -> float:
        if self._timestamps is not None and index < len(self._timestamps):
            return float(self._timestamps[index])
        return index * 0.1  # KITTI runs at ~10 Hz

    def _load_image(self, index: int) -> Optional[np.ndarray]:
        if not self._image_files or index >= len(self._image_files):
            return None
        from PIL import Image  # lazy: only the camera path needs Pillow
        with Image.open(self._image_files[index]) as im:
            return np.asarray(im.convert("RGB"))

    def read_frame(self, index: int) -> Frame:
        lidar = np.fromfile(self._velo_files[index], dtype=np.float32).reshape(-1, 4)
        ego_pose = self._poses[index]
        gt = [
            Detection(
                box=transform_box(box, ego_pose), score=1.0, label=label,
                source="ground_truth", attributes={"gt_id": float(tid)},
            )
            for box, label, tid in self._gt[index]
        ]
        images = {}
        img = self._load_image(index)
        if img is not None:
            images[self.camera] = img
        return Frame(
            timestamp=self._timestamp(index), frame_id=index,
            lidar=lidar.astype(float), calib=self.calib.sensor_calibration(),
            ego_pose=ego_pose, ground_truth=gt, images=images,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self.read_frame(i)

