"""数据装载:data/capture_XXXX 布局。

- 位姿: capture.json → pose.values = [X,Y,Z mm, Rx,Ry,Rz °] → T_base_flange。
  欧拉约定 XYZ 外旋 R = Rz·Ry·Rx(与参考工程 config.yaml euler_convention: XYZ 一致,
  用户已确认;intrinsic 仅作自检对照)。
- PCD: ascii x/y/z/intensity,无效回波已在导出时剔除(无 NaN)。
  帧完整性 = 声明点数 == 实际行数 且 ≥ ratio × 本 capture 最大声明数
  (帧 0 常为残帧,尾帧可能截断)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import bootref  # noqa: F401  必须最先 import(注入参考工程 sys.path)
from calib.geometry.euler import euler_xyz_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
LIDARS = ("airy_202", "airy_205")

CONVENTIONS = ("extrinsic", "intrinsic")


def values_to_T(values, convention: str = "extrinsic") -> np.ndarray:
    """pose.values → T_base_flange 4x4。extrinsic: R = Rz·Ry·Rx。"""
    v = np.asarray(values, dtype=float)
    if v.shape != (6,):
        raise ValueError(f"pose.values 应为 6 维, got {v.shape}")
    T = np.eye(4)
    T[:3, 3] = v[:3] / 1000.0
    rx, ry, rz = np.deg2rad(v[3]), np.deg2rad(v[4]), np.deg2rad(v[5])
    T[:3, :3] = euler_xyz_to_matrix(rx, ry, rz, convention=convention)
    return T


def list_captures(data_root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    return sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith("capture_")
        and (p / "capture.json").is_file()
    )


def load_capture_meta(data_root: Path, capture: str) -> dict:
    with open(data_root / capture / "capture.json") as f:
        return json.load(f)


def load_all_poses(
    data_root: Path = DEFAULT_DATA_ROOT, convention: str = "extrinsic"
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """全部 capture 的位姿,按目录名升序。

    Returns:
        names: ["capture_0001", ...]
        Ts: [T_base_flange 4x4, ...]
        values: (N, 6) 原始 mm/deg
    """
    names, Ts, vals = [], [], []
    for cap in list_captures(data_root):
        meta = load_capture_meta(data_root, cap)
        v = meta["pose"]["values"]
        names.append(cap)
        Ts.append(values_to_T(v, convention))
        vals.append(v)
    return names, Ts, np.array(vals)


# ---------------- PCD 帧校验 ----------------

@dataclass
class FrameInfo:
    frame: int
    path: Path
    declared: int
    actual: int
    status: str = ""   # complete / truncated / partial / bad_header
    reason: str = ""


def _scan_pcd(path: Path) -> tuple[int, int] | None:
    """解析 header POINTS + 数实际数据行数(不解析坐标,mm 级开销)。仅支持 ascii。"""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
            m_pts = re.search(rb"^POINTS (\d+)", head, re.M)
            m_data = re.search(rb"^DATA (\w+)", head, re.M)
            if not m_pts or not m_data or m_data.group(1) != b"ascii":
                return None
            declared = int(m_pts.group(1))
            data_line_end = head.find(b"\n", m_data.end())
            if data_line_end < 0:
                return None
            header_newlines = head[:data_line_end].count(b"\n") + 1
            total_newlines = 0
            last_byte = b"\n"
            f.seek(0)
            while True:
                chunk = f.read(4 << 20)
                if not chunk:
                    break
                total_newlines += chunk.count(b"\n")
                last_byte = chunk[-1:]
        actual = total_newlines - header_newlines
        if last_byte != b"\n":
            actual += 1
        return declared, actual
    except OSError:
        return None


def _frame_key(p: Path) -> int:
    return int(p.stem) if p.stem.isdigit() else 10 ** 9


def scan_lidar_frames(lidar_dir: Path) -> list[FrameInfo]:
    out: list[FrameInfo] = []
    for p in sorted(lidar_dir.glob("*.pcd"), key=_frame_key):
        r = _scan_pcd(p)
        if r is None:
            out.append(FrameInfo(_frame_key(p), p, -1, -1, "bad_header", "无法解析 header 或非 ascii"))
        else:
            out.append(FrameInfo(_frame_key(p), p, r[0], r[1]))
    return out


def classify_frames(frames: list[FrameInfo], ratio: float = 0.9) -> list[FrameInfo]:
    """complete = declared==actual 且 declared ≥ ratio × 本 capture 最大 declared。"""
    if not frames:
        return []
    max_declared = max(f.declared for f in frames if f.declared > 0)
    thr = ratio * max_declared
    for f in frames:
        if f.status == "bad_header":
            continue
        if f.actual != f.declared:
            f.status, f.reason = "truncated", f"actual={f.actual} != declared={f.declared}"
        elif f.declared < thr:
            f.status, f.reason = "partial", f"declared={f.declared} < {thr:.0f}"
        else:
            f.status, f.reason = "complete", ""
    return frames
