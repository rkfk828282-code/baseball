"""월드 좌표 기반 관절 각도 계산.

주의: MediaPipe 월드 좌표의 깊이(z)는 단안 추정값이라 측면 촬영에서 오차가 크다.
외회전·몸통 회전처럼 깊이 방향 성분이 큰 각도는 '참고값'으로만 사용할 것.
"""
from __future__ import annotations

import numpy as np

from .landmarks import sides
from .signals import Track

CAMERA_UP = np.array([0.0, -1.0, 0.0])  # MediaPipe 좌표는 y가 아래 방향


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 벡터 사이 각도(도)."""
    c = np.clip(_dot(_unit(a), _unit(b)), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _reject(v: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    """v에서 axis 방향 성분을 제거 (axis에 수직인 평면으로 투영)."""
    return v - _dot(v, axis_unit)[..., None] * axis_unit


def trunk_frame(tr: Track, arm: str, ref_frame: int | None = None):
    """몸통 좌표계: up(척추 위), right(선수 오른쪽), forward(가슴 방향).

    forward의 부호는 좌표계 손잡이 가정에 의존하므로, ref_frame(보통 릴리스)에서
    투구 손이 몸통 앞쪽에 있도록 부호를 보정한다.
    """
    w = tr.world
    mid_sh = (w["left_shoulder"] + w["right_shoulder"]) / 2
    mid_hip = (w["left_hip"] + w["right_hip"]) / 2
    up = _unit(mid_sh - mid_hip)
    right = _unit(_reject(w["right_shoulder"] - w["left_shoulder"], up))
    forward = _unit(np.cross(up, right))
    if ref_frame is not None:
        throw, _ = sides(arm)
        lo, hi = max(0, ref_frame - 2), min(tr.n, ref_frame + 3)
        hand = w[f"{throw}_wrist"][lo:hi] - mid_sh[lo:hi]
        if np.nanmedian(_dot(hand, forward[lo:hi])) < 0:
            forward = -forward
    return up, right, forward


def shoulder_angles(tr: Track, arm: str, ref_frame: int | None = None) -> dict[str, np.ndarray]:
    """투구 팔의 어깨 외전, 어깨 외회전, 팔꿈치 굴곡 (도)."""
    throw, _ = sides(arm)
    w = tr.world
    S, E, W = w[f"{throw}_shoulder"], w[f"{throw}_elbow"], w[f"{throw}_wrist"]
    up, _, forward = trunk_frame(tr, arm, ref_frame)
    upper = _unit(E - S)
    fore = W - E

    # 어깨 외전: 위팔과 몸통 아래 방향 사이 각
    abd = _angle(upper, -up)
    # 팔꿈치 굴곡: 0 = 완전 신전
    elbow = _angle(upper, fore)

    # 어깨 외회전: 위팔 축에 수직인 평면에서 전완의 방향.
    #   0°  = 전완이 몸통 앞쪽(forward)을 향함
    #   90° = 전완이 위(머리 쪽)를 향함 (90° 외전 시 '코킹' 자세)
    #   MER은 보통 150~180° 이상 (전완이 뒤로 젖혀짐)
    e1 = _unit(_reject(forward, upper))
    sign = 1.0 if throw == "right" else -1.0
    e2 = sign * np.cross(upper, e1)
    f_p = _reject(fore, upper)
    er = np.degrees(np.arctan2(_dot(f_p, e2), _dot(f_p, e1)))
    er = _unwrap_deg(er)
    return {"shoulder_abd": abd, "shoulder_er": er, "elbow_flex": elbow}


def _unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    good = ~np.isnan(a)
    if good.sum() > 1:
        out[good] = np.degrees(np.unwrap(np.radians(a[good])))
        # 90° 근처 외전 자세 기준으로 0~360 범위에 오도록 이동
        med = np.nanmedian(out)
        out -= 360.0 * np.round((med - 90.0) / 360.0)
    return out


def rotation_angles(tr: Track) -> dict[str, np.ndarray]:
    """골반·몸통(어깨선)의 수평면 회전각과 엉덩이-어깨 분리각 (도).

    수평면 = 카메라 기준 수평 (자이로 수평계로 폰을 수평으로 맞춘다는 전제).
    """
    w = tr.world

    def yaw(a, b):
        v = w[b] - w[a]
        return _unwrap_deg_plain(np.degrees(np.arctan2(v[:, 2], v[:, 0])))

    pelvis = yaw("left_hip", "right_hip")
    trunk = yaw("left_shoulder", "right_shoulder")
    sep = (trunk - pelvis + 180.0) % 360.0 - 180.0
    return {"pelvis_rot": pelvis, "trunk_rot": trunk, "hip_shoulder_sep": sep}


def _unwrap_deg_plain(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    good = ~np.isnan(a)
    if good.sum() > 1:
        out[good] = np.degrees(np.unwrap(np.radians(a[good])))
    return out
