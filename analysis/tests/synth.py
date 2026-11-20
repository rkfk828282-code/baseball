"""정답 이벤트를 알고 있는 합성 투구 데이터 (landmarks.csv 형식).

우투수(R), 240fps, 1.5초. 화면 오른쪽(+x)이 투구 방향.
  - 레그킥 정점: frame 150
  - FC          : frame 250 (앞발=왼발이 빠르게 내려와 착지)
  - 골반/몸통/팔꿈치 신전 각속도 최고점: 270 / 285 / 295
  - MER         : frame 285 (외회전 175°)
  - BR          : frame 300 (손목 속력 최고점)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import erf

from pitchlab.landmarks import NAMES
from pitchlab.pose_extract import columns

FPS = 240.0
N = 360
TRUTH = {"knee_peak": 150, "FC": 250, "MER": 285, "BR": 300}


def _sig(i, center, width):
    return 1.0 / (1.0 + np.exp(-(i - center) / width))


def _ramp(i, a, b, va, vb):
    """a→b 구간에서 va→vb 코사인 보간, 구간 밖은 끝값 유지."""
    s = np.clip((i - a) / (b - a), 0, 1)
    return va + (vb - va) * (1 - np.cos(np.pi * s)) / 2


def arm_world(psi_deg, kappa_deg=90.0, alpha_deg=0.0, yaw_deg=0.0, arm="R"):
    """한 프레임의 몸통·투구 팔 월드 좌표.

    psi: 외회전, kappa: 팔꿈치 굴곡, alpha: 수평 내전, yaw: 몸통이 향한 방향.
    """
    up = np.array([0.0, -1.0, 0.0])
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    right = np.cross(fwd, up)
    mid_hip = np.zeros(3)
    mid_sh = mid_hip + 0.5 * up
    side = right if arm == "R" else -right
    a = np.radians(alpha_deg)
    u = np.cos(a) * side + np.sin(a) * fwd                  # 위팔 방향
    e1 = fwd - np.dot(fwd, u) * u
    e1 /= np.linalg.norm(e1)
    e2 = (1 if arm == "R" else -1) * np.cross(u, e1)
    psi, k = np.radians(psi_deg), np.radians(kappa_deg)
    fore = np.cos(k) * u + np.sin(k) * (np.cos(psi) * e1 + np.sin(psi) * e2)
    S = mid_sh + 0.2 * side
    E = S + 0.3 * u
    W = E + 0.27 * fore
    return {
        "mid_sh": mid_sh, "right": right, "fwd": fwd,
        "right_shoulder": mid_sh + 0.2 * right, "left_shoulder": mid_sh - 0.2 * right,
        "right_hip": mid_hip + 0.15 * right, "left_hip": mid_hip - 0.15 * right,
        "S": S, "E": E, "W": W,
    }


def make_df(arm: str = "R") -> pd.DataFrame:
    i = np.arange(N, dtype=float)
    throw, lead = ("right", "left") if arm == "R" else ("left", "right")

    # --- 픽셀 좌표 (몸통 길이 200px) ---
    px = {n: np.tile([400.0, 400.0], (N, 1)) for n in NAMES}
    for n in NAMES:
        if "shoulder" in n:
            px[n] = np.tile([400.0, 300.0], (N, 1))
        elif "hip" in n:
            px[n] = np.tile([400.0, 500.0], (N, 1))
    # 앞발: 바닥(800) → 레그킥(650, frame150) → 가속하며 내려와 frame250 착지
    fy = np.full(N, 800.0)
    fx = np.full(N, 300.0)
    up_ = (i >= 100) & (i < 150)
    fy[up_] = _ramp(i[up_], 100, 150, 800, 650)
    down = (i >= 150) & (i < 250)
    tau = (i[down] - 150) / 100
    fy[down] = 650 + 150 * tau**2
    fx[down] = 300 + 300 * tau
    fx[i >= 250] = 600
    for k, dy in (("ankle", 0), ("heel", 5), ("foot_index", 8)):
        px[f"{lead}_{k}"] = np.column_stack([fx + (10 if k == "foot_index" else 0), fy + dy])
    px[f"{lead}_knee"] = np.column_stack([fx, fy - 150])
    for k in ("ankle", "heel", "foot_index"):
        px[f"{throw}_{k}"] = np.tile([200.0, 800.0], (N, 1))
    # 투구 손목: 속력이 BR에서 최대인 가우시안 프로파일
    sig = 0.02 * FPS
    wx = 400 + 500 * (1 + erf((i - TRUTH["BR"]) / (np.sqrt(2) * sig))) / 2
    px[f"{throw}_wrist"] = np.column_stack([wx, np.full(N, 280.0)])

    # --- 월드 좌표 ---
    psi = np.where(i < 250, 60.0, 0.0)
    psi = np.where((i >= 250) & (i < 285), _ramp(i, 250, 285, 60, 175), psi)
    psi = np.where((i >= 285) & (i < 300), _ramp(i, 285, 300, 175, 100), psi)
    psi = np.where(i >= 300, _ramp(i, 300, 340, 100, 60), psi)
    kappa = 90 - 70 * _sig(i, 295, 3)
    alpha = _ramp(i, 285, 300, 0, 25)
    trunk_yaw = -90 + 90 * _sig(i, 285, 4)
    pelvis_yaw = -90 + 90 * _sig(i, 270, 4)

    world = {n: np.zeros((N, 3)) for n in NAMES}
    for j in range(N):
        a = arm_world(psi[j], kappa[j], alpha[j], trunk_yaw[j], arm)
        p = arm_world(0, 90, 0, pelvis_yaw[j], arm)
        world["right_shoulder"][j], world["left_shoulder"][j] = a["right_shoulder"], a["left_shoulder"]
        world["right_hip"][j], world["left_hip"][j] = p["right_hip"], p["left_hip"]
        world[f"{throw}_elbow"][j], world[f"{throw}_wrist"][j] = a["E"], a["W"]

    rows = {"frame": i.astype(int), "t": i / FPS, "detected": np.ones(N, dtype=int)}
    for n in NAMES:
        rows[f"{n}_x"], rows[f"{n}_y"] = px[n][:, 0], px[n][:, 1]
        rows[f"{n}_v"] = np.full(N, 0.99)
        rows[f"{n}_wx"], rows[f"{n}_wy"], rows[f"{n}_wz"] = world[n].T
    return pd.DataFrame(rows)[columns()]
