"""랜드마크 CSV 로드, 결측 보간, 저역통과 필터."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .landmarks import NAMES


@dataclass
class Track:
    fps: float
    frames: np.ndarray                      # 원본 영상 프레임 번호
    px: dict[str, np.ndarray]               # name -> (n,2) 픽셀 좌표, y는 아래가 +
    world: dict[str, np.ndarray]            # name -> (n,3) 월드 좌표(m)
    vis: dict[str, np.ndarray]              # name -> (n,) visibility
    missing: np.ndarray                     # (n,) 포즈 미검출 프레임
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def t(self) -> np.ndarray:
        return self.frames / self.fps

    def mid(self, a: str, b: str, space: str = "px") -> np.ndarray:
        d = self.px if space == "px" else self.world
        return (d[a] + d[b]) / 2.0


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """NaN 구간을 선형 보간. max_gap보다 긴 구간은 NaN 유지. 양 끝은 최근접값."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    nan = np.isnan(x)
    if nan.all() or not nan.any():
        return x
    idx = np.arange(n)
    good = ~nan
    interp = np.interp(idx, idx[good], x[good])
    # 긴 결측 구간은 보간하지 않음
    i = 0
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            inner = i > 0 and j < n
            if (j - i) <= max_gap or not inner:
                x[i:j] = interp[i:j]
            i = j
        else:
            i += 1
    return x


def lowpass(x: np.ndarray, fps: float, cutoff: float, order: int = 4) -> np.ndarray:
    """영위상(filtfilt) 버터워스 저역통과. NaN 구간은 건너뛰고 구간별로 필터."""
    x = np.asarray(x, dtype=float)
    cutoff = min(cutoff, 0.4 * fps)  # 나이퀴스트 여유
    if cutoff <= 0:
        return x.copy()
    b, a = butter(order, cutoff / (fps / 2.0))
    out = np.full_like(x, np.nan)
    good = ~np.isnan(x)
    # 연속된 유효 구간마다 필터
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
    pad = 3 * max(len(a), len(b))
    for s, e in zip(edges[::2], edges[1::2]):
        seg = x[s:e]
        out[s:e] = filtfilt(b, a, seg) if len(seg) > pad else seg
    return out


def load_track(
    csv_path: str | Path,
    fps: float | None = None,
    cutoff_hz: float = 15.0,
    min_vis: float = 0.3,
    max_gap_s: float = 0.05,
) -> Track:
    """landmarks.csv를 읽어 보간·필터를 적용한 Track을 만든다.

    fps가 없으면 같은 폴더의 meta.json에서 읽는다.
    min_vis 미만의 픽셀 좌표는 결측 처리 후 보간한다.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if fps is None:
        meta_path = csv_path.with_name("meta.json")
        if meta_path.exists():
            fps = float(json.loads(meta_path.read_text())["fps"])
        else:
            dt = np.median(np.diff(df["t"].to_numpy()))
            fps = 1.0 / dt
    max_gap = max(1, int(round(max_gap_s * fps)))

    px, world, vis = {}, {}, {}
    for n in NAMES:
        v = df[f"{n}_v"].to_numpy(dtype=float)
        low = ~(v >= min_vis)
        xy = df[[f"{n}_x", f"{n}_y"]].to_numpy(dtype=float)
        xy[low] = np.nan
        w = df[[f"{n}_wx", f"{n}_wy", f"{n}_wz"]].to_numpy(dtype=float)
        px[n] = np.column_stack([lowpass(fill_gaps(xy[:, k], max_gap), fps, cutoff_hz) for k in range(2)])
        world[n] = np.column_stack([lowpass(fill_gaps(w[:, k], max_gap), fps, cutoff_hz) for k in range(3)])
        vis[n] = v
    return Track(
        fps=fps,
        frames=df["frame"].to_numpy(dtype=int),
        px=px,
        world=world,
        vis=vis,
        missing=df["detected"].to_numpy() == 0,
        params={"cutoff_hz": cutoff_hz, "min_vis": min_vis, "max_gap_s": max_gap_s},
    )


def speed(xy: np.ndarray, fps: float) -> np.ndarray:
    """(n,d) 좌표의 프레임별 속력 (단위/초)."""
    return np.linalg.norm(np.gradient(xy, axis=0), axis=1) * fps
