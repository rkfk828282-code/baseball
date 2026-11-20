"""결과 저장: events.json, 시계열 CSV, 신호 그래프 PNG, 오버레이 영상."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import SKELETON

EVENT_COLORS = {"FC": "tab:green", "MER": "tab:orange", "BR": "tab:red"}


def save_result(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {k: v for k, v in result["_series"].items() if v is not None}
    pd.DataFrame(series).to_csv(out_dir / "series.csv", index=False, float_format="%.4f")
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    path = out_dir / "events.json"
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))
    return path


def plot_signals(result: dict, out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["_series"]
    t = s["t"]
    panels = [
        ("wrist_speed", "Throwing wrist speed (torso/s)"),
        ("foot_y", "Lead foot height (px, down=+)"),
        ("foot_speed", "Lead foot speed (torso/s)"),
        ("shoulder_er", "Shoulder ER (deg, 3D est.)"),
        ("shoulder_abd", "Shoulder abduction (deg)"),
        ("elbow_flex", "Elbow flexion (deg)"),
        (("w_pelvis", "w_trunk", "w_elbow_ext"), "Angular velocity (deg/s)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.0 * len(panels)), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        keys = key if isinstance(key, tuple) else (key,)
        for k in keys:
            if s.get(k) is not None:
                ax.plot(t, s[k], lw=1.2, label=k if len(keys) > 1 else None)
        if key == "foot_y":
            ax.invert_yaxis()
        if len(keys) > 1:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        for name, ev in result["events"].items():
            if ev:
                ax.axvline(ev["t"], color=EVENT_COLORS[name], ls="--", lw=1)
    for name, ev in result["events"].items():
        if ev:
            axes[0].text(ev["t"], axes[0].get_ylim()[1], f" {name}", color=EVENT_COLORS[name],
                         va="top", fontsize=9)
    # 투구 구간 확대
    br = result["events"]["BR"]["t"]
    fc = result["events"]["FC"]["t"] if result["events"]["FC"] else br - 0.3
    axes[-1].set_xlim(fc - 0.4, br + 0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)


def render_overlay(
    video: str | Path,
    landmarks_csv: str | Path,
    result: dict | None,
    out_path: str | Path,
    out_fps: float = 30.0,
    margin_s: float = 0.6,
) -> Path:
    """원본 영상에 뼈대·프레임 번호·이벤트를 그려 저장. 수동 라벨링에도 사용.

    out_fps를 낮게(30) 두면 240fps 영상이 8배 슬로모션으로 재생된다.
    result가 있으면 FC 전 margin_s ~ BR 후 margin_s 구간만 저장.
    """
    import cv2

    df = pd.read_csv(landmarks_csv).set_index("frame")
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first, last = int(df.index.min()), int(df.index.max())
    ev_frames = {}
    if result:
        fps = result["fps"]
        ev_frames = {k: v["frame"] for k, v in result["events"].items() if v}
        anchor = ev_frames.get("FC", ev_frames["BR"])
        first = max(first, int(anchor - margin_s * fps))
        last = min(last, int(ev_frames["BR"] + margin_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    writer = None
    for f in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            H, W = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        if f in df.index and df.at[f, "detected"] == 1:
            row = df.loc[f]
            for a, b in SKELETON:
                pa = (row[f"{a}_x"], row[f"{a}_y"])
                pb = (row[f"{b}_x"], row[f"{b}_y"])
                if np.isnan(pa + pb).any():
                    continue
                color = (0, 200, 255) if "right" in a and "right" in b else (255, 160, 0)
                cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), color, 2, cv2.LINE_AA)
        scale = max(H, W) / 1000
        label = f"frame {f}"
        hit = [k for k, v in ev_frames.items() if v == f]
        cv2.putText(img, label, (20, int(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale,
                    (255, 255, 255), max(2, int(3 * scale)), cv2.LINE_AA)
        if hit:
            cv2.putText(img, " / ".join(hit), (20, int(110 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6 * scale, (0, 0, 255), max(2, int(4 * scale)), cv2.LINE_AA)
        writer.write(img)
        # 이벤트 프레임은 0.5초 정지
        if hit:
            for _ in range(int(out_fps * 0.5)):
                writer.write(img)
    cap.release()
    if writer is not None:
        writer.release()
    return Path(out_path)
