"""영상 → 프레임별 포즈 랜드마크 CSV.

출력:
  <out_dir>/landmarks.csv  프레임별 33개 관절의 픽셀 좌표(x,y), visibility(v),
                           월드 좌표(wx,wy,wz; 미터, 엉덩이 중심 원점)
  <out_dir>/meta.json      fps, 해상도, 원본 경로 등
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .landmarks import NAMES

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_model(variant: str = "heavy") -> Path:
    """lite / full / heavy 모델을 받아 analysis/models/ 에 저장."""
    if variant not in ("lite", "full", "heavy"):
        raise ValueError("model은 lite, full, heavy 중 하나")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        print(f"[모델] {variant} 모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    return path


def columns() -> list[str]:
    cols = ["frame", "t", "detected"]
    for n in NAMES:
        cols += [f"{n}_x", f"{n}_y", f"{n}_v", f"{n}_wx", f"{n}_wy", f"{n}_wz"]
    return cols


def extract(
    video: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    model: str = "heavy",
    start: int = 0,
    end: int | None = None,
) -> Path:
    """영상에서 포즈를 추출해 landmarks.csv를 만든다.

    fps: 실제 촬영 fps. 갤럭시 슬로모션 파일은 재생용 fps(30 등)로
         저장되는 경우가 있으므로, 240fps로 찍었다면 --fps 240을 꼭 지정.
    """
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    use_fps = float(fps) if fps else file_fps
    if use_fps <= 0:
        raise ValueError("fps를 알 수 없습니다. --fps로 지정하세요.")
    if fps is None and file_fps < 100:
        print(f"[주의] 파일 fps가 {file_fps:.1f}입니다. 240fps로 찍었다면 --fps 240을 지정하세요.")

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model(model))),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows = []
    width = height = None
    t0 = time.time()
    last_ts = -1
    with PoseLandmarker.create_from_options(opts) as landmarker:
        idx = start
        while True:
            if end is not None and idx >= end:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # VIDEO 모드는 단조 증가하는 ms 타임스탬프가 필요
            ts = max(int(round(idx * 1000.0 / use_fps)), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ts)

            row = [idx, idx / use_fps]
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                wl = res.pose_world_landmarks[0]
                row.append(1)
                for p, w in zip(lm, wl):
                    vis = p.visibility if p.visibility is not None else np.nan
                    row += [p.x * width, p.y * height, vis, w.x, w.y, w.z]
            else:
                row.append(0)
                row += [np.nan] * (6 * len(NAMES))
            rows.append(row)
            idx += 1
            if idx % 200 == 0:
                print(f"  {idx - start}/{(end or n_total) - start} 프레임 처리 ({time.time() - t0:.0f}s)")
    cap.release()

    df = pd.DataFrame(rows, columns=columns())
    csv_path = out_dir / "landmarks.csv"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    meta = {
        "video": str(video),
        "fps": use_fps,
        "file_fps": file_fps,
        "width": width,
        "height": height,
        "model": model,
        "frames": len(df),
        "detected_frames": int(df["detected"].sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    rate = meta["detected_frames"] / max(len(df), 1) * 100
    print(f"[추출 완료] {csv_path}  ({len(df)} 프레임, 검출률 {rate:.1f}%)")
    return csv_path
