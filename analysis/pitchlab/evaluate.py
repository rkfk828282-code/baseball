"""자동 검출 vs 수동 라벨 비교 (0단계 기술 검증 판정).

labels.csv 형식 (프레임 번호는 오버레이 영상 좌상단에 표시되는 번호):
    clip,FC,MER,BR
    pitch01,812,850,861
    pitch02,640,,690          <- MER은 비워도 됨

결과 폴더 구조: <results>/<clip>/events.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = ("FC", "MER", "BR")


def compare(labels_csv: str | Path, results_dir: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, dtype={"clip": str})
    rows = []
    for _, lab in labels.iterrows():
        path = Path(results_dir) / lab["clip"] / "events.json"
        if not path.exists():
            print(f"[건너뜀] 결과 없음: {path}")
            continue
        res = json.loads(path.read_text())
        fps = res["fps"]
        for ev in EVENTS:
            if ev not in lab or pd.isna(lab[ev]):
                continue
            auto = res["events"].get(ev)
            err = None if auto is None else auto["frame"] - int(lab[ev])
            rows.append({
                "clip": lab["clip"],
                "event": ev,
                "manual": int(lab[ev]),
                "auto": None if auto is None else auto["frame"],
                "err_frames": err,
                "err_ms": None if err is None else round(err / fps * 1000, 1),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, tol_frames: int = 2, pass_rate: float = 0.8) -> dict:
    """이벤트별 오차 요약과 통과 여부.

    기본 기준(제안값): FC·BR이 각각 클립의 80% 이상에서 ±2프레임(240fps에서 약 8ms) 이내.
    MER은 3D 추정 참고값이라 판정에서 제외.
    """
    out = {}
    for ev in EVENTS:
        d = df[df["event"] == ev]
        if d.empty:
            continue
        e = d["err_frames"].astype(float)
        found = e.notna()
        ae = e[found].abs()
        out[ev] = {
            "clips": int(len(d)),
            "detected": int(found.sum()),
            "mean_err_frames": round(float(e[found].mean()), 2) if found.any() else None,
            "mae_frames": round(float(ae.mean()), 2) if found.any() else None,
            "mae_ms": round(float(d["err_ms"].dropna().abs().mean()), 1) if found.any() else None,
            "within_1": round(float((ae <= 1).sum() / len(d)), 3),
            "within_2": round(float((ae <= 2).sum() / len(d)), 3),
            "within_4": round(float((ae <= 4).sum() / len(d)), 3),
            f"pass(±{tol_frames}f ≥{int(pass_rate * 100)}%)": (
                None if ev == "MER" else bool((ae <= tol_frames).sum() / len(d) >= pass_rate)
            ),
        }
    return out


def report(labels_csv, results_dir, tol_frames=2, pass_rate=0.8) -> tuple[pd.DataFrame, dict]:
    df = compare(labels_csv, results_dir)
    if df.empty:
        print("비교할 데이터가 없습니다.")
        return df, {}
    summary = summarize(df, tol_frames, pass_rate)
    with pd.option_context("display.width", 120):
        print(df.to_string(index=False))
    print()
    for ev, s in summary.items():
        print(f"[{ev}] " + ", ".join(f"{k}={v}" for k, v in s.items()))
    out_dir = Path(results_dir)
    df.to_csv(out_dir / "evaluation.csv", index=False)
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    # 평균 오차가 한쪽으로 치우치면(예: 항상 2프레임 늦음) 파라미터 보정으로 해결 가능
    for ev, s in summary.items():
        if s["mean_err_frames"] is not None and s["mae_frames"] and abs(s["mean_err_frames"]) > 0.7 * s["mae_frames"]:
            direction = "늦게" if s["mean_err_frames"] > 0 else "일찍"
            print(f"  → {ev}가 일관되게 {direction} 검출됨 (평균 {s['mean_err_frames']}프레임): 오프셋/파라미터 보정 여지 있음")
    return df, summary
