"""명령줄 도구.

  python -m pitchlab analyze  영상.mp4 --arm R --fps 240 -o output/pitch01
  python -m pitchlab extract  영상.mp4 --fps 240 -o output/pitch01
  python -m pitchlab detect   output/pitch01 --arm R
  python -m pitchlab overlay  영상.mp4 output/pitch01
  python -m pitchlab evaluate labels.csv output/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import evaluate, report
from .events import DetectParams, analyze
from .signals import load_track


def _detect(out_dir: Path, arm: str, fps: float | None, cutoff: float) -> dict:
    tr = load_track(out_dir / "landmarks.csv", fps=fps, cutoff_hz=cutoff)
    res = analyze(tr, arm, DetectParams())
    report.save_result(res, out_dir)
    report.plot_signals(res, out_dir / "signals.png")
    _print_result(res)
    return res


def _print_result(res: dict) -> None:
    print(f"\n=== 결과 (fps {res['fps']:.0f}, 투구 팔 {res['arm']}) ===")
    for name, ev in res["events"].items():
        if ev:
            print(f"  {name:3s}: frame {ev['frame']:>6d}  t={ev['t']:.3f}s")
        else:
            print(f"  {name:3s}: 검출 실패")
    if res["fc_to_br_ms"] is not None:
        print(f"  FC→BR: {res['fc_to_br_ms']} ms")
    for name, a in res["angles"].items():
        print(f"  [{name}] " + ", ".join(f"{k}={v}" for k, v in a.items()))
    kc = res["kinetic_chain"]
    print(f"  회전 순서(골반→몸통→팔): {kc['order_ok']}, "
          f"몸통 회전 시작(FC 기준): {kc['trunk_rot_onset_ms_from_fc']} ms")
    print(f"  품질: {res['quality']}")
    for w in res["warnings"]:
        print(f"  [경고] {w}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pitchlab", description="투구 영상 포즈 분석 (0단계 기술 검증)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_video(p):
        p.add_argument("video")
        p.add_argument("--fps", type=float, help="실제 촬영 fps (슬로모션 파일이면 필수)")
        p.add_argument("--model", default="heavy", choices=["lite", "full", "heavy"])
        p.add_argument("--start", type=int, default=0, help="시작 프레임")
        p.add_argument("--end", type=int, help="끝 프레임(미포함)")

    p = sub.add_parser("analyze", help="추출 + 검출 + 오버레이 한 번에")
    common_video(p)
    p.add_argument("--arm", required=True, choices=["R", "L", "r", "l"])
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--cutoff", type=float, default=15.0, help="저역통과 차단 주파수(Hz)")
    p.add_argument("--no-overlay", action="store_true")

    p = sub.add_parser("extract", help="영상 → landmarks.csv")
    common_video(p)
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("detect", help="landmarks.csv → events.json, signals.png")
    p.add_argument("out")
    p.add_argument("--arm", required=True, choices=["R", "L", "r", "l"])
    p.add_argument("--fps", type=float)
    p.add_argument("--cutoff", type=float, default=15.0)

    p = sub.add_parser("overlay", help="뼈대·프레임 번호·이벤트 오버레이 영상")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--full", action="store_true", help="전체 구간 저장(수동 라벨링용)")
    p.add_argument("--out-fps", type=float, default=30.0)

    p = sub.add_parser("evaluate", help="수동 라벨과 비교")
    p.add_argument("labels")
    p.add_argument("results")
    p.add_argument("--tol", type=int, default=2, help="허용 오차(프레임)")
    p.add_argument("--rate", type=float, default=0.8, help="통과 비율")

    a = ap.parse_args(argv)

    if a.cmd in ("analyze", "extract"):
        from .pose_extract import extract

        extract(a.video, a.out, fps=a.fps, model=a.model, start=a.start, end=a.end)
    if a.cmd in ("analyze", "detect"):
        res = _detect(Path(a.out), a.arm, a.fps, a.cutoff)
        if a.cmd == "analyze" and not a.no_overlay:
            path = report.render_overlay(a.video, Path(a.out) / "landmarks.csv", res,
                                         Path(a.out) / "overlay.mp4")
            print(f"  오버레이: {path}")
    if a.cmd == "overlay":
        out = Path(a.out)
        res = None
        if not a.full and (out / "events.json").exists():
            res = json.loads((out / "events.json").read_text())
        name = "overlay_full.mp4" if a.full else "overlay.mp4"
        path = report.render_overlay(a.video, out / "landmarks.csv", res, out / name, out_fps=a.out_fps)
        print(f"오버레이: {path}")
    if a.cmd == "evaluate":
        evaluate.report(a.labels, a.results, a.tol, a.rate)


if __name__ == "__main__":
    main()
