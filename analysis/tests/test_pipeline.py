import json

import numpy as np
import pytest

from pitchlab.__main__ import main
from pitchlab.events import analyze
from pitchlab.kinematics import shoulder_angles
from pitchlab.signals import Track, fill_gaps, load_track

from .synth import FPS, TRUTH, arm_world, make_df


def _static_track(frames: list[dict]) -> Track:
    names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    world = {n: np.zeros((len(frames), 3)) for n in names}
    for j, f in enumerate(frames):
        for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            world[n][j] = f[n]
        side = "right" if f["arm"] == "R" else "left"
        world[f"{side}_elbow"][j], world[f"{side}_wrist"][j] = f["E"], f["W"]
    return Track(fps=FPS, frames=np.arange(len(frames)), px={}, world=world, vis={},
                 missing=np.zeros(len(frames), bool))


@pytest.mark.parametrize("arm", ["R", "L"])
@pytest.mark.parametrize("yaw", [0, 37, -90])
def test_shoulder_angles_geometry(arm, yaw):
    psis = [30, 60, 90, 120, 150, 175]
    frames = [dict(arm_world(p, 90, 0, yaw, arm), arm=arm) for p in psis]
    ang = shoulder_angles(_static_track(frames), arm, ref_frame=0)  # psi=30: 손이 몸 앞
    np.testing.assert_allclose(ang["shoulder_er"], psis, atol=0.5)
    np.testing.assert_allclose(ang["shoulder_abd"], 90, atol=0.5)
    np.testing.assert_allclose(ang["elbow_flex"], 90, atol=0.5)


def test_fill_gaps_respects_max_gap():
    x = np.array([0, np.nan, 2, np.nan, np.nan, np.nan, 6, np.nan], float)
    out = fill_gaps(x, max_gap=1)
    assert out[1] == 1
    assert np.isnan(out[3:6]).all()
    assert out[7] == 6  # 끝은 최근접값


@pytest.mark.parametrize("arm", ["R", "L"])
def test_event_detection_on_synthetic_pitch(tmp_path, arm):
    csv = tmp_path / "landmarks.csv"
    make_df(arm).to_csv(csv, index=False)
    res = analyze(load_track(csv, fps=FPS), arm)
    ev = {k: v["frame"] for k, v in res["events"].items()}
    print(ev, res["debug"])
    assert abs(ev["BR"] - TRUTH["BR"]) <= 1
    assert abs(ev["FC"] - TRUTH["FC"]) <= 1
    assert abs(ev["MER"] - TRUTH["MER"]) <= 2
    assert res["angles"]["MER"]["shoulder_er"] == pytest.approx(175, abs=5)
    assert res["kinetic_chain"]["order_ok"] is True


def test_cli_detect_and_evaluate(tmp_path):
    clip = tmp_path / "pitch01"
    clip.mkdir()
    make_df("R").to_csv(clip / "landmarks.csv", index=False)
    (clip / "meta.json").write_text(json.dumps({"fps": FPS}))
    main(["detect", str(clip), "--arm", "R"])
    assert (clip / "events.json").exists() and (clip / "signals.png").exists()

    labels = tmp_path / "labels.csv"
    labels.write_text(f"clip,FC,MER,BR\npitch01,{TRUTH['FC']},,{TRUTH['BR']}\n")
    main(["evaluate", str(labels), str(tmp_path)])
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert summary["BR"]["within_1"] == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_detection_with_pixel_noise(tmp_path, seed):
    """실제 포즈 추정처럼 좌표에 잡음(표준편차 2px)이 있어도 FC·BR이 ±2프레임 이내."""
    rng = np.random.default_rng(seed)
    df = make_df("R")
    for c in df.columns:
        if c.endswith("_x") or c.endswith("_y"):
            df[c] += rng.normal(0, 2.0, len(df))
    csv = tmp_path / "landmarks.csv"
    df.to_csv(csv, index=False)
    res = analyze(load_track(csv, fps=FPS), "R")
    ev = {k: v["frame"] for k, v in res["events"].items()}
    print(seed, ev, res["debug"])
    assert abs(ev["BR"] - TRUTH["BR"]) <= 2
    assert abs(ev["FC"] - TRUTH["FC"]) <= 2
