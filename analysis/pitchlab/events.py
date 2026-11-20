"""투구 이벤트 자동 검출: 앞발 착지(FC), 최대 외회전(MER), 릴리스(BR).

전제: 클립 하나에 투구 1개. 측면 촬영(투구 방향과 수직).

검출 순서
  1) BR  : 투구 손목 속력(몸통 길이로 정규화)의 최댓값 프레임
  2) FC  : BR 이전, 앞다리 무릎이 가장 높았던 시점(레그킥 정점) 이후
           앞발이 바닥 근처에서 가장 급격히 감속한 프레임(착지 충격).
           비교용으로 '바닥 높이 도달 + 거의 정지' 방식(fc_stop)도 함께 기록
  3) MER : FC~BR 구간에서 어깨 외회전 각도 최댓값 (3D 추정, 참고값)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .kinematics import rotation_angles, shoulder_angles
from .landmarks import sides
from .signals import Track, speed


@dataclass
class DetectParams:
    # BR
    br_search_from_s: float = 0.0          # 클립 앞부분 잡음 제외용
    # FC
    knee_peak_window_s: float = 2.0        # BR 이전 몇 초 안에서 레그킥 정점을 찾을지
    ground_window_s: float = 0.03          # BR 주변 앞발 높이 = 바닥 높이
    ground_tol_torso: float = 0.08         # 바닥 높이 허용 오차 (몸통 길이 배수)
    foot_speed_frac: float = 0.15          # 보폭 중 최대 발 속력 대비 '멈춤' 기준
    hold_s: float = 0.015                  # 멈춤 조건이 유지되어야 하는 시간
    fc_method: str = "impact"              # impact(착지 충격) | stop(정지)
    impact_zone_torso: float = 0.25        # 충격 탐색: 바닥에서 이 높이 이내
    # 키네틱 체인
    onset_frac: float = 0.2                # 몸통 회전 시작 = 최대 각속도의 20% 도달


def torso_length(tr: Track) -> float:
    sh = tr.mid("left_shoulder", "right_shoulder")
    hip = tr.mid("left_hip", "right_hip")
    return float(np.nanmedian(np.linalg.norm(sh - hip, axis=1)))


def detect_br(tr: Track, arm: str, p: DetectParams) -> tuple[int, np.ndarray]:
    throw, _ = sides(arm)
    v = speed(tr.px[f"{throw}_wrist"], tr.fps) / torso_length(tr)
    start = int(p.br_search_from_s * tr.fps)
    if np.all(np.isnan(v[start:])):
        raise ValueError("투구 손목이 검출되지 않았습니다")
    return start + int(np.nanargmax(v[start:])), v


def lead_foot(tr: Track, arm: str) -> np.ndarray:
    _, lead = sides(arm)
    return np.nanmean(
        np.stack([tr.px[f"{lead}_{k}"] for k in ("ankle", "heel", "foot_index")]), axis=0
    )


def detect_fc(tr: Track, arm: str, br: int, p: DetectParams) -> tuple[int | None, dict]:
    _, lead = sides(arm)
    fps = tr.fps
    torso = torso_length(tr)
    foot = lead_foot(tr, arm)
    foot_y = foot[:, 1]
    foot_v = speed(foot, fps) / torso
    knee_y = tr.px[f"{lead}_knee"][:, 1]

    start = max(0, br - int(p.knee_peak_window_s * fps))
    if br - start < 3 or np.all(np.isnan(knee_y[start:br])):
        return None, {}
    knee_peak = start + int(np.nanargmin(knee_y[start:br]))  # y가 작을수록 높음

    g = max(1, int(p.ground_window_s * fps))
    ground_y = float(np.nanmedian(foot_y[max(0, br - g): br + g + 1]))
    tol = p.ground_tol_torso * torso
    seg_v = foot_v[knee_peak: br + 1]
    thr = p.foot_speed_frac * float(np.nanmax(seg_v)) if np.any(~np.isnan(seg_v)) else np.inf

    # (A) 착지 충격: 앞발이 바닥 근처에서 아래 방향 속도를 가장 급격히 잃는 프레임
    #     = 수직 가속도(y 아래 +)의 최솟값. 영위상 필터라 시점 이동이 거의 없다.
    vy = np.gradient(foot_y) * fps
    ay = np.gradient(vy) * fps
    near = foot_y > ground_y - p.impact_zone_torso * torso
    idx = np.arange(knee_peak, br + 1)
    idx = idx[near[idx] & ~np.isnan(ay[idx])]
    fc_impact = int(idx[np.argmin(ay[idx])]) if len(idx) else None

    # (B) 정지: 바닥 높이에 도달하고 거의 멈춘 첫 프레임 (보조·비교용)
    cond = (foot_v < thr) & (foot_y > ground_y - tol)
    hold = max(2, int(round(p.hold_s * fps)))
    fc_stop = None
    for i in range(knee_peak, br + 1):
        if cond[i: i + hold].all():
            fc_stop = i
            break
    fc = fc_impact if p.fc_method == "impact" and fc_impact is not None else fc_stop
    info = {
        "knee_peak": knee_peak,
        "fc_impact": fc_impact,
        "fc_stop": fc_stop,
        "ground_y_px": ground_y,
        "foot_speed_thr": thr,
        "foot_y": foot_y,
        "foot_speed": foot_v,
    }
    return fc, info


def _peak_time(sig: np.ndarray, lo: int, hi: int) -> int | None:
    seg = sig[lo:hi]
    if len(seg) == 0 or np.all(np.isnan(seg)):
        return None
    return lo + int(np.nanargmax(seg))


def kinetic_chain(tr: Track, arm: str, fc: int | None, br: int, p: DetectParams) -> dict:
    """골반 → 몸통 → 팔(팔꿈치 신전) 각속도 최고점 순서와 몸통 회전 시작 시점."""
    fps = tr.fps
    rot = rotation_angles(tr)
    ang = shoulder_angles(tr, arm, ref_frame=br)
    w_pel = np.gradient(rot["pelvis_rot"]) * fps
    w_trk = np.gradient(rot["trunk_rot"]) * fps
    w_elb = -np.gradient(ang["elbow_flex"]) * fps  # 신전 방향을 +로

    lo = max(0, (fc if fc is not None else br) - int(0.25 * fps))
    hi = min(tr.n, br + int(0.05 * fps) + 1)
    # 회전 방향 부호: FC~BR 동안 몸통이 돈 방향
    direction = np.sign(np.nansum(w_trk[lo:hi])) or 1.0
    w_pel, w_trk = w_pel * direction, w_trk * direction

    pk_p = _peak_time(w_pel, lo, hi)
    pk_t = _peak_time(w_trk, lo, hi)
    pk_e = _peak_time(w_elb, lo, hi)

    onset = None
    if pk_t is not None:
        thr = p.onset_frac * w_trk[pk_t]
        i = pk_t
        while i > 0 and w_trk[i - 1] >= thr:
            i -= 1
        onset = i

    def ms(frame):
        return None if frame is None else round(frame / fps * 1000, 1)

    def fr(i):
        return None if i is None else int(tr.frames[i])

    order_ok = None
    if None not in (pk_p, pk_t, pk_e):
        order_ok = bool(pk_p <= pk_t <= pk_e)
    return {
        "pelvis_peak_frame": fr(pk_p),
        "trunk_peak_frame": fr(pk_t),
        "elbow_ext_peak_frame": fr(pk_e),
        "order_ok": order_ok,
        "trunk_rot_onset_frame": fr(onset),
        "trunk_rot_onset_ms_from_fc": (
            None if onset is None or fc is None else round((onset - fc) / fps * 1000, 1)
        ),
        "pelvis_peak_ms_from_br": None if pk_p is None else ms(pk_p - br),
        "trunk_peak_ms_from_br": None if pk_t is None else ms(pk_t - br),
        "elbow_ext_peak_ms_from_br": None if pk_e is None else ms(pk_e - br),
        "series": {"w_pelvis": w_pel, "w_trunk": w_trk, "w_elbow_ext": w_elb},
    }


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def analyze(tr: Track, arm: str, p: DetectParams | None = None) -> dict:
    """이벤트 검출 + 이벤트 시점 각도. 결과는 JSON 직렬화 가능한 dict(+series)."""
    p = p or DetectParams()
    warnings: list[str] = []
    throw, lead = sides(arm)

    br, wrist_v = detect_br(tr, arm, p)
    fc, fc_info = detect_fc(tr, arm, br, p)
    if fc is None:
        warnings.append("FC를 찾지 못했습니다 (앞발이 가려졌거나 클립이 짧음)")

    ang = shoulder_angles(tr, arm, ref_frame=br)
    rot = rotation_angles(tr)
    mer = None
    if fc is not None and br > fc:
        seg = ang["shoulder_er"][fc: br + 1]
        if np.any(~np.isnan(seg)):
            mer = fc + int(np.nanargmax(seg))
    if mer is None:
        warnings.append("MER을 계산하지 못했습니다")

    chain = kinetic_chain(tr, arm, fc, br, p)

    # 품질 지표
    lo = max(0, (fc if fc is not None else br) - int(0.1 * tr.fps))
    hi = min(tr.n, br + int(0.05 * tr.fps) + 1)
    arm_vis = np.nanmean(
        [np.nanmean(tr.vis[f"{throw}_{k}"][lo:hi]) for k in ("shoulder", "elbow", "wrist")]
    )
    foot_vis = np.nanmean(
        [np.nanmean(tr.vis[f"{lead}_{k}"][lo:hi]) for k in ("ankle", "heel", "foot_index")]
    )
    if arm_vis < 0.6:
        warnings.append(f"투구 팔 가시성 낮음 ({arm_vis:.2f}) - 카메라 위치 확인")
    if foot_vis < 0.6:
        warnings.append(f"앞발 가시성 낮음 ({foot_vis:.2f}) - 발이 화면에 다 보이는지 확인")
    if tr.fps < 120:
        warnings.append(f"fps가 {tr.fps:.0f}로 낮습니다. 릴리스 시점 오차가 커집니다")

    def ev(i):
        if i is None:
            return None
        return {"index": int(i), "frame": int(tr.frames[i]), "t": round(float(tr.t[i]), 4)}

    events = {"FC": ev(fc), "MER": ev(mer), "BR": ev(br)}
    angles = {}
    for name, i in (("FC", fc), ("MER", mer), ("BR", br)):
        if i is None:
            continue
        angles[name] = {
            "shoulder_abd": _r(ang["shoulder_abd"][i]),
            "shoulder_er": _r(ang["shoulder_er"][i]),
            "elbow_flex": _r(ang["elbow_flex"][i]),
            "hip_shoulder_sep": _r(rot["hip_shoulder_sep"][i]),
        }

    series = {
        "t": tr.t,
        "wrist_speed": wrist_v,
        "foot_y": fc_info.get("foot_y"),
        "foot_speed": fc_info.get("foot_speed"),
        **ang,
        **rot,
        **chain.pop("series"),
    }
    return {
        "fps": tr.fps,
        "arm": arm.upper(),
        "events": events,
        "fc_to_br_ms": None if fc is None else round((br - fc) / tr.fps * 1000, 1),
        "angles": angles,
        "kinetic_chain": chain,
        "debug": {
            "knee_peak_frame": None if not fc_info else int(tr.frames[fc_info["knee_peak"]]),
            "fc_impact_frame": None if fc_info.get("fc_impact") is None else int(tr.frames[fc_info["fc_impact"]]),
            "fc_stop_frame": None if fc_info.get("fc_stop") is None else int(tr.frames[fc_info["fc_stop"]]),
            "foot_speed_thr": _r(fc_info.get("foot_speed_thr")),
        },
        "quality": {
            "throwing_arm_visibility": _r(arm_vis * 100),
            "lead_foot_visibility": _r(foot_vis * 100),
            "missing_frames_pct": _r(tr.missing.mean() * 100),
        },
        "params": {**asdict(p), **tr.params},
        "warnings": warnings,
        "_series": series,
    }
