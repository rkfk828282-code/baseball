"""MediaPipe BlazePose 33개 랜드마크 정의.

left/right는 '선수 본인 기준' 좌우다 (화면 기준 아님).
"""

NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
INDEX = {n: i for i, n in enumerate(NAMES)}

# 오버레이에 그릴 뼈대 연결
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]


def sides(arm: str) -> tuple[str, str]:
    """투구 팔(R/L) → (투구 쪽 접두어, 디딤발=앞발 쪽 접두어)."""
    arm = arm.upper()
    if arm not in ("R", "L"):
        raise ValueError("arm은 'R' 또는 'L'이어야 합니다")
    return ("right", "left") if arm == "R" else ("left", "right")
