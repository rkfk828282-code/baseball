# 0단계 기술 검증: 투구 영상 포즈 분석 (pitchlab)

240fps 투구 영상에 MediaPipe Pose를 적용해서 **앞발 착지(FC)·최대 외회전(MER)·릴리스(BR)** 를
자동으로 찾을 수 있는지 확인하는 Python 시제품입니다. 앱에 넣기 전에 PC에서 알고리즘을 검증하는 용도입니다.

## 설치 (Windows PC)

```powershell
cd baseball\analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Python 3.10 이상. 첫 실행 때 포즈 모델(heavy, 약 30MB)을 `analysis/models/`에 자동으로 받습니다.

## 촬영 가이드

| 항목 | 권장 |
|---|---|
| fps | 240fps 슬로모션 (갤럭시: 카메라 → 더보기 → 슬로우 모션) |
| 위치 | 투구 방향과 **수직인 옆쪽**, 3~5m 거리, 허리 높이 |
| 화면 | 레그킥부터 팔로스루까지 **발끝·손끝이 잘리지 않게** 전신이 들어오도록 |
| 폰 | 삼각대 고정, 수평 맞추기 (몸통 회전 계산이 카메라 수평을 기준으로 함) |
| 클립 | **영상 하나에 투구 1개** (자동 자르기는 다음 단계) |
| 기타 | 밝은 곳, 몸에 붙는 옷, 배경에 다른 사람 없이 |

> **fps 주의**: 갤럭시 슬로모션 파일은 재생용 fps(예: 30)로 저장될 수 있습니다.
> 240fps로 찍었다면 항상 `--fps 240`을 붙이세요. 안 붙이면 모든 시간(ms)이 틀립니다.

## 사용법

```powershell
# 1) 한 번에: 포즈 추출 → 이벤트 검출 → 그래프·오버레이 영상
python -m pitchlab analyze videos\pitch01.mp4 --arm R --fps 240 -o output\pitch01
```

`output\pitch01\` 에 생기는 파일:

| 파일 | 내용 |
|---|---|
| `landmarks.csv` | 프레임별 33개 관절 좌표 (픽셀·3D) |
| `events.json` | FC/MER/BR 프레임, 이벤트 시점 각도, 회전 순서, 품질, 경고 |
| `signals.png` | 손목 속력·앞발 높이·외회전 등 그래프 + 이벤트 표시 |
| `overlay.mp4` | 뼈대 + **프레임 번호** + 이벤트 표시 (30fps로 저장 → 8배 슬로모션) |
| `series.csv` | 각도·속도 시계열 |

단계별로 따로 실행할 수도 있습니다.

```powershell
python -m pitchlab extract videos\pitch01.mp4 --fps 240 -o output\pitch01
python -m pitchlab detect  output\pitch01 --arm R          # 파라미터 바꿔가며 재실행
python -m pitchlab overlay videos\pitch01.mp4 output\pitch01 --full   # 전체 구간 (라벨링용)
```

## 검출 방법

| 이벤트 | 방법 | 신뢰도 |
|---|---|---|
| **BR** 릴리스 | 투구 손목 속력(몸통 길이로 정규화)의 최대 프레임 | 2D로 충분 |
| **FC** 앞발 착지 | 레그킥 정점(앞무릎 최고점) 이후, 바닥 근처에서 앞발이 **가장 급격히 감속한 프레임**(착지 충격) | 2D로 충분 |
| **MER** 최대 외회전 | FC~BR 구간에서 3D 어깨 외회전 각도 최대 프레임 | **참고값** (깊이 추정 오차 큼) |

- 비교용으로 "바닥 높이 도달 + 거의 정지" 방식 결과도 `events.json`의 `debug.fc_stop_frame`에 남깁니다.
- 이벤트 시점 각도: 어깨 외전, 어깨 외회전, 팔꿈치 굴곡, 엉덩이-어깨 분리각.
- 키네틱 체인: 골반 → 몸통 → 팔꿈치 신전 각속도 최고점 순서(`order_ok`), 몸통 회전 시작 시점(FC 기준 ms).
- 외회전·몸통 회전은 MediaPipe의 3D 깊이 추정에 의존하므로 **측면 영상 하나로는 부정확할 수 있습니다.**
  이번 단계의 합격 판정은 FC·BR 타이밍으로만 합니다.

## 검증 절차 (0단계 합격 판정)

1. 투구 영상 10개 이상 촬영 → 각각 `analyze` 실행
2. `overlay_full.mp4`(또는 원본)를 한 프레임씩 넘기며 **직접** FC·BR 프레임 번호를 기록
   - FC: 앞발(뒤꿈치든 발끝이든)이 처음 바닥에 닿은 프레임
   - BR: 공이 손가락에서 떨어진 첫 프레임
   - MER(선택): 전완이 가장 뒤로 젖혀진 프레임
3. `labels.csv` 작성 (프레임 번호는 오버레이 왼쪽 위 숫자):
   ```csv
   clip,FC,MER,BR
   pitch01,812,850,861
   pitch02,640,,690
   ```
4. 비교:
   ```powershell
   python -m pitchlab evaluate labels.csv output
   ```
   → `output\evaluation.csv`, `output\evaluation_summary.json`

**합격 기준(제안)**: FC·BR 각각 클립의 **80% 이상에서 ±2프레임(240fps 기준 약 8ms) 이내**.
`--tol`, `--rate`로 바꿀 수 있습니다. 오차가 한쪽으로 치우치면(예: 항상 2프레임 늦음) 보정으로 해결 가능하다고 알려줍니다.

> 주의: 공은 포즈 추정 대상이 아니라 BR은 "손목 속력 최대" 시점으로 추정합니다.
> 실제 릴리스와 몇 프레임 차이가 일정하게 난다면 오프셋 보정을 넣으면 됩니다.

## 테스트

```powershell
python -m pytest -q
```

정답 이벤트를 알고 있는 합성 투구 데이터로 검출 알고리즘·각도 계산·CLI를 검사합니다.
좌표에 잡음(표준편차 2px)을 넣어도 FC·BR이 ±2프레임 안에 들어오는지도 확인합니다.
**실제 영상 검증은 아직 안 됨** → 영상이 생기면 위 검증 절차로 확인해야 합니다.

## 구조

```
analysis/
  pitchlab/
    pose_extract.py  영상 → landmarks.csv (MediaPipe Tasks PoseLandmarker)
    signals.py       결측 보간, 저역통과 필터(기본 15Hz), Track
    kinematics.py    어깨 외전·외회전, 팔꿈치 굴곡, 골반·몸통 회전
    events.py        FC / BR / MER 검출, 키네틱 체인
    report.py        events.json, 그래프, 오버레이 영상
    evaluate.py      수동 라벨 비교·합격 판정
  tests/             합성 데이터 테스트
```
