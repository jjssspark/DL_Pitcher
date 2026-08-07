# 다음 세션 착수 문서 — CV 구종 분류 파일럿

마지막 갱신: 2026-08-07 (3차 세션) · 관련 ADR: ADR-0009, ADR-0010, ADR-0011

---

## 붙여넣을 프롬프트

```
CV 구종 분류 파일럿을 이어서 한다. docs/NEXT_SESSION.md를 먼저 읽고,
docs/TROUBLESHOOTING.md의 TS-023(개선 지표가 다른 클래스를 가리킴)을 확인해라.

지난 세션 결론: 좌표 기반 특징 7개를 추가해 전체 정확도가 0.587 -> 0.673으로,
경기 단위 leave-one-game-out 4폴드로도 0.613 -> 0.659로 올랐다. 그런데 겨냥했던
OFFSPEED는 f1 0.14 -> 0.07로 나빠졌다. one-vs-rest AUC가 0.641 -> 0.648로
제자리라, 좌표 특징이 OFFSPEED에 정보를 하나도 안 넣었다는 뜻이다.
픽셀 좌표로 절대 속도를 복원할 수 없다는 구조적 한계가 실측으로 확인됐다.

그래서 박스 크기(속도 대리 지표)를 담는 v2 캐시로 재수집했다(ADR-0011).
먼저 output/pitch_type_cv/clip_trajectory_cache_v2/ 가 4경기 1193건 다 찼는지
확인하고, 안 찼으면 scripts/build_pitch_group_clips_dataset.py를 다시 띄워라.

오늘 할 일은 박스 특징의 정식 판정이다. 판정은 정확도가 아니라
**OFFSPEED one-vs-rest AUC**로 한다 — 정확도와 f1은 클래스 가중치로도 움직여서
'정보가 늘었나'와 '동작점을 옮겼나'가 섞인다(TS-023).

주의: 조기 신호가 나쁘다. 813026 부분 캐시 69건에서 box_growth_per_frame의
단변량 AUC가 0.540(무신호)이었다. 중계 카메라가 중견수 뒤라 공이 포수 쪽으로
멀어져 박스가 커지는 게 아니라 작아지고, 변화량이 10프레임에 0.2~0.7px라
16px 박스의 정수 양자화 노이즈에 묻힌다.

박스가 죽으면 대비책은 2분류 재정의다. 이미 재봤고 수치가 있다 —
FASTBALL vs BREAKING만 두면 LOGO 정확도 0.775, 기준선 대비 +0.232, AUC 0.874다.

플랜 먼저 세우고 확인받은 뒤 실행해라.
```

---

## 지금 상태

| 항목 | 값 | 비고 |
|---|---|---|
| 데이터 | 4경기 1193투구, 궤적 1030개 | `output/pitch_type_cv/dataset_clips.csv` |
| 특징 | **16개** | 기하 7 + 시간 3 + 기하확장 4 + 박스 2 |
| 홀드아웃 정확도 | **0.673** vs 최빈값 0.483 (+0.190) | game_pk=813027, 300샘플 |
| LOGO 4폴드 | **0.659** vs 기준선 0.449 (+0.210) | 경기별 0.66/0.64/0.65/0.67 |
| 특징 중요도비 | 21.96 | TS-014 무신호 당시 1.23 |
| 그룹별 f1 | BREAKING 0.69 / FASTBALL 0.78 / **OFFSPEED 0.07** | |
| OFFSPEED AUC | **0.648** (기존 7개일 때 0.641) | 제자리 — 이번 세션의 핵심 결과 |
| 2분류 대비책 | 정확도 0.775, AUC 0.874 | FASTBALL vs BREAKING, LOGO 4폴드 |

### 이번 세션에서 확인된 것

- **좌표 특징 확장은 전체 정확도에 확실히 기여한다.** 홀드아웃 1경기가 아니라 LOGO
  4폴드에서 경기별로 일관되게 올랐다. `vertical_accel_px`(중요도 0.42) 하나가 끌었다
- **그 기여는 FASTBALL·BREAKING에서만 왔다.** AUC로 보면 명확하다

  ```
                  기준(7)  ->  전체(14)
  FASTBALL AUC     0.792   ->   0.851
  BREAKING AUC     0.725   ->   0.819
  OFFSPEED AUC     0.641   ->   0.648   <- 제자리
  ```

- **시간 계열 3개(`frame_span`, `end_frame`, `speed_ratio_late_early`)는 헛방이었다.**
  `+시간(10)` 집합은 0.587 -> 0.600으로 거의 안 움직였다. 사슬이 버리던 frame_idx를
  되살린 것 자체는 옳았지만(감지 누락 프레임을 세게 됐다), 구종을 가르지는 못했다
- **클래스 가중치로 OFFSPEED f1 0.35는 만들 수 있다. 다만 새 정보가 아니다.**
  가중치 8에서 f1 0.35 / recall 0.57이지만 전체 정확도가 0.673 -> 0.587로 정확히
  원위치한다. AUC가 안 움직였으므로 같은 ROC 곡선 위의 동작점 이동이다

### 진행 중 / 확인 필요

- **v2 재수집** (2026-08-07 09:20 시작). `clip_trajectory_cache_v2/`에 4경기 1193건.
  병렬 6연결로 약 7.2초/클립, 총 약 2.4시간. 다시 띄우면 캐시 히트분은 건너뛴다
- **박스 특징 정식 판정 미완.** 조기 신호(69건, 홀드아웃 아님)는 나쁘다:
  `box_growth_per_frame` 단변량 AUC 0.540, `release_box_size` 0.614

## 수집 파이프라인 (경로 B, 확정)

```
라벨      https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
          liveData.plays.allPlays[].playEvents[] 에서 isPitch=True
          -> playId + details.type.code (Statcast 코드). 판독률 100%

클립      https://baseballsavant.mlb.com/sporty-videos?playId={playId}
          페이지의 .mp4 링크를 html.unescape 후 GET
          1280x720, 약 59.6fps, 4~5MB, 길이 6~15초로 가변

윈도우    클립 시작 기준 2.8 ~ 4.2초 고정  <- 바꾸지 말 것. 근거는 ADR-0010
감지      models/ball_broadcast_v1.pt, imgsz=960, conf=0.05
궤적      longest_moving_chain_frames(candidates, max_jump_px=60, min_total_move_px=30)
캐시      clip_trajectory_cache_v2/  (x, y, conf, w, h)   <- v1은 (x, y, conf)
```

**긴 작업은 하네스 백그라운드로 띄우지 말 것** (TS-016 재발).
```bash
nohup venv/bin/python3 -c "
import os, runpy
os.setsid()
runpy.run_path('scripts/build_pitch_group_clips_dataset.py', run_name='__main__')
" > output/pitch_type_cv/build_clips_v2.log 2>&1 < /dev/null & disown
```

**Savant는 연결당 대역폭을 제한한다.** 2026-08-07 실측 단일 연결 0.1MB/s(클립당 30초),
6연결 0.52MB/s(클립당 7.2초). 기본값이 `--download-workers 6`이다. 하루 전 같은 4경기가
100분이었으니 조건이 날마다 다르다 — 시작하고 5분 뒤 캐시 증가 속도를 재서 예상 시간을 잡아라.

## 다음 갈래

### (A) 박스 특징 정식 판정 — 먼저 할 것

v2 캐시가 다 차면 데이터셋을 재생성하고(다운로드 0, 캐시만) 판정한다.

```bash
venv/bin/python3 scripts/build_pitch_group_clips_dataset.py   # 캐시 히트, 수 분
venv/bin/python3 scripts/eval_feature_sets.py
```

판정은 **OFFSPEED one-vs-rest AUC**로 한다. 0.648에서 유의하게 오르지 않으면 실패다.
정확도와 f1은 클래스 가중치로도 움직이므로 단독 근거가 될 수 없다(TS-023).

`scripts/eval_feature_sets.py`의 `FEATURE_SETS`에 박스 포함/미포함 집합을 넣어 비교한다.

### (B) 2분류로 목표 재정의 — (A)가 실패하면

FASTBALL vs BREAKING만 남기면 이미 쓸 만하다. **이미 측정했다.**

```
2분류 LOGO 정확도 0.775  (경기별 0.73 / 0.77 / 0.83 / 0.77)
      최빈값 기준선 0.543 -> +0.232
      AUC 0.874
대조: 3분류 0.659, 기준선 0.449 -> +0.210
```

OFFSPEED 190구(전체의 16%)를 버리는 대신 나머지 858구에서 신뢰할 수 있는 예측을 준다.
"중계 영상만으로 되는 것과 안 되는 것"의 경계를 명시하는 결론이라 오히려 정직하다.

### (C) 경기를 더 늘린다

경로 B는 임의 game_pk를 쓸 수 있다. 경기당 약 30분(6연결 기준).
단, OFFSPEED AUC가 특징 추가에 반응하지 않는 상태에서 표본만 늘리면 같은 결과다.
(A) 판정 후에 결정한다.

### (D) 확보율 편차의 원인 규명 — 보류 중

FASTBALL 궤적 확보율이 경기별 71~92%로 21%p 흔들린다. "빠른 공이라 끊긴다"는 단일
설명으로는 안 맞는다(813027에서는 FASTBALL 92.4%로 BREAKING을 넘는다).
분류 성능에 직접 기여하지 않아 계속 뒤로 밀리고 있다.

## 재사용 가능한 것

- `output/pitch_type_cv/clip_trajectory_cache_v2/*.jsonl` — 박스 크기 포함 감지 후보 전체.
  사슬 임계값·특징 정의를 바꿔도 영상 재처리 없이 재계산 가능
- `output/pitch_type_cv/clip_trajectory_cache/*.jsonl` — v1(좌표만). 지우지 말 것,
  좌표 기반 재실험 시 재취득 2시간을 아낀다
- `scripts/eval_feature_sets.py` — 특징 집합 절제 실험. 정확도·중요도비·그룹별
  precision/recall/f1을 함께 찍는다. `--permutations N`으로 순열검정
- `output/pitch_type_cv/feature_ablation.csv` / `.png` — 4개 집합 비교 결과
- `scripts/calibrate_clip_window.py` — 새 시즌·다른 중계사 클립의 윈도우 재확인용

## 판정 규칙 (바꾸지 말 것)

- **기준선은 최빈값이다.** 33% 랜덤 아님. 불균형 때문에 아무것도 학습하지 않아도 48%가 나온다
- **소수 클래스를 겨냥한 작업은 전체 정확도로 판정하지 않는다.** 대상 클래스의
  one-vs-rest AUC를 본다. AUC는 임계값·클래스 가중치와 무관해서 "정보가 늘었나"와
  "동작점을 옮겼나"를 갈라준다 (TS-023에서 이 구분이 유일한 단서였다)
- **정확도만 보지 않는다.** 특징 중요도 최대/최소 비가 1에 가까우면 무신호다 (TS-014)
- **비율 지표는 분자와 분모가 각각 무엇을 세는지 확인한다.** 요약 통계에 중앙값만
  찍으면 못 본다 — 분포의 꼬리를 본다 (TS-021)
- **게이트 통과 후에도 궤적 좌표를 눈으로 본다** (TS-019)

## 건드리면 안 되는 것

- `runs/`, `models/`, `*.pt` — YOLO 자동 생성 / 재학습 비용이 크다
- `src/pose_detector.py`, `src/feature_engineering.py`, `src/yolo_detector.py` — 기존 데모
  파이프라인. 수정은 `src/pitch_type_cv/` 안에서만
- `src/pitch_type_cv/dataset.py`, `scripts/build_pitch_group_dataset.py` — 구 OCR 경로.
  경로 B와 병렬로 남겨둔 것이므로 삭제하지 않는다 (ADR-0010)
- 스캔 윈도우 2.8~4.2초, 감지 설정 imgsz 960 / conf 0.05 (ADR-0010)
