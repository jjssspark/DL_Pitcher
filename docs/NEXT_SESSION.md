# 다음 세션 착수 문서 — CV 구종 분류 파일럿

마지막 갱신: 2026-08-06 (2차 세션) · 관련 ADR: ADR-0009, ADR-0010

---

## 붙여넣을 프롬프트

```
CV 구종 분류 파일럿을 이어서 한다. docs/NEXT_SESSION.md를 먼저 읽고,
docs/TROUBLESHOOTING.md의 TS-020(구종 매핑 구멍)과 TS-021(지표가 다른 대상을 셈)을
확인해라.

지난 세션 결론: 데이터 수집을 중계 OCR에서 MLB StatsAPI + Baseball Savant 투구별
클립으로 전환했다(ADR-0010). 4경기 1193투구, 궤적 확보 1030개.
game_pk 단위 홀드아웃이 성립했고 누수를 제거해도 성능 차이가 유지됐다
(+10.2%p -> +10.3%p, 순열검정 p < 0.001).

남은 문제는 두 개다:
  (1) OFFSPEED f1 0.14 - 표본은 4개에서 40개로 늘었는데 recall이 0.10이다
  (2) FASTBALL 궤적 확보율 80.9%로 여전히 최하위. 다만 경기별 편차가 21%p라
      "빠른 공이라 끊긴다"는 단일 설명으로는 안 맞는다

오늘 할 일을 정하기 전에 아래 "다음 갈래"를 읽고 나에게 추천안을 말해라.
플랜 먼저 세우고 확인받은 뒤 실행해라.
```

---

## 지금 상태

| 항목 | 값 | 비고 |
|---|---|---|
| 데이터 | 4경기 1193투구, 궤적 1030개 | `output/pitch_type_cv/dataset_clips.csv` |
| 홀드아웃 | 813027, 300샘플 (OFFSPEED 40) | game_pk 단위, 누수 없음 |
| 정확도 | **0.587** vs 최빈값 기준선 0.483 (+10.3%p) | |
| 순열검정 | **p < 0.001** (셔플 1000회 중 0회) | 셔플 평균 0.419 |
| 특징 중요도 비 | 3.75 | TS-014 무신호 당시 1.23 |
| 그룹별 f1 | BREAKING 0.62 / FASTBALL 0.65 / **OFFSPEED 0.14** | |

### 이번 세션에서 확인된 것

- **누수를 제거해도 성능 차이가 유지됐다.** 1경기 시절 +10.2%p는 투구 단위 분할이라
  누수 허용 상한이었는데, 진짜 game_pk 홀드아웃에서 +10.3%p가 나왔다. 이전 수치가
  부풀려진 것이 아니었다
- **위상 정렬 효과가 특징 순위에 나타났다.** OCR 타임스탬프 시절 1위였던
  `apparent_speed_px_per_frame`(0.301)이 4위(0.128)로 내려가고, 물리적으로 구종을 가르는
  `horizontal_deviation_px`(0.261)가 1위가 됐다. 속도 특징이 원근 압축으로 체계적 왜곡을
  받고 있었다는 가설과 일치한다

  ```
  horizontal_deviation_px        0.261
  vertical_drop_px               0.190
  duration_frames                0.153
  apparent_speed_px_per_frame    0.128
  curvature_ratio                0.127
  path_length_px                 0.073
  straight_line_px               0.070
  ```

## 수집 파이프라인 (경로 B, 확정)

```
라벨      https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
          liveData.plays.allPlays[].playEvents[] 에서 isPitch=True
          -> playId + details.type.code (Statcast 코드). 판독률 100%

클립      https://baseballsavant.mlb.com/sporty-videos?playId={playId}
          페이지의 .mp4 링크를 html.unescape 후 GET. Referer 불필요(문서의 403 주장은 오류)
          1280x720, 약 59.6fps, 4~5MB, 길이 6~15초로 가변

윈도우    클립 시작 기준 2.8 ~ 4.2초 고정  <- 바꾸지 말 것. 근거는 ADR-0010
감지      models/ball_broadcast_v1.pt, imgsz=960, conf=0.05  <- OCR 경로와 동일
궤적      longest_moving_chain(candidates, max_jump_px=60, min_total_move_px=30)
```

실행:
```bash
venv/bin/python3 scripts/build_pitch_group_clips_dataset.py
```

**긴 작업은 하네스 백그라운드로 띄우지 말 것** (TS-016 재발). 1193투구 빌드는 약 100분이다.
```bash
nohup venv/bin/python3 -c "
import os, runpy
os.setsid()
runpy.run_path('scripts/build_pitch_group_clips_dataset.py', run_name='__main__')
" > output/pitch_type_cv/build_clips.log 2>&1 < /dev/null & disown
```

## 남은 문제 2개

### (1) OFFSPEED f1 0.14 — 표본을 늘렸는데도 낮다

```
              precision    recall  f1-score   support
    BREAKING       0.51      0.78      0.62       115
    FASTBALL       0.77      0.57      0.65       145
    OFFSPEED       0.24      0.10      0.14        40
```

표본 부족(4개 -> 40개)은 해결됐는데 recall이 0.10이다. 40개 중 4개만 맞힌다.
**표본 문제가 아니라 특징 문제일 가능성이 크다.** 체인지업·스플리터는 패스트볼과 궤적이
비슷하고(속도만 다름), 픽셀 좌표만으로는 절대 속도를 알 수 없다.

### (2) FASTBALL 확보율 80.9% — 경기별 편차가 21%p

```
group    FASTBALL  BREAKING  OFFSPEED
775294      77.6%     88.7%     95.5%
813024      71.4%     92.9%     88.9%
813026      82.2%    100.0%     84.6%
813027      92.4%     88.5%    100.0%   <- FASTBALL이 BREAKING을 넘는다
```

평균으로는 FASTBALL이 최하위지만 813027에서는 역전된다. "빠른 공이라 프레임 간 이동이
커서 사슬이 끊긴다"는 설명이 맞다면 경기와 무관하게 일정해야 한다. 중계 카메라 위치·
화질 같은 경기별 조건이 섞여 있다.

## 다음 갈래

### (A) 특징을 늘린다 — OFFSPEED를 겨냥

현재 7개 특징은 전부 궤적의 기하학이고, 구종을 가르는 핵심인 **속도**가 픽셀 단위라
원근에 오염돼 있다. 후보:
- 릴리스 포인트 좌표 (클립 내 위치가 고정이라 이제 비교 가능해졌다)
- 궤적 후반부/전반부의 곡률 비 (체인지업은 후반 낙차가 크다)
- 박스 크기 변화율 (공이 카메라에 가까워지는 속도 = 실제 속도의 대리 지표)

장점: 데이터 재수집이 필요 없다. `trajectory_cache`에 감지 후보 전체가 남아 있어
박스 크기까지 재계산 가능하다. 단점: 특징 추가가 성능으로 이어진다는 보장이 없다.

### (B) 경기를 더 늘린다

경로 B는 임의 game_pk를 쓸 수 있어 확장이 쉽다. 경기당 약 100분.
장점: 홀드아웃을 2경기로 늘려 결과 안정성이 오른다. 단점: **OFFSPEED recall 0.10은
표본이 10배 늘어도 그대로일 수 있다.** (1)의 진단이 맞다면 헛수고다.

### (C) 확보율 편차의 원인을 규명한다

경기별로 FASTBALL 확보율이 71~92%로 흔들리는 이유를 먼저 밝힌다. 중계 카메라 앵글,
해상도, 조명 중 무엇인지. 장점: 감지 파이프라인 개선의 방향이 나온다.
단점: 분류 성능에 직접 기여하지 않는다.

**추천: (A) -> (C) -> (B) 순.** (B)는 (1)의 원인을 모르는 상태에서 돌리면 100분 × N을
쓰고 같은 f1을 볼 위험이 크다. (A)는 재수집 없이 캐시만으로 검증 가능해 가장 싸다.

## 재사용 가능한 것

- `output/pitch_type_cv/clip_trajectory_cache/*.jsonl` — 4경기 1193투구의 **프레임별 감지
  후보 전체**. 사슬 임계값이나 특징 정의를 바꿔도 영상 재처리 없이 재계산 가능
- `output/pitch_type_cv/clip_window_calibration.csv` — 60투구의 사슬 시작 시각 실측
- `scripts/calibrate_clip_window.py` — 새 시즌·다른 중계사 클립을 넣을 때 윈도우 재확인용
- `output/pitch_type_cv/ocr_cache/` — 구 OCR 경로 산출물. 경로 B로 전환했으므로 참고용

## 판정 규칙 (바꾸지 말 것)

- **기준선은 최빈값이다.** 33% 랜덤 아님. 클래스 불균형 때문에 아무것도 학습하지 않아도
  48%가 나온다
- **정확도만 보지 않는다.** 특징 중요도 최대/최소 비가 1에 가까우면 정확도와 무관하게
  무신호다. TS-014에서 이 지표만이 유일한 단서였다
- **비율 지표는 분자와 분모가 각각 무엇을 세는지 확인한다.** TS-021에서 "확보율 93.3%"가
  타구 비행까지 세고 있었다. 요약 통계에 중앙값만 찍으면 못 본다 — 분포의 꼬리를 본다
- **게이트 통과 후에도 궤적 좌표를 눈으로 본다** (TS-019)

## 건드리면 안 되는 것

- `runs/`, `models/`, `*.pt` — YOLO 자동 생성 / 재학습 비용이 크다
- `src/pose_detector.py`, `src/feature_engineering.py`, `src/yolo_detector.py` — 기존 데모
  파이프라인. 수정은 `src/pitch_type_cv/` 안에서만
- `src/pitch_type_cv/dataset.py`, `scripts/build_pitch_group_dataset.py` — 구 OCR 경로.
  경로 B와 병렬로 남겨둔 것이므로 삭제하지 않는다 (ADR-0010)
