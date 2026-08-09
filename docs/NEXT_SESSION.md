# 다음 세션 착수 문서 — CV 구종 분류 파일럿

마지막 갱신: 2026-08-07 (4차 세션) · 관련 ADR: ADR-0009, ADR-0010, ADR-0011, ADR-0012

---

## 붙여넣을 프롬프트

```
CV 구종 분류 파일럿을 이어서 한다. docs/NEXT_SESSION.md를 먼저 읽고,
docs/TROUBLESHOOTING.md의 TS-024(속도 대리 지표의 전제가 이 카메라에서 안 성립함)와
docs/ADR.md의 ADR-0012(2분류 축소 결정)를 확인해라.

지난 세션 결론: 3분류를 접었다. OFFSPEED(체인지업·스플리터)를 두 갈래로 공략했고
둘 다 실패했다 — 좌표 기반 특징 7개(재수집 불필요), 박스 크기 특징 2개(전량 재수집
100분). 특징을 7개에서 16개로 늘리는 동안 OFFSPEED one-vs-rest AUC는 0.710 -> 0.728로
0.02 안에서만 움직였다. 같은 기간 FASTBALL 0.778 -> 0.850, BREAKING 0.715 -> 0.822다.
정보가 들어간 곳과 안 들어간 곳이 명확히 갈렸다.

원인은 물리다. 중견수 뒤 중계 카메라에서 공은 카메라로부터 멀어지므로 박스가 커지지
않고 작아진다(실측 -0.148 px/frame). 10프레임 누적 1.5px인데 박스 자체가 17px이고
정수 양자화라, 재려는 신호가 측정 단위보다 작았다.

그래서 FASTBALL vs BREAKING 2분류로 재정의했다. LOGO 4폴드 정확도 0.783,
최빈값 기준선 0.543 대비 +0.240, 평균 AUC 0.881. 3분류(0.665, +0.216)보다 낫고
경기별 편차도 작다.

그 뒤 앱에 붙였고, 앱에서는 안 된다는 것이 확인됐다(TS-025). 판정률은 고쳤지만
(32% -> 74%) 정확도가 41.0%로 기준선 54.7% 아래다. 배선 버그는 아니다 — 캐시된
학습 궤적을 같은 경로로 통과시키면 0.942가 나온다. 남은 것은 도메인 격차다.
클립은 투구가 항상 같은 위치에 오도록 잘려 있는데 원본 영상은 그렇지 않다.

오늘 할 일은 아래 (A-2)~(C) 중에서 고른다. 기본 추천은 (A-2), 사슬이 투구인지
판별하는 게이트다 — TS-025에 남은 문제가 그것 하나로 좁혀져 있다.

판정 규칙은 바꾸지 마라. 특히 소수 클래스를 겨냥한 작업은 전체 정확도가 아니라
대상 클래스의 one-vs-rest AUC로 판정한다(TS-023, TS-024). 그리고 실험실 수치가
배포 환경에서 재현된다고 가정하지 마라 — 이번에 78.3%가 41.0%가 됐다(TS-025).

플랜 먼저 세우고 확인받은 뒤 실행해라.
```

---

## 지금 상태

| 항목 | 값 | 비고 |
|---|---|---|
| 데이터 | 4경기 1193투구, 궤적 1029개 (86.3%) | `output/pitch_type_cv/dataset_clips.csv` |
| 특징 | **16개** | 기하 7 + 시간 3 + 기하확장 4 + 박스 2 |
| **채택 모델** | **2분류 FASTBALL vs BREAKING** | ADR-0012 |
| 2분류 LOGO | **0.783** vs 기준선 0.543 (+0.240) | 경기별 0.735/0.794/0.827/0.777 |
| 2분류 AUC | **0.881** | 경기별 0.857/0.901/0.912/0.852 |
| 2분류 홀드아웃 f1 | FASTBALL 0.80 / BREAKING 0.75 | game_pk=813027, 260샘플 |
| **앱 경로 판정률** | **73.6%** (39/53) | 전체 중계 영상. 고치기 전 32.1% |
| **앱 경로 정확도** | **41.0%** vs 기준선 0.547 | 무신호 — 실험실 0.783이 재현 안 됨 (TS-025) |
| 3분류 LOGO (참고) | 0.665 vs 기준선 0.449 (+0.216) | 경기별 0.66/0.66/0.63/0.70 |
| 3분류 홀드아웃 (참고) | 0.700 vs 최빈값 0.483 (+0.217) | 300샘플 |
| 특징 중요도비 | 35.53 | TS-014 무신호 당시 1.23 |
| OFFSPEED AUC | 0.710 → **0.728** (특징 7→16개) | 실패 확정. 기준 +0.03 미달 |

### 이번 세션에서 확정된 것

- **박스 크기는 이 카메라 앵글에서 속도 대리 지표가 되지 못한다.** 재수집 100분을 들여
  실측했다. 원인은 알고리즘이 아니라 촬영 조건이다 (TS-024)

  ```
  OFFSPEED LOGO AUC
    기준(7)      0.710
    +기하(11)    0.730   <- 박스 없이 이게 최고
    좌표전체(14) 0.724
    +박스(16)    0.728
  ```

- **전체 정확도로는 `+박스(16)`이 최고다** (홀드아웃 0.700, LOGO 0.665). 박스 특징을
  코드에서 빼지 않은 이유다. `box_growth_per_frame`은 중요도 3위(0.054)
- **OFFSPEED는 두 축 모두에서 반응하지 않았다.** 표본 10배(4→40개)에도, 특징 9개 추가에도.
  이 상태에서 경기만 늘리는 것은 근거가 없다
- **재수집 자체는 깨끗했다.** 다운로드 실패 0건, 궤적 1029개로 v1(1030개)과 1개 차이 —
  v2 파이프라인이 같은 궤적을 재현한다

### 특징 중요도 (16개, 3분류 기준)

```
vertical_accel_px           0.398   <- 지배적
vertical_drop_px            0.087
box_growth_per_frame        0.054
curvature_ratio             0.053
speed_ratio_late_early      0.052
release_y                   0.048
release_x                   0.048
release_box_size            0.042
end_frame                   0.042
apparent_speed_px_per_frame 0.040
horizontal_deviation_px     0.031
frame_span                  0.029
duration_frames             0.026
late_drop_ratio             0.024
path_length_px              0.014
straight_line_px            0.011
```

`vertical_accel_px` 하나가 0.4를 먹는다. 4경기 4폴드에서 일관되지만 다른 시즌·구장에서
유지되는지는 미확인이다. 남은 리스크로 계속 들고 간다.

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

**Savant는 연결당 대역폭을 제한한다.** 실측 단일 연결 0.1MB/s(클립당 30초),
6연결 0.52MB/s(클립당 7.2초). 다만 6연결은 약 40분 뒤 `ConnectionResetError(54)`가 났다.
**기본값을 4연결로 내렸고 재시도(4회, 지수 백오프)를 넣었다 — 이 조합에서 1193건 실패 0건.**
연결 수를 다시 올리려면 재시도가 실패를 삼키지 않는지부터 확인해라(TS-021 구조).

**venv 셔뱅이 깨져 있다.** venv가 `~/Desktop/`에서 현재 경로로 옮겨져 `venv/bin/pip`이
`bad interpreter`로 죽는다. 패키지 설치는 `venv/bin/python3 -m pip`을 쓴다.
같은 이유로 `jupyter nbconvert`도 안 되고 `venv/bin/python3 -m nbconvert`를 써야 한다.

## 다음 갈래

### (A) 2분류 모델을 Streamlit에 붙인다 — **완료, 단 미검증** (2026-08-07)

붙였다. 그리고 앱에서는 안 된다는 것이 실측으로 확인됐다 (TS-025).

```
                        판정률        정확도       기준선
Savant 클립 (실험실)     86.3%        78.3%        54.3%
전체 중계 영상 (앱)      73.6%        41.0%        54.7%
```

**판정률 문제는 고쳤다.** 고정 창을 쓸 수 없다는 것이 원인이었다 — 오버레이 시각과
실제 투구 시각의 간격이 0.4~7.4초로 흩어진다. 넓은 창(t-4.0 ~ t+0.5초)으로 찾고
사슬 프레임을 평행이동해 `end_frame`을 학습 중앙값에 맞추는 것으로 32% → 74%가 됐다.

**정확도는 미해결이다.** 41.0%(16/39)는 기준선 아래이고 동전 던지기와 구분되지 않는다.
배선 버그는 아니다 — 캐시된 학습 궤적을 같은 경로로 통과시키면 0.942가 나온다(학습셋 기준).
남은 것은 도메인 격차다. 클립에서 뽑은 궤적과 원본 영상에서 뽑은 궤적이 같은 것이 아니다.

만들어진 것:
- `src/pitch_type_cv/live_classifier.py` — 판정 경로. `PitchVerdict`가 falsy라 실패를
  실수로 성공처럼 쓸 수 없다
- `scripts/train_two_class_model.py` → `output/pitch_type_cv/two_class_classifier.pkl`
- `scripts/measure_app_cv_accuracy.py` — 앱 경로 채점 (OCR 라벨 기준, `--sweep` 지원)
- `scripts/diagnose_app_window.py` — 공의 실제 오프셋 분포 진단
- `streamlit_app/app.py` — 실측 카드 아래 CV 패널. "실험적" 배지 + 실측 수치 병기

### (A-2) 사슬이 투구인지 판별하는 게이트 — TS-025 (2)를 풀려면 필요

넓은 창은 사슬 알고리즘이 "가장 긴 움직임"을 고를 뿐 그게 투구라는 보장이 없다.
리플레이·컷 전환·다른 앵글이 창 안에 들어온다. 실측 오프셋 중 7.40초짜리는 직전 투구나
리플레이일 가능성이 높다.

시작점으로 쓸 수 있는 것:
- 학습 궤적 1029개의 특징 분포가 있다. 앱에서 뽑은 사슬이 그 분포 안에 드는지로
  1차 필터가 가능하다 (마할라노비스 거리 등)
- 방향으로는 안 갈린다 — 학습 데이터에도 위로 가는 궤적이 39.1% 있다 (확인함)

### (A-3) 원래 계획이었던 것 (참고)

여기까지 만든 것이 한 번도 서비스에 붙지 않았다. 수치는 이미 쓸 만하다(LOGO 0.783, AUC 0.881).

**먼저 정할 것 — 앱이 무엇을 보여줄 것인가.** 이게 명세된 적이 없다.
`streamlit_app/`은 `pitch_type_cv`를 한 줄도 import하지 않는다. 두 모델이 **다른 문제를
푼다**: 앱의 `models/pitch_predictor.h5`(BiLSTM, ADR-0002)는 Statcast 시퀀스로 *다음* 투구를
예측하고, CV 경로는 영상으로 *방금 던진* 공을 분류한다. 붙이는 방식이 두 가지다.

1. CV 예측을 독립 화면으로 — 영상 업로드 → 궤적 → FASTBALL/BREAKING 확률
2. **예측 vs 실측 대조** — BiLSTM이 예측한 다음 구종 옆에 영상에서 실측한 실제 구종.
   앱 UI에 이미 "실측 / 예측" 라벨이 있고 이쪽이 원래 그림으로 보이지만 적힌 적은 없다

**모델 직렬화 상태**: `save_classifier`/`load_classifier`는 `src/pitch_type_cv/group_classifier.py`에
이미 있고, 노트북이 3분류 `output/pitch_type_cv/group_classifier.pkl`을 저장한다.
**없는 것은 2분류 모델 저장**이다 — `scripts/eval_two_class.py`는 매번 학습하고 버린다.

**궤적을 못 잡는 14%를 숨기지 말 것.** 확보율이 그룹별 80.9~91.3%, 경기별 FASTBALL은
71~92%로 흔들린다(아래 (C)). LOGO 0.783은 '궤적이 잡힌 투구'에 한정된 수치다.

앱 UI는 이전 세션에서 레이아웃·가독성만 손봤다(고정 480px iframe 제거, 스코어보드
flex 정렬, 확률 차트 가로 막대, 이모지 제거). 야구 분석 시스템다운 방향으로는 아직 안 갔다

### (B) 경기를 더 늘린다

경로 B는 임의 game_pk를 쓸 수 있다. 경기당 약 30분(4연결 기준).
**목적을 2분류 정밀도 향상으로 한정할 때만 유효하다.** OFFSPEED 부활 목적이라면 근거가 없다 —
표본 10배와 특징 9개 추가 두 축 모두에서 반응이 없었다(ADR-0012 기각 사유).

### (C) 확보율 편차의 원인 규명 — 계속 보류 중

FASTBALL 궤적 확보율이 경기별 71~92%로 21%p 흔들린다. "빠른 공이라 끊긴다"는 단일
설명으로는 안 맞는다(813027에서는 FASTBALL 92.4%로 BREAKING을 넘는다).
분류 성능에 직접 기여하지 않아 네 세션째 밀리고 있다. 서비스에 붙이면(A) 14%가
"판정 불가"로 노출되므로 그때 우선순위가 올라간다.

### 닫힌 갈래

- **OFFSPEED 3분류** — ADR-0012로 닫았다. 다시 열리는 조건은 속도 정보가 있는 입력
  (트래킹 데이터, 다중 카메라, 고프레임 촬영)이 생길 때다. Statcast `release_speed`를
  입력으로 쓰는 것은 "중계 영상만으로 예측"이라는 전제를 깨서 기각했다

## 재사용 가능한 것

- `output/pitch_type_cv/clip_trajectory_cache_v2/*.jsonl` — 박스 크기 포함 감지 후보 전체.
  1192건 확보. 사슬 임계값·특징 정의를 바꿔도 영상 재처리 없이 재계산 가능
- `output/pitch_type_cv/clip_trajectory_cache/*.jsonl` — v1(좌표만). 지우지 말 것
- `scripts/eval_feature_sets.py` — 특징 집합 절제 실험. 정확도·중요도비·그룹별
  precision/recall/f1·one-vs-rest AUC·LOGO 4폴드를 함께 찍고 판정선까지 출력한다.
  `--permutations N`으로 순열검정(대상은 정확도 최고 집합)
- `scripts/eval_two_class.py` — 2분류 LOGO + ROC + 혼동행렬
- `output/pitch_type_cv/feature_ablation.csv` / `.png` — 5개 집합 비교
- `output/pitch_type_cv/two_class_folds.csv` / `two_class_result.png` — 2분류 결과
- `scripts/calibrate_clip_window.py` — 새 시즌·다른 중계사 클립의 윈도우 재확인용

## 판정 규칙 (바꾸지 말 것)

- **기준선은 최빈값이다.** 33% 랜덤 아님. 불균형 때문에 아무것도 학습하지 않아도 48%가 나온다
- **소수 클래스를 겨냥한 작업은 전체 정확도로 판정하지 않는다.** 대상 클래스의
  one-vs-rest AUC를 본다. AUC는 임계값·클래스 가중치와 무관해서 "정보가 늘었나"와
  "동작점을 옮겼나"를 갈라준다 (TS-023에서 이 구분이 유일한 단서였다)
- **대리 지표는 물리적 전제가 이 데이터에서 성립하는지 한 클립으로 먼저 잰다.**
  박스 크기는 부호가 반대였고 변화량이 측정 해상도 아래였다 — 100분 전에 알 수 있었다 (TS-024)
- **실험실 수치는 배포 환경에 대한 약속이 아니다.** 학습 데이터를 균일 규격으로 자르면
  그 규격이 모델의 암묵적 전제가 된다. 클립 78.3%가 원본 영상에서 41.0%가 됐다 (TS-025)
- **한 증상 뒤에 문제가 둘 있을 수 있다.** 판정률과 정확도를 따로 보지 않았다면 창 수정이
  성공한 것을 못 보고 되돌렸을 것이다 — 전체 숫자는 안 움직였다 (TS-025)
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
