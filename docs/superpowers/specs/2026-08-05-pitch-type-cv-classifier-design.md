# 영상 기반 구종 그룹 분류기 (CV 파일럿) — 설계

## 배경

기존 서비스는 `game_pk`로 확보한 Statcast 데이터를 기반으로 다음 구종을 예측하고, YOLO는 투구
타이밍 감지·구속 추정에만 쓰인다. 방금 던진 공의 구종 자체는 Statcast 정답값을 그대로 표시할 뿐,
영상(공의 궤적)을 직접 보고 판단하지는 않는다.

목표는 "중계 영상 아무거나 가져와서 공을 보고 구종을 맞추는" 더 고도화된 서비스로 가는 것. 다만
이는 두 가지 서로 다른 난이도의 문제를 포함한다 — (1) 영상만으로 구종 판별(CV 분류), (2) 그 결과를
다음 구종 예측에 연결. 이번 스펙은 **(1)의 파일럿 검증까지만** 다룬다. (2)는 (1)의 정확도가 나온
뒤 별도 설계로 진행한다 — 기존 BiLSTM은 Statcast의 정밀 수치 피처(`pfx_x`, `pfx_z`, `plate_x`,
`plate_z`)로 학습돼 있어, CV로 뽑은 대략적인 궤적 신호를 그대로 이어붙일 수 없고 이 자체가 별도
설계 결정을 필요로 한다.

기존 `pose_detector.py`의 `scan_pitch_overlays()`가 Fox 중계 오버레이(예: "SLIDER 87 MPH")를
OCR로 읽는 기능을 이미 갖고 있으나, 이번 목표는 오버레이 텍스트에 의존하지 않고 **궤적 자체**로
구종을 판단하는 모델을 만드는 것이다. OCR은 이 모델의 학습 정답(라벨) 타이밍을 잡는 용도로만
재사용한다.

## 목표

1. 5-10경기(Fox 중계, YouTube에 영상 존재, `game_pk` 확보)를 파일럿 데이터셋으로 구성
2. 투구 궤적(YOLO 픽셀 좌표 시퀀스) → 수작업 특징(속도 추정치, 곡률, 낙하폭 등) 추출
3. 특징 → 3그룹(FASTBALL / BREAKING / OFFSPEED) 분류기 학습 (GradientBoosting/RandomForest)
4. 홀드아웃 경기로 정확도 검증 — 랜덤 베이스라인(33%) 유의미하게 상회하는지가 성공 기준
5. 결과를 `notebooks/01_pitch_group_classifier.ipynb`에서 그래프(혼동행렬, 그룹별 정확도)로 확인

## 비범위 (Non-goals)

- 다음 구종 예측 파이프라인과의 통합 — 별도 스펙
- Streamlit 앱(`streamlit_app/`) 통합 — 이번엔 오프라인 연구 파이프라인만
- 7개 세분류(FF/SI/FC/SL/CU/CH/FS) 분류 — 3그룹으로 한정
- Fox 외 방송사(ESPN, Apple TV+ 등) 지원 — OCR 라벨링이 Fox 오버레이 포맷에 의존하므로 파일럿은
  Fox 중계로 한정. (분류기 자체는 오버레이 없는 영상에도 추론 가능하지만, 학습 데이터 확보 범위는
  Fox로 제한)
- YouTube 영상 자동 검색/수집 — game_pk+URL 리스트를 직접 수동 지정
- 딥러닝(3D-CNN, End-to-end 시퀀스 모델) — 파일럿 데이터 규모(5-10경기)에 과적합 위험이 커서
  제외. 수작업 특징 + 경량 ML로 먼저 방향을 검증하고, 데이터가 늘어나면 재검토
- 기존 `yolo_detector.py`, `pose_detector.py`, `feature_engineering.py` 수정 — import만 하고
  원본은 건드리지 않음 (기존 데모 파이프라인 영향 없음)

## 아키텍처 & 컴포넌트

새 패키지 `src/pitch_type_cv/`:

| 파일 | 역할 | 의존 |
|---|---|---|
| `dataset.py` | game_pk+URL 리스트 → OCR 타임스탬프 추출 → Statcast 라벨 매칭 → 학습셋(csv/parquet) 생성 | `pose_detector.scan_pitch_overlays`, `pybaseball` |
| `trajectory_features.py` | 궤적 윈도우 추출(`extract_trajectory_window`) + 궤적 픽셀 좌표 시퀀스 → 특징 벡터 계산 | `yolo_detector.detect_ball_in_frame`, `estimate_speed` |
| `group_classifier.py` | 특징 벡터 → 3그룹 학습·추론 (GradientBoosting 또는 RandomForest) | scikit-learn |
| `pitch_group_map.py` | Statcast `pitch_type` → 3그룹 매핑 상수 (순수 룩업 테이블) | 없음 |

`pitch_group_map.py`의 매핑 기준:

| 그룹 | Statcast `pitch_type` |
|---|---|
| FASTBALL | FF(포심), SI(싱커), FC(커터) |
| BREAKING | SL(슬라이더), CU(커브) |
| OFFSPEED | CH(체인지업), FS(스플리터) |

`OTHER`(너클볼 등 희귀 구종)로 매핑되는 값은 학습 데이터셋 생성 단계에서 제외한다.

실행 스크립트: `scripts/build_pitch_group_dataset.py` (데이터셋 생성, `generate_fixed_demo_scan.py`
스타일의 1회성 로컬 실행). 학습·검증은 `notebooks/01_pitch_group_classifier.ipynb`에서 진행 —
프로젝트 컨벤션(그래프로 진행 상황 확인)에 맞춰 노트북으로 둔다. 산출물(혼동행렬, 정확도 그래프,
학습된 모델)은 `output/pitch_type_cv/`에 저장.

## 데이터 흐름

OCR 타임스탬프를 단일 기준점(anchor)으로 삼아 "영상 시간 ↔ Statcast 순서 ↔ 궤적 구간"의 매칭을
1축으로 단순화한다.

```
game_pk, youtube_url
    │
    ├─→ statcast_single_game(game_pk) → 정렬된 pitch_type 시퀀스 (정답)
    │
    └─→ 영상 다운로드 (resolve_video_path, 캐시 재사용)
            │
            └─→ scan_pitch_overlays() → 투구 타임스탬프 리스트 (오버레이 갱신 시점)
                    │
                    ├─→ [정답 매칭] i번째 타임스탬프 ↔ i번째 Statcast row (둘 다 시간순)
                    │
                    └─→ [궤적 추출] 각 타임스탬프 기준 (t-3.0s ~ t-0.3s) 구간만 잘라
                         `extract_trajectory_window()`로 그 안에서만 프레임별 공 감지
                         → 픽셀 좌표 시퀀스 (궤적)
```

**핵심 결정**: 기존 `yolo_detector.process_video()`의 독립적인 모션 기반 이벤트 감지는 재사용하지
않는다. 그걸 쓰면 "OCR 이벤트 ↔ YOLO 이벤트"를 또 매칭해야 하는 이중 정합 문제가 생긴다. 대신
`trajectory_features.py`에 OCR 타임스탬프 구간만 잘라 그 안에서 `detect_ball_in_frame()`을
프레임별로 직접 호출하는 새 함수(`extract_trajectory_window`)를 추가해 정합축을 하나로 줄인다.

타임스탬프 윈도우 값(`t-3.0s ~ t-0.3s`)은 방송 딜레이 추정치이며, 파일럿 첫 경기로 실측 보정이
필요하다.

**개수 불일치 처리**: OCR 감지 개수와 Statcast 투구 수가 다를 수 있다 (오버레이 인식 실패 등).
파일럿 규모라 퍼지 재정렬 로직은 만들지 않고, `min(len)`까지만 앞에서부터 매칭 + 불일치 시 해당
경기를 로그로 경고한다.

## 특징 추출 (`trajectory_features.py`)

궤적(픽셀 좌표 시퀀스)에서 다음을 계산:

- 추정 속도 (기존 `estimate_speed` 로직 재사용 — `px_per_meter` 캘리브레이션 필요)
- 궤적 곡률 (직선 대비 편차)
- 수직 낙하폭 / 수평 편차량
- 궤적 지속 프레임 수 (릴리스~홈플레이트 통과 추정 구간 길이)

이 특징 벡터가 `group_classifier.py`의 입력이 된다.

## 에러 처리

파일럿 배치 작업이므로 "하나 실패해도 전체는 계속 진행"이 원칙:

| 상황 | 처리 |
|---|---|
| YouTube 다운로드 실패 (네트워크 등) | 해당 경기 스킵 + 로그 경고, 나머지 경기 계속 |
| OCR 타임스탬프 0개 감지 (오버레이 인식 실패) | 해당 경기 전체 제외 + 로그 경고 |
| 특정 구간 궤적 포인트 부족 (공 가려짐 등, 3점 미만) | 그 투구 샘플만 제외, 제외 건수 로그 |
| OCR 개수 vs Statcast 개수 불일치 | `min(len)`까지 매칭 + 경기별 불일치 수 로그 |
| 특정 그룹 샘플 수 과소 (예: 오프스피드 20개 미만) | 학습은 진행하되 노트북에 경고 표시 — 파일럿 규모 한계로 명시 |

## 검증

- 순수 함수(`pitch_group_map.py`의 매핑, `trajectory_features.py`의 특징 계산)는 `tests/`에
  유닛 테스트 작성
- 분류기 성능은 유닛 테스트 대상이 아니라 **홀드아웃 경기 기준 정확도·혼동행렬**로
  `notebooks/01_pitch_group_classifier.ipynb`에서 확인 (5-10경기 중 1-2경기를 홀드아웃으로 분리)
- 성공 기준: 완벽한 정확도가 아니라 **3그룹 랜덤 베이스라인(33%)을 유의미하게 상회**하는지 —
  파일럿의 목적은 "이 방향이 통하는지" 검증

## 다음 단계 (이번 스펙 이후, 별도 진행)

- 파일럿 정확도가 유의미하면: 데이터 규모 확대(자동 수집), Fox 외 방송사 확장 검토
- 분류기 결과를 다음 구종 예측(BiLSTM)에 연결하는 방법 설계
- Streamlit 앱 통합 (game_pk 없이 URL만으로 동작하는 "범용 모드")
