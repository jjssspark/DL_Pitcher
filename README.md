![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![Git LFS](https://img.shields.io/badge/Git%20LFS-model%20files-lightgrey)

# ⚾ PitchIQ — MLB 실시간 투구 분석 & 다음 구종 예측

> 이전 투구 패턴과 경기 상황을 분석해 투수가 **다음에 던질 구종**을 예측하고,
> 중계 영상만으로 **구종과 투구 시각**까지 자동으로 판정하는 서비스입니다.

---

## 📌 Project Overview

기존 중계 화면은 투수가 공을 던진 **이후**에야 구종·구속을 보여줍니다.
PitchIQ는 여기서 한발 더 나가 "다음엔 어떤 공을 던질까?"를 AI가 미리 예측해서 보여주는 것을 목표로 합니다.

세 갈래가 하나의 Streamlit 화면에서 만납니다.

| 갈래 | 입력 | 하는 일 |
|---|---|---|
| **예측** | Statcast API | BiLSTM이 다음 구종을 예측 (8분류 48.5%) |
| **영상 구종 판정** | 중계 영상만 | YOLOv8 공 궤적으로 속구/변화구 판정 (2분류 66.4%) |
| **타임라인 정렬** | 중계 영상만 | 스코어버그 OCR 앵커로 영상 시각 ↔ 투구 번호 매칭 |

가운데 갈래가 이 프로젝트의 핵심 질문입니다 — **Statcast API 없이 화면만 보고 구종을 맞힐 수 있는가.**
같은 값을 API가 알려주는 정답과 나란히 띄워 화면에서 바로 채점되도록 만들었습니다.

## 🚀 Live Demo

### 데모 영상 (1분)

**[▶ youtu.be/XnbVtPsR3V8](https://youtu.be/XnbVtPsR3V8)**

로컬에서 실제로 돌아가는 화면을 녹화한 것입니다. 설치 없이 이것만 봐도 전체 흐름이 보입니다.

| 시각 | 내용 |
|---|---|
| 0:03 | Statcast 데이터 · 중계 영상 · 예측을 한 화면에 |
| 0:06 | 구종 10종을 Statcast 실측 무브먼트(`pfx_x`·`pfx_z`)로 표현 |
| 0:14 | 방금 던진 공 채점 — 직전 예측이 맞았는지 그 자리에서 |
| 0:28 | 영상만으로 낸 구종 판정 (YOLOv8 궤적) — 적중 |
| 0:36 | 같은 판정의 오답 사례 — 채점 125구 66.4% · 기준선 53.6% |
| 0:54 | 투수별 구종 분포와 누적 통계 |

영상 편집은 [`scripts/build_demo.py`](scripts/build_demo.py)로 재현할 수 있습니다.

### 배포된 앱

**[pitchiq-dlpitcher.streamlit.app](https://pitchiq-dlpitcher.streamlit.app/)**

사이드바에 `game_pk`(예: `745735`)를 입력하면 실제 MLB Statcast 데이터를 불러와 다음 구종을 실시간으로 예측합니다.
YouTube 영상 연동 · 실시간 스캔 기능은 무료 배포 환경(리소스 제약, 로컬 영상 서버 미노출)에서
정상 동작하지 않을 수 있습니다 — 자세한 내용은 [Known Limitations](#-known-limitations) 참고.

## 🎯 Features

- MLB Statcast 2025 시즌 전체 데이터 기반 학습
- BiLSTM + 투수/타자 Embedding + 경기 상황(카운트·주자·점수차 등) 기반 다음 구종 예측
- 투수 성향 피처(구종별 비율, 카운트별 최빈 구종, 상대 타자 전적) 반영
- YOLOv8 커스텀 모델로 중계 영상에서 야구공을 탐지하고 궤적을 추적
- **궤적 모양만으로 속구/변화구 판정** — Statcast API 없이 영상만 사용, 채점 125구에서 66.4% (기준선 53.6%)
- **스코어버그 OCR로 타임라인 앵커 200개 생성** — 영상 재생 시각과 투구 번호를 자동 정렬
- YouTube 영상 자동 다운로드 및 캐싱 (yt-dlp), Range 지원 로컬 HTTP 서버로 대용량 영상 탐색 재생
- 실측 구종 · 영상 판정 · 다음 구종 예측을 한 화면에 세로로 쌓아, 예측 적중 여부가 다음 투구에서 바로 채점되는 Streamlit 대시보드

## 🏗 Architecture

```
   MLB Statcast (pybaseball)              중계 영상 (YouTube / 로컬)
              │                                      │
              ▼                          ┌───────────┴───────────┐
     Feature Engineering                 ▼                       ▼
  (시퀀스 3구 + 상황 + 성향)      YOLOv8 공 탐지            스코어버그 OCR
              │                    → 궤적 특징            (pytesseract)
              ▼                          │                       │
       BiLSTM 학습/추론                  ▼                       ▼
      다음 구종 8분류 예측         속구/변화구 2분류        타임라인 앵커 200개
              │                          │                       │
              │                          │                       ▼
              │                          │              index_at_time() 보간
              │                          │              영상 시각 ↔ 투구 번호
              └──────────────┬───────────┴───────────────────────┘
                             ▼
                  Streamlit 실시간 대시보드
        (실측 구종 · 영상 판정 · 다음 구종 예측을 한 화면에)
```

> 영상 쪽 두 갈래는 **재생 중에 돌지 않습니다.** 구종 판정은 `scripts/batch_cv_verdicts.py`가,
> 타임라인 앵커는 `scripts/build_anchors_from_overlay.py`가 미리 계산해 JSON으로 저장하고
> 앱은 읽기만 합니다. 재생 중 추론은 부하가 크고, 무엇보다 **사용자가 실제로 본 투구만
> 채점되어 표본이 경기 초반에 치우치는** 문제가 있었습니다 (자세한 배경은 TS-036).

## 🛠 Tech Stack

| 분야 | 기술 | 비고 |
|---|---|---|
| Language | Python 3.13 | |
| 데이터 수집 | pybaseball (Statcast) | |
| 딥러닝 | TensorFlow 2.21 / Keras 3 | BiLSTM + Embedding (8분류) |
| 객체 탐지 | YOLOv8 (Ultralytics) | 중계 도메인 데이터셋으로 재파인튜닝 → 궤적 추적 |
| 궤적 분류 | scikit-learn | 궤적 특징 → 속구/변화구 2분류 |
| 영상 처리 | OpenCV, yt-dlp | YouTube 다운로드 + 프레임 처리 |
| OCR | pytesseract | 스코어버그 판독 → 타임라인 앵커 (시스템 tesseract 필요) |
| 시각화 | Matplotlib, Seaborn, Plotly | |
| 서비스 | Streamlit | 커스텀 영상 컴포넌트 + 사전 계산 JSON 조회 |
| 테스트 | pytest | 115개 |
| 모델 파일 관리 | Git LFS | `models/*.h5`, `*.pt` |

## 📈 Model Performance

### 1) 다음 구종 예측 — BiLSTM (Statcast 기반, 8분류)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Guess (8-class) | 12.5% | - |
| Initial LSTM | 39% | - |
| BiLSTM (Embedding + 투수 성향 피처, seq_len=5) | 46.14% | 39.2% |
| **BiLSTM (Embedding + 투수 성향 피처, seq_len=3) — 현재 모델** | **48.5%** | **43.4%** |

### 2) 영상만으로 구종 판정 — YOLOv8 궤적 (2분류)

미학습 경기의 원본 중계 영상에 그대로 돌린 결과입니다. Statcast를 정답으로 채점합니다.

| 항목 | 값 |
|---|---|
| 판정 시도 | 181구 |
| 판정 성공 | 137구 (76%) |
| 채점 대상 | 125구 (OFFSPEED 제외 — 2분류로 표현 불가) |
| **정확도** | **66.4%** (83/125) |
| 기준선 (다수 클래스로만 찍기) | 53.6% |

기준선 대비 12.8%p, 이항검정 p≈0.004로 유의합니다.

> ⚠️ 초기에는 이 값을 **76.7% (n=43)** 로 표기했습니다. 재생 중 실제로 본 투구만 채점되는
> 구조라 표본이 경기 초반(투수 1명·카메라 안정 구간)에 치우쳐 있었고, 표본을 181구로 늘리자
> 66.4%로 내려앉았습니다. **화면의 실시간 집계를 대표 수치로 굳히면 안 된다**는 걸 배운
> 지점이라 낮아진 값을 그대로 싣습니다 (전말은 [TS-036](docs/TROUBLESHOOTING.md)).

Day 2에서 시퀀스 길이(3/5/8)와 클래스 불균형 보정(class weight) 실험을 진행했고, `seq_len=3`이
가장 안정적으로 좋은 결과를 보여 기본값으로 채택했습니다. 실험 과정과 트레이드오프는
[`docs/blog/day2.md`](docs/blog/day2.md)에 정리했습니다.

혼동행렬·클래스별 정확도는 `python src/evaluate.py` 실행 후 `notebooks/figures/`에서 확인할 수 있습니다.

<p align="center">
  <img src="notebooks/figures/07_confusion_matrix.png" width="49%" />
  <img src="notebooks/figures/08_accuracy_by_class.png" width="49%" />
</p>

## 📂 Project Structure

```
baseball-pitch-predictor/
├── data/
│   ├── raw/            # Statcast CSV (gitignore — 직접 수집 필요)
│   └── yolo/            # YOLO 학습용 데이터셋
├── models/
│   ├── pitch_predictor.h5   # BiLSTM 모델 (Git LFS)
│   ├── scaler.pkl            # 피처 스케일러 (Git LFS)
│   ├── ball_broadcast_v1.pt  # 현재 쓰는 공 탐지기 — 중계 도메인 재파인튜닝 (Git LFS)
│   └── baseball_detector.pt  # 폐기된 초기 탐지기 — 중계 감지율 3% (TS-014, ADR-0009)
├── notebooks/
│   ├── eda.py
│   └── figures/
├── src/
│   ├── data_collector.py       # Statcast 수집
│   ├── feature_engineering.py  # 시퀀스/컨텍스트/성향 피처 생성
│   ├── model.py                 # BiLSTM 모델 정의 및 학습
│   ├── evaluate.py              # 성능 평가 + 시각화
│   ├── train_yolo.py            # YOLO 커스텀 모델 학습 (Roboflow)
│   ├── yolo_detector.py         # 야구공 탐지 + 궤적/구속 추정
│   ├── pose_detector.py         # 프레임 차분 모션 감지 + 스코어버그 OCR
│   ├── timeline_anchor.py       # 앵커 보간 (영상 시각 ↔ 투구 번호)
│   ├── pitch_type_cv/           # 궤적 → 속구/변화구 2분류
│   │   ├── trajectory_features.py
│   │   ├── group_classifier.py
│   │   └── ...
│   └── pipeline.py              # CLI용 통합 파이프라인 (앱은 미사용)
├── scripts/                      # 오프라인 사전 계산 (앱이 읽는 JSON 생성)
│   ├── batch_cv_verdicts.py     # → streamlit_app/fixed_demo_cv.json
│   ├── build_anchors_from_overlay.py  # → fixed_demo_anchors.json
│   └── ...
├── streamlit_app/
│   ├── app.py                    # 실시간 대시보드
│   ├── local_video_player/       # Range 지원 커스텀 영상 컴포넌트
│   └── fixed_demo_*.json         # 사전 계산 결과 (판정 · 앵커 · 스캔)
├── tests/                        # pytest 115개
├── docs/
│   ├── TROUBLESHOOTING.md       # 문제 해결 기록 38건 (실패한 시도 포함)
│   ├── ADR.md                    # 설계 결정 기록
│   ├── ROADMAP.md
│   └── blog/                     # Day1 · Day2 기술 블로그
├── requirements.txt
└── requirements-dev.txt
```

## 🚀 Getting Started

```bash
git clone https://github.com/jjssspark/DL_Pitcher.git
cd DL_Pitcher

# 대용량 모델 파일은 Git LFS로 관리됩니다
git lfs install
git lfs pull

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # 서비스 실행용
# pip install -r requirements-dev.txt  # 노트북/모델 학습까지 필요하면

# 1) Statcast 데이터 수집 (최초 1회, 20~30분 소요)
python src/data_collector.py --full --year 2025

# 2) 모델 학습 (models/pitch_predictor.h5, scaler.pkl 생성)
python src/model.py

# 3) 서비스 실행
streamlit run streamlit_app/app.py
```

> OCR 기능을 쓰려면 `brew install tesseract`(macOS) 등으로 시스템 tesseract가 별도로 필요합니다.
> `src/train_yolo.py`로 YOLO 모델을 직접 학습하려면 Roboflow 계정과 API 키가 필요합니다.

## ⚙ Pipeline

```
[학습 · 1회]
Statcast 2025 전체 → Preprocessing → Feature Engineering → BiLSTM 학습
중계 도메인 데이터셋 → YOLOv8 재파인튜닝 → 야구공 탐지기
궤적 특징 추출     → 2분류 분류기 학습

[사전 계산 · 경기당 1회]
중계 영상 → YOLOv8 궤적 추적 → 구종 판정 JSON
중계 영상 → 스코어버그 OCR   → 타임라인 앵커 JSON
Statcast → BiLSTM 배치 추론  → 예측 캐시 (.bilstm_cache/*.pkl)

[재생 중 · 실시간]
영상 재생 시각 → 앵커 보간 → 현재 투구 번호 → 세 결과를 조회해 렌더
```

## 🚧 Known Limitations

측정해서 알게 된 한계를 그대로 적습니다.

- **영상 구종 판정은 2분류까지만 됩니다.** OFFSPEED(체인지업·스플리터)는 중계 카메라 궤적에서
  속구와 갈리지 않아 범위에서 제외했습니다. 모델이 아니라 입력의 한계입니다 (ADR-0012).
- **판정 불가 44구가 남아 있습니다.** 사유는 궤적 없음 27 · 궤적 짧음 16 · 공 미감지 1로,
  카메라 각도나 가림 문제라 탐색 창을 옮겨도 잡히지 않습니다.
- **판정 신뢰도는 쓸모가 없습니다.** 적중 평균 0.83, 오답 0.79로 구분이 안 됩니다.
  화면에 표시하지만 판단 근거로 쓰면 안 됩니다.
- **BiLSTM 예측은 전체 투구를 덮지 못합니다.** 데모 경기 320구 중 173구에만 값이 나옵니다.
  나머지는 해당 투수의 2025 학습 데이터가 없거나(교체 투수) 시퀀스 3구가 안 쌓인 구간입니다.
  없는 예측을 지어내지 않고 화면에 "학습 이력 부족"으로 표시합니다.
- **타임라인은 앵커 사이를 선형 보간합니다.** 앵커가 200개라 프레임 단위로 정확하지는 않습니다.
- `pose_detector.py`는 이름과 달리 포즈 추정이 아니라 프레임 차분 + OCR로 동작합니다 (모듈명 정리 예정).
- YOLO 학습 스크립트의 `device="mps"`는 Apple Silicon 전용입니다. 다른 환경에서는 `cuda`/`cpu`로 수정이 필요합니다.

### 폐기한 접근

막다른 길도 결과의 일부라 남깁니다.

| 시도 | 왜 버렸나 |
|---|---|
| 마운드 프레임 차분으로 투구 시각 검출 | 홀드아웃에서 균등 분할보다 나빴음 (표준편차 24.9s vs 7.1s) — TS-032 |
| 재생 중 실시간 OCR로 투구 감지 | 오버레이가 투구 +2.8~4.2초에야 떠서 구조적으로 최소 +4.4초 지연 — TS-031 |
| 포즈 추정(YOLOv8-pose)으로 투구 동작 인식 | 정확도·속도 모두 미달 |

## 📝 Engineering Notes

문제를 어떻게 좁혀 나갔는지가 남아 있는 문서입니다.

- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — 해결 기록 38건.
  증상 / 재현 조건 / 표면·근본 원인 / **시도했지만 안 된 것** / 해결 / 검증 순으로 씁니다.
  틀린 가설과 잘못 짚은 방향도 그대로 남겼습니다.
- [`docs/ADR.md`](docs/ADR.md) — 설계 결정과 그 근거.
- [`docs/blog/`](docs/blog/) — Day1·Day2 기술 정리.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — 초기 5일 로드맵.

특히 볼 만한 기록:

| 번호 | 내용 |
|---|---|
| TS-037 | BiLSTM이 영원히 "계산 중"에 멈춘 건 pyarrow/TF의 absl 심볼 충돌 데드락 + shape 불일치였고, 예외가 dict에 갇혀 화면엔 정상 문구로 보였음 |
| TS-036 | 정확도 76.7%(n=43)가 표본을 늘리자 66.4%(n=125)로 내려앉은 이유 |
| TS-033 | 오버레이 픽셀 차분이 실패해 글자만 이진화해 XOR — 앵커 33개 → 200개 |
| TS-028 | 학습·추론의 프레임 샘플링 불일치로 모델이 전부 직구로 붕괴 (0.410 → 0.767) |

## 🚀 Future Work

- 실시간 경기 영상 분석 정확도 개선
- Pitch Tracking 자동화 고도화
- MLB → KBO 확장
- 선수 맞춤 분석
- 모바일 서비스 개발

## 👨‍💻 Developer

**박지수** — AI Deep Learning Project, 2026
