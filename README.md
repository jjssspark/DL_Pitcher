<div align="center">

# PitchIQ

### 중계 화면만으로 다음 공을 읽는 MLB 투구 분석 서비스

투수가 다음에 던질 구종을 예측하고, Statcast API 없이 영상만으로 방금 던진 구종을 판정합니다.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00B8D4)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-model%20files-8A8A8A)

[데모 영상 1분](https://youtu.be/XnbVtPsR3V8) · [배포된 앱](https://pitchiq-dlpitcher.streamlit.app/) · [트러블슈팅 39건](docs/TROUBLESHOOTING.md) · [설계 결정 기록](docs/ADR.md)

<a href="https://youtu.be/XnbVtPsR3V8">
  <img src="https://img.youtube.com/vi/XnbVtPsR3V8/maxresdefault.jpg" width="640" alt="PitchIQ 데모 영상" />
</a>

<br/>

<table>
<tr>
  <td align="center" width="220"><h2>48.5%</h2>다음 구종 예측<br/><sub>8분류 · 무작위 12.5%</sub></td>
  <td align="center" width="220"><h2>66.4%</h2>영상만으로 구종 판정<br/><sub>2분류 · 기준선 53.6%</sub></td>
  <td align="center" width="220"><h2>200개</h2>타임라인 앵커<br/><sub>영상 시각 ↔ 투구 번호</sub></td>
</tr>
</table>

</div>

---

## 목차

[무엇을 푸는가](#무엇을-푸는가) · [세 갈래](#세-갈래) · [Architecture](#architecture) · [Tech Stack](#tech-stack) · [성능](#성능) · [Getting Started](#getting-started) · [Project Structure](#project-structure) · [Known Limitations](#known-limitations) · [Engineering Notes](#engineering-notes)

---

## 무엇을 푸는가

중계 화면은 공이 이미 포수 미트에 꽂힌 다음에야 구속과 구종을 보여줍니다. 정작 가장 궁금한
순간, 그러니까 공이 손을 떠나기 전에는 화면에 아무 정보도 없습니다.

```
0.0s              ~0.45s                    +2.8~4.2s
릴리스   ────────  포수 미트  ────────────  화면에 구종 표시
         ↑ 이 구간에는 아무 정보도 없음
```

`+2.8~4.2초`는 스코어버그가 실제로 뜨는 시각을 재서 나온 값입니다. 재생 중 OCR로 투구를
잡으려던 시도를 접은 이유이기도 합니다 ([TS-031](docs/TROUBLESHOOTING.md)).

이 프로젝트의 핵심 질문은 하나입니다.

> Statcast API 없이, 화면만 보고 구종을 맞힐 수 있는가.

API가 알려주는 정답을 같은 화면에 나란히 띄워, 예측과 판정이 다음 투구에서 곧바로
채점되도록 만들었습니다. 맞으면 맞았다고, 틀리면 틀렸다고 화면에 그대로 남습니다.

## 세 갈래

| | 입력 | 하는 일 | 결과 |
|---|---|---|---|
| 예측 | Statcast API | BiLSTM이 직전 3구와 경기 상황으로 다음 구종을 8분류 예측 | 48.5% |
| 영상 구종 판정 | 중계 영상만 | YOLOv8 공 궤적의 모양만으로 속구·변화구를 가름 | 66.4% |
| 타임라인 정렬 | 중계 영상만 | 스코어버그 OCR 앵커로 영상 시각과 투구 번호를 매칭 | 앵커 200개 |

가운데 갈래가 핵심 질문입니다. 나머지 둘은 그 답을 화면에서 채점 가능하게 만드는 장치입니다.

### 주요 기능

- MLB Statcast 2025 시즌 전체 데이터로 학습
- BiLSTM에 투수·타자 Embedding과 경기 상황(카운트·주자·점수차)을 결합
- 투수 성향 피처 반영 — 구종별 비율, 카운트별 최빈 구종, 상대 타자 전적
- 중계 도메인 데이터셋으로 파인튜닝한 YOLOv8로 야구공 탐지 및 궤적 추적
- 궤적 모양만으로 속구·변화구 판정. 채점 125구에서 66.4% (기준선 53.6%)
- 스코어버그 OCR로 타임라인 앵커 200개 생성
- YouTube 영상 자동 다운로드·캐싱(yt-dlp), Range 지원 로컬 HTTP 서버로 대용량 영상 탐색 재생
- 실측 구종·영상 판정·다음 구종 예측을 한 화면에 쌓아 다음 투구에서 바로 채점

## Architecture

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

영상 쪽 두 갈래는 재생 중에 돌지 않습니다. 구종 판정은 `scripts/batch_cv_verdicts.py`가,
타임라인 앵커는 `scripts/build_anchors_from_overlay.py`가 미리 계산해 JSON으로 저장하고
앱은 읽기만 합니다. 재생 중 추론은 부하도 크지만, 무엇보다 사용자가 실제로 본 투구만
채점되어 표본이 경기 초반에 치우치는 문제가 있었습니다 ([TS-036](docs/TROUBLESHOOTING.md)).

### 실행 단계

```
[학습 · 1회]
Statcast 2025 전체    → Preprocessing → Feature Engineering → BiLSTM 학습
중계 도메인 데이터셋   → YOLOv8 파인튜닝 → 야구공 탐지기
궤적 특징 추출         → 2분류 분류기 학습

[사전 계산 · 경기당 1회]
중계 영상 → YOLOv8 궤적 추적 → 구종 판정 JSON
중계 영상 → 스코어버그 OCR    → 타임라인 앵커 JSON
Statcast  → BiLSTM 배치 추론  → 예측 캐시 (.bilstm_cache/*.pkl)

[재생 중 · 실시간]
영상 재생 시각 → 앵커 보간 → 현재 투구 번호 → 세 결과를 조회해 렌더
```

## Tech Stack

| 분야 | 기술 | 비고 |
|---|---|---|
| Language | Python 3.13 | |
| 데이터 수집 | pybaseball (Statcast) | |
| 딥러닝 | TensorFlow 2.21 / Keras 3 | BiLSTM + Embedding (8분류) |
| 객체 탐지 | YOLOv8 (Ultralytics) | 중계 도메인 데이터셋으로 파인튜닝 → 궤적 추적 |
| 궤적 분류 | scikit-learn | 궤적 특징 → 속구·변화구 2분류 |
| 영상 처리 | OpenCV, yt-dlp | YouTube 다운로드 + 프레임 처리 |
| OCR | pytesseract | 스코어버그 판독 → 타임라인 앵커 (시스템 tesseract 필요) |
| 시각화 | Matplotlib, Seaborn, Plotly | |
| 서비스 | Streamlit | 커스텀 영상 컴포넌트 + 사전 계산 JSON 조회 |
| 테스트 | pytest | 115개 |
| 모델 파일 관리 | Git LFS | `models/*.h5`, `*.pt` |

## 성능

### 다음 구종 예측 — BiLSTM, Statcast 기반 8분류

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Guess (8-class) | 12.5% | — |
| Initial LSTM | 39.0% | — |
| BiLSTM (Embedding + 투수 성향 피처, seq_len=5) | 46.6% | 39.2% |
| BiLSTM (Embedding + 투수 성향 피처, seq_len=3) — 현재 모델 | 48.5% | 43.4% |

시퀀스 길이 3·5·8과 클래스 불균형 보정(class weight)을 비교해 `seq_len=3`을 채택했습니다.
문맥을 길게 줄수록 좋아지지는 않았습니다. 길이를 늘리면 조건을 만족하는 타석이 줄어
학습 시퀀스가 22.7만개에서 6.1만개로 떨어집니다. 실험 과정은
[`docs/blog/day2.md`](docs/blog/day2.md)에 있습니다.

데모 경기 320구 중 173구에만 예측이 나옵니다. 나머지는 해당 투수의 2025 학습 데이터가
없거나(교체 투수) 시퀀스 3구가 아직 안 쌓인 구간입니다. 없는 예측을 지어내지 않고 화면에
"학습 이력 부족"으로 표시합니다.

<p align="center">
  <img src="notebooks/figures/07_confusion_matrix.png" width="49%" />
  <img src="notebooks/figures/08_accuracy_by_class.png" width="49%" />
</p>

### 영상만으로 구종 판정 — YOLOv8 궤적, 2분류

미학습 경기의 원본 중계 영상에 그대로 돌린 결과입니다. Statcast를 정답으로 채점합니다.

| 항목 | 값 |
|---|---:|
| 판정 시도 | 181구 |
| 판정 실패 (궤적 없음 27 · 짧음 16 · 미감지 1) | −44구 |
| 판정 성공 | 137구 (76%) |
| 채점 제외 (OFFSPEED — 2분류로 표현 불가) | −12구 |
| 채점 대상 | 125구 |
| 정확도 | 66.4% (83/125) |
| 기준선 (다수 클래스로만 찍기) | 53.6% |

기준선 대비 12.8%p, 이항검정 p≈0.004로 유의합니다.

> 초기에는 이 값을 76.7%(n=43)로 표기했습니다. 재생 중 실제로 본 투구만 채점되는 구조라
> 표본이 경기 초반(투수 한 명·카메라 안정 구간)에 치우쳐 있었고, 표본을 181구로 늘리자
> 66.4%로 내려앉았습니다. 화면의 실시간 집계를 대표 수치로 굳히면 안 된다는 걸 배운
> 지점이라 낮아진 값을 그대로 싣습니다 (전말은 [TS-036](docs/TROUBLESHOOTING.md)).

## Getting Started

```bash
git clone https://github.com/jjssspark/DL_Pitcher.git
cd DL_Pitcher

# 대용량 모델 파일은 Git LFS로 관리됩니다
git lfs install
git lfs pull

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # 서비스 실행용
# pip install -r requirements-dev.txt  # 노트북·모델 학습까지 필요하면

# 1) Statcast 데이터 수집 (최초 1회, 20~30분 소요)
python src/data_collector.py --full --year 2025

# 2) 모델 학습 (models/pitch_predictor.h5, scaler.pkl 생성)
python src/model.py

# 3) 서비스 실행
streamlit run streamlit_app/app.py
```

배포된 앱은 사이드바에 `game_pk`(예: `745735`)를 넣으면 실제 MLB Statcast 데이터를 불러옵니다.
YouTube 영상 연동과 실시간 스캔은 무료 배포 환경의 리소스 제약 때문에 로컬만큼 안정적이지
않을 수 있습니다.

OCR을 쓰려면 `brew install tesseract`(macOS) 등으로 시스템 tesseract가 별도로 필요합니다.
`src/train_yolo.py`로 탐지기를 직접 학습하려면 Roboflow 계정과 API 키가 필요합니다.

## Project Structure

```
baseball-pitch-predictor/
├── data/
│   ├── raw/                          # Statcast CSV (gitignore — 직접 수집)
│   └── yolo/                         # YOLO 학습용 데이터셋
├── models/
│   ├── pitch_predictor.h5            # BiLSTM 모델 (Git LFS)
│   ├── scaler.pkl                    # 피처 스케일러 (Git LFS)
│   ├── ball_broadcast_v1.pt          # 현재 쓰는 공 탐지기 (Git LFS)
│   └── baseball_detector.pt          # 폐기된 초기 탐지기 — 감지율 3% (TS-014)
├── src/
│   ├── data_collector.py             # Statcast 수집
│   ├── feature_engineering.py        # 시퀀스·컨텍스트·성향 피처 생성
│   ├── model.py                      # BiLSTM 정의 및 학습
│   ├── evaluate.py                   # 성능 평가 + 시각화
│   ├── train_yolo.py                 # 탐지기 학습 (Roboflow 데이터셋)
│   ├── yolo_detector.py              # 공 탐지 + 궤적·구속 추정
│   ├── pose_detector.py              # 프레임 차분 모션 감지 + 스코어버그 OCR
│   ├── timeline_anchor.py            # 앵커 보간 (영상 시각 ↔ 투구 번호)
│   ├── pitch_type_cv/                # 궤적 → 속구·변화구 2분류
│   └── pipeline.py                   # CLI용 통합 파이프라인 (앱은 미사용)
├── scripts/                          # 오프라인 사전 계산 (앱이 읽는 JSON 생성)
│   ├── batch_cv_verdicts.py          # → streamlit_app/fixed_demo_cv.json
│   └── build_anchors_from_overlay.py # → fixed_demo_anchors.json
├── streamlit_app/
│   ├── app.py                        # 실시간 대시보드
│   ├── local_video_player/           # Range 지원 커스텀 영상 컴포넌트
│   └── fixed_demo_*.json             # 사전 계산 결과 (판정 · 앵커 · 스캔)
├── tests/                            # pytest 115개
├── notebooks/figures/                # 혼동행렬·클래스별 정확도 그래프
└── docs/
    ├── TROUBLESHOOTING.md            # 문제 해결 기록 39건 (실패한 시도 포함)
    ├── ADR.md                        # 설계 결정 기록 14건
    ├── ROADMAP.md
    └── blog/                         # Day1 · Day2 기술 정리
```

## Known Limitations

측정해서 알게 된 한계를 그대로 적습니다.

- 영상 구종 판정은 2분류까지만 됩니다. OFFSPEED(체인지업·스플리터)는 중계 카메라 궤적에서
  속구와 갈리지 않아 범위에서 제외했습니다. 모델이 아니라 입력의 한계입니다 (ADR-0012)
- 판정 불가 44구가 남아 있습니다. 궤적 없음 27 · 짧음 16 · 미감지 1로, 카메라 각도나 가림
  문제라 탐색 창을 옮겨도 잡히지 않습니다
- 판정 신뢰도는 판단 근거로 쓸 수 없습니다. 적중 평균 0.83, 오답 0.79로 구분이 안 됩니다.
  화면에 표시하되 근거로는 쓰지 않습니다
- BiLSTM 예측은 320구 중 173구만 덮습니다
- 타임라인은 앵커 사이를 선형 보간합니다. 앵커가 200개라 프레임 단위로 정확하지는 않습니다
- `pose_detector.py`는 이름과 달리 포즈 추정이 아니라 프레임 차분 + OCR로 동작합니다
- YOLO 학습 스크립트의 `device="mps"`는 Apple Silicon 전용입니다

### 폐기한 접근

막다른 길도 결과의 일부라 남깁니다.

| 시도 | 왜 버렸나 |
|---|---|
| 마운드 프레임 차분으로 투구 시각 검출 | 홀드아웃에서 균등 분할보다 나빴음. 오차 표준편차 24.9초 대 7.1초 (TS-032) |
| 재생 중 실시간 OCR로 투구 감지 | 오버레이가 투구 +2.8~4.2초에야 떠서 구조적으로 최소 +4.4초 지연 (TS-031) |
| 포즈 추정(YOLOv8-pose)으로 투구 동작 인식 | 정확도·속도 모두 미달 |

## Engineering Notes

문제를 어떻게 좁혀 나갔는지가 남아 있는 문서입니다.

| 문서 | 내용 |
|---|---|
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | 해결 기록 39건. 증상 / 재현 조건 / 표면·근본 원인 / 시도했지만 안 된 것 / 해결 / 검증 |
| [`docs/ADR.md`](docs/ADR.md) | 설계 결정 14건과 그 근거. 대체되거나 틀린 것으로 판명된 결정도 상태를 갱신해 남김 |
| [`docs/RETROSPECT.md`](docs/RETROSPECT.md) | 회고(KPT). 3주간 작업 방식이 어떻게 바뀌었는지 전후 비교 |
| [`docs/blog/`](docs/blog/) | Day1 · Day2 기술 정리 |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 초기 5일 로드맵 |

틀린 가설과 잘못 짚은 방향도 지우지 않았습니다. 특히 볼 만한 기록입니다.

| 번호 | 내용 |
|---|---|
| TS-014 | 학습셋 mAP 0.852를 보고 탐지가 된다고 판단했으나 중계 영상 감지율은 3%, 실제 공 크기 감지는 0건. YOLO가 잡던 건 공이 아니라 투수 글러브였음 |
| TS-028 | 학습과 추론의 프레임 샘플링이 어긋나 가속도가 1/4로 눌렸고 모델이 전부 직구로 붕괴. 맞추자 0.410 → 0.767 |
| TS-036 | 정확도 76.7%(n=43)가 표본을 늘리자 66.4%(n=125)로 내려앉은 이유 |
| TS-033 | 오버레이 픽셀 차분이 실패해 글자만 이진화해 XOR. 앵커 33개 → 200개 |

## Future Work

측정하다 남긴 숙제들입니다. 실제로 문서에 근거가 있는 것만 적습니다.

- 투수 성향 피처의 데이터 누수 제거. 지금은 시즌 전체 데이터로 계산해서, 시즌 중 실시간
  서비스로 쓰면 미래 정보가 새어 들어갑니다. 예측 시점 이전까지만 누적하도록 바꿔야 합니다 (ADR-0002)
- 단방향 LSTM과의 정량 비교. 양방향이 유리하다고 판단해 BiLSTM으로 갔는데 A/B 비교를
  실제로 돌리지는 않았습니다 (ADR-0002)
- 클래스 불균형은 balanced 가중치로는 안 됐습니다. 가중치 상한이나 focal loss가 남은 후보입니다 (ADR-0006)
- OFFSPEED를 가르려면 중계 카메라 궤적 말고 다른 입력이 필요합니다 (ADR-0012)
- `pose_detector.py` 모듈명 정리

---

<div align="center">

박지수 · AI Deep Learning Project · 2026

</div>
