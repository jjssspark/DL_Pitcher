# [PitchIQ 개발기 Day 1] 프로젝트 소개와 기반 다지기

## 이 프로젝트가 뭔가요

PitchIQ는 MLB Statcast 데이터로 학습한 딥러닝 모델이 "투수가 다음에 어떤 공을 던질지"를 예측하는 서비스입니다.
중계 화면은 투수가 공을 던진 *이후*에야 구종·구속을 알려주는데, PitchIQ는 그 전에 예측값을 보여줘서
야구를 좀 더 분석적으로 볼 수 있게 하는 것이 목표입니다.

여기에 더해 실제 중계 영상(YouTube)을 넣으면 YOLOv8로 공을 탐지하고 투구 타이밍을 자동으로 잡아내서,
Statcast 데이터 기반 예측과 실시간 영상을 하나의 Streamlit 대시보드에서 연결하는 구조까지 만들었습니다.

## 왜 이 문제를 골랐나

야구는 "다음 수"를 예측하는 게임입니다. 투수와 포수는 카운트, 이전 투구, 상대 타자 성향을 종합해서
다음 구종을 정하는데, 이 의사결정 과정 자체가 시퀀스 데이터(직전 몇 개의 투구)와 구조화된 컨텍스트(카운트,
주자, 점수차)가 함께 필요한 전형적인 시계열+표형 데이터 결합 문제입니다. 그래서 LSTM 계열 모델과
Embedding, 정형 피처를 같이 쓰는 구조가 자연스럽게 나왔습니다.

## 데이터: MLB Statcast가 뭔가요

Statcast는 MLB 구장에 설치된 추적 시스템이 모든 투구·타구에 대해 기록하는 데이터입니다. 구속, 회전수,
좌우/상하 변화량(무브먼트), 홈플레이트 통과 위치는 물론 카운트, 아웃카운트, 주자 상황, 점수차까지 투구
단위로 다 들어있습니다. Python의 `pybaseball` 라이브러리가 이 데이터를 API처럼 긁어올 수 있게 해줘서,
2025 시즌 전체(4월~10월)를 월별로 나눠 수집했습니다. 월별로 나눈 이유는 한 번에 시즌 전체를 요청하면
메모리와 타임아웃 문제가 생기기 때문입니다.

## 전체 아키텍처

```
Statcast 데이터 ─→ 피처 엔지니어링 ─→ BiLSTM 학습 ──┐
                                                    ├─→ 통합 파이프라인 ─→ Streamlit 서비스
중계 영상 ─→ YOLOv8 공 탐지 ─→ 투구 타이밍 인식 ──────┘
```

두 파이프라인(구종 예측 / 영상 분석)이 원래 독립적으로 개발됐고, `src/pipeline.py`의
`PitchPipeline` 클래스가 이 둘을 연결하는 역할을 합니다. YOLO가 투구를 감지할 때마다
`on_pitch_detected()`가 호출되면서 Statcast 시퀀스를 한 칸씩 전진시키고 다음 구종을 예측합니다.

## 기술 스택과 선택 이유

- **TensorFlow/Keras (BiLSTM)**: 시퀀스 방향성(과거→현재뿐 아니라 양방향 문맥)을 활용하려고 단방향 LSTM
  대신 Bidirectional을 선택했습니다.
- **Embedding Layer (투수/타자 ID)**: 투수마다 구종 레퍼토리와 패턴이 완전히 다르기 때문에, 원-핫 대신
  저차원 벡터로 투수/타자의 "성향"을 학습하게 했습니다.
- **YOLOv8 (Ultralytics)**: 실시간에 가까운 속도로 작은 객체(야구공)를 탐지해야 해서, 가볍고 커스텀
  학습이 쉬운 YOLOv8n을 base로 골랐습니다.
- **Streamlit**: 빠르게 프로토타입을 서비스 형태로 보여줄 수 있어서 선택했습니다. 다만 백그라운드
  스레드/캐시 관리에서 겪은 어려움은 Day 4에 따로 정리할 예정입니다.

## Day 1에 실제로 한 일

코드/모델은 이미 상당 부분 만들어져 있었어서, Day 1은 "이걸 남이 재현할 수 있는 상태로 만들기"에
집중했습니다.

1. **`requirements.txt` 작성**: 실행 중인 venv의 `site-packages` 안 `*.dist-info` 폴더명에서 정확한
   버전을 추출해서 고정했습니다. (뒤에 트러블슈팅에 이유가 나옵니다)
2. **`.gitignore` / `.gitattributes` 재정리**: 서비스에 실제로 필요한 모델 파일 3개
   (`pitch_predictor.h5`, `scaler.pkl`, `baseball_detector.pt`)만 Git LFS로 추적하고, 사전학습
   체크포인트나 학습 실험 산출물(수백 MB짜리 이미지 포함 디렉터리)은 계속 제외하도록 정리했습니다.
3. **README 전면 개편**: 기존 README는 초기 기획 수준이었는데, 실제로는 YouTube 영상 연동·실시간 스캔·
   OCR·모션 감지까지 붙은 훨씬 큰 서비스였습니다. 실제 구현 기준으로 다시 썼습니다.
4. **Git 히스토리 새로 시작**: 아래 트러블슈팅 참고.

## 트러블슈팅

### 1) `.git`이 2GB — 왜 이렇게 커졌나

`git count-objects -vH`로 확인해보니 `.git` 내부가 로컬로 500MB+, 패킹된 상태로 1.4GB 이상이었습니다.
원인을 추적해보니 커밋이 4개뿐인데도 각 커밋마다 수십 MB짜리 Statcast CSV(월별로 6개, 총 87MB짜리
전체 파일 포함), YOLO 사전학습 가중치, 학습된 모델 파일이 그대로 커밋되어 있었습니다. 나중에
`.gitignore`로 막았지만 이미 커밋된 이력은 그대로 남기 때문에 `.gitignore`만으로는 저장소 용량이
줄지 않습니다.

선택지는 두 가지였습니다: `git filter-repo`로 과거 이력에서 대용량 blob만 골라 제거하거나, 현재
상태를 기준으로 새 히스토리를 시작하는 것. 커밋이 4개뿐이라 잃을 게 많지 않았고, 이미 public 저장소라
과거 이력에 민감한 정보가 없는지도 다시 확인해야 했기 때문에 **새 히스토리로 시작**하는 쪽을 택했습니다.
(대신 기존 히스토리도 로컬 브랜치로 남겨뒀습니다.)

### 2) `pip freeze`가 안 먹힘 — venv 경로 이식성 문제

`venv/bin/pip freeze`를 실행했더니 `bad interpreter: No such file or directory` 에러가 났습니다.
`pip`, `pip3`, `ipython` 등 venv의 console script들을 열어보니 첫 줄(shebang)이 전부
`#!/Users/tina/Desktop/baseball-pitch-predictor/venv/bin/python3.13`으로 박혀 있었습니다.
`venv/pyvenv.cfg`를 확인해보니 이 venv는 애초에 `/Users/tina/Desktop/baseball-pitch-predictor`
경로에서 만들어졌던 것이었고, 그 뒤 프로젝트 폴더가 `/Users/tina/Project/baseball-pitch-predictor`로
옮겨지면서 이 절대경로가 더는 존재하지 않게 된 것이었습니다. venv를 만들 때 생성되는 `pip`/`ipython`
같은 콘솔 스크립트들은 인터프리터 경로가 shebang에 하드코딩되기 때문에, venv 폴더(또는 상위 프로젝트
폴더)를 옮기기만 해도 깨집니다. 반면 `venv/bin/python3.13` 자체는 시스템 Python으로의 심볼릭 링크라
경로 이동의 영향을 받지 않아 살아있었고, 그래서 `venv/bin/python3.13 -m pip ...`처럼 인터프리터를
직접 호출하면 우회할 수 있다는 것도 확인했습니다. 다만 이번 조사는 `pip`가 정상 동작하는지까지는
확인했고, `requirements.txt` 자체는 `site-packages`의 `*.dist-info` 폴더명에서 패키지명·버전을
직접 파싱하는 방식으로 만들었습니다. 근본적으로는 venv를 새로 만드는 게 제일 깔끔합니다.

### 3) `opencv-python`과 `opencv-python-headless`가 동시에 설치되어 있었음

두 패키지가 같은 `cv2` 모듈을 제공해서 같이 설치되면 어느 게 로드되는지가 설치 순서에 따라
달라지고, 서버 환경(GUI 없음)에서는 `opencv-python`이 불필요한 GUI 의존성 때문에 문제를 일으킬 수
있습니다. `requirements.txt`에는 `opencv-python-headless` 하나만 남겼습니다.

## 다음 단계 (Day 2 예고)

모델 코드(`feature_engineering.py`, `model.py`, `evaluate.py`)를 정리하고, 클래스 불균형 보정이나
시퀀스 길이 튜닝 같은 성능 개선 실험을 1~2개 시도해볼 예정입니다. BiLSTM+Embedding 구조를 왜 이렇게
설계했는지, 46.14%라는 수치가 8-클래스 분류 문제에서 어떤 의미인지도 자세히 다룰 예정입니다.
