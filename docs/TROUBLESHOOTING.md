# 트러블슈팅 기록 (PitchIQ / baseball-pitch-predictor)

이 프로젝트에서 실제로 발생했던 문제만 기록한다. 추측성 원인은 쓰지 않는다.
원본 근거: docs/blog/day1.md, docs/blog/day2.md, git 커밋 히스토리, 실제 코드.

## 인덱스

TS-010 (2026-08-05, Infra, 심각도 Medium, 상태 해결됨)
git worktree로 전환하기 직전 커밋 없이 작성한 파일(계획 문서)이 워크트리에는 보이지 않아 유실

TS-009 (2026-08-04, FE, 심각도 High, 상태 우회 적용)
Streamlit 컴포넌트 안에 YouTube iframe을 재중첩해 오류 153으로 영상 재생 자체가 안 됨

TS-008 (2026-08-04, FE, 심각도 Medium, 상태 해결됨)
시간 비례 싱크 전환 후에도 경계값(t≈0)에서 투구 인덱스가 실제보다 앞서 표시

TS-007 (2026-08-04, FE, 심각도 High, 상태 우회 적용)
OCR 이벤트 기반 투구-영상 싱크가 20% 커버리지 한계로 2구째부터 어긋남

TS-006 (2026-08-04, FE, 심각도 Medium, 상태 해결됨)
YouTube IFrame API 로딩 레이스 컨디션(YT is not defined)으로 자동재생 안 됨

TS-005 (2026-08-03, Infra, 심각도 Medium, 상태 해결됨)
Streamlit Cloud에 tesseract-ocr 바이너리가 없어 OCR 기능이 조용히 실패

TS-004 (2026-08-03, ML/Viz, 심각도 Low, 상태 해결됨)
헤드리스 환경에서 matplotlib 한글 그래프가 네모(tofu)로 깨짐

TS-003 (2026-07-27, Build, 심각도 Medium, 상태 해결됨)
opencv-python과 opencv-python-headless 동시 설치로 충돌 위험

TS-002 (2026-07-27, Build, 심각도 Medium, 상태 우회 적용)
프로젝트 폴더 이동 후 venv의 pip 콘솔 스크립트가 shebang 경로 깨짐

TS-001 (2026-07-27, Infra, 심각도 High, 상태 해결됨)
대용량 데이터/모델이 커밋 이력에 쌓여 .git이 2GB까지 비대화

영역 구분: FE / BE / ML / Viz / Build / Infra / Test
심각도 구분: Critical / High / Medium / Low
상태 구분: 해결됨 / 우회 적용 / 미해결

---

## TS-001 · 대용량 CSV·모델이 커밋 이력에 쌓여 .git이 2GB까지 비대화

날짜: 2026-07-27
영역: Infra
심각도: High
상태: 해결됨 (히스토리 재시작)

증상
커밋은 4개뿐인데 git count-objects -vH로 확인한 .git 크기가 로컬 기준 500MB 이상, 패킹된 상태로도 1.4GB를 넘었다.

재현 조건
각 커밋에 Statcast CSV(월별 6개, 전체 87MB), YOLO 사전학습 가중치, 학습된 모델(.h5)이 그대로 커밋되어 있던 상태에서 git count-objects -vH 실행.
재현율: 항상 (이 이력을 갖고 있는 한).

원인
표면 - .git이 비정상적으로 크다.
근본 - .gitignore를 나중에 추가해도 이미 커밋된 이력의 blob은 그대로 남는다. 매달 수십 MB짜리 CSV와 대용량 바이너리가 초기 커밋들에 반복적으로 쌓여 있었다.
확인 방법 - git count-objects -vH로 로컬/패킹 용량을 직접 측정.

시도했지만 안 된 것
시도: .gitignore만 추가
결과: 용량 그대로
이유: 이미 커밋된 과거 이력은 .gitignore의 영향을 받지 않는다

해결
두 선택지(git filter-repo로 과거 이력에서 대용량 blob 제거 / 현재 상태 기준으로 새 히스토리 시작) 중, 커밋이 4개뿐이라 잃을 게 적고 이미 public 저장소라 과거 이력에 민감정보가 없는지도 재확인해야 했기 때문에 새 히스토리로 시작하는 쪽을 선택했다. 기존 히스토리는 로컬 브랜치로 보존.
커밋: 85298e6 Restart project history with a clean, reproducible snapshot

검증
새 root 커밋 이후 서비스가 의존하는 모델 3개(pitch_predictor.h5, scaler.pkl, baseball_detector.pt)만 Git LFS로 추적되고, 재수집·재학습 가능한 대용량 산출물(Statcast CSV, YOLO 학습셋, 사전학습 체크포인트)은 .gitignore로 제외됨을 확인.

추후 관리
재발 방지 - .gitattributes에 LFS 추적 대상을 명시하고, .gitignore에 재생성 가능한 대용량 산출물을 등록해뒀다.
남은 리스크 - 새 기여자가 대용량 파일을 실수로 다시 커밋하면 동일 문제가 재발할 수 있다. pre-commit 파일 크기 훅은 아직 없음.

배운 점
.gitignore는 미래의 커밋만 막는다. 이미 이력에 들어간 대용량 파일은 별도로 제거하지 않는 한 저장소에 영구히 남는다. 커밋 수가 적을 때 정리하는 비용이 가장 낮다.

---

## TS-002 · 프로젝트 폴더 이동 후 venv의 pip가 shebang 경로 문제로 깨짐

날짜: 2026-07-27
영역: Build
심각도: Medium
상태: 우회 적용

증상
venv/bin/pip freeze를 실행하면 다음 에러가 났다.

bad interpreter: No such file or directory

재현 조건
환경: macOS, Python 3.13 venv
재현 절차: venv를 /Users/tina/Desktop/baseball-pitch-predictor에서 생성 → 프로젝트 폴더를 /Users/tina/Project/baseball-pitch-predictor로 이동 → venv/bin/pip freeze 실행
재현율: 항상

원인
표면 - pip 실행이 안 됨.
근본 - venv 생성 시 pip, pip3, ipython 등 콘솔 스크립트의 첫 줄(shebang)에 인터프리터의 절대경로가 하드코딩된다(#!/Users/tina/Desktop/.../venv/bin/python3.13). 폴더를 옮기면 이 경로가 더는 존재하지 않아 셸이 인터프리터를 못 찾는다.
확인 방법 - venv/pyvenv.cfg와 콘솔 스크립트 첫 줄을 직접 열어 옛 경로(/Desktop/...)가 박혀 있는 것을 확인.

시도했지만 안 된 것
시도: venv/bin/pip 직접 실행
결과: 동일 에러
이유: 스크립트 자체의 shebang이 깨진 것이라 실행 방식을 바꿔도 소용없음

해결
venv/bin/python3.13은 시스템 Python으로의 심볼릭 링크라 경로 이동의 영향을 받지 않는다는 점을 이용해, 인터프리터를 직접 호출하는 방식으로 우회했다.

venv/bin/python3.13 -m pip freeze

requirements.txt 자체는 pip freeze에 의존하지 않고, site-packages의 *.dist-info 폴더명에서 패키지명·버전을 직접 파싱해서 만들었다.

검증
위 명령으로 패키지 목록을 정상적으로 얻어 requirements.txt를 작성했다.

추후 관리
재발 방지 - 근본 해결은 아니다. venv 자체를 새로 만드는 것이 가장 깔끔한 해법으로 남아 있다.
남은 리스크 - 이 venv를 다시 옮기면 동일 증상이 재발한다.

배운 점
venv의 콘솔 스크립트(pip, ipython 등)는 인터프리터 경로가 생성 시점에 고정된다. venv 폴더(또는 상위 프로젝트 폴더)를 옮기면 항상 깨진다고 가정해야 하고, python -m pip 형태로 인터프리터를 직접 호출하면 임시 우회가 가능하다.

재발 (2026-08-05) - 같은 venv를 git worktree(`.claude/worktrees/pitch-type-cv-classifier/venv` 심볼릭 링크)에서 참조했을 때 `venv/bin/jupyter`, `venv/bin/pip`가 동일하게 `bad interpreter` 에러로 깨져 있는 것을 재확인. `venv/bin/python3 -m nbconvert`, `venv/bin/python3 -m pip`처럼 `python3 -m <module>` 형태로 우회해 해결 — 근본 원인과 우회법 모두 이 항목과 동일.

---

## TS-003 · opencv-python과 opencv-python-headless 동시 설치로 인한 충돌 위험

날짜: 2026-07-27
영역: Build
심각도: Medium
상태: 해결됨

증상
의존성 점검 중 opencv-python과 opencv-python-headless가 같은 환경에 동시에 설치되어 있는 것을 발견했다. 둘 다 cv2 모듈을 제공한다.

원인
표면 - 같은 모듈(cv2)을 제공하는 패키지가 두 개 설치되어 있다.
근본 - 어느 패키지가 실제로 로드되는지가 설치 순서에 따라 달라진다. GUI가 없는 서버/배포 환경(Streamlit Cloud 등)에서는 opencv-python이 불필요한 GUI 의존성(X11 등) 때문에 임포트 에러를 일으킬 수 있다.

해결
requirements.txt에 opencv-python-headless 하나만 남기고 주석으로 이유를 명시했다.

opencv-python-headless==4.10.0.84
(주석: opencv-python 과 opencv-python-headless 가 동시에 설치되어 있으면 로드되는 cv2가 설치 순서에 따라 달라질 수 있음. GUI가 필요 없으므로 headless 하나만 남기는 것을 권장)

검증
pip list에서 opencv-python이 없고 opencv-python-headless만 남아 있는 것을 확인.

추후 관리
재발 방지 - 새 의존성 추가 시 pip list | grep opencv로 중복 설치 여부를 확인하는 습관을 남겼다.

배운 점
같은 최상위 모듈(cv2)을 제공하는 패키지가 여러 개 설치돼도 pip나 import는 에러를 내지 않는다. 배포 환경(GUI 없음)을 기준으로 어떤 변형이 필요한지 미리 결정해야 한다.

---

## TS-004 · 헤드리스 환경에서 matplotlib 한글 그래프가 네모(tofu)로 깨짐

날짜: 2026-08-03
영역: ML/Viz
심각도: Low
상태: 해결됨

증상
src/evaluate.py로 혼동행렬·클래스별 정확도 그래프를 재생성했더니, 그래프의 한글 라벨(구종명, "정확도" 등)이 전부 빈 네모(tofu)로 나왔다.

재현 조건
환경: macOS 로컬에서는 발생하지 않고, CI/서버처럼 시스템 한글 폰트가 없는 헤드리스 환경에서 재현.
재현 절차: plt.rcParams["font.family"]를 지정하지 않은 채 한글 라벨이 포함된 그래프 생성.
재현율: 항상 (한글 폰트 미탑재 환경에서).

원인
표면 - 그래프의 한글 텍스트가 깨짐.
근본 - matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프가 없다. 로컬 macOS에서는 matplotlib이 시스템 폰트(AppleGothic 등)를 알아서 찾아 쓰기 때문에 문제가 드러나지 않았을 뿐이다.
확인 방법 - 로컬(정상)과 폰트가 다른 환경을 비교해 동일 코드에서 결과가 갈리는 것을 확인.

해결
src/evaluate.py에 폰트 우선순위를 지정하고, 마이너스 기호 깨짐도 함께 방지했다.

plt.rcParams["font.family"] = ["NanumGothic", "AppleGothic", "Malgun Gothic"]
plt.rcParams["axes.unicode_minus"] = False

검증
그래프를 재생성해 한글 라벨이 정상 출력되는 것을 확인 (notebooks/figures/07_confusion_matrix.png 등).

추후 관리
재발 방지 - 우선순위 리스트 방식이라 실행 환경에 셋 중 하나만 있어도 동작한다.
남은 리스크 - Streamlit Cloud 배포 환경에는 위 폰트가 기본으로 없을 수 있다. 배포 시 폰트 설치 단계가 필요하다는 것을 로드맵에 남겨뒀다(docs/ROADMAP.md).

배운 점
로컬에서만 테스트하면 시스템에 우연히 있던 것 덕분에 숨어 있는 문제를 놓치기 쉽다. 다국어 텍스트가 들어가는 시각화는 폰트를 명시적으로 지정해야 배포 환경에서도 동일하게 보인다.

---

## TS-005 · Streamlit Cloud에 tesseract-ocr 바이너리가 없어 OCR 기능이 조용히 실패

날짜: 2026-08-03
영역: Infra
심각도: Medium
상태: 해결됨

증상
로컬에서는 방송 오버레이 OCR(pose_detector.ocr_check_pitch_overlay, scan_pitch_overlays)이 정상 동작하는데, Streamlit Cloud에 배포하면 OCR 관련 기능이 크래시 없이 조용히 실패할 수 있는 상태였다.

재현 조건
환경: Streamlit Cloud (Debian 기반 컨테이너), requirements.txt만 있고 packages.txt는 없는 상태.
재현 절차: pytesseract가 파이썬 패키지로는 설치되지만, pytesseract는 시스템에 설치된 tesseract 바이너리를 서브프로세스로 호출하는 얇은 wrapper다.
재현율: 항상 (시스템 바이너리가 없는 배포 환경에서).

원인
표면 - pip install pytesseract가 성공해도 OCR이 동작하지 않을 수 있다.
근본 - requirements.txt 26번째 줄에 이미 남겨둔 주석대로, pytesseract는 시스템에 tesseract 바이너리가 별도로 설치되어 있어야 한다(brew install tesseract 등). 로컬 macOS에는 Homebrew로 이미 설치돼 있어서 문제를 못 느꼈지만, Streamlit Cloud는 이 바이너리를 기본 제공하지 않는다.
확인 방법 - requirements.txt의 기존 주석과 streamlit_app/app.py의 _run_ocr_check_bg()가 pytesseract 호출을 try/except Exception으로 감싸 {"status": "error", "error": str(e)}로 저장하도록 되어 있는 것을 확인. 즉 실패해도 앱 전체는 죽지 않고 OCR 결과만 조용히 비게 되는 구조였다.

시도했지만 안 된 것
해당 없음 — requirements.txt의 사전 주석 덕분에 원인을 바로 특정할 수 있었다.

해결
Streamlit Cloud는 packages.txt에 적힌 패키지를 apt-get으로 설치해준다는 점을 이용해, 루트에 packages.txt를 추가했다.

tesseract-ocr

커밋: 889da2f chore: Streamlit Cloud 배포용 packages.txt 추가 (tesseract-ocr)

검증
배포 후 README에 라이브 데모 링크를 추가(51b93e1)하고 실제 배포 URL에서 기능이 동작하는 것을 확인.

추후 관리
재발 방지 - Python 패키지가 시스템 바이너리에 의존하는 경우(pytesseract, ffmpeg 등) requirements.txt 주석뿐 아니라 packages.txt도 같이 챙기는 것을 배포 체크리스트에 남긴다.
남은 리스크 - README에 명시했듯, YouTube 영상 연동·실시간 스캔은 무료 배포 환경의 리소스 제약으로 여전히 로컬만큼 안정적이지 않을 수 있다.

배운 점
Python 패키지가 설치됐다는 것과 그 기능이 동작한다는 것은 다르다. 특히 OCR·이미지·오디오 처리 라이브러리는 서브프로세스로 시스템 바이너리를 호출하는 경우가 많아서, 로컬에 이미 깔려 있던 시스템 의존성을 배포 환경 설정(packages.txt, Dockerfile 등)에 옮기는 걸 잊기 쉽다. 이런 실패는 예외를 삼키는 코드와 만나면 에러 로그도 없이 그냥 안 되는 상태로만 보인다.

참고
Streamlit Cloud packages.txt 문서: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

---

## TS-006 · YouTube IFrame API 로딩 레이스 컨디션(YT is not defined)으로 자동재생 안 됨

날짜: 2026-08-04
영역: FE
심각도: Medium
상태: 해결됨 (이후 다른 원인으로 재발 — TS-009 참고)

증상
페이지에 임베드된 YouTube 영상이 자동재생되지 않았다. 브라우저 콘솔 에러:

ReferenceError: YT is not defined

재현 조건
환경: streamlit_app/youtube_player/frontend/index.html (Streamlit 커스텀 컴포넌트).
재현 절차: 컴포넌트 로드 시 YouTube IFrame API 스크립트(https://www.youtube.com/iframe_api)가 비동기로 로드되는 도중 플레이어 초기화 코드가 먼저 실행됨.
재현율: 항상 (스크립트 로딩 타이밍 구조 자체의 문제).

원인
표면 - ReferenceError: YT is not defined.
근본 - componentArgs && componentArgs.video_id && !player 조건만으로 initPlayer()를 호출했는데, 이 시점에 비동기 로드되는 window.YT(IFrame API 전역 객체)가 아직 없을 수 있었다. 또한 playerVars에 autoplay/mute가 아예 없어 API가 정상 로드돼도 자동재생되지 않았다.
확인 방법 - 브라우저 콘솔의 정확한 에러 메시지가 원인(API 로딩 타이밍)을 직접 지목.

시도했지만 안 된 것
시도: claude-in-chrome 자동화 탭에서 곧바로 재생 여부 확인
결과: 재생 상태 불확실
이유: 자동화 탭이 document.hasFocus()===false / visibilityState==='hidden' 상태라 브라우저 자체 동영상 스로틀링에 걸려, 수정이 실제로 동작하는지와 자동화 환경의 한계인지 구분이 안 됨.

해결
streamlit_app/youtube_player/frontend/index.html 두 곳 수정:
1. if (componentArgs && componentArgs.video_id && !player) → ... && window.YT && window.YT.Player 로 API 로드 완료 가드 추가.
2. playerVars에 autoplay: 1, mute: 1 추가 (브라우저 자동재생 정책상 음소거 없이는 자동재생 불가).

검증
독립 테스트 HTML(yt_test.html)로 플레이어 단독 로직 검증 — onReady 이후 state=3(buffering) 도달 확인. 컴포넌트 파일을 최상위 페이지로 직접 로드해 중첩 iframe 없는 환경에서 정상 동작 확인. 자동화 환경 한계로 완전한 시각 확인은 못 해 사용자에게 공개 후, 실제 배포 환경에서 사용자가 최종 확인("아 되네").

추후 관리
재발 방지 - 없음.
남은 리스크 - 레이스 컨디션 자체는 해결됐으나, 이후 별도 원인(이중 iframe 구조에서 발생하는 유튜브 오류 153)으로 영상 재생이 다시 안 되는 문제가 재발함(TS-009). 결과적으로 이 컴포넌트 자체를 폐기하고 st.video()로 교체.

배운 점
서드파티 JS API를 비동기 스크립트 태그로 로드할 때는, 전역 콜백뿐 아니라 다른 트리거로도 초기화 함수가 호출될 수 있는 모든 경로에 전역 객체 존재 여부 가드를 걸어야 한다. 브라우저 자동화 도구로 비디오 자동재생을 검증할 때는 탭의 포커스/가시성 상태가 실제 사용자 환경과 다를 수 있다는 한계를 스스로 인지하고 사용자에게 투명하게 공개해야 한다.

---

## TS-007 · OCR 이벤트 기반 투구-영상 싱크가 20% 커버리지 한계로 2구째부터 어긋남

날짜: 2026-08-04
영역: FE
심각도: High
상태: 우회 적용

증상
1구는 정확했지만 2구째부터 표시되는 구종/싱크가 실제 영상과 어긋났다.

재현 조건
환경: Streamlit Community Cloud 배포, 고정 데모 모드(game_pk=775300, 고정 YouTube 영상).
재현 절차: 영상 재생하며 2번째 투구 이후 표시되는 구종/구속을 실제 중계와 대조.
재현율: 항상 (OCR 커버리지가 구조적으로 낮음).

원인
표면 - 특정 투구부터 표시되는 싱크가 실제와 안 맞음.
근본 - 정적 OCR 스캔이 320구 중 65개(~20%)만 감지했다. 감지 이벤트 사이 간격 동안 실제로는 여러 투구가 지나가는 경우가 흔해, 이벤트 기반 순차/점프 로직으로는 근본적으로 정확한 싱크가 불가능한 구조였다.
확인 방법 - "1~5이닝 캡이 조기 종료 원인"이라는 가설로 OCR 스캔을 이닝 제한 없이 재생성했으나, 결과 파일이 이전과 MD5 완전 동일 → 가설 기각. 캡이 아니라 OCR 인식률 자체가 낮은 것이 원인임을 확인.

시도했지만 안 된 것
시도: OCR 정적 스캔 타임스탬프 기반 순차(+1) 싱크, "P:N" 카운트 점프 감지
결과: 1구는 맞았지만 2구부터 어긋남
이유: 320구 중 65개(~20%)만 감지되는 낮은 OCR 커버리지 — 감지 간 간격이 실제로는 여러 투구를 건너뛰는 경우가 흔함

시도: 이닝 캡(max_pitches=140)이 조기 종료 원인이라는 가설로 OCR 스캔 재생성
결과: 재생성 파일이 이전과 MD5 동일 — 가설 기각
이유: 애초에 이닝 캡에 도달한 적이 없었고, 진짜 원인은 OCR 자체의 낮은 인식률이었음

해결
근본적으로 OCR 이벤트 기반 싱크를 포기하고 "영상 재생시간 ÷ 전체 길이 × 총 투구수"로 인덱스를 추정하는 시간 비례 방식으로 전환했다 (streamlit_app/app.py, 커밋 2112ebc). 이 전환 직후 남아있던 경계값 잔여 버그 2건은 TS-008 참고.

검증
배포 사이트에서 투구가 끊기거나 한 인덱스에 고정되지 않고 매끄럽게 진행되는 것을 확인.

추후 관리
재발 방지 - 없음.
남은 리스크 - 시간 비례 방식은 프레임 단위 정확도를 포기한 근사치다. 이닝별로 투구 간 실제 간격이 크게 다를 수 있어(공수교대, 중계 지연 등) 표시되는 투구가 점점 실제와 벌어질 수 있다 — 사용자가 인지하고 승인한 트레이드오프.

배운 점
감지 커버리지가 낮은 이벤트 기반 시스템은 "이벤트 사이 간격 동안 상태가 그대로"라고 가정하면 안 된다. 원인 불명의 증상을 만났을 때 가장 그럴듯한 가설(이닝 캡)부터 검증했지만 틀렸고, 재생성 파일의 MD5를 비교하는 저비용 검증으로 가설을 빠르게 기각할 수 있었다.

---

## TS-008 · 시간 비례 싱크 전환 후에도 경계값(t≈0)에서 투구 인덱스가 실제보다 앞서 표시

날짜: 2026-08-04
영역: FE
심각도: Medium
상태: 해결됨

증상
TS-007에서 시간 비례 방식으로 전환한 뒤에도: (1) 영상 재생이 시작되지 않았는데도(영상 시간 0초, playing:false) "방금 던진 구종" 카드에 이미 데이터가 표시됨. (1차 수정 후) (2) 재생을 막 시작한 시점(거의 0초)에도 투구 인덱스가 0→1로 즉시 넘어가 "공 던지지도 않았는데 투구 타임라인에 1로 표시되고 방금 던진 구종이 뜬다"는 사용자 재보고가 있었다.

재현 조건
환경: Streamlit Community Cloud 배포, 고정 데모 모드.
재현 절차: 페이지 새로고침 → 자동 로드 완료 직후 / 재생 시작 직후 화면 확인.
재현율: 항상 (조건이 성립하면 매번).

원인
표면 - 영상이 거의 진행되지 않았는데 투구 인덱스가 이미 앞서 있음.
근본(1차) - YouTube 플레이어가 onReady 직후, 재생이 시작되기 전에도 {time:0, playing:false} 이벤트를 먼저 보내는데, 싱크 로직이 _vid_t is not None이기만 하면(실제 재생 여부와 무관하게) 인덱스를 진행시키도록 되어 있었다.
근본(2차, 1차 수정 후에도 재발) - 시간 비례 인덱스 계산에 target_idx + 1이라는 "선행 오프셋"이 있어, 재생이 막 시작돼 _vid_t가 0에 가까운 값이어도 인덱스가 무조건 최소 1로 올라갔다.
확인 방법 - 코드를 직접 추적해 _vid_t/_vid_pl 세션 상태 갱신 순서와 조건문을 특정. 수정 후 배포 사이트 스크린샷으로 "투구 타임라인 0" 상태가 재생 시작 전까지 유지되는 것을 확인.

시도했지만 안 된 것
시도: _vid_pl(재생 중 여부) 게이팅만 추가
결과: 재생 시작 "전" 오작동은 해결됐지만 시작 "직후" 오작동은 재발
이유: target_idx + 1 선행 오프셋이 별개 원인으로 남아 있었음

해결
streamlit_app/app.py 두 곳 수정:
1. _vid_t is not None and loaded → _vid_t is not None and loaded and _vid_pl 로 조건 강화 (커밋 0f9822b)
2. _new_cidx_ts = min(_target_idx + 1, len(pitches)-1) → _new_cidx_ts = max(0, min(_target_idx, len(pitches)-1)) 로 선행 오프셋 제거 (커밋 566aed8)

검증
배포 사이트 새로고침 직후 스크린샷 — "투구 타임라인 0" 유지, "방금 던진 구종" 카드는 "경기 로드 후 재생" placeholder만 표시되는 것 확인.

추후 관리
재발 방지 - 없음 (자동 테스트 없이 수동 스크린샷 검증에 의존).
남은 리스크 - 없음 (경계값 버그 자체는 완전히 해결. 시간 비례 방식 자체의 근사치 한계는 TS-007 참고).

배운 점
값이 None이 아니라는 것만으로 "이미 시작됐다"고 판단하면 초기화 시점의 더미/초기값 이벤트에도 반응해버린다. "1구 앞서가게" 같은 휴리스틱 오프셋은 항상 경계값(0, 시작 시점)에서 의도치 않은 부작용을 만들 가능성이 크므로, 도입 전에 t=0 케이스를 먼저 손으로 검증해야 한다.

---

## TS-009 · Streamlit 컴포넌트 안에 YouTube iframe을 재중첩해 오류 153으로 영상 재생 자체가 안 됨

날짜: 2026-08-04
영역: FE
심각도: High
상태: 우회 적용

증상
배포된 Streamlit 앱에서 영상 영역이 완전히 검은 화면으로 나오고, 재생 버튼도 안 보이고 재생도 안 됐다. 사용자 보고: "영상 재생바 안뜨고 / 영상 재생도 안됏는데 바로 방금던진구종이 뜸" (이후 "이번엔 영상 재생이 아예 안된느데?"로 재확인).

오류 153 / 동영상 플레이어 구성 오류 (embed URL 직접 접속 시 유튜브 네이티브 오류 화면)

재현 조건
환경: Streamlit Community Cloud 배포, streamlit_app/youtube_player 커스텀 컴포넌트 (Streamlit 컴포넌트 iframe 안에 YouTube IFrame API iframe을 다시 중첩하는 구조).
재현 절차: 배포 URL 접속 → 고정 데모 자동 로드 → 영상 영역 확인.
재현율: 이 세션에서는 100% (자동화 브라우저, 배포 페이지에서 실제 사용된 embed URL을 직접 열어도 동일 재현). 이전 라운드에는 정상 재생됐다는 보고("아 되네")도 있어 간헐적 성격도 있음.

원인
표면 - 유튜브 iframe 자체가 렌더링되지 않고 완전히 검은 화면으로 남음.
근본 - youtube_player 커스텀 컴포넌트가 이미 Streamlit이 만든 컴포넌트 iframe 안에서 다시 YouTube IFrame API로 또 다른 iframe을 로드하는 "iframe 안의 iframe" 구조였다. 배포 페이지에서 실제 network 요청으로 사용된 embed URL(정확한 origin/referrer/enablejsapi 파라미터 포함)을 그대로 브라우저에서 직접 열어보니 유튜브가 "오류 153: 동영상 플레이어 구성 오류"를 반환 — 이중 iframe 중첩 상황에서 origin 파라미터가 실제 최상위 origin과 일치하지 않아 유튜브 측 검증에 실패하는 것으로 추정.
확인 방법 - (1) 배포 페이지의 network 요청에서 실제 embed URL을 캡처, (2) 그 URL을 그대로 브라우저에서 열어 유튜브 네이티브 오류 화면(오류 153)을 재현 확인.

시도했지만 안 된 것
시도: TS-006의 레이스 컨디션/autoplay 수정 (window.YT 가드 + playerVars autoplay/mute)
결과: 단독 테스트 페이지에서는 재생 상태 도달, 배포 환경에서는 여전히 검은 화면
이유: 레이스 컨디션은 별개 문제였고, 오류 153의 근본 원인(이중 iframe origin 불일치)은 그대로 남아 있었음

시도: origin/referrer 파라미터를 애플리케이션 코드에서 직접 조정
결과: 검토 후 폐기, 시도하지 않음
이유: Streamlit 컴포넌트 iframe이 생성하는 origin은 Streamlit 프레임워크 내부 영역이라 애플리케이션 코드로 제어 불가능

해결
streamlit_app/app.py의 영상 렌더링 분기 수정 — 로컬 파일 재생(local_video_player, 로컬 개발 전용)은 그대로 두고, YouTube 재생 경로만 커스텀 youtube_player 컴포넌트 대신 Streamlit 기본 st.video(url, autoplay=True, muted=True)로 교체. 더 이상 쓰이지 않는 _yt_id()/_yt 제거. (커밋 f24213d)

검증
배포 사이트 스크린샷으로 유튜브 네이티브 플레이어(제목·재생바·컨트롤 포함)가 정상 렌더링되는 것을 확인.

추후 관리
재발 방지 - 없음 — 근본 원인(Streamlit 컴포넌트 iframe 안에서의 YouTube IFrame API 중첩 자체)은 회피했을 뿐 고치지 않음.
남은 리스크 - 실시간 재생 시간 이벤트를 더 이상 받을 수 없어, 자동 투구 싱크(시간 비례 방식, TS-007/TS-008 참고)가 YouTube 경로에서는 동작하지 않게 됨 — 수동 슬라이더/이전·다음 버튼으로 대체. 향후 실시간 싱크가 다시 필요해지면 iframe 중첩이 없는 방식(단일 iframe에 직접 YT.Player 마운트, 또는 서버 사이드 폴링)을 검토해야 한다.

배운 점
Streamlit components.v1.declare_component로 만든 컴포넌트는 그 자체가 이미 하나의 iframe이라, 그 안에서 또 다른 서드파티 iframe(YouTube 등)을 API 기반으로 제어하려 하면 "iframe 안의 iframe"이 되어 origin 검증에 실패할 수 있다. 서드파티 임베드는 가능하면 최상위 페이지에 가장 가까운 곳에서 렌더링하거나, 프레임워크가 제공하는 네이티브 위젯이 있다면 커스텀 컴포넌트보다 그걸 먼저 검토하는 게 안전하다.

---

## TS-010 · git worktree로 전환하기 직전 커밋 없이 작성한 파일이 워크트리에는 보이지 않아 유실

날짜: 2026-08-05
영역: Infra
심각도: Medium
상태: 해결됨

증상
브레인스토밍 스킬로 설계 문서를 작성·커밋한 뒤, 이어서 구현 계획 문서(`docs/superpowers/plans/2026-08-05-pitch-type-cv-classifier-pilot.md`)를 메인 체크아웃에 Write로 작성했다. 곧바로 `EnterWorktree`로 격리된 작업공간으로 전환한 뒤 `sdd-workspace` 스크립트로 그 계획 파일을 읽으려 하자 `no such plan file` 에러가 났다.

재현 조건
환경: 같은 저장소 안에서 Write 도구로 새 파일 생성 → git add/commit 하지 않은 상태 → EnterWorktree(name=...)로 새 워크트리 진입.
재현율: 항상 (커밋 없이 작성한 파일이 있는 상태에서 워크트리 전환 시).

원인
표면 - 방금 만든 파일이 새 워크트리 디렉터리에 없다.
근본 - git worktree는 같은 `.git` 객체 저장소를 공유하지만 워크트리마다 독립된 작업 디렉터리(파일시스템)를 갖는다. 커밋되지 않은 워킹트리 변경분은 git 객체가 아니라 순수 파일시스템 상태이므로, 다른 워크트리에서는 애초에 존재할 방법이 없다. `EnterWorktree`는 브랜치(커밋 이력) 기준으로 워크트리를 만들 뿐, 원본 체크아웃의 미커밋 변경분을 복사해오지 않는다.
확인 방법 - 워크트리 진입 후 `ls docs/superpowers/plans/`로 해당 파일이 없는 것을 확인, `git status --short`로 원본 체크아웃 쪽에는 아직 untracked 상태로 남아있었을 것으로 추정(직접 재확인은 샌드박스 제약으로 워크트리 세션에서 다른 워크트리 경로에 대해 `git -C`를 쓸 수 없어 못함).

시도했지만 안 된 것
없음 — 원인이 명확해서 바로 우회로 넘어감.

해결
같은 내용을 워크트리 안에서 Write로 다시 작성하고 그 자리에서 커밋했다. 이후 `sdd-workspace`/`task-brief` 스크립트가 정상적으로 파일을 찾음.

검증
`sdd-workspace docs/superpowers/plans/...pilot.md` 재실행 시 정상적으로 워크스페이스 경로 출력, 이후 태스크별 `task-brief` 생성도 문제없이 진행됨.

추후 관리
재발 방지 - 앞으로 이런 워크플로(브레인스토밍 → writing-plans → subagent-driven-development)를 이어서 진행할 때는, 워크트리 전환 직전에 만든 파일은 반드시 커밋까지 마치고 나서 EnterWorktree를 호출한다.
남은 리스크 - 없음. 다만 workflow 스킬 문서 자체에는 이 순서 의존성이 명시돼 있지 않아, 다른 세션에서도 동일하게 재발할 수 있다.

배운 점
git worktree는 "같은 저장소의 다른 브랜치를 보는 창"이지 "같은 작업 디렉터리의 스냅샷"이 아니다. 커밋되지 않은 파일은 그 워크트리에만 속한다. 워크트리 도구(EnterWorktree 등)로 전환하기 직전에는 "지금 만든 파일을 커밋했는가"를 항상 체크리스트로 확인해야 한다.
