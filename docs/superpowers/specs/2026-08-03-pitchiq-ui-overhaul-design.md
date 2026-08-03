# PitchIQ Streamlit UI 전면 개편 — 설계

## 배경

`streamlit_app/app.py` (1,537줄)의 UI가 다닥다닥 붙어있고 글씨가 작아 가독성이 떨어짐.
게임 미로드 상태의 화면도 빈 placeholder 수준이라 첫인상이 약함. 정적이고 호버 인터랙션이 없음.

## 목표

1. 게임 미로드 상태를 제대로 된 랜딩 화면으로 교체
2. 전역 타이포그래피·여백 스케일을 키워 밀도 문제 해소
3. 지정된 4개 영역에 호버 인터랙션 추가

## 비범위 (Non-goals)

- 영상 동기화 / OCR / 스레딩 등 비즈니스 로직 변경 금지
- 색상·다크 테마 방향 변경 금지 (기존 네이비/블루·퍼플 그라데이션 유지)
- session_state 구조 변경 금지
- 백엔드(BiLSTM, YOLO) 로직 변경 금지

## A. 랜딩 화면 (`loaded == False`일 때)

현재: 영상 영역에 작은 `🎬` placeholder, 우측 패널에 "경기를 먼저 로드하세요" 텍스트만 존재 (app.py:1140-1149, 1220-1224).

교체안: `col_video` + `col_panel` 두 영역에 나눠 그리던 것 대신, 게임이 로드되지 않았을 때는 전체 폭을 쓰는 히어로 섹션 하나로 렌더링.

- 상단: 로고/타이틀(⚾ PitchIQ) + 태그라인 1줄 + 모델 정확도(48.5%) 강조 배지
- 기능 소개 카드 3~4개 (그리드): 구종 예측 / 실시간 영상 동기화 / 구종 분포 분석 / (선택) YOLO 투구 감지
  - 기본 상태: 아이콘 + 제목만 노출
  - hover 상태: 설명 텍스트가 아래로 슬라이드다운 (순수 CSS `max-height`/`opacity` 트랜지션, Python rerun 없음)
- 하단: "← 사이드바에서 game_pk를 입력하고 경기 로드를 눌러보세요" 안내 + 예시 game_pk(745735) 표기

`loaded == True`일 때의 기존 2단 레이아웃(`col_video`/`col_panel`)은 그대로 유지.

## B. 타이포그래피 · 여백 스케일

CSS 블록(app.py:22-74)에 스케일 조정. 절대값 변경 예시 (기존 → 신규):

| 요소 | 기존 | 신규 |
|---|---|---|
| `.panel-title` | 0.62rem | 0.72rem |
| `.player-name` | 1rem | 1.15rem |
| `.pitch-code` | 2rem | 2.4rem |
| `.pitch-name` | 0.75rem | 0.85rem |
| `.panel` padding | .85rem 1rem | 1.1rem 1.3rem |
| `.panel` margin-bottom | .6rem | .85rem |
| `.pitch-card` padding | .7rem 1rem | .95rem 1.25rem |

인라인 스타일로 하드코딩된 폰트 크기(`font-size:.6x rem` 형태)도 같은 비율(대략 +15~20%)로 조정. 색상값·레이아웃 구조(columns 비율 등)는 유지.

## C. 호버 인터랙션 (4개 영역)

1. **랜딩 기능 카드** — CSS `:hover`로 설명 영역 `max-height:0→60px`, `opacity:0→1` 트랜지션.
2. **다음 구종 예측 확률 막대(Plotly bar, app.py:1440)** — `hovertemplate`을 추가해 구종 코드·한글명·정확한 확률(%) 툴팁 표시. 현재 파이차트(app.py:1468)에는 이미 있으므로 동일 패턴 적용.
3. **투구 타임라인 / 최근 투구 행(`.pitch-row`, app.py:1194 이하)** — hover 시 배경 강조 + 좌측 보더 액센트, 구속/타석결과 상세 정보를 hover 시 노출(순수 CSS).
4. **통계 카드 전반** (하단 4개 카드 app.py:1486-1510, `.panel` 카드) — hover 시 `border-color` glow + `translateY(-2px)` lift, `transition` 적용.

## 구현 파일

- `streamlit_app/app.py` 만 수정 (CSS 블록 확장 + 랜딩 섹션 함수 추가 + 기존 인라인 스타일 폰트/패딩 값 조정)
- 신규 파일 없음

## 검증

- `streamlit run streamlit_app/app.py`로 로컬 실행
  - 게임 미로드 상태: 랜딩 화면 렌더링 확인, 기능 카드 hover 시 설명 노출 확인
  - 게임 로드 후: 기존 기능(예측, 타임라인, 통계) 정상 동작 확인, 폰트/여백 커진 것 육안 확인, 각 hover 대상 동작 확인
- 브라우저 스크린샷으로 개편 전/후 비교
