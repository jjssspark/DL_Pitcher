# PitchIQ 게임 HUD 스타일 UI 강화 — 설계

## 배경

방송 중계 스코어보드 개편(`2026-08-04-broadcast-ui-redesign-design.md`) 이후에도 "더 직관적이고
게임처럼, 더 야구서비스 같고, 더 동적이었으면 좋겠다"는 요청을 받음. 스타일 방향은 스포츠 게임
HUD(MLB The Show류 인게임 UI), 기존 색상(네이비·퍼플·골드)과 레이아웃 골격은 유지, 아래 3가지
동적 요소를 추가하기로 확정:
1. 투구 갱신 시 리빌/카드 애니메이션
2. 실시간감 있는 상태표시(LIVE 배지)
3. 예측 적중/연속 항기(스트릭) 시각화

## 목표

1. "다음 투구 예측" 카드를 게임 HUD 스탯 게이지처럼 시각화
2. 연속 예측 적중을 COMBO 배지로 게임화
3. 투구가 바뀔 때 카드가 자연스럽게 나타나는 애니메이션
4. 스코어보드에 LIVE 감을 주는 펄스 배지
5. 예측 적중/실패 순간의 시각적 피드백 강화

## 비범위 (Non-goals)

- 레이아웃 구조(2단 컬럼, 카드 배치 순서) 변경 없음
- 색상 팔레트·타이포그래피 재정의 없음 (기존 네이비/퍼플/골드 유지)
- 하단 구종별 확률 막대그래프는 그대로 유지 — 원형 게이지는 "예측 카드"에만 추가
- 사이드바, 스코어보드 팀컬러/세그먼트 카운트 등 이전 개편 요소는 변경하지 않음
- 외부 JS/애니메이션 라이브러리 도입 없음 (순수 CSS만 사용)
- `session_state` 필드 추가는 허용하되 기존 필드 의미 변경 금지

## A. 예측 카드 — 원형 신뢰도 게이지

`_predict_next`/BiLSTM 결과의 `confidence`(0~1)를 텍스트 `%` 대신 원형 게이지로 표시한다.

- 구현: `conic-gradient`로 링을 그리는 `div` + 안쪽에 카드 배경색 원을 겹쳐 도넛 형태를
  만든다 (`mask`/`border-radius` 트릭, 외부 라이브러리 불필요).
  ```css
  .conf-gauge {
    width: 84px; height: 84px; border-radius: 50%; position: relative;
    background: conic-gradient(var(--gauge-color) calc(var(--pct) * 3.6deg), rgba(255,255,255,.08) 0deg);
    display: flex; align-items: center; justify-content: center;
  }
  .conf-gauge::before {
    content: ""; position: absolute; width: 64px; height: 64px; border-radius: 50%;
    background: #0f172a;
  }
  ```
  `--pct`와 `--gauge-color`는 인라인 style로 파이썬에서 주입 (`_cf*100`, `_cc` 기존 변수 재사용).
- 게이지 중앙에 구종 코드(`FF` 등)를 배치, 게이지 아래/옆에 기존 텍스트(구종명, 신뢰도 %, 근거
  문구)는 그대로 둔다 — 기존 `pred-hero` 카드 내부 레이아웃을 좌우 배치(게이지 | 텍스트)로 조정.
- 색상은 기존 `_cc`(신뢰도 구간별 초록/주황/빨강) 로직 그대로 재사용.

## B. COMBO 스트릭 배지

- 신규 세션 상태: `pred_streak`(int, 기본 0), `_streak_calc_idx`(int, 기본 -1 — 마지막으로 스트릭을
  계산한 `c_idx`, 재렌더 시 중복 계산 방지용).
- "방금 던진 구종" 카드 블록의 기존 적중 판정(`_hit = _prev_pred_type == prev["pitch_type"]`) 직후,
  `c_idx != st.session_state._streak_calc_idx`일 때만:
  - 적중이면 `pred_streak += 1`
  - 빗나가면 `pred_streak = 0`
  - `_streak_calc_idx = c_idx`로 갱신
- `pred_streak >= 2`일 때만 예측 카드 상단에 `🔥 COMBO x{n}` 배지를 표시 (pop-in CSS 애니메이션).
  스트릭이 끊겼을 때 별도의 리셋 애니메이션은 넣지 않는다 — 배지가 조건 미충족으로 자연스럽게
  사라지는 것으로 충분 (불필요한 복잡도 배제).

## C. 투구 갱신 애니메이션

- "방금 던진 구종" 카드와 예측 카드에 `@keyframes cardReveal`(살짝 아래→위 슬라이드 + 페이드인,
  `0.35s ease-out`)을 적용.
- 별도 JS나 컴포넌트 키 조작 없이도 동작하는 이유: 두 카드는 매번 `st.markdown(..., unsafe_allow_html=True)`
  로 새로 렌더링되는데, `c_idx`가 바뀌면 텍스트 내용(구종/구속/신뢰도 등)이 달라져 Streamlit
  프론트엔드가 DOM 노드를 교체 → CSS 애니메이션이 자연 재생된다. `c_idx`가 그대로인 채 다른 위젯
  조작으로 rerun이 일어나면 내용이 동일해 노드가 재생성되지 않아 애니메이션도 재생되지 않는다 —
  "투구가 실제로 바뀔 때만" 애니메이션이 트리거되는 원하는 동작과 일치.

## D. LIVE 상태 배지

- 스코어보드(`.scoreboard` 블록) 좌상단 또는 팀명 옆에 `🔴 LIVE` 배지 추가.
- `@keyframes livePulse` — `opacity`와 `transform: scale()`를 0.6↔1 사이로 1.5s 주기 반복 (컴포지터
  친화적 속성만 사용).

## E. 적중/실패 카드 글로우

- 기존 "✓ 예측 적중" / "✗ 빗나감" 텍스트 배지가 표시되는 시점(위 B와 동일한 `_hit` 판정)에 맞춰
  "방금 던진 구종" 카드 테두리에 1회성 글로우를 추가:
  - 적중: `@keyframes glowHit` — 초록(`#34d399`) `box-shadow` 펄스 1회.
  - 실패: `@keyframes glowMiss` — 옅은 빨강(`#f87171`) `box-shadow` 플래시 1회.
- 카드가 갱신될 때만 재생되는 원리는 C와 동일 (내용이 바뀌어야 노드가 재생성됨).

## 구현 파일

- `streamlit_app/app.py`만 수정 (CSS 블록에 5개 `@keyframes` 및 관련 클래스 추가, 예측 카드
  마크업을 게이지+텍스트 레이아웃으로 조정, 스트릭 계산 로직 추가, LIVE 배지 마크업 추가)
- 신규 파일 없음

## 검증

- `streamlit run streamlit_app/app.py`로 로컬 실행 후:
  - 예측 카드에 원형 게이지가 신뢰도만큼 채워져 표시되는지 확인
  - "다음 투구" 버튼으로 투구를 3회 이상 연속 적중시켜 COMBO 배지가 뜨는지, 실패 시 사라지는지 확인
  - 투구 전환 시 카드 슬라이드+페이드 애니메이션이 재생되는지, 슬라이더를 같은 값으로 재조작했을 때
    불필요하게 재생되지 않는지 확인
  - LIVE 배지가 계속 펄스되는지 확인
  - 적중/실패 시 카드 글로우가 색상 구분되어 1회 재생되는지 확인
- 회귀 확인: 기존 예측 로직(BiLSTM/통계 폴백), 슬라이더·이전/다음 버튼 네비게이션이 그대로 동작하는지
