# PitchIQ 방송 중계 스타일 UI 개편 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `streamlit_app/app.py`의 UI를 방송 중계·스코어보드 스타일로 개편해 랜딩 순간을 만들고, 우측 패널의 정보 위계와 카드별 시각 구분을 명확히 한다.

**Architecture:** CSS 블록 확장(신규 클래스 추가, 기존 클래스는 그대로 두고 인라인 스타일로 팀컬러 등 동적 값 적용) + 자동로드 블록을 2단계 rerun 구조로 분리해 스플래시를 넣는다. `session_state`/예측/영상동기화 로직은 건드리지 않는다.

**Tech Stack:** Python, Streamlit, 순수 CSS(애니메이션 포함, JS 없음).

## Global Constraints

- 스펙: `docs/superpowers/specs/2026-08-04-broadcast-ui-redesign-design.md`
- `session_state` 구조 변경 금지 (필드 추가만 허용)
- BiLSTM/YOLO 예측 로직, 영상 동기화(OCR/타임스탬프) 로직 변경 금지
- 2단 레이아웃(`col_video`/`col_panel`) 골격 유지 — 카드 내부 마크업·CSS만 변경
- 이 프로젝트는 pytest 인프라가 없음 — 검증은 `streamlit run streamlit_app/app.py` 실행 후 브라우저로 관찰

---

### Task 1: 랜딩 스플래시 (2단계 rerun)

**Files:**
- Modify: `streamlit_app/app.py` — CSS 블록(108행 `</style>` 직전)에 스플래시 클래스 추가, 자동로드 블록(622~628행) 2단계화, `_render_intro_splash()` 함수 추가, `_DEFAULTS`에 `_intro_shown` 추가

**Interfaces:**
- Produces: `_render_intro_splash() -> None` — 신규 함수, 이 태스크 내에서만 호출됨

- [ ] **Step 1: CSS에 스플래시 스타일 추가**

`</style>""", unsafe_allow_html=True)` 직전에 삽입:

```css
.intro-splash{position:fixed;inset:0;background:#080e1a;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:1rem;z-index:9999}
.intro-splash .intro-logo{font-size:3.4rem;font-weight:900;letter-spacing:-.02em;
  background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;animation:introPulse 1.4s ease-in-out infinite}
.intro-splash .intro-tagline{font-size:1rem;color:#94a3b8}
.intro-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.35rem .9rem;
  border-radius:999px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);
  color:#f87171;font-size:.78rem;font-weight:700;letter-spacing:.04em}
.intro-badge::before{content:"";width:7px;height:7px;border-radius:50%;background:#ef4444;
  animation:introPulse 1.1s ease-in-out infinite}
@keyframes introPulse{0%,100%{opacity:1}50%{opacity:.45}}
```

- [ ] **Step 2: `_render_intro_splash()` 함수 추가**

`_load_game()` 함수 정의 바로 뒤(그리고 자동로드 블록 622행보다 반드시 위)에 삽입:

```python
def _render_intro_splash() -> None:
    st.markdown(
        '<div class="intro-splash">'
        '<div class="intro-logo">⚾ PitchIQ</div>'
        '<div class="intro-tagline">MLB 실시간 투구 분석 &amp; 다음 구종 예측 시스템</div>'
        '<div class="intro-badge">LIVE 데모 준비 중</div>'
        '</div>', unsafe_allow_html=True)
```

주의: 기존 `_render_landing_hero()`는 801행 근처(자동로드 블록보다 한참 아래)에 있어도 문제
없다 — 그건 `if loaded: ... else: _render_landing_hero()` 분기에서 스크립트 뒷부분에 가서야
호출되기 때문. 하지만 `_render_intro_splash()`는 622행 근처 자동로드 블록 안에서 곧바로
호출되므로, 함수 정의가 그보다 앞에 와야 `NameError`가 나지 않는다.

- [ ] **Step 3: `_DEFAULTS`에 플래그 추가**

`_DEFAULTS` 딕셔너리(583행)에 한 줄 추가:

```python
    "_intro_shown":          False,  # 세션당 스플래시 1회만 표시
```

- [ ] **Step 4: 자동로드 블록을 2단계로 분리**

기존(622~628행):
```python
if not st.session_state._demo_auto_loaded and st.session_state.game_pk == "":
    st.session_state._demo_auto_loaded = True
    st.session_state.video_src = FIXED_DEMO_VIDEO_URL
    st.session_state.is_playing = True  # iframe autoplay와 상태 일치 — 다음 rerun에서 pauseVideo() 방지
    _demo_ok, _demo_msg = _load_game(str(FIXED_DEMO_GAME_PK))
    if not _demo_ok:
        print(f"[FixedDemo] 자동 로드 실패: {_demo_msg}")
```

교체 후:
```python
if not st.session_state._demo_auto_loaded and st.session_state.game_pk == "":
    if not st.session_state._intro_shown:
        st.session_state._intro_shown = True
        _render_intro_splash()
        time.sleep(1.4)
        st.rerun()
    else:
        st.session_state._demo_auto_loaded = True
        st.session_state.video_src = FIXED_DEMO_VIDEO_URL
        st.session_state.is_playing = True  # iframe autoplay와 상태 일치 — 다음 rerun에서 pauseVideo() 방지
        _demo_ok, _demo_msg = _load_game(str(FIXED_DEMO_GAME_PK))
        if not _demo_ok:
            print(f"[FixedDemo] 자동 로드 실패: {_demo_msg}")
```

- [ ] **Step 5: `_render_landing_hero()`의 안내 문구 수정 (사이드바 수동 입력 삭제로 깨진 문구)**

기존 828~829행:
```python
    st.info("💡 왼쪽 사이드바에서 **game_pk**를 입력하고 경기 로드를 눌러보세요 — 예시: **745735** "
            "(2024년 6월 8일 LAD @ NYY)", icon="⚾")
```

교체 후 (더 이상 수동 입력 UI가 없으므로 재시도 안내로 변경):
```python
    st.info("💡 데이터 로드에 실패했습니다. 페이지를 새로고침해 다시 시도해보세요.", icon="⚾")
```

- [ ] **Step 6: 검증**

Run: `python3 -c "import ast; ast.parse(open('streamlit_app/app.py').read())"`
Expected: 에러 없음

- [ ] **Step 7: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "feat: 최초 진입 시 브랜드 스플래시 추가"
```

---

### Task 2: 방송 스코어보드 헤더

**Files:**
- Modify: `streamlit_app/app.py` — CSS에 세그먼트 카운트 스타일 추가, `PITCH_META` 근처에 `TEAM_COLORS` 상수 추가, 스코어보드 마크업(현재 844~876행) 수정

**Interfaces:**
- Produces: `TEAM_COLORS: dict[str, str]` — Task 4에서도 참고 가능(선택적, 필수 아님)

- [ ] **Step 1: `TEAM_COLORS` 상수 추가**

`FIXED_DEMO_VIDEO_URL` 정의 바로 아래에 추가:

```python
TEAM_COLORS = {"NYY": "#0C2340", "LAD": "#005A9C"}  # 고정 데모 게임은 이 두 팀만 등장
```

- [ ] **Step 2: CSS에 세그먼트 카운트 스타일 추가**

`</style>` 직전(Task 1에서 추가한 `.intro-*` 규칙 아래)에 추가:

```css
.seg-dot{display:inline-block;width:13px;height:13px;border-radius:3px;margin:0 2px}
.team-score{font-size:2.1rem}
```

(`.team-score` 기존 정의는 1.7rem — 이 규칙이 CSS 순서상 뒤에 오므로 덮어써서 확대된다.)

- [ ] **Step 3: 세그먼트 카운트 헬퍼 함수 추가**

`_count_dots_simple` 정의 바로 아래에 추가:

```python
def _seg_dots(n: int, total: int, color_on: str, color_off: str = "rgba(100,116,139,.2)") -> str:
    dots = ""
    for i in range(total):
        c = color_on if i < n else color_off
        dots += f'<span class="seg-dot" style="background:{c}"></span>'
    return dots
```

- [ ] **Step 4: 스코어보드 마크업 수정**

현재 840~876행의 `balls_html`/`strikes_html`/`outs_html` 생성과 스코어보드 `st.markdown` 블록에서:
- `_count_dots_simple(...)` 호출 3곳을 `_seg_dots(...)` 호출로 교체 (파라미터 동일하게 유지)
- 원정팀 팀명 표시 줄:
  ```python
  f'<div style="font-size:.8rem;font-weight:800;color:#94a3b8;letter-spacing:.08em">{aw}</div>'
  ```
  을
  ```python
  f'<div style="font-size:.8rem;font-weight:800;color:{TEAM_COLORS.get(aw, "#94a3b8")};'
  f'letter-spacing:.08em;border-bottom:2px solid {TEAM_COLORS.get(aw, "#94a3b8")};'
  f'padding-bottom:.15rem;display:inline-block">{aw}</div>'
  ```
  로 교체. 홈팀 줄도 `hw` 기준으로 동일하게 교체.

- [ ] **Step 5: 검증**

Run: `python3 -c "import ast; ast.parse(open('streamlit_app/app.py').read())"`
Expected: 에러 없음

- [ ] **Step 6: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "style: 스코어보드 헤더에 팀컬러·세그먼트 카운트 적용"
```

---

### Task 3: 우측 패널 — 예측 카드 히어로화 + 보조 카드 격하

**Files:**
- Modify: `streamlit_app/app.py` — CSS에 `.panel-secondary`, `.card-badge` 추가, "투수·타자" 패널(1213행)과 "다음 투구 예측" 카드(1354~1371행) 마크업 수정

**Interfaces:**
- Consumes: 없음 (기존 `bilstm_status`, `_bilstm_res`, `pred`, `_nc`, `_nm`, `_cf`, `_cc` 등 기존 로컬 변수 재사용)

- [ ] **Step 1: CSS에 보조 카드/카드 배지 스타일 추가**

`</style>` 직전에 추가:

```css
.panel-secondary{background:rgba(15,23,42,.5);border:1px solid rgba(59,130,246,.08);
  border-radius:10px;padding:.8rem 1rem;margin-bottom:.65rem}
.panel-secondary .panel-title{font-size:.66rem}
.card-badge{display:inline-flex;align-items:center;gap:.3rem;font-size:.66rem;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;padding:.15rem .5rem;border-radius:999px;margin-bottom:.4rem}
.card-badge-pred{background:rgba(167,139,250,.14);color:#c4b5fd;border:1px solid rgba(167,139,250,.35)}
.card-badge-actual{background:rgba(52,211,153,.12);color:#6ee7b7;border:1px solid rgba(52,211,153,.3)}
.pred-hero{border-width:1.5px!important;padding:1.3rem 1.5rem!important}
.pred-hero .pitch-code{font-size:3rem!important}
```

- [ ] **Step 2: "투수·타자" 패널을 `.panel-secondary`로 격하**

현재 1213행 `f'<div class="panel">'`를 `f'<div class="panel-secondary">'`로 교체.

- [ ] **Step 3: "다음 투구 예측" 카드에 근거 배지 추가 + 히어로 스타일 적용**

현재 1354~1358행(레이블):
```python
                st.markdown(
                    f'<div class="panel-title" style="font-size:.72rem;font-weight:700;letter-spacing:.1em;'
                    f'text-transform:uppercase;color:#64748b;margin:.2rem 0 .35rem">'
                    f'다음 투구 예측 {src_badge}</div>',
                    unsafe_allow_html=True)
```

교체 후 (근거 문구를 `card-badge`로 표시):
```python
                _pred_basis = "BiLSTM 모델 예측 — 직전 투구 흐름 기반" if _bilstm_res else "통계 기반 예측 (BiLSTM 계산 중)"
                st.markdown(
                    f'<div class="card-badge card-badge-pred">🔮 예측 · {_pred_basis}</div>',
                    unsafe_allow_html=True)
```

현재 1359~1361행(카드 본문 시작)의
```python
                st.markdown(
                    f'<div class="pitch-card" style="background:linear-gradient(135deg,rgba(15,23,42,.8),'
                    f'rgba(46,27,75,.4));border-color:rgba(167,139,250,.3)">'
```
을
```python
                st.markdown(
                    f'<div class="pitch-card pred-hero" style="background:linear-gradient(135deg,rgba(15,23,42,.8),'
                    f'rgba(46,27,75,.4));border-color:rgba(167,139,250,.3)">'
```
로 교체 (클래스 `pred-hero` 추가만, 나머지 동일).

- [ ] **Step 4: "방금 던진 구종" 카드에도 실측 배지 추가**

현재 1281행 레이블
```python
            st.markdown('<div class="panel-title" style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.35rem">방금 던진 구종</div>', unsafe_allow_html=True)
```
바로 위에 삽입:
```python
            st.markdown('<div class="card-badge card-badge-actual">📊 실측</div>', unsafe_allow_html=True)
```

- [ ] **Step 5: 검증**

Run: `python3 -c "import ast; ast.parse(open('streamlit_app/app.py').read())"`
Expected: 에러 없음

- [ ] **Step 6: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "style: 예측 카드 히어로화 + 보조 카드 격하 + 예측/실측 배지 추가"
```

---

### Task 4: 전체 회귀 검증

**Files:** 없음 (검증 전용)

- [ ] **Step 1: 로컬 실행 — 새 세션에서 스플래시부터 확인**

`streamlit run streamlit_app/app.py`를 브라우저 시크릿 창으로 열어:
1. 진입 직후 ⚾ PitchIQ 스플래시가 약 1.4초 보이는지 확인
2. 이후 자동으로 대시보드(스코어보드+영상+예측)로 전환되는지 확인
3. 스코어보드에 팀컬러 언더라인·확대된 점수·사각 세그먼트 카운트가 보이는지 확인
4. 우측 패널에서 "다음 투구 예측" 카드가 다른 카드보다 눈에 띄게 크고, "🔮 예측 · ..." 배지가
   보이는지 확인. "방금 던진 구종"에는 "📊 실측" 배지, "투수·타자"는 더 작은 보조 카드로
   보이는지 확인

- [ ] **Step 2: 회귀 — 오늘 고친 자동재생/자동로드**

영상이 자동재생(음소거) 시도되는지, `_demo_auto_loaded`/`_intro_shown` 덕분에 같은 세션
새로고침 시 스플래시가 다시 뜨지 않는지 확인.

- [ ] **Step 3: 브라우저 스크린샷으로 개편 전/후 비교 (선택, 시간 되면)**
