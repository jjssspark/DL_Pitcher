# PitchIQ UI 전면 개편 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `streamlit_app/app.py`의 UI를 랜딩 히어로 섹션 신설 + 타이포/여백 확대 + 4개 영역 hover 인터랙션으로 전면 개편한다.

**Architecture:** 단일 파일(`streamlit_app/app.py`, 1,537줄)만 수정한다. CSS 블록 확장 → 랜딩 섹션 함수 추가 → 기존 조각난 "미로드" 표시 3곳을 하나로 통합 → 나머지 패널들의 인라인 스타일 폰트/패딩을 상향 조정 → hover 인터랙션 CSS/Plotly hovertemplate 추가.

**Tech Stack:** Streamlit 1.58, 순수 CSS(`unsafe_allow_html`), Plotly `go.Figure`.

## Global Constraints

- 구현 파일은 `streamlit_app/app.py` 하나만 (신규 파일 없음)
- 영상 동기화 / OCR / 스레딩 / BiLSTM / YOLO 로직 변경 금지 — 스타일·마크업만 수정
- 색상 팔레트·다크 테마 방향 변경 금지 (기존 네이비 `#080e1a` / 블루 `#60a5fa` / 퍼플 `#a78bfa` / 그린 `#34d399` 그대로)
- `session_state` 키 이름/구조 변경 금지
- 자동화 테스트 프레임워크 없음 — 각 태스크의 검증은 `python -m py_compile streamlit_app/app.py`(구문 오류 확인) + 최종 태스크에서 브라우저 스크린샷으로 시각 확인

---

### Task 1: CSS 파운데이션 — 타이포 스케일 + hover 유틸리티 클래스

**Files:**
- Modify: `streamlit_app/app.py:22-74` (CSS `<style>` 블록)

**Interfaces:**
- Produces: 신규 CSS 클래스 `.hero-title`, `.hero-tagline`, `.feature-grid`, `.feature-card`(hover 시 `.feature-card-desc` 노출), `.panel:hover`, `.pitch-card:hover`, `.stat-card`, `.stat-card:hover`, `.pitch-row:hover .pitch-row-detail`. 이후 Task 2~7이 이 클래스명을 그대로 사용한다.
- 기존 클래스 폰트 크기 변경: `.panel-title`(0.62rem→0.72rem), `.player-name`(1rem→1.15rem), `.player-sub`(.72rem→.8rem), `.pitch-code`(2rem→2.4rem), `.pitch-name`(.75rem→.85rem), `.pitch-speed`(.8rem→.9rem), `.pitch-row`(.73rem→.82rem), `.badge`(.65rem→.7rem)

- [ ] **Step 1: CSS 블록 교체**

`streamlit_app/app.py:22-74`를 아래로 교체:

```python
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#080e1a;min-height:100vh}
[data-testid="stSidebar"]{background:rgba(10,16,30,.97)!important;border-right:1px solid rgba(59,130,246,.18)}
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;color:#e2e8f0}
#MainMenu,footer,header{visibility:hidden}

/* rerun 깜빡임 방지 */
[data-stale="true"]{opacity:1!important;transition:none!important}
.element-container{transition:none!important}
iframe{transition:none!important}

/* 스코어보드 */
.scoreboard{background:linear-gradient(135deg,rgba(15,23,42,.95),rgba(20,30,55,.95));
  border:1px solid rgba(59,130,246,.22);border-radius:14px;padding:.7rem 1.2rem;
  margin-bottom:.8rem;backdrop-filter:blur(16px)}
.team-score{font-size:1.7rem;font-weight:900;letter-spacing:-.03em;line-height:1}
.team-name{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin-top:.05rem}
.inning-box{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);
  border-radius:8px;padding:.3rem .65rem;font-size:.8rem;font-weight:700;color:#93c5fd;text-align:center}
.count-dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 2px}

/* 패널 카드 — 타이포 확대 + hover glow */
.panel{background:rgba(15,23,42,.7);border:1px solid rgba(59,130,246,.12);
  border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.85rem;backdrop-filter:blur(8px);
  transition:border-color .2s,transform .2s}
.panel:hover{border-color:rgba(59,130,246,.35);transform:translateY(-2px)}
.panel-title{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#64748b;margin-bottom:.5rem}
.player-name{font-size:1.15rem;font-weight:800;color:#e2e8f0;line-height:1.15}
.player-sub{font-size:.8rem;color:#64748b;margin-top:.15rem}

/* 구종 카드 */
.pitch-card{border-radius:10px;padding:.95rem 1.25rem;margin-bottom:.6rem;
  border:1px solid rgba(255,255,255,.06);transition:border-color .2s,transform .2s}
.pitch-card:hover{transform:translateY(-2px)}
.pitch-code{font-size:2.4rem;font-weight:900;letter-spacing:-.02em;line-height:1}
.pitch-name{font-size:.85rem;color:#94a3b8;margin-top:.1rem}
.pitch-speed{font-size:.9rem;font-weight:700;color:#64748b;margin-top:.18rem}

/* 타임라인 행 — hover 시 상세정보 노출 */
.pitch-row{display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;
  border-radius:7px;margin-bottom:.22rem;font-size:.82rem;transition:background .15s}
.pitch-row-detail{max-height:0;opacity:0;overflow:hidden;font-size:.68rem;color:#64748b;
  transition:max-height .2s ease,opacity .2s ease}
.pitch-row:hover{background:rgba(59,130,246,.08)!important}
.pitch-row:hover .pitch-row-detail{max-height:40px;opacity:1;margin-top:.15rem}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;
  font-size:.7rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.badge-actual{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.badge-pred{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}
.badge-sim{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.25)}

/* 하단 통계 카드 — hover lift */
.stat-card{background:rgba(15,23,42,.6);border:1px solid rgba(59,130,246,.1);
  border-radius:10px;padding:.9rem 1.1rem;text-align:center;margin-bottom:.5rem;
  transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:rgba(59,130,246,.4);transform:translateY(-3px)}

/* 랜딩 히어로 */
.hero-wrap{padding:2.4rem 1rem 1.6rem;text-align:center}
.hero-title{font-size:2.6rem;font-weight:900;letter-spacing:-.02em;line-height:1.1;
  background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text}
.hero-tagline{font-size:1.05rem;color:#94a3b8;margin-top:.6rem}
.hero-badge{display:inline-block;margin-top:1rem;padding:.4rem 1rem;border-radius:999px;
  background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.35);color:#34d399;
  font-size:.85rem;font-weight:700}
.feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:1rem;margin-top:2.2rem}
.feature-card{background:rgba(15,23,42,.7);border:1px solid rgba(59,130,246,.14);
  border-radius:14px;padding:1.4rem 1.2rem;text-align:left;transition:border-color .2s,transform .2s;
  cursor:default}
.feature-card:hover{border-color:rgba(167,139,250,.5);transform:translateY(-4px)}
.feature-icon{font-size:1.8rem;margin-bottom:.5rem}
.feature-title{font-size:1.02rem;font-weight:800;color:#e2e8f0}
.feature-card-desc{max-height:0;opacity:0;overflow:hidden;font-size:.82rem;color:#94a3b8;
  line-height:1.5;margin-top:0;transition:max-height .25s ease,opacity .25s ease,margin-top .25s ease}
.feature-card:hover .feature-card-desc{max-height:120px;opacity:1;margin-top:.55rem}

div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#1d4ed8,#6d28d9)!important;color:#fff!important;
  border:none!important;border-radius:9px!important;font-weight:600!important;
  padding:.45rem 1.1rem!important;transition:opacity .18s!important}
div[data-testid="stButton"]>button:hover{opacity:.82!important}
[data-testid="stFileUploader"]{border:1.5px dashed rgba(59,130,246,.3)!important;
  border-radius:10px!important;background:rgba(15,23,42,.4)!important}
</style>""", unsafe_allow_html=True)
```

- [ ] **Step 2: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료 (exit code 0)

- [ ] **Step 3: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: CSS 파운데이션 확장 — 타이포 스케일 + hover 유틸리티 클래스"
```

---

### Task 2: 랜딩 히어로 섹션 — 3곳에 흩어진 미로드 표시 통합

**Files:**
- Modify: `streamlit_app/app.py:897-949` (헤더의 else 분기)
- Modify: `streamlit_app/app.py:1001-1002` (메인 레이아웃 컬럼 분기)
- Modify: `streamlit_app/app.py:1140-1149` (영상 placeholder)
- Modify: `streamlit_app/app.py:1220-1224` (우측 패널 placeholder)

**Interfaces:**
- Consumes: Task 1의 `.hero-wrap`, `.hero-title`, `.hero-tagline`, `.hero-badge`, `.feature-grid`, `.feature-card`, `.feature-icon`, `.feature-title`, `.feature-card-desc` 클래스
- Produces: 함수 `_render_landing_hero() -> None` — `loaded == False`일 때 전체 폭 히어로를 렌더링. 이후 태스크는 이 함수를 건드리지 않음.

- [ ] **Step 1: `_render_landing_hero()` 함수 추가**

`streamlit_app/app.py:896` (헤더 섹션 시작 주석 `# ══ 메인 헤더 ═...` 바로 위)에 삽입:

```python
def _render_landing_hero() -> None:
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-title">⚾ PitchIQ</div>'
        '<div class="hero-tagline">MLB 실시간 투구 분석 & 다음 구종 예측 시스템</div>'
        '<div class="hero-badge">🎯 BiLSTM 모델 정확도 48.5%</div>'
        '<div class="feature-grid">'
        '<div class="feature-card">'
        '<div class="feature-icon">🧠</div>'
        '<div class="feature-title">다음 구종 예측</div>'
        '<div class="feature-card-desc">BiLSTM + 투수·타자 Embedding으로 직전 투구 흐름과 '
        '경기 상황을 분석해 다음 구종을 실시간으로 예측합니다.</div>'
        '</div>'
        '<div class="feature-card">'
        '<div class="feature-icon">🎬</div>'
        '<div class="feature-title">실시간 영상 동기화</div>'
        '<div class="feature-card-desc">YouTube 중계 영상을 불러오면 YOLOv8 + 모션 감지로 '
        '투구 타이밍을 자동으로 찾아 예측과 동기화합니다.</div>'
        '</div>'
        '<div class="feature-card">'
        '<div class="feature-icon">📊</div>'
        '<div class="feature-title">구종 분포 분석</div>'
        '<div class="feature-card-desc">투수별 이번 경기 구종 비율과 카운트별 성향을 '
        '실시간으로 집계해 보여줍니다.</div>'
        '</div>'
        '</div>'
        '</div>', unsafe_allow_html=True)
    st.info("💡 왼쪽 사이드바에서 **game_pk**를 입력하고 경기 로드를 눌러보세요 — 예시: **745735** "
            "(2024년 6월 8일 LAD @ NYY)", icon="⚾")


```

- [ ] **Step 2: 헤더 else 분기를 새 함수 호출로 교체**

`streamlit_app/app.py:942-949`의 기존 코드:

```python
else:
    st.markdown(
        '<div style="padding:.6rem 0 .5rem">'
        '<div style="font-size:2rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">⚾ PitchIQ</div>'
        '<div style="font-size:.85rem;color:#475569;margin-top:.15rem">사이드바에서 경기를 로드하세요 (game_pk 입력)</div>'
        '</div>', unsafe_allow_html=True)
    st.info("💡 예시 — 2024년 6월 8일 LAD @ NYY : **game_pk = 745735**", icon="⚾")
```

를 다음으로 교체:

```python
else:
    _render_landing_hero()
```

- [ ] **Step 3: 메인 레이아웃을 `loaded`일 때만 렌더링하도록 조건 분기**

`streamlit_app/app.py:1001-1002`의 기존 코드:

```python
# ══ 메인 레이아웃 ═════════════════════════════════════════════════
col_video, col_panel = st.columns([3.2, 1.2], gap="medium")
```

를 다음으로 교체 (이후 `with col_video:` / `with col_panel:` 블록 전체를 `if loaded:` 안으로 들여쓰기):

```python
# ══ 메인 레이아웃 ═════════════════════════════════════════════════
if loaded:
    col_video, col_panel = st.columns([3.2, 1.2], gap="medium")
else:
    col_video = col_panel = None
```

이어서 `streamlit_app/app.py:1004`의 `with col_video:` 줄부터 `1482`줄(`with col_panel:` 블록의 마지막 줄, `st.plotly_chart(fig2, ...)`)까지 — 즉 **`with col_video:` 블록 전체 + `with col_panel:` 블록 전체(사이의 구분 주석 `# ── 오른쪽: 경기 상황 패널 ──` 포함)** — 를 4칸 들여쓰기 추가하고 최상위에 `if loaded:`를 씌운다. **`1485`줄의 `# ══ 하단 통계 ═` 주석과 `if loaded and c_idx > 0:` 이하 블록은 이미 자체적으로 `loaded`를 체크하고 있고 `col_video`/`col_panel`에 속하지 않으므로 들여쓰기 대상에서 제외한다** (건드리지 않음). 즉:

```python
if loaded:
    with col_video:
        ...(기존 col_video 블록 전체, line 1004~1215, 그대로)...

    # ── 오른쪽: 경기 상황 패널 ────────────────────────────────────────
    with col_panel:
        ...(기존 col_panel 블록 전체, line 1219~1482, 그대로)...

# ══ 하단 통계 ═════════════════════════════════════════════════════  (기존 line 1485, 들여쓰기 변경 없음)
if loaded and c_idx > 0:
    ...
```

이 작업은 코드 내용 변경이 아니라 들여쓰기 추가이므로, 에디터의 블록 인덴트 기능이나 `sed`가 아닌 **Edit 도구로 각 줄을 직접 4칸 들여쓰기**해야 한다 (특히 문자열 리터럴 내부의 개행에 잘못 들여쓰기가 들어가지 않도록 라인 단위로 확인). 빈 줄(예: 1216-1217)은 들여쓰기 없이 그대로 두어도 무방하다.

이제 `streamlit_app/app.py:1140-1149`(영상 placeholder)와 `1220-1224`(우측 패널 placeholder)는 `if loaded:` 블록 안에서만 도달하므로 **그대로 두되 else 분기(영상 없을 때, 우측 패널 없을 때)는 삭제하지 않는다** — 이 두 곳의 else는 "게임은 로드됐지만 아직 영상 URL을 안 넣은 경우"에 대응하는 것이라 랜딩과는 다른 상태이므로 유지한다. `1140-1149`, `1220-1224`는 이번 태스크에서 수정하지 않는다.

- [ ] **Step 4: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료. 들여쓰기가 깨졌다면 `IndentationError`가 여기서 잡힌다.

- [ ] **Step 5: 브라우저로 시각 확인**

Run: `streamlit run streamlit_app/app.py` (로컬)
- 게임을 로드하지 않은 초기 상태: 히어로 섹션(타이틀 + 태그라인 + 정확도 배지 + 기능 카드 3개)이 전체 폭으로 보이는지 확인
- 기능 카드에 마우스를 올렸을 때 설명 텍스트가 슬라이드다운으로 나타나는지 확인
- 사이드바에서 `game_pk=745735` 입력 후 "경기 로드" 클릭 → 기존 대시보드(영상/패널 2단 레이아웃)가 정상적으로 나오는지 확인 (회귀 없음)

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/app.py
git commit -m "feat: 게임 미로드 상태를 랜딩 히어로 섹션으로 통합"
```

---

### Task 3: 사이드바 라벨 폰트 크기 상향

**Files:**
- Modify: `streamlit_app/app.py:672-894` (사이드바 내 인라인 스타일)

**Interfaces:**
- Consumes: 없음 (인라인 스타일 값만 변경)
- Produces: 없음 (다른 태스크가 참조하지 않음)

이 태스크는 사이드바 섹션의 하드코딩된 `font-size` 값을 아래 규칙으로 일괄 상향한다 (약 +15%, 반올림):

| 기존 | 신규 |
|---|---|
| `.62rem` (섹션 라벨: 경기 로드/경기 영상/타임라인 보정/경기 현황/구종 범례) | `.72rem` |
| `.68rem` (MLB 투구 예측 시스템 서브타이틀) | `.78rem` |
| `.7rem` (영상 URL 상태, 대기 중 등) | `.8rem` |
| `.72rem` (BiLSTM 상태 배지) | `.82rem` |
| `.65rem` (싱크 상태) | `.75rem` |
| `.76rem` (경기 현황 본문) | `.85rem` |
| `.7rem` (구종 범례 행) | `.8rem` |
| `1.35rem` (⚾ PitchIQ 로고) | `1.5rem` |

- [ ] **Step 1: 사이드바 로고/서브타이틀 (line 672-677)**

기존:
```python
    st.markdown(
        '<div style="padding:.8rem 0 .4rem">'
        '<div style="font-size:1.35rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">⚾ PitchIQ</div>'
        '<div style="font-size:.68rem;color:#475569;margin-top:.05rem">MLB 투구 예측 시스템</div>'
        '</div>', unsafe_allow_html=True)
```

신규:
```python
    st.markdown(
        '<div style="padding:.8rem 0 .4rem">'
        '<div style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">⚾ PitchIQ</div>'
        '<div style="font-size:.78rem;color:#475569;margin-top:.05rem">MLB 투구 예측 시스템</div>'
        '</div>', unsafe_allow_html=True)
```

- [ ] **Step 2: 나머지 사이드바 섹션 라벨/본문 (line 679-894 범위)**

`streamlit_app/app.py:679-894` 안에서 위 표에 나열된 `font-size` 값을 문자열 그대로 찾아 신규 값으로 치환한다 (예: `font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569` 패턴이 5곳 — 경기 로드/경기 영상/타임라인 보정/경기 현황/구종 범례 라벨 — 모두 동일 문자열이므로 `replace_all`로 일괄 치환 가능). 값 변경 외 다른 내용(마크업 구조, 색상, 텍스트)은 손대지 않는다.

- [ ] **Step 3: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료

- [ ] **Step 4: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: 사이드바 라벨 폰트 크기 상향"
```

---

### Task 4: 투수·타자 패널 + "방금 던진 구종" 카드 — 폰트 상향

**Files:**
- Modify: `streamlit_app/app.py:1226-1397`

**Interfaces:**
- Consumes: Task 1의 `.panel`, `.pitch-card`, `.badge` 클래스 (자동으로 hover/크기 적용됨 — 별도 작업 불필요)
- Produces: 없음

이 구간의 카드들은 이미 `class="panel"` / `class="pitch-card"`를 쓰고 있어 **Task 1의 CSS 클래스 변경만으로 폰트/hover가 자동 적용된다.** 이 태스크에서는 클래스를 안 쓰고 인라인으로 하드코딩된 폰트만 상향한다.

- [ ] **Step 1: 투수 교체 / 타자 교체 알림 카드 (line 1237-1267)**

`font-size:.6rem`(라벨) → `.68rem`, `font-size:.82rem`(이전 투수명) → `.9rem`, `font-size:.95rem`(신규 투수/타자명) → `1.05rem`, `font-size:.65rem`(팀·타석 정보) → `.72rem` 로 각각 치환.

- [ ] **Step 2: 투수·타자 정보 패널 (line 1271-1293)**

`font-size:.6rem`(라벨 "투수"/"타자") → `.68rem`, `font-size:.88rem`(이름) → `1rem`, `font-size:.65rem`(투구폼/타석) → `.72rem`, `font-size:.68rem`(주자 정보) → `.75rem`.

- [ ] **Step 3: 타석 투구 결과 점 (line 1332-1337)**

`font-size:.6rem` → `.68rem`.

- [ ] **Step 4: 방금 던진 구종 카드 (line 1340-1396)**

`font-size:.62rem`(패널 타이틀) → `.72rem`, `font-size:.78rem`(구속) → `.85rem`, `font-size:.6rem`(뱃지류: 결과, 예측적중) → `.68rem`, `font-size:.65rem`(이벤트/예측적중 텍스트) → `.72rem`, `font-size:.8rem`(placeholder "경기 로드 후 재생") → `.9rem`.

- [ ] **Step 5: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: 투수·타자 패널 및 구종 카드 폰트 크기 상향"
```

---

### Task 5: 다음 구종 예측 카드 — 폰트 상향 + Plotly hover 툴팁

**Files:**
- Modify: `streamlit_app/app.py:1398-1456`

**Interfaces:**
- Consumes: `PITCH_META` dict (기존, 변경 없음)
- Produces: 없음

- [ ] **Step 1: 예측 카드 폰트 상향 (line 1413-1430)**

`font-size:.62rem`(패널 타이틀) → `.72rem`, `font-size:1.05rem`(신뢰도 %) → `1.2rem`, `font-size:.65rem`("신뢰도" 라벨) → `.72rem`.

- [ ] **Step 2: 확률 막대 차트에 hovertemplate 추가 (line 1440-1446)**

기존:
```python
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker=dict(color=colors, opacity=0.82,
                            line=dict(color="rgba(255,255,255,.04)", width=1)),
                text=[f"{v:.0%}" for v in vals], textposition="outside",
                textfont=dict(size=8, color="#94a3b8"),
            ))
```

신규 (구종 코드·한글명·확률을 hover 툴팁으로 노출, `customdata`로 한글명 전달):

```python
            _pitch_kor_names = [PITCH_META.get(c, PITCH_META["OTHER"])["name"] for c in codes]
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker=dict(color=colors, opacity=0.82,
                            line=dict(color="rgba(255,255,255,.04)", width=1)),
                text=[f"{v:.0%}" for v in vals], textposition="outside",
                textfont=dict(size=8, color="#94a3b8"),
                customdata=list(zip(codes, _pitch_kor_names)),
                hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>확률: %{y:.1%}<extra></extra>",
            ))
```

- [ ] **Step 3: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료

- [ ] **Step 4: 브라우저로 hover 확인**

Run: `streamlit run streamlit_app/app.py` → 게임 로드 후 "다음 투구 예측" 막대 그래프의 막대에 마우스를 올려 구종 코드·한글명·정확한 확률(%)이 툴팁으로 뜨는지 확인.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: 예측 카드 폰트 상향 및 확률 막대 hover 툴팁 추가"
```

---

### Task 6: 투구 타임라인 / 최근 투구 행 — 폰트 상향 + hover 상세정보

**Files:**
- Modify: `streamlit_app/app.py:1186-1219`

**Interfaces:**
- Consumes: Task 1의 `.pitch-row`, `.pitch-row-detail` 클래스
- Produces: 없음

- [ ] **Step 1: 라벨 폰트 상향 (line 1186, 1195)**

`font-size:.63rem` (투구 타임라인 / 최근 투구 라벨) → `.72rem` 로 치환 (두 곳 모두 동일 패턴이므로 `replace_all`).

- [ ] **Step 2: 최근 투구 행에 hover 상세정보 추가 (line 1197-1219 부근)**

현재 `.pitch-row`는 구번호/구종/구속/이벤트만 표시한다. 각 행 안에 `.pitch-row-detail` div를 추가해 hover 시 카운트·타석결과 상세가 보이도록 한다. 기존 루프에서 각 `_r`(pitch dict)에 대해 만드는 `st.markdown(f'<div class="pitch-row" ...')` 블록의 닫는 `</div>` 직전에 아래 줄을 추가:

```python
                f'<div class="pitch-row-detail">카운트 {_r["balls"]}-{_r["strikes"]} · '
                f'{_r["inning"]}회 {"초" if _r["inning_topbot"]=="Top" else "말"}</div>'
```

(정확한 삽입 위치는 해당 f-string의 마지막 `f'</div>'` 바로 앞 줄. `_r["balls"]`, `_r["strikes"]`, `_r["inning"]`, `_r["inning_topbot"]`은 `pitches` 리스트의 각 항목에 이미 존재하는 키이므로 추가 데이터 처리 불필요.)

- [ ] **Step 3: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료

- [ ] **Step 4: 브라우저로 hover 확인**

게임 로드 후 "최근 투구" 목록의 행에 마우스를 올려 카운트/이닝 상세가 아래로 나타나는지 확인.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: 투구 타임라인 라벨 폰트 상향 및 최근 투구 행 hover 상세정보 추가"
```

---

### Task 7: 하단 통계 카드 — `.stat-card` 클래스 적용

**Files:**
- Modify: `streamlit_app/app.py:1495-1510`

**Interfaces:**
- Consumes: Task 1의 `.stat-card` 클래스
- Produces: 없음

- [ ] **Step 1: 인라인 스타일을 `.stat-card` 클래스로 교체**

기존 (line 1503-1510):
```python
        with _col:
            st.markdown(
                f'<div style="background:rgba(15,23,42,.6);border:1px solid rgba(59,130,246,.1);'
                f'border-radius:10px;padding:.7rem 1rem;text-align:center;margin-bottom:.5rem">'
                f'<div style="font-size:.6rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.22rem">{_lbl}</div>'
                f'<div style="font-size:1.35rem;font-weight:800;color:#e2e8f0;line-height:1">{_val}</div>'
                f'<div style="font-size:.62rem;color:#64748b;margin-top:.08rem">{_sub}</div>'
                f'</div>', unsafe_allow_html=True)
```

신규:
```python
        with _col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.28rem">{_lbl}</div>'
                f'<div style="font-size:1.55rem;font-weight:800;color:#e2e8f0;line-height:1">{_val}</div>'
                f'<div style="font-size:.7rem;color:#64748b;margin-top:.1rem">{_sub}</div>'
                f'</div>', unsafe_allow_html=True)
```

- [ ] **Step 2: 구문 확인**

Run: `python -m py_compile streamlit_app/app.py`
Expected: 에러 없이 종료

- [ ] **Step 3: Commit**

```bash
git add streamlit_app/app.py
git commit -m "style: 하단 통계 카드에 hover lift 효과 적용"
```

---

### Task 8: 전체 시각 검증 (최종)

**Files:** 없음 (검증 전용)

**Interfaces:** 없음

- [ ] **Step 1: 로컬 실행**

Run: `streamlit run streamlit_app/app.py`

- [ ] **Step 2: 랜딩 상태 스크린샷**

브라우저에서 앱 최초 접속 화면(게임 미로드) 스크린샷 — 히어로 타이틀/태그라인/정확도 배지/기능 카드 3개가 전체 폭으로 보이는지, 기능 카드 hover 시 설명이 나타나는지 확인.

- [ ] **Step 3: 대시보드 상태 스크린샷**

사이드바에 `game_pk=745735` 입력 후 "경기 로드" 클릭 → 로드 완료 후 스크린샷 — 스코어보드/투수·타자 패널/구종 카드/예측 확률 막대/타임라인/하단 통계 카드가 모두 이전 대비 폰트가 커지고 여백이 넓어졌는지, 각 hover 대상(예측 막대, 최근 투구 행, 통계 카드, 패널)이 정상 동작하는지 확인.

- [ ] **Step 4: 회귀 확인**

투구 타임라인 슬라이더로 여러 투구를 이동해보며 예측/방금 던진 구종/구종 분포 도넛 차트가 기존과 동일하게 갱신되는지 확인 (기능 로직은 변경하지 않았으므로 회귀가 없어야 정상).

- [ ] **Step 5: 최종 커밋 (변경사항 있을 경우만)**

검증 중 발견된 사소한 스타일 오차를 수정했다면:
```bash
git add streamlit_app/app.py
git commit -m "style: UI 개편 최종 시각 검증 후 미세 조정"
```
