# PitchIQ 고정 데모 모드 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 방문자가 아무 입력 없이 페이지를 열면 game_pk=775300(2024 WS 1차전) + 고정 YouTube 영상이 자동으로 로드되어 재생과 동시에 예측이 시작되도록 만들고, 배포 환경(Streamlit Community Cloud)에서 로컬 다운로드/실시간 OCR 없이 안정적으로 동작하게 한다.

**Architecture:** `streamlit_app/app.py`의 세션 상태 초기화 직후, 최초 진입을 감지해 기존 "경기 로드" 버튼과 동일한 절차(`_load_game` 헬퍼로 추출)를 1회 자동 실행한다. 영상은 항상 `youtube_player`(iframe) 경로를 타도록 로컬 다운로드를 트리거하지 않는다. 투구 타이밍 동기화는 로컬에서 1회 생성한 정적 JSON(`streamlit_app/fixed_demo_scan.json`)을 읽어 채운다. 기존 수동 game_pk/영상 URL 입력 UI는 `st.expander`로 접어 고급 옵션으로 격하한다.

**Tech Stack:** Python, Streamlit, pybaseball(Statcast), 기존 `pose_detector.scan_pitch_overlays`/`yolo_detector.resolve_video_path`(로컬 1회 스크립트에서만 사용).

## Global Constraints

- 고정 game_pk = `775300`, 고정 영상 URL = `https://youtu.be/gMm3EODDb6w` (스펙 원문 그대로, 다른 값으로 대체 금지)
- 배포 환경에서는 로컬 다운로드/실시간 OCR 스캔을 절대 트리거하지 않는다
- BiLSTM/YOLO 모델 로직 변경 금지 (스펙 비범위)
- 기존 수동 game_pk/영상 URL 입력 기능 완전 삭제 금지 — `st.expander`로 접어 고급 옵션으로 격하 (사용자 확인: "expander로 접기")
- `streamlit_app/fixed_demo_scan.json`은 git 커밋 대상 (`streamlit_app/.scan_cache/`와는 별도 경로이며 `.gitignore`에 걸리지 않음)
- 이 프로젝트에는 pytest 인프라가 없고 `app.py`는 Streamlit 세션 상태에 강결합된 스크립트라 순수 단위 테스트가 어렵다. 각 태스크의 검증은 `streamlit run streamlit_app/app.py` 실행 후 관찰 가능한 동작/로그로 확인한다.

---

### Task 1: `_load_game()` 헬퍼 추출

기존 "경기 로드" 버튼 클릭 시 로직(`app.py` 사이드바 내부, `load_btn` 핸들러)을 재사용 가능한 함수로 추출한다. 자동 데모 로드(Task 2)와 수동 버튼 클릭이 동일한 코드를 공유하기 위함 — 로직 중복 방지.

**Files:**
- Modify: `streamlit_app/app.py` (함수는 `_run_bilstm_bg` 정의 직후, `# ══ 세션 상태 초기화 ══` 주석 이전에 추가)
- Modify: `streamlit_app/app.py` 사이드바 `load_btn` 핸들러 (현재 `if load_btn and gk_input.strip():` 블록)

**Interfaces:**
- Produces: `_load_game(gk_str: str) -> tuple[bool, str]` — Task 2에서 사용. 성공 시 `(True, "✅ ... 로드 완료")`, 실패 시 `(False, "로드 실패: ...")`를 반환하고 `st.rerun()`은 호출하지 않는다(호출부 책임).

- [ ] **Step 1: `_load_game` 함수 추가**

`_run_bilstm_bg` 함수 정의(현재 523~529행) 바로 뒤에 삽입:

```python
def _load_game(gk_str: str) -> tuple[bool, str]:
    """game_pk 문자열로 Statcast 데이터 로드 + BiLSTM 백그라운드 계산 트리거.
    성공 시 (True, 안내메시지), 실패 시 (False, 에러메시지)를 반환한다.
    """
    try:
        gk = int(gk_str.strip())
        with st.spinner("Statcast 데이터 수집 중..."):
            _pitches, _meta = fetch_game_pitches(gk)
        _pid_tuple = tuple(sorted({p["pitcher_id"] for p in _pitches}))
        st.session_state.game_pk            = gk_str.strip()
        st.session_state.game_pitches       = _pitches
        st.session_state.game_meta          = _meta
        st.session_state.bilstm_preds       = []
        st.session_state.bilstm_status      = "computing"
        st.session_state.current_pitch_idx  = 0
        st.session_state._last_ocr_mlb_idx      = -1
        st.session_state._sync_activated         = False
        st.session_state.video_pitch_data        = []
        st.session_state._scan_raw_data          = []
        st.session_state._next_scan_idx          = 0
        st.session_state._sixth_inning_alert     = False
        st.session_state._last_pitch_video_time  = -30.0
        if gk not in _bilstm_tasks or _bilstm_tasks[gk].get("status") != "done":
            _bilstm_tasks[gk] = {"status": "computing"}
            threading.Thread(
                target=_run_bilstm_bg, args=(gk, _pid_tuple), daemon=True
            ).start()
        return True, f"✅ {_meta['game_date']} {_meta['away_team']} @ {_meta['home_team']} — {len(_pitches)}구 로드 완료 (BiLSTM 계산 중...)"
    except Exception as _e:
        return False, f"로드 실패: {_e}"
```

- [ ] **Step 2: 기존 `load_btn` 핸들러를 헬퍼 호출로 교체**

기존:
```python
    if load_btn and gk_input.strip():
        try:
            gk = int(gk_input.strip())
            with st.spinner("Statcast 데이터 수집 중..."):
                _pitches, _meta = fetch_game_pitches(gk)
            _pid_tuple = tuple(sorted({p["pitcher_id"] for p in _pitches}))
            st.session_state.game_pk            = gk_input.strip()
            st.session_state.game_pitches       = _pitches
            st.session_state.game_meta          = _meta
            st.session_state.bilstm_preds       = []
            st.session_state.bilstm_status      = "computing"
            st.session_state.current_pitch_idx  = 0
            st.session_state._last_ocr_mlb_idx      = -1
            st.session_state._sync_activated         = False
            st.session_state.video_pitch_data        = []
            st.session_state._scan_raw_data          = []
            st.session_state._next_scan_idx          = 0
            st.session_state._sixth_inning_alert     = False
            st.session_state._last_pitch_video_time  = -30.0
            # BiLSTM은 백그라운드에서 계산 (앱 즉시 시작)
            if gk not in _bilstm_tasks or _bilstm_tasks[gk].get("status") != "done":
                _bilstm_tasks[gk] = {"status": "computing"}
                threading.Thread(
                    target=_run_bilstm_bg, args=(gk, _pid_tuple), daemon=True
                ).start()
            st.success(f"✅ {_meta['game_date']} {_meta['away_team']} @ {_meta['home_team']} — {len(_pitches)}구 로드 완료 (BiLSTM 계산 중...)")
            st.rerun()
        except Exception as _e:
            st.error(f"로드 실패: {_e}")
```

교체 후:
```python
    if load_btn and gk_input.strip():
        _ok, _msg = _load_game(gk_input.strip())
        if _ok:
            st.success(_msg)
            st.rerun()
        else:
            st.error(_msg)
```

- [ ] **Step 3: 동작 확인**

`streamlit run streamlit_app/app.py` 실행 → 사이드바에 game_pk `745735` 입력 후 "경기 로드" 클릭 → 기존과 동일하게 경기 데이터가 로드되고 BiLSTM 계산 중 표시가 뜨는지 확인 (리팩터 전과 동일 동작이어야 함 — 회귀 없음).

- [ ] **Step 4: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "refactor: 경기 로드 로직을 _load_game 헬퍼로 추출"
```

---

### Task 2: 최초 진입 시 고정 데모 자동 로드

**Files:**
- Modify: `streamlit_app/app.py` — 상수 추가(PITCH_META 근처) + 자동 로드 트리거 블록 추가(`_DEFAULTS` 초기화 루프 직후)

**Interfaces:**
- Consumes: `_load_game(gk_str: str) -> tuple[bool, str]` (Task 1)
- Produces: `FIXED_DEMO_GAME_PK = 775300`, `FIXED_DEMO_VIDEO_URL = "https://youtu.be/gMm3EODDb6w"` 상수 — Task 4에서 재사용. `_demo_auto_loaded` 세션 키 — 재진입/초기화 판별에 사용.

- [ ] **Step 1: 고정 데모 상수 추가**

`PITCH_META` 딕셔너리 정의 위(또는 `ROOT` 상수 근처)에 추가:

```python
# ══ 고정 데모 설정 ════════════════════════════════════════════════
FIXED_DEMO_GAME_PK   = 775300
FIXED_DEMO_VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
```

- [ ] **Step 2: `_DEFAULTS`에 가드 플래그 추가**

`_DEFAULTS` 딕셔너리에 한 줄 추가:

```python
    "_demo_auto_loaded":     False,  # 최초 진입 자동 데모 로드 완료 여부 (초기화 버튼으로 리셋 안 됨)
```

- [ ] **Step 3: 자동 로드 트리거 블록 추가**

`_DEFAULTS` 초기화 루프(`for _k, _v in _DEFAULTS.items(): ...`) 바로 뒤, "앱 재로드 후 `_pose_tasks` 사라진 경우" 주석 이전에 삽입:

```python
# ══ 최초 진입: 고정 데모 자동 로드 ═══════════════════════════════════
# game_pk가 비어있고(한 번도 로드 안 됨) 아직 자동로드를 시도한 적 없을 때만 1회 실행.
# "초기화" 버튼은 _demo_auto_loaded를 리셋하지 않으므로 재자동로드되지 않는다.
if not st.session_state._demo_auto_loaded and st.session_state.game_pk == "":
    st.session_state._demo_auto_loaded = True
    st.session_state.video_src = FIXED_DEMO_VIDEO_URL
    _demo_ok, _demo_msg = _load_game(str(FIXED_DEMO_GAME_PK))
    if not _demo_ok:
        print(f"[FixedDemo] 자동 로드 실패: {_demo_msg}")
```

주의: 이 블록은 어떤 다운로드 트리거도 호출하지 않는다 — `video_src`만 문자열로 설정하고 `_local_video_path`는 건드리지 않으므로, 아래 메인 레이아웃의 `_use_local_player = bool(_local_play_path and ...)` 판정이 자연히 `False`가 되어 `youtube_player`(iframe) 경로로만 재생된다 (스펙 B 요구사항 — 별도 코드 불필요).

- [ ] **Step 4: 동작 확인**

브라우저 시크릿 창(또는 `st.session_state` 캐시가 없는 완전 새 세션)으로 앱을 열어:
1. 사이드바 "경기 로드" 입력 없이도 game_pk 775300의 스코어보드/투수/타자 패널이 즉시 렌더링되는지 확인
2. 터미널 로그에 `[FixedDemo]` 에러가 없는지 확인
3. "초기화" 버튼 클릭 후에도 재자동로드되지 않고 빈 랜딩 화면으로 돌아가는지 확인 (스펙 A 마지막 조건)

- [ ] **Step 5: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "feat: 최초 진입 시 고정 데모 게임 자동 로드"
```

---

### Task 3: 정적 스캔 캐시 생성 스크립트 (로컬 1회 실행용)

**Files:**
- Create: `scripts/generate_fixed_demo_scan.py`

**Interfaces:**
- Produces: `streamlit_app/fixed_demo_scan.json` 파일 — Task 4에서 앱이 읽음. 스키마: `{"version": str, "pitch_times": list[float], "pitch_data": list[dict]}` (기존 `_scan_cache_path`가 쓰는 스키마와 동일).

- [ ] **Step 1: 스크립트 작성**

```python
"""
고정 데모 영상(https://youtu.be/gMm3EODDb6w, game_pk=775300)의 방송 오버레이 OCR
스캔 결과를 streamlit_app/fixed_demo_scan.json에 정적으로 저장한다.
배포 환경은 이 파일만 읽고, 실시간 다운로드/OCR을 하지 않는다. 로컬에서 1회만 실행.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "streamlit_app"))

GAME_PK   = 775300
VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
SCAN_VER  = "v5-ocr-overlay"  # app.py의 _SCAN_VER와 동일하게 유지
OUT_PATH  = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")


def main() -> None:
    from pybaseball import statcast_single_game
    from yolo_detector import resolve_video_path
    from pose_detector import scan_pitch_overlays

    print(f"[1/3] Statcast 투구 수 조회: game_pk={GAME_PK}")
    df = statcast_single_game(GAME_PK)
    n_pitches   = len(df)
    max_pitches = int((df["inning"] <= 5).sum())
    print(f"  총 {n_pitches}구, 1~5이닝 {max_pitches}구")

    print(f"[2/3] 영상 다운로드: {VIDEO_URL}")
    cache_dir  = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
    os.makedirs(cache_dir, exist_ok=True)
    video_path = resolve_video_path(VIDEO_URL, download_dir=cache_dir)
    print(f"  다운로드 완료: {video_path}")

    print("[3/3] 방송 오버레이 OCR 스캔 (1~5이닝)...")
    pitch_times, pitch_data = scan_pitch_overlays(
        video_path, expected_count=n_pitches,
        max_pitches=max_pitches, skip_start_sec=0.0,
    )
    print(f"  {len(pitch_times)}개 투구 타임스탬프 감지")

    with open(OUT_PATH, "w") as f:
        json.dump({"version": SCAN_VER, "pitch_times": pitch_times, "pitch_data": pitch_data}, f)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스크립트 실행 (시간 소요 — 영상 다운로드 + 5이닝 분량 OCR 스캔)**

Run: `python scripts/generate_fixed_demo_scan.py`
Expected: `streamlit_app/fixed_demo_scan.json` 생성, `pitch_times` 길이가 0보다 큼 (수십~백여 개 수준 예상, 1~5이닝 범위)

- [ ] **Step 3: 결과 파일 검증**

Run: `python -c "import json; d=json.load(open('streamlit_app/fixed_demo_scan.json')); print(d['version'], len(d['pitch_times']), len(d['pitch_data']))"`
Expected: `v5-ocr-overlay <N> <N>` 형태로 0이 아닌 개수 출력

- [ ] **Step 4: 커밋**

```bash
git add scripts/generate_fixed_demo_scan.py streamlit_app/fixed_demo_scan.json
git commit -m "feat: 고정 데모 영상 정적 스캔 캐시 생성 스크립트 + 결과 파일 추가"
```

---

### Task 4: 앱에서 정적 스캔 캐시 로드

**Files:**
- Modify: `streamlit_app/app.py` — 기존 "재시작 후 캐시 영상이 있고 스캔이 안 됐거나..." 블록(오프라인 스캔 완료 체크 이후, `_auto_vpath` 블록 이전 또는 이후) 근처에 추가

**Interfaces:**
- Consumes: `FIXED_DEMO_GAME_PK`(Task 2), `streamlit_app/fixed_demo_scan.json`(Task 3)

- [ ] **Step 1: 정적 캐시 로드 블록 추가**

`_SCAN_VER = "v5-ocr-overlay"` 정의 직후, `_auto_vpath` 관련 블록 이전에 삽입 (스크립트의 `SCAN_VER`와 반드시 동일한 문자열 유지):

```python
# ══ 고정 데모 게임의 정적 스캔 캐시 로드 (다운로드/실시간 OCR 없음) ═════════
if (
    st.session_state.get("game_pk") == str(FIXED_DEMO_GAME_PK)
    and not st.session_state.get("video_pitch_times")
    and st.session_state.get("_scan_status", "idle") == "idle"
):
    _fixed_scan_path = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
    if os.path.exists(_fixed_scan_path):
        try:
            with open(_fixed_scan_path) as _fsf:
                _fs = json.load(_fsf)
            if _fs.get("version") == _SCAN_VER:
                st.session_state.video_pitch_times = _fs["pitch_times"]
                st.session_state._scan_raw_data    = _fs.get("pitch_data", [])
                st.session_state.video_pitch_data  = []
                st.session_state._next_scan_idx    = 0
                st.session_state._scan_status      = "done"
                st.session_state._scan_version     = _SCAN_VER
                print(f"[FixedDemo] 정적 스캔 캐시 로드: {len(_fs['pitch_times'])}개")
            else:
                print(f"[FixedDemo] 스캔 캐시 버전 불일치: {_fs.get('version')} != {_SCAN_VER}")
        except Exception as _fse:
            print(f"[FixedDemo] 스캔 캐시 로드 실패: {_fse}")
    else:
        print(f"[FixedDemo] 스캔 캐시 파일 없음: {_fixed_scan_path}")
```

- [ ] **Step 2: 동작 확인**

`streamlit run streamlit_app/app.py`로 완전 새 세션 열기 → 자동 로드 완료 후 터미널에 `[FixedDemo] 정적 스캔 캐시 로드: N개` 출력 확인 → 영상 재생 시 투구 타임라인이 자동으로 갱신되는지(투구 카드가 넘어가는지) 확인

- [ ] **Step 3: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "feat: 고정 데모 게임에 정적 스캔 캐시 적용"
```

---

### Task 5: 수동 입력 UI를 고급 옵션(expander)으로 격하

**Files:**
- Modify: `streamlit_app/app.py` 사이드바 섹션 — "경기 로드" 라벨(`st.markdown('<p ... 경기 로드 ...')`)부터 "영상 싱크 상태/타임라인 보정" 블록(`if st.session_state._local_video_path and st.session_state.video_src: ... 타임라인 보정 ...` 끝) 직전까지, `st.divider()`(현재 898행) 이전까지의 전체 구간

**Interfaces:**
- 없음 (순수 UI 구조 변경, 로직/상태 변경 없음)

- [ ] **Step 1: 대상 구간을 `st.expander`로 감싸기**

"경기 로드" 라벨 markdown부터 (현재 코드 기준) "타임라인 보정" 블록의 `else: st.warning("영상을 재생 중일 때 클릭하세요")`까지 — 즉 `st.divider()`(898행, "경기 진행 요약" 앞) 바로 이전까지의 모든 코드를 한 단계 들여쓰기하고 다음으로 감싼다:

```python
    with st.expander("⚙️ 고급: 다른 경기 직접 불러오기", expanded=False):
        # (기존 "경기 로드" 라벨부터 타임라인 보정 블록까지 — 로직 변경 없이 들여쓰기만 추가)
        ...
```

이때 다음은 **로직을 절대 바꾸지 않는다** — 오직 들여쓰기만 4칸 추가:
- "경기 로드" 라벨 + `gk_input` + `load_btn`/`clear_btn` + 두 버튼 핸들러
- BiLSTM 상태 표시(`_bstatus` 블록) — 사용자가 다른 경기를 로드했을 때 BiLSTM 진행 상황을 바로 옆에서 보게 하기 위해 expander **안**에 포함한다
- `st.divider()`(776행)
- "경기 영상" 라벨 + `yt_url` 입력 + 다운로드 트리거 + 다운로드 상태 표시
- `if st.session_state._local_video_path and st.session_state.video_src:` 스캔/싱크 상태 + 타임라인 보정 블록 전체

이후 원래 898행의 `st.divider()`, "경기 진행 요약", "구종 범례" 블록은 expander **밖**, 기존과 같은 들여쓰기(사이드바 최상위)로 유지한다.

- [ ] **Step 2: 동작 확인**

`streamlit run streamlit_app/app.py` 실행:
1. 사이드바에 "⚙️ 고급: 다른 경기 직접 불러오기"가 접힌 상태로 보이고, 기본 화면에는 game_pk/URL 입력창이 바로 보이지 않는지 확인
2. expander를 펼치면 기존 game_pk 입력, 경기 로드/초기화 버튼, YouTube URL 입력이 그대로 동작하는지 확인 (다른 game_pk, 예: `745735` 입력 후 로드 테스트)
3. "경기 진행 요약", "구종 범례"는 expander 밖에서 평소처럼 보이는지 확인

- [ ] **Step 3: 커밋**

```bash
git add streamlit_app/app.py
git commit -m "style: 수동 경기 입력 UI를 고급 옵션 expander로 격하"
```

---

### Task 6: 전체 회귀 검증

**Files:** 없음 (검증 전용 태스크)

- [ ] **Step 1: 완전 새 세션 — 고정 데모 자동 재생 확인**

브라우저 시크릿 창으로 `streamlit run streamlit_app/app.py` 접속 → 입력 없이 game_pk 775300 로드 + YouTube iframe 재생 + BiLSTM 예측이 자동으로 시작되는지 확인 (스펙 "검증 — 로컬" 항목)

- [ ] **Step 2: 로컬 파일 재생 경로가 선택되지 않는지 확인**

터미널 로그(`[SYNC] ... lpath=False`) 또는 `st.session_state._local_video_path`가 `None`으로 유지되는지 확인 — iframe(`youtube_player`) 경로만 사용됨을 재확인

- [ ] **Step 3: 회귀 — 수동 game_pk 입력 흐름**

expander를 펼쳐 다른 game_pk(예: `745735`)를 입력하고 로드 → 기존과 동일하게 동작하는지 확인 (Task 1 리팩터 이후에도 정상)

- [ ] **Step 4: "초기화" 버튼 이후 재자동로드 안 됨 확인**

"초기화" 클릭 → 빈 랜딩 화면으로 복귀 → 같은 세션에서 새로고침(F5)해도 다시 775300이 자동 로드되지 않는지 확인. (완전히 새 세션/시크릿 창에서는 다시 자동 로드되는 것이 정상 동작.)

- [ ] **Step 5: 배포 전 안내**

로컬 검증이 끝나면 `streamlit_app/fixed_demo_scan.json`이 커밋되어 있어야 Streamlit Community Cloud 배포판에서도 동일하게 동작한다 — 배포 자체(재배포 트리거)는 이 계획의 범위 밖이며 사용자가 별도로 푸시/배포한다.
