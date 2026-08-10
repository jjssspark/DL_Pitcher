"""
PitchIQ — MLB 실시간 투구 분析 & 다음 구종 예측
Statcast 경기 데이터 기반 · YOLO 투구 타이밍 감지 · BiLSTM 예측
"""
import sys, os, time, threading, uuid, warnings, pickle, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(page_title="PitchIQ", page_icon="⚾", layout="wide",
                   initial_sidebar_state="expanded")

# ══ CSS ══════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700;900&display=swap');

[data-testid="stAppViewContainer"]{
  min-height:100vh;
  background:
    linear-gradient(115deg,transparent 49.6%,rgba(96,165,250,.05) 49.6% 50%,transparent 50%),
    linear-gradient(65deg,transparent 49.6%,rgba(96,165,250,.05) 49.6% 50%,transparent 50%),
    radial-gradient(ellipse 80% 50% at 50% -10%,rgba(59,130,246,.06),transparent),
    linear-gradient(180deg,#0a1120 0%,#080e1a 40%,#05080f 100%);
  background-attachment:fixed}
[data-testid="stSidebar"]{background:rgba(10,16,30,.97)!important;border-right:1px solid rgba(59,130,246,.18)}
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;color:#e2e8f0}
#MainMenu,footer,header{visibility:hidden}

/* 스코어보드 타이포 — 큰 숫자/코드 요소는 콘덴스드 디스플레이 폰트 */
.team-score,.pitch-code,.conf-gauge-code,.hero-title,.intro-logo,.inning-box{
  font-family:'Oswald',sans-serif}

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

/* 랜딩 스플래시 */
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

/* 스코어보드 강화 */
.seg-dot{display:inline-block;width:13px;height:13px;border-radius:3px;margin:0 2px}
.team-score{font-size:2.1rem}

/* 보조 카드 / 카드 배지 */
.panel-secondary{background:rgba(15,23,42,.5);border:1px solid rgba(59,130,246,.08);
  border-radius:10px;padding:.8rem 1rem;margin-bottom:.65rem}
.panel-secondary .panel-title{font-size:.66rem}
.card-badge{display:inline-flex;align-items:center;gap:.3rem;font-size:.66rem;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;padding:.15rem .5rem;border-radius:999px;margin-bottom:.4rem}
.card-badge-pred{background:rgba(167,139,250,.14);color:#c4b5fd;border:1px solid rgba(167,139,250,.35)}
.card-badge-actual{background:rgba(52,211,153,.12);color:#6ee7b7;border:1px solid rgba(52,211,153,.3)}
.pred-hero{border-width:1.5px!important;padding:1.3rem 1.5rem!important}
.pred-hero .pitch-code{font-size:3rem!important}

/* 게임 HUD — 원형 신뢰도 게이지 */
.conf-gauge{width:84px;height:84px;border-radius:50%;position:relative;flex-shrink:0;
  background:conic-gradient(var(--gauge-color) calc(var(--pct) * 3.6deg), rgba(255,255,255,.08) 0deg);
  display:flex;flex-direction:column;align-items:center;justify-content:center}
.conf-gauge::before{content:"";position:absolute;inset:9px;border-radius:50%;background:#0f172a}
.conf-gauge>*{position:relative;z-index:1}
.conf-gauge-code{font-size:1.25rem;font-weight:900;line-height:1}
.conf-gauge-pct{font-size:.6rem;font-weight:700;color:#94a3b8;margin-top:.1rem}

/* 게임 HUD — COMBO 스트릭 배지 */
.combo-badge{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .6rem;border-radius:999px;
  background:rgba(251,191,36,.16);border:1px solid rgba(251,191,36,.4);color:#fbbf24;
  font-size:.72rem;font-weight:800;letter-spacing:.03em;margin-bottom:.4rem;
  animation:comboPop .35s ease-out}
@keyframes comboPop{0%{transform:scale(.6);opacity:0}60%{transform:scale(1.12);opacity:1}100%{transform:scale(1)}}

/* 게임 HUD — 투구 갱신 리빌 애니메이션 */
.card-reveal{animation:cardReveal .35s ease-out}
@keyframes cardReveal{0%{opacity:0;transform:translateY(8px)}100%{opacity:1;transform:translateY(0)}}

/* 게임 HUD — LIVE 배지 */
.live-badge{display:inline-flex;align-items:center;gap:.35rem;padding:.2rem .55rem;border-radius:999px;
  background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.4);color:#f87171;
  font-size:.68rem;font-weight:800;letter-spacing:.06em}
.live-badge::before{content:"";width:6px;height:6px;border-radius:50%;background:#ef4444;
  animation:livePulse 1.5s ease-in-out infinite}
@keyframes livePulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}

/* 게임 HUD — 적중/실패 글로우 (리빌 애니메이션 겸함) */
.glow-hit{animation:cardReveal .35s ease-out,glowHit 1s ease-out}
.glow-miss{animation:cardReveal .35s ease-out,glowMiss .8s ease-out}
@keyframes glowHit{0%{box-shadow:0 0 0 0 rgba(52,211,153,.6)}100%{box-shadow:0 0 0 14px rgba(52,211,153,0)}}
@keyframes glowMiss{0%{box-shadow:0 0 0 0 rgba(248,113,113,.55)}100%{box-shadow:0 0 0 10px rgba(248,113,113,0)}}

/* 사이드바 — 스티치 구분선 & 카드 */
.stitch-divider{height:5px;margin:.7rem 0;
  border-top:1px dashed rgba(148,163,184,.28);border-bottom:1px dashed rgba(148,163,184,.28)}
.sidebar-card{background:rgba(15,23,42,.55);border:1px solid rgba(59,130,246,.1);
  border-radius:10px;padding:.75rem .9rem;margin-bottom:.7rem}
</style>""", unsafe_allow_html=True)

# ══ 고정 데모 설정 ════════════════════════════════════════════════
FIXED_DEMO_GAME_PK   = 775300
FIXED_DEMO_VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
TEAM_COLORS = {"NYY": "#0C2340", "LAD": "#005A9C"}  # 고정 데모 게임은 이 두 팀만 등장
# 어두운 배경(#0a1120)에 쓸 수 있게 밝기를 올린 변형. 구단 원색을 그대로 쓰면
# NYY 네이비(#0C2340)가 배경과 명도 차가 거의 없어 팀명이 보이지 않는다.
TEAM_ACCENTS = {"NYY": "#8aa9d6", "LAD": "#4d9fe0"}
FIXED_DEMO_VIDEO_DURATION_SEC = 8231  # 고정 데모 YouTube 영상 총 길이(초)

# ── 타임라인 앵커 ────────────────────────────────────────────────
# 영상 길이를 투구 수로 균등 분할하면 25.72초에 한 칸씩 넘어간다. 그런데 실제 중계의
# 투구 간격은 광고·리플레이·타자 교체 때문에 12초에서 626초까지 널뛴다. 균등 가정으로는
# 구조적으로 어긋나고 화면에는 "타임라인이 공 한 개 느리다"로 보인다 (TS-007, TS-008).
#
# 방송 스코어버그의 투수 투구수(P:N)를 읽어둔 지점을 앵커로 삼고 사이를 보간한다.
# 앵커 파일이 없으면 빈 목록이 되어 기존 균등 분할 그대로 동작한다.
from timeline_anchor import index_at_time, resolve_anchors, time_at_index  # noqa: E402

TIMELINE_COUNTER_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_anchors.json")


@st.cache_data(show_spinner=False)
def _load_timeline_counters() -> list[tuple[float, int | None]]:
    """사전에 읽어둔 (영상 시각, P:N) 관측. 파일이 없으면 빈 목록."""
    if not os.path.exists(TIMELINE_COUNTER_PATH):
        return []
    try:
        rows = json.load(open(TIMELINE_COUNTER_PATH))
    except Exception:
        return []
    return [(float(r["t"]), r.get("counter")) for r in rows if "t" in r]


def _timeline_anchors(pitches: list[dict]) -> list[tuple[float, int]]:
    """관측을 앵커로 푼다. 경기가 바뀌면 결과도 달라지므로 투구 수로 캐시를 가른다."""
    key = len(pitches)
    cache = st.session_state.setdefault("_anchor_cache", {})
    if key not in cache:
        cache[key] = resolve_anchors(
            _load_timeline_counters(), pitches, FIXED_DEMO_VIDEO_DURATION_SEC
        )
    return cache[key]

# ══ 구종 메타 ══════════════════════════════════════════════════════
PITCH_META = {
    "FF": {"name": "포심 패스트볼", "color": "#ef4444", "emoji": "🔴"},
    "FA": {"name": "패스트볼",      "color": "#f87171", "emoji": "🔴"},
    "SI": {"name": "싱커",          "color": "#f97316", "emoji": "🟠"},
    "FC": {"name": "커터",          "color": "#eab308", "emoji": "🟡"},
    "SL": {"name": "슬라이더",      "color": "#3b82f6", "emoji": "🔵"},
    "ST": {"name": "스위퍼",        "color": "#06b6d4", "emoji": "🩵"},
    "CU": {"name": "커브",          "color": "#8b5cf6", "emoji": "🟣"},
    "KC": {"name": "너클커브",      "color": "#a855f7", "emoji": "🔮"},
    "CH": {"name": "체인지업",      "color": "#10b981", "emoji": "🟢"},
    "FS": {"name": "스플리터",      "color": "#0891b2", "emoji": "💧"},
    "FO": {"name": "포크볼",        "color": "#0e7490", "emoji": "💧"},
    "KN": {"name": "너클볼",        "color": "#78716c", "emoji": "🪨"},
    "EP": {"name": "이피어스",      "color": "#6b7280", "emoji": "🐢"},
    "CS": {"name": "슬로커브",      "color": "#7c3aed", "emoji": "💜"},
    "OTHER": {"name": "기타",       "color": "#475569", "emoji": "⚫"},
}
FASTBALLS = {"FF", "FA", "SI", "FC"}
BREAKING  = {"SL", "ST", "CU", "KC", "CS"}
OFFSPEED  = {"CH", "FS", "FO", "KN", "EP"}

# ══ 프로세스 레벨 캐시 (Streamlit 재로드 시에도 유지) ════════════
@st.cache_resource
def _make_pose_tasks() -> dict:
    return {}

@st.cache_resource
def _make_bilstm_tasks() -> dict:
    return {}

@st.cache_resource
def _make_model_ref() -> list:
    return [None]  # [0]에 모델 저장

@st.cache_resource
def _make_scan_tasks() -> dict:
    return {}

@st.cache_resource
def _make_cv_tasks() -> dict:
    return {}

@st.cache_resource
def _make_cv_model_ref() -> list:
    return [None, None]  # [분류기, 공 감지기]

_pose_tasks   = _make_pose_tasks()
_bilstm_tasks = _make_bilstm_tasks()
_model_ref    = _make_model_ref()
_scan_tasks   = _make_scan_tasks()
_cv_tasks     = _make_cv_tasks()
_cv_model_ref = _make_cv_model_ref()
_game_cache: dict = {}


# ══ 경기 데이터 로딩 ══════════════════════════════════════════════
def precompute_bilstm(game_pk: int, pitcher_id_tuple: tuple) -> list:
    """CSV 1번, 모델 1번 로드 후 배치 예측. 디스크 캐시 사용."""
    import joblib
    from pybaseball import statcast_single_game

    # 디스크 캐시 확인 (같은 game_pk는 재계산 불필요)
    _cache_dir  = os.path.join(ROOT, "streamlit_app", ".bilstm_cache")
    _cache_file = os.path.join(_cache_dir, f"{game_pk}.pkl")
    os.makedirs(_cache_dir, exist_ok=True)
    if os.path.exists(_cache_file):
        with open(_cache_file, "rb") as _f:
            return pickle.load(_f)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from feature_engineering import build_features

    _LABEL_MAP  = {0:"FF", 1:"SI", 2:"FC", 3:"SL", 4:"CU", 5:"CH", 6:"FS", 7:"OTHER"}
    _SEQ_LEN    = 5
    _SEQ_COLS   = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z", "pitch_label"]
    _CTX_COLS   = [
        "balls", "strikes", "out_0", "out_1", "out_2",
        "inning", "inning_top", "on_1b_flag", "on_2b_flag", "on_3b_flag",
        "score_diff", "stand_enc", "p_throws_enc",
        "p_ff_pct", "p_si_pct", "p_fc_pct", "p_sl_pct",
        "p_cu_pct", "p_ch_pct", "p_fs_pct",
        "pitcher_count_top_enc", "matchup_count",
    ]
    _SCALE_COLS = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]

    data_path   = os.path.join(ROOT, "data", "raw", "statcast_2025_full.csv")
    model_path  = os.path.join(ROOT, "models", "pitch_predictor.h5")
    scaler_path = os.path.join(ROOT, "models", "scaler.pkl")

    df_raw   = pd.read_csv(data_path)
    df_train = df_raw[df_raw["pitcher"].isin(pitcher_id_tuple)].copy()

    from tensorflow import keras
    model  = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    pitcher_dfs: dict = {}
    for pid in pitcher_id_tuple:
        sub = df_train[df_train["pitcher"] == pid].copy()
        if len(sub) == 0:
            continue
        try:
            sub   = build_features(sub)
            avail = [c for c in _SCALE_COLS if c in sub.columns]
            sub[avail] = scaler.transform(sub[avail])
            pitcher_dfs[pid] = sub.sort_values(
                ["game_date", "at_bat_number", "pitch_number"]
            ).reset_index(drop=True)
        except Exception:
            pass

    df_game = statcast_single_game(game_pk)
    df_game = df_game.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)

    pitcher_pos: dict   = {pid: 0 for pid in pitcher_id_tuple}
    game_to_batch: list = []
    X_seq_list, X_ctx_list, X_pit_list, X_bat_list = [], [], [], []

    for game_i, row in df_game.iterrows():
        pid  = int(row["pitcher"])
        df_p = pitcher_dfs.get(pid)
        if df_p is None:
            continue
        idx = pitcher_pos[pid]
        pitcher_pos[pid] += 1
        if idx < _SEQ_LEN or idx >= len(df_p):
            continue
        window   = df_p.iloc[idx - _SEQ_LEN : idx]
        row_feat = df_p.iloc[idx]
        seq_c = [c for c in _SEQ_COLS if c in window.columns]
        ctx_c = [c for c in _CTX_COLS  if c in row_feat.index]
        if len(seq_c) < 2 or len(ctx_c) < 2:
            continue
        X_seq_list.append(window[seq_c].values.astype(np.float32))
        X_ctx_list.append(row_feat[ctx_c].values.astype(np.float32))
        X_pit_list.append([int(row_feat["pitcher"]) % 2000])
        X_bat_list.append([int(row_feat.get("batter", 0)) % 3000])
        game_to_batch.append((int(game_i), len(X_seq_list) - 1))

    results = [None] * len(df_game)
    if X_seq_list:
        probs_all = model.predict(
            {
                "seq_input":     np.array(X_seq_list),
                "ctx_input":     np.array(X_ctx_list),
                "pitcher_input": np.array(X_pit_list),
                "batter_input":  np.array(X_bat_list),
            },
            batch_size=64,
            verbose=0,
        )
        for game_i, b_i in game_to_batch:
            probs    = probs_all[b_i]
            next_idx = int(np.argmax(probs))
            results[game_i] = {
                "next_pitch":    _LABEL_MAP[next_idx],
                "confidence":    float(probs[next_idx]),
                "probabilities": {_LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
                "source":        "BiLSTM",
            }

    # 디스크 캐시 저장
    with open(_cache_file, "wb") as _f:
        pickle.dump(results, _f)
    return results


@st.cache_data(show_spinner=False)
def fetch_game_pitches(game_pk: int) -> tuple:
    """Statcast에서 경기 전체 투구 데이터 수집 및 가공"""
    from pybaseball import statcast_single_game, playerid_reverse_lookup

    df = statcast_single_game(game_pk)
    df = df.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)

    # 투수 이름 조회
    p_ids = [int(x) for x in df["pitcher"].dropna().unique()]
    try:
        p_lu  = playerid_reverse_lookup(p_ids, key_type="mlbam")
        p_map = {
            int(r["key_mlbam"]): f"{r['name_last'].title()}, {r['name_first'].title()}"
            for _, r in p_lu.iterrows()
        }
    except Exception:
        p_map = {}

    meta = {
        "home_team":    df["home_team"].iloc[0],
        "away_team":    df["away_team"].iloc[0],
        "game_date":    str(df["game_date"].iloc[0])[:10],
        "total_pitches": len(df),
    }

    pitches = []
    for _, row in df.iterrows():
        pid  = int(row["pitcher"]) if pd.notna(row["pitcher"]) else 0
        # inning_topbot: Top = away bats, Bot = home bats
        is_top = str(row.get("inning_topbot", "Top")).lower().startswith("t")
        b_score = int(row["bat_score"]) if pd.notna(row.get("bat_score")) else 0
        f_score = int(row["fld_score"]) if pd.notna(row.get("fld_score")) else 0
        if is_top:
            away_score, home_score = b_score, f_score
        else:
            home_score, away_score = b_score, f_score

        pt = str(row.get("pitch_type", "") or "")
        pitches.append({
            "pitch_idx":       len(pitches),
            "at_bat":          int(row["at_bat_number"]),
            "pitch_num_in_ab": int(row["pitch_number"]),
            "pitcher_id":      pid,
            "pitcher_name":    p_map.get(pid, f"투수 #{pid}"),
            "pitcher_hand":    str(row.get("p_throws", "R")),
            "batter_name":     str(row.get("player_name", f"타자 #{int(row['batter'])}")),
            "batter_hand":     str(row.get("stand", "R")),
            "inning":          int(row["inning"]),
            "inning_topbot":   "Top" if is_top else "Bot",
            "balls":           int(row["balls"]),
            "strikes":         int(row["strikes"]),
            "outs":            int(row["outs_when_up"]),
            "on_1b":           pd.notna(row.get("on_1b")),
            "on_2b":           pd.notna(row.get("on_2b")),
            "on_3b":           pd.notna(row.get("on_3b")),
            "away_score":      away_score,
            "home_score":      home_score,
            "pitch_type":      pt if pt in PITCH_META else ("OTHER" if pt else ""),
            "pitch_type_raw":  pt,
            "release_speed":   float(row["release_speed"]) if pd.notna(row.get("release_speed")) else None,
            "description":     str(row.get("description", "")),
            "events":          str(row.get("events", "") or ""),
            "away_team":       meta["away_team"],
            "home_team":       meta["home_team"],
        })

    return pitches, meta


def _predict_next(pitches: list, idx: int) -> dict:
    """
    현재까지 투구 이력 기반으로 다음 구종 예측.
    파이프라인 모델이 없을 때 사용하는 통계 기반 예측.
    """
    codes = [c for c in PITCH_META if c != "OTHER"]
    cur   = pitches[idx]
    pid   = cur["pitcher_id"]

    # 이 투수가 이번 경기에서 던진 구종 분포
    history = [p["pitch_type"] for p in pitches[:idx]
               if p["pitcher_id"] == pid and p["pitch_type"] and p["pitch_type"] != "OTHER"]

    if history:
        cnt   = Counter(history)
        total = sum(cnt.values())
        base  = {c: cnt.get(c, 0) / total for c in codes}
    else:
        # 투구 이력 없음 → 일반적 패스트볼 비중 높은 기본값
        base = {c: 0.01 for c in codes}
        for c in ["FF", "FA", "SI"]: base[c] = 0.28
        for c in ["SL", "ST"]:       base[c] = 0.18
        for c in ["CH", "CU"]:       base[c] = 0.10
        total = sum(base.values()); base = {c: v/total for c, v in base.items()}

    # 카운트 상황 보정
    balls, strikes = cur["balls"], cur["strikes"]
    adj = dict(base)
    if strikes == 2:          # 2스트라이크: 변화구 ↑
        for c in BREAKING | OFFSPEED: adj[c] = adj.get(c, 0) * 1.35
    if balls == 3:            # 3볼: 스트라이크 필요 → 패스트볼 ↑
        for c in FASTBALLS:  adj[c] = adj.get(c, 0) * 1.40
    if balls == 0 and strikes == 0:  # 첫 구: 패스트볼 선호
        for c in FASTBALLS:  adj[c] = adj.get(c, 0) * 1.20

    total = sum(adj.values())
    probs = {c: round(adj.get(c, 0) / total, 4) for c in codes} if total > 0 \
            else {c: round(1/len(codes), 4) for c in codes}

    best = max(probs, key=probs.get)
    return {"next_pitch": best, "confidence": probs[best], "probabilities": probs}


# ══ UI 헬퍼 ══════════════════════════════════════════════════════
def _diamond_svg(on_1b: bool, on_2b: bool, on_3b: bool) -> str:
    def sq(on, x, y):
        c = "#fbbf24" if on else "rgba(100,116,139,.2)"
        return f'<rect x="{x}" y="{y}" width="12" height="12" rx="2" fill="{c}" transform="rotate(45,{x+6},{y+6})"/>'
    return (
        '<svg width="50" height="50" viewBox="0 0 50 50">'
        + sq(on_2b, 19, 2) + sq(on_3b, 2, 19) + sq(on_1b, 36, 19)
        + '<rect x="19" y="36" width="12" height="12" rx="2" fill="rgba(100,116,139,.12)" transform="rotate(45,25,42)"/>'
        + '</svg>'
    )


def _count_dots(n: int, total: int, color_on: str, color_off: str = "rgba(100,116,139,.2)") -> str:
    return "".join(
        f'<span class="count-dot" style="background:{"' + color_on + '" if i < n else "' + color_off + '"}"></span>'
        for i in range(total)
    )


def _count_dots_simple(n, total, color_on, color_off="rgba(100,116,139,.2)"):
    dots = ""
    for i in range(total):
        c = color_on if i < n else color_off
        dots += f'<span class="count-dot" style="background:{c}"></span>'
    return dots


def _seg_dots(n: int, total: int, color_on: str, color_off: str = "rgba(100,116,139,.2)") -> str:
    dots = ""
    for i in range(total):
        c = color_on if i < n else color_off
        dots += f'<span class="seg-dot" style="background:{c}"></span>'
    return dots


def _score_color(diff: int) -> str:
    if diff > 0: return "#34d399"
    if diff < 0: return "#f87171"
    return "#94a3b8"


# ══ 포즈 감지 + 영상 다운로드 백그라운드 ════════════════════════════
def _get_pose_model():
    if _model_ref[0] is None:
        from pose_detector import load_pose_model
        _model_ref[0] = load_pose_model()
    return _model_ref[0]


def _run_pose_check_bg(task_id: str, video_path: str, check_time: float) -> None:
    """현재 영상 시간 주변 프레임에서 투구 모션 감지 (백그라운드)"""
    try:
        from pose_detector import extract_frames, detect_pitch_motion
        model  = _get_pose_model()
        frames = extract_frames(video_path, check_time, duration=2.5, step=2, max_frames=20)
        result = detect_pitch_motion(model, frames)
        is_pitch, max_score, avg_score = result if isinstance(result, tuple) else (result, 0.0, 0.0)
        print(f"[PitchDetect] t={check_time:.1f}s  frames={len(frames)}  max={max_score:.4f}  avg={avg_score:.4f}  PITCH={is_pitch}")
        _pose_tasks[task_id] = {
            "status": "done", "is_pitch": is_pitch,
            "check_time": check_time, "max_score": max_score, "avg_score": avg_score,
        }
    except Exception as e:
        print(f"[PitchDetect] ERROR t={check_time:.1f}s: {e}")
        _pose_tasks[task_id] = {"status": "error", "is_pitch": False,
                                "error": str(e), "check_time": check_time}


def _start_pose_check(video_path: str, check_time: float) -> str:
    task_id = str(uuid.uuid4())[:8]
    _pose_tasks[task_id] = {"status": "processing", "check_time": check_time}
    threading.Thread(
        target=_run_pose_check_bg, args=(task_id, video_path, check_time), daemon=True
    ).start()
    return task_id


# 재생 중 오버레이를 OCR 하던 _start_ocr_check / _run_ocr_check_bg 는 제거했다 (TS-031).
# 오버레이가 투구보다 늦게 뜨는 구조라 지연을 못 줄인다. pose_detector의
# ocr_check_pitch_overlay 자체는 남아 있다 — 오프라인 앵커 생성이 그걸 쓴다
# (scripts/build_timeline_anchors.py).


# ══ CV 궤적 구종 판정 ═════════════════════════════════════════════
# 위 실측 표시는 Statcast API에서 온다 — 정답이지만 API가 있어야만 나온다.
# 아래 경로는 같은 값을 영상만으로 낸다. 범위는 FASTBALL vs BREAKING 2분류다
# (OFFSPEED는 중계 궤적으로 안 갈린다 — ADR-0012).
def _get_cv_models():
    """분류기 + 공 감지기 지연 로드. 둘 다 무거워서 최초 판정 때 한 번만 올린다."""
    if _cv_model_ref[0] is None:
        from pitch_type_cv.group_classifier import load_classifier
        from pitch_type_cv.live_classifier import TWO_CLASS_MODEL_PATH
        _cv_model_ref[0] = load_classifier(TWO_CLASS_MODEL_PATH)
    if _cv_model_ref[1] is None:
        from ultralytics import YOLO
        _cv_model_ref[1] = YOLO(os.path.join(ROOT, "models", "ball_broadcast_v1.pt"))
    return _cv_model_ref[0], _cv_model_ref[1]


def _run_cv_check_bg(task_id: str, video_path: str, pitch_time: float, pitch_idx: int) -> None:
    """영상 궤적으로 구종 판정 (백그라운드). 실패는 실패로 남긴다."""
    try:
        from pitch_type_cv.live_classifier import classify_video_pitch
        classifier, detector = _get_cv_models()
        verdict = classify_video_pitch(classifier, detector, video_path, pitch_time)
        _cv_tasks[task_id] = {
            "status": "done", "pitch_idx": pitch_idx, "pitch_time": pitch_time,
            "group": verdict.group, "confidence": verdict.confidence,
            "probabilities": verdict.probabilities,
            "n_points": verdict.n_points, "reason": verdict.reason,
        }
        print(f"[CV] idx={pitch_idx} t={pitch_time:.1f}s → "
              f"{verdict.group or '판정불가(' + verdict.reason + ')'} "
              f"conf={verdict.confidence:.2f} pts={verdict.n_points}")
    except Exception as e:
        print(f"[CV] ERROR idx={pitch_idx}: {e}")
        _cv_tasks[task_id] = {
            "status": "error", "pitch_idx": pitch_idx, "group": None,
            "reason": "error", "error": str(e), "n_points": 0, "confidence": 0.0,
        }


def _start_cv_check(video_path: str, pitch_time: float, pitch_idx: int) -> str:
    task_id = str(uuid.uuid4())[:8]
    _cv_tasks[task_id] = {"status": "processing", "pitch_idx": pitch_idx}
    threading.Thread(
        target=_run_cv_check_bg, args=(task_id, video_path, pitch_time, pitch_idx), daemon=True
    ).start()
    return task_id


def _statcast_to_two_class(pitch_type: str | None) -> str | None:
    """Statcast 구종 코드 → 2분류 정답. OFFSPEED와 미매핑은 채점에서 뺀다."""
    if pitch_type in FASTBALLS:
        return "FASTBALL"
    if pitch_type in BREAKING:
        return "BREAKING"
    return None


@st.cache_resource
def _get_video_server(directory: str) -> int:
    """
    로컬 비디오 파일을 HTTP로 serve. 포트 번호 반환.

    Range 요청을 직접 구현한다. SimpleHTTPRequestHandler는 Range를 처리하지 않는데
    Accept-Ranges 헤더만 붙어 있어서, 브라우저가 탐색 가능하다고 믿고 Range를 보내면
    서버가 그걸 무시하고 파일 전체를 200으로 돌려줬다. 1.1GB 데모 영상에서 재생이
    0:00에 멈춰 있던 원인이다. 헤더가 없었으면 브라우저가 순차 다운로드로 폴백했을
    텐데, 있다고 광고하니 되지도 않는 Range 재생을 시도했다.

    서버도 ThreadingTCPServer로 바꾼다. 단일 스레드로는 영상 요청 하나가 커넥션을
    붙잡고 있는 동안 나머지가 전부 대기한다 — 브라우저는 영상에 커넥션을 여럿 연다.
    """
    import http.server, re, shutil, socketserver

    class _Handler(http.server.SimpleHTTPRequestHandler):
        # HTTP/1.0으로는 Chrome 미디어 스택이 영상을 못 연다. fetch()로는 206과 바이트가
        # 정상적으로 오는데 <video>는 readyState 0에서 멈춰 있었다 — 미디어 재생은
        # 바이트 레인지 탐색에 지속 연결을 요구한다. Content-Length는 두 경로 모두에서
        # 반드시 나가므로 1.1로 올려도 프레이밍이 깨지지 않는다.
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Accept-Ranges", "bytes")
            # 교차 출처에서는 이 둘이 기본으로 JS에 노출되지 않는다.
            self.send_header("Access-Control-Expose-Headers",
                             "Content-Range, Accept-Ranges, Content-Length")
            super().end_headers()

        def send_head(self):
            self._range_remaining = None
            range_header = self.headers.get("Range")
            path = self.translate_path(self.path)
            if not range_header or os.path.isdir(path):
                return super().send_head()

            matched = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not matched:
                return super().send_head()   # 형식을 모르면 전체를 준다

            try:
                stream = open(path, "rb")
            except OSError:
                self.send_error(404, "File not found")
                return None

            size = os.fstat(stream.fileno()).st_size
            start_raw, end_raw = matched.groups()
            if start_raw == "":
                # bytes=-N : 마지막 N바이트
                if end_raw == "":
                    stream.close()
                    self.send_error(400, "Bad Range")
                    return None
                length = min(int(end_raw), size)
                start, end = size - length, size - 1
            else:
                start = int(start_raw)
                end = min(int(end_raw), size - 1) if end_raw else size - 1

            if start >= size or start > end:
                stream.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None

            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            stream.seek(start)
            self._range_remaining = end - start + 1
            return stream

        def copyfile(self, source, outputfile):
            remaining = self._range_remaining
            if remaining is None:
                shutil.copyfileobj(source, outputfile)
                return
            # 요청받은 구간만 보낸다. 그냥 copyfileobj를 부르면 seek 이후 전부가 나간다.
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)

        def log_message(self, *args):
            pass

    class _Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    for port in range(8510, 8530):
        try:
            server = _Server(("", port), _Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return port
        except OSError:
            continue
    return 8510




def _run_bilstm_bg(game_pk: int, pitcher_id_tuple: tuple) -> None:
    """BiLSTM 예측을 백그라운드에서 실행. 완료 시 _bilstm_tasks에 저장."""
    try:
        results = precompute_bilstm(game_pk, pitcher_id_tuple)
        _bilstm_tasks[game_pk] = {"status": "done", "results": results}
    except Exception as e:
        _bilstm_tasks[game_pk] = {"status": "error", "error": str(e)}


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
        # BiLSTM은 백그라운드에서 계산 (앱 즉시 시작)
        if gk not in _bilstm_tasks or _bilstm_tasks[gk].get("status") != "done":
            _bilstm_tasks[gk] = {"status": "computing"}
            threading.Thread(
                target=_run_bilstm_bg, args=(gk, _pid_tuple), daemon=True
            ).start()
        return True, f"✅ {_meta['game_date']} {_meta['away_team']} @ {_meta['home_team']} — {len(_pitches)}구 로드 완료 (BiLSTM 계산 중...)"
    except Exception as _e:
        return False, f"로드 실패: {_e}"


def _render_intro_splash() -> None:
    st.markdown(
        '<div class="intro-splash">'
        '<div class="intro-logo">⚾ PitchIQ</div>'
        '<div class="intro-tagline">MLB 실시간 투구 분석 &amp; 다음 구종 예측 시스템</div>'
        '<div class="intro-badge">LIVE 데모 준비 중</div>'
        '</div>', unsafe_allow_html=True)


def _scan_cache_path(video_path: str, version: str) -> str:
    import hashlib
    key = hashlib.md5(f"{video_path}|{version}".encode()).hexdigest()[:12]
    cache_dir = os.path.join(ROOT, "streamlit_app", ".scan_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.json")


def _run_scan_bg(task_id: str, video_path: str, total_pitches: int, max_pitches: int = 0) -> None:
    """방송 오버레이 OCR 스캔. max_pitches>0이면 해당 수 찾고 조기 종료."""
    try:
        from pose_detector import scan_pitch_overlays
        times, pdata = scan_pitch_overlays(
            video_path, expected_count=total_pitches,
            max_pitches=max_pitches, skip_start_sec=0.0,
        )
        _scan_tasks[task_id] = {"status": "done", "pitch_times": times, "pitch_data": pdata}
        # 디스크 캐시 저장 — 재시작 시 재스캔 방지
        try:
            cache_path = _scan_cache_path(video_path, _SCAN_VER)
            with open(cache_path, "w") as _cf:
                json.dump({"version": _SCAN_VER, "pitch_times": times, "pitch_data": pdata}, _cf)
            print(f"[VideoScan] 캐시 저장: {cache_path}")
        except Exception as _ce:
            print(f"[VideoScan] 캐시 저장 실패: {_ce}")
    except Exception as e:
        print(f"[VideoScan] ERROR: {e}")
        _scan_tasks[task_id] = {"status": "error", "error": str(e)}


# ══ 세션 상태 초기화 ══════════════════════════════════════════════
_DEFAULTS = {
    "game_pk":               "",
    "game_pitches":          [],
    "game_meta":             {},
    "bilstm_preds":          [],
    "bilstm_status":         "idle",
    "current_pitch_idx":     0,
    "video_src":             None,
    "_upload_name":          "",
    "_local_video_path":     None,   # 다운로드된 로컬 MP4 경로
    "_pose_task_id":         None,   # 투구 감지 태스크
    "_pose_last_check_time": -99.0,  # 마지막 포즈 체크 시각
    "_last_pitch_video_time": 0.0,  # 처음 12초 자동 쿨다운 (영상 로딩 false positive 방지)
    "seek_to":               None,
    "is_playing":            False,
    "video_synced":          False,
    "video_pitch_times":     [],    # 스캔 타임스탬프: [video_sec, ...]
    "video_pitch_data":      [],    # 영상에서 읽은 구종·구속 (list, MLB idx로 확장).
                                    # TS-031 이후 채우는 곳이 없어 항상 비어 있고,
                                    # 표시는 Statcast 값으로 폴백한다. 오프라인 스캔이
                                    # 투구 시각을 전부 확보하면 여기를 다시 채운다.
    "_scan_raw_data":        [],    # 스캔 순서 기준 raw OCR 데이터 (scan idx → {type, speed})
    "_next_scan_idx":        0,     # 다음 처리할 스캔 타임스탬프 인덱스
    "cv_enabled":            False, # CV 궤적 판정 사용 여부 (사이드바 토글, 기본 꺼짐)
    "_cv_verdicts":          {},    # 투구 idx -> CV 판정 결과 (영상만으로 낸 실측)
    "_cv_task_idx":          {},    # 투구 idx -> 진행 중 태스크 id (중복 실행 방지)
    "_cv_hits":              0,     # API 정답 대비 CV 적중 수
    "_cv_scored":            0,     # 채점된 수 (판정 성공 & 정답이 2분류인 것만)
    "_cv_unavailable":       0,     # 궤적을 못 잡아 판정 불가한 수
    "_scan_task_id":         None,
    "_scan_status":          "idle",  # idle | scanning | done | error
    "_scan_version":         "",
    "_sync_activated":       False,   # 첫 투구 감지 후 True
    "_last_ocr_mlb_idx":    -1,      # 마지막으로 확인된 MLB 투구 인덱스
    "_sixth_inning_alert":   False,   # 6회초 알림 플래그
    "_vid_t":                None,    # 저장된 영상 재생 시각
    "_vid_t_wall":           0.0,     # _vid_t 수신 당시 벽시계 (wall clock)
    "_vid_pl":               False,   # 저장된 재생 중 여부
    "_demo_auto_loaded":     False,  # 최초 진입 자동 데모 로드 완료 여부 (초기화 버튼으로 리셋 안 됨)
    "_intro_shown":          False,  # 세션당 스플래시 1회만 표시
    "pred_streak":           0,      # 연속 예측 적중 횟수 (COMBO 배지)
    "_streak_calc_idx":     -1,      # pred_streak을 마지막으로 갱신한 c_idx (재렌더 중복 방지)
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══ 최초 진입: 고정 데모 자동 로드 ═══════════════════════════════════
# game_pk가 비어있고(한 번도 로드 안 됨) 아직 자동로드를 시도한 적 없을 때만 1회 실행.
# "초기화" 버튼은 _demo_auto_loaded를 리셋하지 않으므로 재자동로드되지 않는다.
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

# 앱 재로드 후 _pose_tasks 사라진 경우: 캐시 디렉토리에서 영상 파일 자동 감지
_cache_dir = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
if not st.session_state._local_video_path and os.path.exists(_cache_dir):
    _cached = sorted(
        [f for f in os.listdir(_cache_dir) if f.endswith((".mp4", ".mkv", ".webm"))],
        key=lambda f: os.path.getmtime(os.path.join(_cache_dir, f)),
        reverse=True,
    )
    if _cached:
        st.session_state._local_video_path = os.path.join(_cache_dir, _cached[0])

# 태스크 ID가 있는데 _pose_tasks에 없으면 (재로드로 결과 소실) 초기화
if st.session_state._pose_task_id and st.session_state._pose_task_id not in _pose_tasks:
    st.session_state._pose_task_id = None

# 재시작 후 캐시 영상이 있고 스캔이 안 됐거나 파라미터 버전이 다르면 재스캔
_SCAN_VER   = "v5-ocr-overlay"

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

# 오프라인 스캔 완료 체크
_stid_check = st.session_state.get("_scan_task_id")
if _stid_check and _scan_tasks.get(_stid_check, {}).get("status") == "done":
    _done_task = _scan_tasks[_stid_check]
    st.session_state.video_pitch_times = _done_task["pitch_times"]
    st.session_state._scan_raw_data    = _done_task.get("pitch_data", [])
    st.session_state.video_pitch_data  = []
    st.session_state._next_scan_idx    = 0
    st.session_state._scan_status      = "done"
    st.session_state._scan_version     = _SCAN_VER
    st.session_state._scan_task_id     = None
    st.rerun()
_auto_vpath = st.session_state.get("_local_video_path")
_ver_stale  = st.session_state.get("_scan_version", "") != _SCAN_VER
if _ver_stale and st.session_state.get("_scan_status") == "done":
    # 이전 파라미터로 된 스캔 결과 무효화
    st.session_state.video_pitch_times = []
    st.session_state._scan_status      = "idle"
    st.session_state._sync_activated   = False
if (
    _auto_vpath and os.path.exists(_auto_vpath)
    and not st.session_state.get("video_pitch_times")
    and st.session_state.get("_scan_status", "idle") == "idle"
    and not st.session_state.get("_scan_task_id")
    and st.session_state.get("game_pitches")
):
    # 디스크 캐시 확인 — 있으면 즉시 로드, 없으면 백그라운드 스캔
    _disk_cache = _scan_cache_path(_auto_vpath, _SCAN_VER)
    if os.path.exists(_disk_cache):
        try:
            with open(_disk_cache) as _dcf:
                _dc = json.load(_dcf)
            if _dc.get("version") == _SCAN_VER:
                st.session_state.video_pitch_times = _dc["pitch_times"]
                st.session_state._scan_raw_data    = _dc.get("pitch_data", [])
                st.session_state.video_pitch_data  = []
                st.session_state._next_scan_idx    = 0
                st.session_state._scan_status      = "done"
                st.session_state._scan_version     = _SCAN_VER
                print(f"[VideoScan] 디스크 캐시 로드: {len(_dc['pitch_times'])}개")
        except Exception as _dce:
            print(f"[VideoScan] 캐시 로드 실패: {_dce}")

    if not st.session_state.get("video_pitch_times"):
        _gp          = st.session_state["game_pitches"]
        _n_pitches   = len(_gp)
        _max_pitches = sum(1 for p in _gp if p.get("inning", 0) <= 5)
        _stid = str(uuid.uuid4())[:8]
        _scan_tasks[_stid] = {"status": "scanning"}
        threading.Thread(
            target=_run_scan_bg, args=(_stid, _auto_vpath, _n_pitches, _max_pitches), daemon=True
        ).start()
        st.session_state._scan_task_id = _stid
        st.session_state._scan_status  = "scanning"

# BiLSTM 백그라운드 완료 체크 — 완료됐으면 session state에 반영하고 rerun
_gk_str = st.session_state.get("game_pk", "")
if _gk_str and st.session_state.get("bilstm_status") == "computing":
    _gk_int = int(_gk_str)
    _bt = _bilstm_tasks.get(_gk_int, {})
    if _bt.get("status") == "done":
        st.session_state.bilstm_preds  = _bt["results"]
        st.session_state.bilstm_status = "done"
        st.rerun()

pitches   = st.session_state.game_pitches
meta      = st.session_state.game_meta
c_idx     = st.session_state.current_pitch_idx
loaded    = bool(pitches)

# ── 완료된 CV 판정 수거 & 채점 ──
# 백그라운드 스레드가 _cv_tasks에 넣은 결과를 세션 상태로 옮긴다. 채점은 여기서
# 한 번만 한다 — 렌더 중에 세면 재렌더마다 중복 집계된다 (스트릭 로직이 같은 이유로
# _streak_calc_idx를 둔다).
for _cvi, _cvt in list(st.session_state._cv_task_idx.items()):
    _cvr = _cv_tasks.get(_cvt)
    if _cvr is None:
        # 태스크 딕셔너리가 사라진 경우(앱 재시작, cache_resource 해제). 그냥 두면
        # 이 항목이 영원히 안 지워지고 아래 폴링이 1.5초마다 재실행을 무한히 걸어
        # 버튼 클릭이 전부 재렌더에 묻힌다.
        del st.session_state._cv_task_idx[_cvi]
        continue
    if _cvr.get("status") == "processing":
        continue
    _truth = _statcast_to_two_class(pitches[_cvi]["pitch_type"]) if _cvi < len(pitches) else None
    st.session_state._cv_verdicts[_cvi] = {**_cvr, "truth": _truth}
    if _cvr.get("group") is None:
        st.session_state._cv_unavailable += 1
    elif _truth:
        # 정답이 OFFSPEED면 채점에서 뺀다. 2분류 모델은 그 클래스를 낼 수 없어
        # 무조건 오답이 되고, 그건 모델이 아니라 범위의 문제다 (ADR-0012).
        st.session_state._cv_scored += 1
        if _truth == _cvr["group"]:
            st.session_state._cv_hits += 1
    del st.session_state._cv_task_idx[_cvi]


# ══ 사이드바 ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:.8rem 0 .4rem">'
        '<div style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">⚾ PitchIQ</div>'
        '<div style="font-size:.78rem;color:#475569;margin-top:.05rem">MLB 투구 예측 시스템</div>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)

    # 프레임 샘플링을 학습과 맞춘 뒤(TS-028) 앱 정확도가 0.410 -> 0.767로 올라
    # 실험실 수치(0.783)와 같아졌다. 다만 투구마다 YOLO를 돌리므로 재생이 무거워질 수
    # 있어 기본은 꺼둔다 — 켜고 끄는 판단은 사용자에게 맡긴다.
    st.checkbox(
        "CV 궤적 판정",
        key="cv_enabled",
        help="Statcast 없이 영상 궤적만으로 속구/변화구를 판정한다. 학습에 쓰지 않은 "
             "경기에서 정확도 76.7%(기준선 58.1%). 투구마다 YOLO를 돌려 재생이 무거워질 수 있다.",
    )

    st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)

    # 경기 진행 요약 (로드된 경우)
    if loaded:
        cur = pitches[c_idx]
        half = "▲" if cur["inning_topbot"] == "Top" else "▼"
        _away_s = cur["away_score"]
        _home_s = cur["home_score"]
        st.markdown(
            '<div class="sidebar-card">'
            '<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.4rem">경기 현황</p>'
            f'<div style="font-size:.85rem;line-height:2.1;color:#64748b">'
            f'<div>{cur["away_team"]} <span style="color:#e2e8f0;font-weight:700">{_away_s}</span>'
            f' : <span style="color:#e2e8f0;font-weight:700">{_home_s}</span> {cur["home_team"]}</div>'
            f'<div>이닝 <span style="color:#93c5fd;font-weight:700">{half} {cur["inning"]}회</span></div>'
            f'<div>카운트 <span style="color:#e2e8f0;font-weight:700">{cur["balls"]}-{cur["strikes"]}</span>'
            f'  아웃 <span style="color:#fbbf24;font-weight:700">{cur["outs"]}</span></div>'
            f'<div>진행 <span style="color:#a78bfa;font-weight:700">{c_idx+1}/{len(pitches)}구</span></div>'
            f'</div></div>', unsafe_allow_html=True)

    # 구종 범례
    _legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:.35rem;margin-bottom:.2rem;font-size:.8rem">'
        f'<div style="width:7px;height:7px;border-radius:50%;background:{_m["color"]};flex-shrink:0"></div>'
        f'<span style="color:#64748b;width:2rem">{_c}</span>'
        f'<span style="color:#475569">{_m["name"]}</span></div>'
        for _c, _m in list(PITCH_META.items())[:10] if _c != "OTHER"
    )
    st.markdown(
        '<div class="sidebar-card">'
        '<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.3rem">구종 범례</p>'
        f'{_legend_rows}</div>', unsafe_allow_html=True)


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
    st.info("💡 데이터 로드에 실패했습니다. 페이지를 새로고침해 다시 시도해보세요.", icon="⚾")


# ══ 메인 헤더 ═════════════════════════════════════════════════════
if loaded:
    cur  = pitches[c_idx]
    prev = pitches[c_idx - 1] if c_idx > 0 else None
    half = "▲" if cur["inning_topbot"] == "Top" else "▼"
    aw, hw = cur["away_team"], cur["home_team"]
    aws, hws = cur["away_score"], cur["home_score"]

    balls_html   = _seg_dots(cur["balls"],   3, "#3b82f6")
    strikes_html = _seg_dots(cur["strikes"], 2, "#f59e0b")
    outs_html    = _seg_dots(cur["outs"],    2, "#ef4444")
    _aw_color    = TEAM_ACCENTS.get(aw, "#94a3b8")
    _hw_color    = TEAM_ACCENTS.get(hw, "#94a3b8")

    st.markdown(
        f'<div class="scoreboard">'
        # 폭을 제한하고 가운데 정렬한다. space-between으로 풀어두면 와이드 화면에서
        # 두 팀 점수가 양 끝으로 밀려나고 가운데 900px가 빈 영역이 된다.
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'gap:clamp(1.5rem,6vw,5rem);max-width:760px;margin:0 auto">'
        # 원정팀
        f'<div style="text-align:center;min-width:72px">'
        f'<div style="font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#64748b">원정</div>'
        f'<div style="font-size:.8rem;font-weight:800;color:{_aw_color};letter-spacing:.08em;'
        f'border-bottom:2px solid {_aw_color};padding-bottom:.15rem;display:inline-block">{aw}</div>'
        f'<div class="team-score" style="color:{"#f1f5f9" if aws >= hws else "#475569"}">{aws}</div>'
        f'</div>'
        # 중앙 (이닝 + 카운트)
        f'<div style="text-align:center;flex:1">'
        f'<div style="margin-bottom:.35rem"><span class="live-badge">LIVE</span></div>'
        # inline-block이 없으면 블록으로 늘어나 이닝 표시가 폭 전체를 채우는 빈 막대가 된다.
        f'<div class="inning-box" style="display:inline-block;margin-bottom:.5rem;'
        f'font-size:.85rem;padding:.3rem 1.1rem">{half}&nbsp;{cur["inning"]}회</div>'
        f'<div style="display:flex;justify-content:center;align-items:center;gap:1.1rem">'
        f'<div style="display:flex;align-items:center;gap:.32rem">'
        f'<span style="font-size:.62rem;font-weight:800;color:#60a5fa;letter-spacing:.04em">B</span>'
        f'{balls_html}</div>'
        f'<div style="width:1px;height:12px;background:rgba(148,163,184,.18)"></div>'
        f'<div style="display:flex;align-items:center;gap:.32rem">'
        f'<span style="font-size:.62rem;font-weight:800;color:#f59e0b;letter-spacing:.04em">S</span>'
        f'{strikes_html}</div>'
        f'<div style="width:1px;height:12px;background:rgba(148,163,184,.18)"></div>'
        f'<div style="display:flex;align-items:center;gap:.32rem">'
        f'<span style="font-size:.62rem;font-weight:800;color:#ef4444;letter-spacing:.04em">O</span>'
        f'{outs_html}</div>'
        f'</div></div>'
        # 홈팀
        f'<div style="text-align:center;min-width:72px">'
        f'<div style="font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#64748b">홈</div>'
        f'<div style="font-size:.8rem;font-weight:800;color:{_hw_color};letter-spacing:.08em;'
        f'border-bottom:2px solid {_hw_color};padding-bottom:.15rem;display:inline-block">{hw}</div>'
        f'<div class="team-score" style="color:{"#f1f5f9" if hws >= aws else "#475569"}">{hws}</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True)
else:
    _render_landing_hero()

st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

# 6회초 알림
if st.session_state.get("_sixth_inning_alert"):
    st.warning("🔔 6회초 시작 — 1~5이닝 분析 구간 완료!", icon="⚾")

# ══ OCR 실시간 감지 결과 처리 ════════════════════════════════════════
_OCR_TO_CODE = {
    "Knuckle Curve": "KC", "Curveball": "CU", "4-Seam Fastball": "FF",
    "Fastball": "FF", "Sinker": "SI", "2-Seam Fastball": "SI",
    "Slider": "SL", "Sweeper": "ST", "Changeup": "CH",
    "Cutter": "FC", "Splitter": "FS", "Knuckleball": "KN",
}

# 실시간 OCR 결과로 인덱스를 정하던 블록을 제거했다 (TS-031). 발주 쪽 주석 참고.
#
# 이 블록은 OCR이 "투구다"라고 하면 영상 시각과 무관하게 current_pitch_idx를
# 직전+1로 밀었다. 여기서 읽은 구종·구속을 video_pitch_data[추정 인덱스]에 넣어
# 화면에도 띄웠는데, 인덱스가 추정값이라 엉뚱한 투구에 다른 구종이 붙을 수 있었다.
# 지금은 그 자리를 Statcast 값이 채운다(아래 _display_code 폴백).

# ══ 메인 레이아웃 ═════════════════════════════════════════════════
if loaded:
    col_video, col_panel = st.columns([3.2, 1.2], gap="medium")
else:
    col_video = col_panel = None

# ── 왼쪽: 영상 + 내비게이션 ──────────────────────────────────────
if loaded:
    with col_video:
        _vsrc = st.session_state.video_src
        _local_play_path = st.session_state.get("_local_video_path")
        _use_local_player = bool(_local_play_path and os.path.exists(_local_play_path))

        # 로컬 파일 재생 시에만 커스텀 플레이어(JS 이벤트 기반 자동 싱크) 사용.
        # YouTube는 iframe-in-iframe 구조에서 임베드 오류(153)가 나 재생 자체가
        # 안 되는 경우가 있어 Streamlit 기본 st.video()로 단순화 — 대신 실시간
        # 자동 싱크는 포기하고 아래 슬라이더로 수동 이동한다.
        if _use_local_player:
            sys.path.insert(0, os.path.join(ROOT, "streamlit_app"))
            _seek_to    = st.session_state.get("seek_to")
            _is_playing = st.session_state.get("is_playing", False)

            from local_video_player import local_video_player as _lvp
            _vport     = _get_video_server(os.path.dirname(os.path.abspath(_local_play_path)))
            _video_url = f"http://localhost:{_vport}/{os.path.basename(_local_play_path)}"
            _yt_result = _lvp(video_url=_video_url, seek_to=_seek_to, is_playing=_is_playing, key="local_main")

            if _seek_to is not None:
                st.session_state.seek_to = None

            _yt_data             = _yt_result
            _current_video_time  = None
            _is_actually_playing = False

            if isinstance(_yt_data, str):
                try:
                    _yt_data = json.loads(_yt_data)
                except Exception:
                    _yt_data = None
            if isinstance(_yt_data, dict):
                _current_video_time  = _yt_data.get("time")
                _is_actually_playing = bool(_yt_data.get("playing", False))
            elif isinstance(_yt_data, (int, float)):
                _current_video_time  = float(_yt_data)

            # 영상 시간/재생상태 세션 저장
            if _current_video_time is not None:
                st.session_state._vid_t       = _current_video_time
                st.session_state._vid_t_wall  = time.time()   # 받은 벽시계 시각 기록
            if isinstance(_yt_data, dict):
                _pl = bool(_yt_data.get("playing", False))
                st.session_state._vid_pl    = _pl
                st.session_state.is_playing = _pl   # rerun 때마다 pauseVideo() 방지

            _local_path = _local_play_path

            # ── 시간 비례 자동 싱크 ──
            # 정적 OCR 스캔은 320구 중 65개(~20%)만 감지해 개별 투구 단위 싱크는 신뢰할 수
            # 없음(감지 간 간격이 실제로 여러 투구를 건너뛰는 경우가 흔함). 대신 영상 재생
            # 시간 대비 진행 비율로 투구 인덱스를 추정한다 — 프레임 단위로 정확하진 않지만
            # 항상 부드럽게 앞으로 진행한다. 구종/구속 표시는 이미 실제 Statcast 데이터로
            # 폴백하므로(아래 _display_code 로직) 내용 정확도엔 영향 없다.
            _vid_t_base = st.session_state.get("_vid_t")
            _vid_t_wall = st.session_state.get("_vid_t_wall", 0.0)
            _vid_pl     = st.session_state.get("_vid_pl", False)

            if _current_video_time is not None:
                _vid_t = _current_video_time
            elif _vid_t_base is not None:
                _vid_t = _vid_t_base
            else:
                _vid_t = None

            print(f"[SYNC] vid_t={_vid_t} pl={_vid_pl} loaded={loaded} lpath={bool(_local_path)}")
            if _vid_t is not None and loaded and _vid_pl:
                _new_cidx_ts = index_at_time(
                    _vid_t, _timeline_anchors(pitches),
                    FIXED_DEMO_VIDEO_DURATION_SEC, len(pitches),
                )
                if _new_cidx_ts > st.session_state.get("current_pitch_idx", 0):
                    st.session_state.current_pitch_idx = _new_cidx_ts
                    st.session_state._sync_activated    = True
                    if pitches[_new_cidx_ts]["inning"] >= 6:
                        st.session_state._sixth_inning_alert = True
                    st.rerun()

            # 실시간 OCR 투구 감지는 없앴다 (TS-031).
            #
            # 방송 오버레이는 공이 던져진 **뒤에** 뜬다. 구속 표시가 나타나는 시점이
            # 투구 +2.8~4.2초(ADR-0010 스캔창)이고, 여기에 검사 주기 0.5초와 OCR 1회
            # 1.14초가 더 붙어 아무리 빨라도 +4.4초다. 코드를 최적화해서 줄일 수 있는
            # 지연이 아니다 — 그 시점까지 영상에 정보가 존재하지 않는다.
            #
            # 게다가 그 결과로 인덱스를 정할 때 영상 시각을 보지 않고 무조건 직전+1을
            # 썼다. 앵커 보간(영상 시각 기준, 지연 0)과 드라이버가 둘이 되어 결과가
            # 둘 중 큰 쪽으로 정해졌고, 어느 쪽이 이길지 정해져 있지 않아 계속 어긋났다.
            #
            # 이제 인덱스는 위의 index_at_time() 하나가 정한다.
        elif _vsrc:
            _current_video_time = None
            st.video(_vsrc, autoplay=True, muted=True)
        else:
            _current_video_time = None
            st.markdown(
                '<div style="background:rgba(8,14,26,.9);border:1.5px dashed rgba(59,130,246,.2);'
                'border-radius:12px;height:300px;display:flex;flex-direction:column;'
                'align-items:center;justify-content:center;gap:.6rem">'
                '<div style="font-size:2.8rem">🎬</div>'
                '<div style="color:#334155;font-size:.85rem">사이드바에서 YouTube URL을 입력하거나 영상을 업로드하세요</div>'
                '<div style="color:#1e293b;font-size:.72rem">YOLO가 자동으로 투구를 감지합니다</div>'
                '</div>', unsafe_allow_html=True)

        if loaded:
            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

            _local_ready = bool(st.session_state.get("_local_video_path"))

            # TS-031 이후 재생 중 감지가 없으므로 _pose_task_id는 항상 비어 있다.
            # 아래 분기는 남겨둔다 — 오프라인 스캔이 붙으면 다시 쓰인다.
            _pose_active = bool(st.session_state.get("_pose_task_id"))
            _scan_st_note = st.session_state.get("_scan_status", "idle")
            if not _local_ready:
                _sync_note = "아래 슬라이더로 투구를 이동하세요"
            elif _scan_st_note == "scanning":
                _sync_note = "영상 분석 중…"
            elif _scan_st_note == "done":
                _sync_note = "싱크 준비 완료 — 재생 시각에 맞춰 자동 진행"
            elif _pose_active:
                _sync_note = "투구 모션 감지 중…"
            else:
                _sync_note = ""
            _pitch_label = f"투구 {c_idx+1} / {len(pitches)} | "
            if _sync_note:
                st.caption(f"{_pitch_label}{_sync_note}")

            # 슬라이더 (투구 타임라인)
            st.markdown('<p style="font-size:.72rem;font-weight:700;color:#475569;letter-spacing:.09em;text-transform:uppercase;margin:.5rem 0 .1rem">투구 타임라인</p>', unsafe_allow_html=True)
            # 이동은 전부 on_click/on_change 콜백으로 처리한다.
            #
            # st.button()의 반환값으로 처리하면 안 된다. 이 앱은 영상 재생 중 여러
            # 폴링 루프가 st.rerun()을 계속 걸어서, 클릭이 처리되기 전에 다음 재실행에
            # 버려진다 — 실측으로 '다음 투구'를 눌러도 st.button()이 True를 돌려준 적이
            # 한 번도 없었다. 콜백은 위젯 이벤트를 처리하는 시점에 실행되므로 안 밀린다.
            def _goto_pitch_cb(idx: int) -> None:
                idx = max(0, min(idx, len(pitches) - 1))
                st.session_state.current_pitch_idx = idx
                st.session_state.video_synced = True
                # 영상도 같이 옮긴다. 인덱스만 바꾸면 자동 싱크가 영상 시각에 맞춰
                # 인덱스를 도로 끌고 간다. 표시와 같은 매핑을 써야 둘이 안 어긋난다.
                if pitches:
                    st.session_state.seek_to = time_at_index(
                        idx, _timeline_anchors(pitches),
                        FIXED_DEMO_VIDEO_DURATION_SEC, len(pitches),
                    )

            # 슬라이더 손잡이를 현재 인덱스에 맞춘다. 위젯을 만들기 **전에** 키를 쓰는 것이
            # 정해진 방법이다 — 콜백 안에서 자기 위젯 키를 건드리면 on_change와 얽혀
            # 방금 누른 값이 되돌아온다(실측: 버튼을 눌러도 seek_to가 0.0으로 잡혔다).
            if st.session_state.get("pitch_slider") != c_idx:
                st.session_state.pitch_slider = c_idx

            sel = st.slider("투구 선택", 0, max(len(pitches)-1, 0),
                            key="pitch_slider", label_visibility="collapsed")
            if sel != c_idx:                     # 사용자가 직접 끈 경우
                _goto_pitch_cb(sel)
                st.rerun()

            _bp, _bn = st.columns(2)
            with _bp:
                st.button("◀ 이전 투구", use_container_width=True,
                          disabled=(c_idx <= 0), key="btn_prev",
                          on_click=_goto_pitch_cb, args=(c_idx - 1,))
            with _bn:
                st.button("다음 투구 ▶", use_container_width=True,
                          disabled=(c_idx >= len(pitches) - 1), key="btn_next",
                          on_click=_goto_pitch_cb, args=(c_idx + 1,))

            # 최근 투구 리스트
            st.markdown('<p style="font-size:.72rem;font-weight:700;color:#475569;letter-spacing:.09em;text-transform:uppercase;margin:.4rem 0 .15rem">최근 투구</p>', unsafe_allow_html=True)
            _start = max(0, c_idx - 7)
            _row_desc_map = {
                "called_strike": "스트라이크", "swinging_strike": "헛스윙",
                "swinging_strike_blocked": "헛스윙(블)", "ball": "볼", "blocked_ball": "블로킹볼",
                "foul": "파울", "foul_tip": "파울팁", "hit_into_play": "인플레이",
            }
            for _r in reversed(pitches[_start: c_idx] if c_idx > 0 else []):
                _pt  = _r["pitch_type"] or "—"
                _m   = PITCH_META.get(_pt, PITCH_META["OTHER"])
                _spd = f'{_r["release_speed"]:.1f}' if _r["release_speed"] else "—"
                _is_c = _r["pitch_idx"] == c_idx
                _bg  = "rgba(59,130,246,.1)" if _is_c else "rgba(8,14,26,.6)"
                _bdr = "1px solid rgba(59,130,246,.3)" if _is_c else "1px solid rgba(148,163,184,.07)"
                _ev  = _r["events"] if _r["events"] and _r["events"] not in ("nan", "None", "") else ""
                _ev_html = f' <span style="color:#34d399;font-size:.65rem">[{_ev}]</span>' if _ev else ""
                _row_batter = _r["batter_name"].split(",")[0] if _r.get("batter_name") else "—"
                _row_desc   = _row_desc_map.get(_r.get("description"), _r.get("description") or "—")
                st.markdown(
                    f'<div class="pitch-row" style="background:{_bg};border:{_bdr}">'
                    f'<div style="display:flex;align-items:center;gap:.5rem;width:100%">'
                    f'<span style="color:#475569;width:1.4rem;font-size:.68rem;font-weight:{"700" if _is_c else "400"}">'
                    f'#{_r["pitch_idx"]+1}</span>'
                    f'<span style="font-weight:700;color:{_m["color"]};width:2.8rem">{_m["emoji"]} {_pt}</span>'
                    f'<span style="color:#64748b;font-size:.68rem">{_spd} mph</span>'
                    f'<span style="color:#64748b;font-size:.65rem;margin-left:auto">'
                    f'{_r["inning_topbot"][0]}{_r["inning"]}회 {_r["balls"]}-{_r["strikes"]}'
                    f'{_ev_html}</span>'
                    f'</div>'
                    f'<div class="pitch-row-detail">타자 {_row_batter} · {_row_desc}</div>'
                    f'</div>', unsafe_allow_html=True)


    # ── 오른쪽: 경기 상황 패널 ────────────────────────────────────────
    with col_panel:
        if not loaded:
            st.markdown(
                '<div style="height:400px;display:flex;align-items:center;justify-content:center;'
                'color:#334155;font-size:.85rem;border:1px dashed rgba(59,130,246,.1);border-radius:12px">'
                '경기를 먼저 로드하세요</div>', unsafe_allow_html=True)
        else:
            cur  = pitches[c_idx]
            prev = pitches[c_idx - 1] if c_idx > 0 else None

            # ── 투수 / 타자 교체 알림 (티빙 스타일) ──────────────────
            if prev is not None:
                _pitcher_changed = prev["pitcher_id"] != cur["pitcher_id"]
                _batter_changed  = (not _pitcher_changed) and (prev["at_bat"] != cur["at_bat"])
            else:
                _pitcher_changed = False
                _batter_changed  = False

            if _pitcher_changed:
                _old_p = prev["pitcher_name"].split(",")[0] if prev else "—"
                _new_p = cur["pitcher_name"].split(",")[0]
                _new_hand = cur["pitcher_hand"]
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,rgba(26,10,46,.95),rgba(13,31,60,.95));'
                    f'border:1px solid rgba(167,139,250,.55);border-radius:12px;padding:.75rem 1rem;'
                    f'margin-bottom:.55rem">'
                    f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
                    f'color:#a78bfa;margin-bottom:.3rem">🔄 투수 교체</div>'
                    f'<div style="display:flex;align-items:center;gap:.55rem;flex-wrap:wrap">'
                    f'<span style="color:#94a3b8;font-size:.9rem;text-decoration:line-through">{_old_p}</span>'
                    f'<span style="color:#475569;font-size:1rem;font-weight:300">→</span>'
                    f'<span style="color:#60a5fa;font-weight:800;font-size:1.05rem">{_new_p}</span>'
                    f'<span style="background:rgba(96,165,250,.15);color:#93c5fd;border:1px solid rgba(96,165,250,.3);'
                    f'border-radius:999px;padding:.1rem .45rem;font-size:.68rem;font-weight:700">{_new_hand}투</span>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif _batter_changed:
                _new_b    = cur["batter_name"]
                _new_bh   = cur["batter_hand"]
                _bat_team = cur["away_team"] if cur["inning_topbot"] == "Top" else cur["home_team"]
                st.markdown(
                    f'<div style="background:rgba(15,23,42,.8);border:1px solid rgba(52,211,153,.35);'
                    f'border-radius:12px;padding:.65rem 1rem;margin-bottom:.55rem">'
                    f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
                    f'color:#34d399;margin-bottom:.25rem">⚾ 타자 교체</div>'
                    f'<div style="font-size:1.05rem;font-weight:800;color:#e2e8f0">{_new_b}</div>'
                    f'<div style="font-size:.72rem;color:#64748b;margin-top:.1rem">{_bat_team} · {_new_bh}타</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── 투수 / 타자 정보 ──
            batting_team = cur["away_team"] if cur["inning_topbot"] == "Top" else cur["home_team"]
            st.markdown(
                f'<div class="panel-secondary">'
                f'<div class="panel-title">투수 · 타자</div>'
                f'<div style="display:flex;gap:.6rem;margin-bottom:.5rem">'
                f'<div style="flex:1">'
                f'<div style="font-size:.68rem;color:#475569;margin-bottom:.1rem">투수</div>'
                f'<div style="font-size:1rem;font-weight:800;color:#60a5fa">{cur["pitcher_name"]}</div>'
                f'<div style="font-size:.72rem;color:#475569">투구폼 {cur["pitcher_hand"]}</div>'
                f'</div>'
                f'<div style="flex:1">'
                f'<div style="font-size:.68rem;color:#475569;margin-bottom:.1rem">타자 ({batting_team})</div>'
                f'<div style="font-size:1rem;font-weight:800;color:#e2e8f0">{cur["batter_name"]}</div>'
                f'<div style="font-size:.72rem;color:#475569">{cur["batter_hand"]}타</div>'
                f'</div></div>'
                # 주자 다이아몬드
                f'<div style="display:flex;align-items:center;gap:.8rem">'
                f'{_diamond_svg(cur["on_1b"], cur["on_2b"], cur["on_3b"])}'
                f'<div style="font-size:.75rem;color:#64748b;line-height:1.8">'
                f'{"🟡 1루" if cur["on_1b"] else "○ 1루"}<br>'
                f'{"🟡 2루" if cur["on_2b"] else "○ 2루"}<br>'
                f'{"🟡 3루" if cur["on_3b"] else "○ 3루"}'
                f'</div></div></div>',
                unsafe_allow_html=True)

            # ── 현재 타석 상황 (투구 결과 실시간) ──
            _bilstm_preds = st.session_state.get("bilstm_preds", [])

            # 이번 타석에서 던진 투구 목록 (현재 at_bat 기준)
            _cur_ab  = cur["at_bat"]
            _ab_pitches = [p for p in pitches[:c_idx] if p["at_bat"] == _cur_ab]
            _ab_pitch_n = len(_ab_pitches)  # 이번 타석 누적 투구 수

            _desc_map = {
                "called_strike":    ("스트라이크", "#ef4444"),
                "swinging_strike":  ("헛스윙",     "#ef4444"),
                "swinging_strike_blocked": ("헛스윙(블)", "#f87171"),
                "ball":             ("볼",          "#3b82f6"),
                "blocked_ball":     ("블로킹볼",    "#3b82f6"),
                "foul":             ("파울",        "#fbbf24"),
                "foul_tip":         ("파울팁",      "#fbbf24"),
                "hit_into_play":    ("인플레이",    "#34d399"),
            }
            _ev_map = {
                "strikeout":      "삼진 아웃", "home_run":     "홈런",
                "single":         "1루타",     "double":       "2루타",
                "triple":         "3루타",     "walk":         "볼넷",
                "hit_by_pitch":   "사구",      "field_out":    "아웃",
                "grounded_into_double_play": "병살타", "force_out": "포스 아웃",
                "sac_fly":        "희생플라이", "sac_bunt":     "희생번트",
            }

            # 이번 타석 투구 결과 점들
            if _ab_pitches:
                _dots_html = ""
                for _ap in _ab_pitches:
                    _dinfo = _desc_map.get(_ap["description"], ("", "#475569"))
                    _dot_c = _dinfo[1]
                    _dots_html += (
                        f'<span title="{_dinfo[0]}" style="display:inline-block;width:9px;height:9px;'
                        f'border-radius:50%;background:{_dot_c};margin-right:3px"></span>'
                    )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">'
                    f'<span style="font-size:.68rem;color:#475569;white-space:nowrap">타석 {_ab_pitch_n}구</span>'
                    f'<div>{_dots_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── 방금 던진 구종 ──
            st.markdown('<div class="card-badge card-badge-actual">실측</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title" style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.35rem">방금 던진 구종</div>', unsafe_allow_html=True)

            _vpd = st.session_state.get("video_pitch_data", [])
            _ocr_i = c_idx - 1 if c_idx > 0 else -1
            _ocr_d = _vpd[_ocr_i] if 0 <= _ocr_i < len(_vpd) else {}
            _OCR_TO_CODE_DISP = {
                "Knuckle Curve": "KC", "Curveball": "CU", "4-Seam Fastball": "FF",
                "Fastball": "FF", "Sinker": "SI", "2-Seam Fastball": "SI",
                "Slider": "SL", "Sweeper": "ST", "Changeup": "CH",
                "Cutter": "FC", "Splitter": "FS", "Knuckleball": "KN",
            }
            _ocr_type_str  = _ocr_d.get("pitch_type") if isinstance(_ocr_d, dict) else None
            _ocr_speed_val = _ocr_d.get("speed")       if isinstance(_ocr_d, dict) else None
            _display_code  = (_OCR_TO_CODE_DISP.get(_ocr_type_str, prev["pitch_type"] if prev else "OTHER")
                              if _ocr_type_str else (prev["pitch_type"] if prev else "OTHER"))
            _api_speed     = prev["release_speed"] if prev else None
            _spd_val       = _ocr_speed_val or _api_speed
            _spd           = f'{_spd_val:.0f} mph' if _spd_val else "—"

            if prev and _display_code:
                _m    = PITCH_META.get(_display_code, PITCH_META["OTHER"])
                _desc_raw  = prev.get("description", "")
                _desc_info = _desc_map.get(_desc_raw, (_desc_raw, "#64748b"))
                _desc_kor, _desc_col = _desc_info
                _ev   = prev["events"] if prev["events"] not in ("nan", "None", "", None) else ""
                _ev_kor = _ev_map.get(_ev, _ev)

                # 이전 예측 적중 여부 — prev(방금 던진 구)와 비교
                _prev_pred = ""
                _reveal_cls = "card-reveal"
                if c_idx > 0 and prev and prev.get("pitch_type"):
                    _pb = _bilstm_preds[c_idx - 1] if (c_idx - 1) < len(_bilstm_preds) else None
                    _prev_pred_type = _pb["next_pitch"] if _pb else _predict_next(pitches, c_idx - 1)["next_pitch"]
                    _hit = _prev_pred_type == prev["pitch_type"]
                    _prev_pred = (f'<span style="font-size:.72rem;color:{"#34d399" if _hit else "#f87171"};'
                                  f'font-weight:700;margin-left:.4rem">{"✓ 예측 적중" if _hit else "✗ 빗나감"}</span>')
                    _reveal_cls = "glow-hit" if _hit else "glow-miss"

                    # COMBO 스트릭 갱신 — c_idx당 1회만 (재렌더 시 중복 집계 방지)
                    if c_idx != st.session_state._streak_calc_idx:
                        st.session_state.pred_streak = (st.session_state.pred_streak + 1) if _hit else 0
                        st.session_state._streak_calc_idx = c_idx

                st.markdown(
                    f'<div class="pitch-card {_reveal_cls}" style="background:rgba(15,23,42,.6);border-color:{_m["color"]}44">'
                    f'<div style="display:flex;align-items:center;gap:.8rem">'
                    f'<div style="font-size:2.2rem;line-height:1">{_m["emoji"]}</div>'
                    f'<div style="flex:1">'
                    f'<div class="pitch-code" style="color:{_m["color"]}">{_display_code}</div>'
                    f'<div class="pitch-name">{_ocr_type_str or _m["name"]}</div>'
                    f'<div style="margin-top:.2rem;display:flex;align-items:center;flex-wrap:wrap;gap:.3rem">'
                    f'<span style="color:#64748b;font-size:.85rem;font-weight:600">{_spd}</span>'
                    f'<span style="background:{_desc_col}22;color:{_desc_col};border:1px solid {_desc_col}44;'
                    f'border-radius:999px;padding:.08rem .4rem;font-size:.68rem;font-weight:700">{_desc_kor}</span>'
                    + (f'<span style="color:#34d399;font-size:.72rem;font-weight:700">[{_ev_kor}]</span>' if _ev_kor else "")
                    + (_prev_pred)
                    + f'</div></div></div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="pitch-card" style="min-height:70px;display:flex;align-items:center;'
                    'justify-content:center;background:rgba(8,14,26,.5);border-color:rgba(59,130,246,.1)">'
                    '<span style="color:#475569;font-size:.9rem">경기 로드 후 재생</span></div>',
                    unsafe_allow_html=True)

            # ── 영상만으로 낸 구종 (CV 궤적 판정) ──
            # 위 실측은 Statcast API에서 왔다. 아래는 API 없이 영상 궤적만으로 낸 값이라
            # 둘을 나란히 두면 CV 적중률이 화면에서 바로 채점된다.
            _cv_vd = st.session_state._cv_verdicts.get(_ocr_i)
            _cv_running = _ocr_i in st.session_state._cv_task_idx

            # 판정 발주: 방금 던진 구가 있고, 아직 결과도 진행 중인 것도 없을 때 한 번만.
            _cv_path  = st.session_state.get("_local_video_path")
            # 현재 재생 시각이 아니라 **스캔이 잡은 실제 투구 시각**으로 쏜다.
            # 투구 인덱스는 영상 길이에 비례 매핑되므로(_target_idx), 인덱스가 넘어간
            # 순간의 재생 시각에 공이 날아가고 있다는 보장이 없다. 오버레이가 뜬 시각은
            # 실제로 투구가 있었던 시각이다.
            _cv_now   = st.session_state.get("_vid_t")
            _cv_times = st.session_state.get("video_pitch_times") or []
            _cv_past  = [t for t in _cv_times if _cv_now is not None and t <= _cv_now]
            _cv_vid_t = max(_cv_past) if _cv_past else None
            if (st.session_state.get("cv_enabled")
                    and prev and _ocr_i >= 0 and _cv_vd is None and not _cv_running
                    and _cv_path and os.path.exists(_cv_path)
                    and _cv_vid_t is not None):
                _cv_tid = _start_cv_check(_cv_path, _cv_vid_t, _ocr_i)
                st.session_state._cv_task_idx[_ocr_i] = _cv_tid
                _cv_running = True

            if _cv_vd:
                _cv_group = _cv_vd.get("group")
                if _cv_group:
                    _cv_col  = "#60a5fa" if _cv_group == "FASTBALL" else "#c084fc"
                    _cv_kor  = "속구 계열" if _cv_group == "FASTBALL" else "변화구 계열"
                    _cv_conf = _cv_vd.get("confidence", 0.0)
                    _cv_mark = ""
                    if _cv_vd.get("truth"):
                        _cv_ok = _cv_vd["truth"] == _cv_group
                        _cv_mark = (f'<span style="font-size:.68rem;font-weight:700;margin-left:.35rem;'
                                    f'color:{"#34d399" if _cv_ok else "#f87171"}">'
                                    f'{"✓" if _cv_ok else "✗"}</span>')
                    _cv_body = (
                        f'<span style="color:{_cv_col};font-weight:800;font-size:.9rem">{_cv_kor}</span>'
                        f'<span style="color:#64748b;font-size:.72rem;margin-left:.4rem">'
                        f'{_cv_conf:.0%} · 궤적 {_cv_vd.get("n_points", 0)}점</span>'
                        + _cv_mark
                    )
                else:
                    # 궤적을 못 잡은 14%를 숨기지 않는다. 숨기면 앱 정확도가 실제보다 좋아 보인다.
                    _cv_why = {"no_detections": "공 미감지", "no_trajectory": "궤적 없음",
                               "too_few_points": "궤적 짧음", "error": "오류"}.get(
                                   _cv_vd.get("reason", ""), _cv_vd.get("reason", ""))
                    _cv_body = (f'<span style="color:#64748b;font-size:.82rem">판정 불가</span>'
                                f'<span style="color:#475569;font-size:.7rem;margin-left:.35rem">{_cv_why}</span>')
            elif _cv_running:
                _cv_body = '<span style="color:#475569;font-size:.8rem">궤적 분석 중…</span>'
            elif not st.session_state.get("cv_enabled"):
                _cv_body = ('<span style="color:#334155;font-size:.8rem">'
                            '꺼짐 — 사이드바에서 켠다</span>')
            else:
                _cv_body = '<span style="color:#334155;font-size:.8rem">대기</span>'

            _cv_n, _cv_h = st.session_state._cv_scored, st.session_state._cv_hits
            _cv_rate = (f'{_cv_h / _cv_n:.0%} ({_cv_h}/{_cv_n})' if _cv_n else '—')
            # 학습에 쓰지 않은 경기(775300)의 원본 중계 영상에서 정확도 76.7% / AUC 0.780이다
            # (n=43, 기준선 58.1%). 클립 단위 검증 78.3%와 사실상 같다 — 프레임 샘플링을
            # 학습과 맞추고 나서 도메인 격차가 사라졌다 (TS-028). 실측 근거를 화면에 함께 적는다.
            st.markdown(
                f'<div style="margin-top:.45rem;padding:.5rem .65rem;border-radius:10px;'
                f'background:rgba(8,14,26,.5);border:1px solid rgba(77,189,138,.18)">'
                f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:.5rem">'
                f'<span style="font-size:.62rem;font-weight:700;letter-spacing:.08em;'
                f'text-transform:uppercase;color:#475569">영상만으로 · CV'
                f'<span style="margin-left:.35rem;padding:.05rem .3rem;border-radius:4px;'
                f'background:rgba(77,189,138,.14);color:#4dbd8a;font-size:.58rem;'
                f'letter-spacing:0">검증됨</span></span>'
                f'<span style="font-size:.62rem;color:#475569">이번 세션 {_cv_rate}</span>'
                f'</div>'
                f'<div style="margin-top:.2rem">{_cv_body}</div>'
                f'<div style="margin-top:.3rem;font-size:.58rem;color:#3f4a5c;line-height:1.5">'
                f'미학습 경기 원본 영상 정확도 76.7% · AUC 0.780 (n=43, 기준선 58.1%) — '
                f'클립 단위 검증 78.3%와 동등'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            # ── 다음 구종 예측 ──
            if True:
                _bilstm_res = _bilstm_preds[c_idx] if c_idx < len(_bilstm_preds) else None
                if _bilstm_res:
                    pred = _bilstm_res
                else:
                    pred = _predict_next(pitches, c_idx)

                _nc  = pred["next_pitch"]
                _nm  = PITCH_META.get(_nc, PITCH_META["OTHER"])
                _cf  = pred["confidence"]
                _cc  = "#34d399" if _cf >= 0.45 else "#f59e0b" if _cf >= 0.3 else "#f87171"

                _pred_basis = "BiLSTM 모델 예측 — 직전 투구 흐름 기반" if _bilstm_res else "통계 기반 예측 (BiLSTM 계산 중)"
                if st.session_state.pred_streak >= 2:
                    st.markdown(
                        f'<div class="combo-badge">🔥 COMBO x{st.session_state.pred_streak}</div>',
                        unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card-badge card-badge-pred">예측 · {_pred_basis}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pitch-card pred-hero card-reveal" style="background:linear-gradient(135deg,rgba(15,23,42,.8),'
                    f'rgba(46,27,75,.4));border-color:rgba(167,139,250,.3)">'
                    f'<div style="display:flex;align-items:center;gap:.8rem">'
                    f'<div class="conf-gauge" style="--pct:{_cf*100};--gauge-color:{_cc}">'
                    f'<span class="conf-gauge-code" style="color:{_cc}">{_nc}</span>'
                    f'<span class="conf-gauge-pct">{_cf:.0%}</span>'
                    f'</div>'
                    f'<div style="flex:1">'
                    f'<div class="pitch-code" style="color:{_nm["color"]}">{_nc}</div>'
                    f'<div class="pitch-name">{_nm["name"]}</div>'
                    f'<div style="margin-top:.2rem">'
                    f'<span style="color:{_cc};font-size:1.2rem;font-weight:800">{_cf:.0%}</span>'
                    f'<span style="color:#475569;font-size:.72rem;margin-left:.2rem">신뢰도</span>'
                    f'</div></div></div></div>',
                    unsafe_allow_html=True)

                # 구종별 예측 확률 바
                probs  = pred["probabilities"]
                codes  = [c for c in probs if probs[c] > 0.005]
                codes  = sorted(codes, key=lambda c: -probs[c])[:7]
                vals   = [probs[c] for c in codes]
                colors = [PITCH_META.get(c, PITCH_META["OTHER"])["color"] for c in codes]
                _pitch_kor_names = [PITCH_META.get(c, PITCH_META["OTHER"])["name"] for c in codes]
                # 가로 막대다. 세로로 두면 285px 폭에 7개 막대가 들어가면서 구종명이
                # 7px로 줄고 두 줄로 구겨진다 — 가로로 눕히면 한 행에 한 구종이라
                # 이름이 그대로 읽히고 확률 순위도 위에서부터 바로 보인다.
                labels = [f"{c}  {n}" for c, n in zip(codes, _pitch_kor_names)]

                fig = go.Figure(go.Bar(
                    x=vals, y=labels, orientation="h",
                    marker=dict(color=colors, opacity=0.82,
                                line=dict(color="rgba(255,255,255,.04)", width=1)),
                    text=[f"{v:.0%}" for v in vals], textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                    customdata=list(zip(codes, _pitch_kor_names)),
                    hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>확률: %{x:.1%}<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=8, t=6, b=6), showlegend=False,
                    height=max(120, 26 * len(codes) + 24),
                    bargap=0.28,
                    yaxis=dict(autorange="reversed",   # 확률 높은 구종이 위로
                               tickfont=dict(size=10, color="#94a3b8"),
                               gridcolor="rgba(0,0,0,0)", zeroline=False),
                    xaxis=dict(tickformat=".0%", tickfont=dict(size=8, color="#475569"),
                               gridcolor="rgba(148,163,184,.06)", zeroline=False,
                               range=[0, max(vals) * 1.25 if vals else 1]),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 투수 이번 경기 구종 분포 (누적)
            if c_idx >= 3:
                seen = [p for p in pitches[:c_idx+1]
                        if p["pitcher_id"] == cur["pitcher_id"] and p["pitch_type"]]
                if seen:
                    cnt   = Counter(p["pitch_type"] for p in seen)
                    _pc   = list(cnt.keys())
                    _pv   = [cnt[c] for c in _pc]
                    _pcol = [PITCH_META.get(c, PITCH_META["OTHER"])["color"] for c in _pc]
                    _plab = [f'{c} {PITCH_META.get(c,PITCH_META["OTHER"])["name"]}' for c in _pc]
                    fig2  = go.Figure(go.Pie(
                        labels=_plab, values=_pv,
                        marker=dict(colors=_pcol, line=dict(color="#080e1a", width=2)),
                        hole=0.55, textinfo="label+percent",
                        textfont=dict(size=7.5, color="#94a3b8"),
                        hovertemplate="%{label}: %{value}구 (%{percent})<extra></extra>",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=4, b=0), height=170, showlegend=False,
                        annotations=[dict(text=f"<b>{sum(_pv)}</b><br>구", x=0.5, y=0.5,
                                          font=dict(size=11, color="#e2e8f0"), showarrow=False)],
                    )
                    st.markdown(f'<p style="font-size:.62rem;font-weight:700;color:#475569;letter-spacing:.09em;text-transform:uppercase;margin:.15rem 0 .1rem">{cur["pitcher_name"].split(",")[0]} 구종 분포</p>', unsafe_allow_html=True)
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══ 하단 통계 ═════════════════════════════════════════════════════
if loaded and c_idx > 0:
    seen_so_far = pitches[:c_idx]
    _tot  = c_idx
    _ff   = sum(1 for p in seen_so_far if p["pitch_type"] in FASTBALLS)
    _br   = sum(1 for p in seen_so_far if p["pitch_type"] in BREAKING)
    _os   = sum(1 for p in seen_so_far if p["pitch_type"] in OFFSPEED)
    _spds = [p["release_speed"] for p in seen_so_far if p["release_speed"]]
    _avgspd = float(np.mean(_spds)) if _spds else 0.0

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
    for _col, _icon, _lbl, _val, _sub, _vcol in zip(
        st.columns(4),
        ["⚾", "🔴", "🔵", "💨"],
        ["분석 투구", "패스트볼", "변화구", "평균 구속"],
        [str(_tot), f"{_ff/max(_tot,1):.0%}", f"{(_br+_os)/max(_tot,1):.0%}",
         f"{_avgspd:.1f}" if _avgspd else "—"],
        ["구", f"{_ff}구", f"{_br+_os}구", "mph"],
        ["#e2e8f0", "#ef4444", "#3b82f6", "#fbbf24"],
    ):
        with _col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.28rem">{_icon}&nbsp;{_lbl}</div>'
                f'<div style="font-size:1.55rem;font-weight:800;color:{_vcol};line-height:1">{_val}</div>'
                f'<div style="font-size:.7rem;color:#64748b;margin-top:.1rem">{_sub}</div>'
                f'</div>', unsafe_allow_html=True)

# 오프라인 스캔 폴링 (2초마다 완료 여부 확인)
_poll_stid = st.session_state.get("_scan_task_id")
if _poll_stid and _scan_tasks.get(_poll_stid, {}).get("status") == "scanning":
    time.sleep(2.0)
    st.rerun()

# CV 판정 폴링 — 백그라운드 결과를 화면에 올리려면 재렌더가 필요하다.
# 궤적 판정은 프레임 수십 장에 YOLO를 돌리므로 OCR보다 느리다. 간격을 길게 잡는다.
#
# 조건을 "대기 목록이 비어 있지 않다"가 아니라 "실제로 처리 중인 태스크가 있다"로 둔다.
# 앞의 조건은 태스크가 유실되면 영원히 참이 되어 재실행이 끝없이 돈다.
if any(_cv_tasks.get(_t, {}).get("status") == "processing"
       for _t in st.session_state.get("_cv_task_idx", {}).values()):
    time.sleep(1.5)
    st.rerun()

# ── 포즈/OCR 태스크 폴링 — 제거함 ──
#
# 여기 있던 `time.sleep(0.4); st.rerun()`이 TS-027에서 지운 1초 폴링과 같은 버그다.
# 그때 타임스탬프 루프만 걷어내고 이건 남겨뒀는데, 이쪽이 0.4초 주기라 더 나빴다.
#
# OCR 검사는 0.5초 간격으로 걸리는데 1.19GB 영상에서 한 프레임을 OCR 하는 데 그보다
# 오래 걸린다. 그래서 재생 중에는 태스크가 사실상 항상 "processing"이고, 이 블록이
# 0.4초마다 스크립트를 통째로 재실행시켰다 — 실측으로 실행 80번 중 끝까지 간 것이
# 2번이었다. 위젯 구간에 닿기 전에 잘리니 투구 인덱스도 안 넘어가고 버튼 클릭도
# 처리되기 전에 버려진다.
#
# 폴링이 없어도 결과는 올라온다. 재생 중에는 local_video_player가 2초마다 시각을
# 보고해 재실행이 걸리고, 그 재실행이 위쪽에서 완료된 태스크를 거둬간다. 정지 중에는
# 새 검사가 시작되지 않으므로 거둘 것도 없다.

# ── 타임스탬프 싱크 폴링 — 제거함 ──
#
# 여기 있던 `time.sleep(1.0); st.rerun()`이 버튼을 먹통으로 만들고 있었다.
# 조건(스캔 타임스탬프가 있고 / 영상 시각을 받은 적 있고 / 300초 안)이 재생만 하면
# 사실상 항상 참이라, 스크립트가 1초마다 무한히 재실행됐다. 그 사이에 들어온 위젯
# 클릭은 처리되기 전에 다음 재실행에 버려진다 — 실측으로 '다음 투구'를 눌러도
# st.button()이 True를 돌려준 적이 한 번도 없었다.
#
# 폴링 자체가 불필요하다. local_video_player 컴포넌트가 재생 중 200ms마다 시각을
# setComponentValue로 올리고, 그 값 변경이 이미 Streamlit 재실행을 일으킨다.
# 즉 싱크는 컴포넌트 보고로 굴러가고 이 루프는 중복이었다. 정지 중에는 보고가
# 멈추지만 그때는 갱신할 것도 없다.
