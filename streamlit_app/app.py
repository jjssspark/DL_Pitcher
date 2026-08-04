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
</style>""", unsafe_allow_html=True)

# ══ 고정 데모 설정 ════════════════════════════════════════════════
FIXED_DEMO_GAME_PK   = 775300
FIXED_DEMO_VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
TEAM_COLORS = {"NYY": "#0C2340", "LAD": "#005A9C"}  # 고정 데모 게임은 이 두 팀만 등장
FIXED_DEMO_VIDEO_DURATION_SEC = 8231  # 고정 데모 YouTube 영상 총 길이(초)

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

_pose_tasks   = _make_pose_tasks()
_bilstm_tasks = _make_bilstm_tasks()
_model_ref    = _make_model_ref()
_scan_tasks   = _make_scan_tasks()
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


def _run_ocr_check_bg(task_id: str, video_path: str, check_time: float) -> None:
    """방송 오버레이 OCR 실시간 투구 감지 (백그라운드)"""
    try:
        from pose_detector import ocr_check_pitch_overlay
        is_pitch, pitch_type, speed, pitch_count = ocr_check_pitch_overlay(video_path, check_time)
        _pose_tasks[task_id] = {
            "status": "done", "is_pitch": is_pitch,
            "pitch_type": pitch_type, "speed": speed,
            "pitch_count": pitch_count,
            "check_time": check_time,
        }
        if is_pitch:
            print(f"[OCR실시간] t={check_time:.1f}s → {pitch_type} {speed}mph P:{pitch_count}")
    except Exception as e:
        _pose_tasks[task_id] = {
            "status": "error", "is_pitch": False,
            "check_time": check_time, "error": str(e),
        }


def _start_ocr_check(video_path: str, check_time: float) -> str:
    task_id = str(uuid.uuid4())[:8]
    _pose_tasks[task_id] = {"status": "processing", "check_time": check_time}
    threading.Thread(
        target=_run_ocr_check_bg, args=(task_id, video_path, check_time), daemon=True
    ).start()
    return task_id


@st.cache_resource
def _get_video_server(directory: str) -> int:
    """로컬 비디오 파일을 HTTP로 serve. 포트 번호 반환."""
    import http.server, socketserver

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()
        def log_message(self, *args):
            pass

    for port in range(8510, 8530):
        try:
            server = socketserver.TCPServer(("", port), _Handler)
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
    "video_pitch_data":      [],    # MLB 인덱스 기준 OCR 표시 데이터 (list, MLB idx로 확장)
    "_scan_raw_data":        [],    # 스캔 순서 기준 raw OCR 데이터 (scan idx → {type, speed})
    "_next_scan_idx":        0,     # 다음 처리할 스캔 타임스탬프 인덱스
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


# ══ 사이드바 ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:.8rem 0 .4rem">'
        '<div style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">⚾ PitchIQ</div>'
        '<div style="font-size:.78rem;color:#475569;margin-top:.05rem">MLB 투구 예측 시스템</div>'
        '</div>', unsafe_allow_html=True)

    st.divider()

    # 경기 진행 요약 (로드된 경우)
    if loaded:
        cur = pitches[c_idx]
        st.markdown('<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.4rem">경기 현황</p>', unsafe_allow_html=True)
        half = "▲" if cur["inning_topbot"] == "Top" else "▼"
        _away_s = cur["away_score"]
        _home_s = cur["home_score"]
        _diff   = _away_s - _home_s
        st.markdown(
            f'<div style="font-size:.85rem;line-height:2.1;color:#64748b">'
            f'<div>{cur["away_team"]} <span style="color:#e2e8f0;font-weight:700">{_away_s}</span>'
            f' : <span style="color:#e2e8f0;font-weight:700">{_home_s}</span> {cur["home_team"]}</div>'
            f'<div>이닝 <span style="color:#93c5fd;font-weight:700">{half} {cur["inning"]}회</span></div>'
            f'<div>카운트 <span style="color:#e2e8f0;font-weight:700">{cur["balls"]}-{cur["strikes"]}</span>'
            f'  아웃 <span style="color:#fbbf24;font-weight:700">{cur["outs"]}</span></div>'
            f'<div>진행 <span style="color:#a78bfa;font-weight:700">{c_idx+1}/{len(pitches)}구</span></div>'
            f'</div>', unsafe_allow_html=True)
        st.divider()

    # 구종 범례
    st.markdown('<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.3rem">구종 범례</p>', unsafe_allow_html=True)
    for _c, _m in list(PITCH_META.items())[:10]:
        if _c == "OTHER": continue
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.35rem;margin-bottom:.2rem;font-size:.8rem">'
            f'<div style="width:7px;height:7px;border-radius:50%;background:{_m["color"]};flex-shrink:0"></div>'
            f'<span style="color:#64748b;width:2rem">{_c}</span>'
            f'<span style="color:#475569">{_m["name"]}</span></div>',
            unsafe_allow_html=True)


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
    _aw_color    = TEAM_COLORS.get(aw, "#94a3b8")
    _hw_color    = TEAM_COLORS.get(hw, "#94a3b8")

    st.markdown(
        f'<div class="scoreboard">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.8rem">'
        # 원정팀
        f'<div style="text-align:center;min-width:72px">'
        f'<div style="font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#64748b">원정</div>'
        f'<div style="font-size:.8rem;font-weight:800;color:{_aw_color};letter-spacing:.08em;'
        f'border-bottom:2px solid {_aw_color};padding-bottom:.15rem;display:inline-block">{aw}</div>'
        f'<div class="team-score" style="color:{"#f1f5f9" if aws >= hws else "#475569"}">{aws}</div>'
        f'</div>'
        # 중앙 (이닝 + 카운트)
        f'<div style="text-align:center;flex:1">'
        f'<div class="inning-box" style="margin-bottom:.5rem;font-size:.85rem">{half}&nbsp;{cur["inning"]}회</div>'
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

_pose_tid = st.session_state.get("_pose_task_id")
if _pose_tid and _pose_tid in _pose_tasks:
    _ptask = _pose_tasks[_pose_tid]
    if _ptask["status"] in ("done", "error"):
        _ocr_check_t  = _ptask.get("check_time", -99.0)
        _scan_last_t  = st.session_state.get("_last_pitch_video_time", -99.0)
        _scan_covered = _scan_last_t > 0 and _ocr_check_t <= _scan_last_t + 8.0
        if _ptask.get("is_pitch") and loaded and not _scan_covered:
            _ocr_type   = _ptask.get("pitch_type")
            _ocr_speed  = _ptask.get("speed")
            _ocr_pcount = _ptask.get("pitch_count")

            # 실시간 OCR은 무조건 순차 전진 — P:N OCR 오독 시 큰 점프 방지
            _last_mlb = st.session_state.get("_last_ocr_mlb_idx", -1)
            _best_idx = min(_last_mlb + 1, len(pitches) - 1)
            _method   = "순차"

            _vpd = list(st.session_state.get("video_pitch_data", []))
            while len(_vpd) <= _best_idx:
                _vpd.append({})
            _vpd[_best_idx] = {"pitch_type": _ocr_type, "speed": _ocr_speed}
            _new_cidx = min(_best_idx + 1, len(pitches) - 1)
            st.session_state.video_pitch_data        = _vpd
            st.session_state.current_pitch_idx       = _new_cidx
            st.session_state._last_ocr_mlb_idx       = _best_idx
            st.session_state._last_pitch_video_time  = _ptask["check_time"]
            st.session_state.video_synced            = True
            st.session_state._sync_activated         = True
            if pitches[_new_cidx]["inning"] >= 6:
                st.session_state._sixth_inning_alert = True
            print(f"[싱크] MLB #{_best_idx+1} 확정 → c_idx={_new_cidx} ({_ocr_type} {_ocr_speed}mph) [{_method}]")

        del _pose_tasks[_pose_tid]
        st.session_state._pose_task_id = None
        st.rerun()

# ══ 메인 레이아웃 ═════════════════════════════════════════════════
if loaded:
    col_video, col_panel = st.columns([3.2, 1.2], gap="medium")
else:
    col_video = col_panel = None

# ── 왼쪽: 영상 + 내비게이션 ──────────────────────────────────────
if loaded:
    with col_video:
        # YouTube URL인지 확인 후 iframe 임베드
        import re as _re
        _vsrc = st.session_state.video_src
        def _yt_id(url):
            if not url: return None
            m = _re.search(r"(?:v=|youtu\.be/)([\w-]{11})", str(url))
            return m.group(1) if m else None

        _yt = _yt_id(_vsrc)
        _local_play_path = st.session_state.get("_local_video_path")
        _use_local_player = bool(_local_play_path and os.path.exists(_local_play_path))

        # 로컬 파일 우선 → 스캔 캐시와 타임라인이 정확히 일치
        # 로컬 없을 때만 YouTube 플레이어 폴백
        if _use_local_player or _yt:
            sys.path.insert(0, os.path.join(ROOT, "streamlit_app"))
            _seek_to    = st.session_state.get("seek_to")
            _is_playing = st.session_state.get("is_playing", False)

            if _use_local_player:
                from local_video_player import local_video_player as _lvp
                _vport     = _get_video_server(os.path.dirname(os.path.abspath(_local_play_path)))
                _video_url = f"http://localhost:{_vport}/{os.path.basename(_local_play_path)}"
                _yt_result = _lvp(video_url=_video_url, seek_to=_seek_to, is_playing=_is_playing, key="local_main")
            else:
                from youtube_player import youtube_player as _yt_player
                _yt_result = _yt_player(video_id=_yt, start_sec=0, seek_to=_seek_to, is_playing=_is_playing, key="yt_main")

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
                _target_idx  = int((_vid_t / FIXED_DEMO_VIDEO_DURATION_SEC) * len(pitches))
                _new_cidx_ts = max(0, min(_target_idx, len(pitches) - 1))
                if _new_cidx_ts > st.session_state.get("current_pitch_idx", 0):
                    st.session_state.current_pitch_idx = _new_cidx_ts
                    st.session_state._sync_activated    = True
                    if pitches[_new_cidx_ts]["inning"] >= 6:
                        st.session_state._sixth_inning_alert = True
                    st.rerun()

            # ── 실시간 OCR 투구 감지 (항상 작동 — 사전 스캔 병행) ──
            # _vid_t 기준으로 로컬 파일 프레임 직접 OCR → P: 증가 감지
            _ocr_vid_t = _current_video_time if _current_video_time is not None else _vid_t
            if (_ocr_vid_t is not None and loaded
                    and _local_path and os.path.exists(_local_path)):
                _last_check  = st.session_state.get("_pose_last_check_time", -99.0)
                _last_pitch  = st.session_state.get("_last_pitch_video_time", -30.0)
                _no_task     = not bool(st.session_state.get("_pose_task_id"))
                _check_due   = (_ocr_vid_t - _last_check) >= 0.5
                _cooldown_ok = (_ocr_vid_t - _last_pitch) >= 8.0
                if _check_due and _cooldown_ok and _no_task:
                    _new_tid = _start_ocr_check(_local_path, _ocr_vid_t)
                    st.session_state._pose_task_id         = _new_tid
                    st.session_state._pose_last_check_time = _ocr_vid_t
        elif _vsrc:
            _current_video_time = None
            st.video(_vsrc)
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

            _pose_active = bool(st.session_state.get("_pose_task_id"))
            _scan_st_note = st.session_state.get("_scan_status", "idle")
            if _scan_st_note == "scanning":
                _sync_note = "⏳ 영상 분석 중..."
            elif _scan_st_note == "done":
                _sync_note = "✅ 싱크 준비 완료 — 재생하면 자동 감지"
            elif _pose_active:
                _sync_note = "🔍 투구 모션 감지 중..."
            elif not _local_ready:
                _sync_note = "YouTube URL을 입력하세요"
            else:
                _sync_note = ""
            _pitch_label = f"투구 {c_idx+1} / {len(pitches)} | " if st.session_state.get("_sync_activated") else ""
            if _sync_note:
                st.caption(f"{_pitch_label}{_sync_note}")

            # 슬라이더 (투구 타임라인)
            st.markdown('<p style="font-size:.72rem;font-weight:700;color:#475569;letter-spacing:.09em;text-transform:uppercase;margin:.5rem 0 .1rem">투구 타임라인</p>', unsafe_allow_html=True)
            sel = st.slider("투구 선택", 0, max(len(pitches)-1, 0), c_idx,
                            label_visibility="collapsed")
            if sel != c_idx:
                st.session_state.current_pitch_idx = sel
                st.session_state.video_synced = True
                st.rerun()

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
            st.markdown('<div class="card-badge card-badge-actual">📊 실측</div>', unsafe_allow_html=True)
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
                if c_idx > 0 and prev and prev.get("pitch_type"):
                    _pb = _bilstm_preds[c_idx - 1] if (c_idx - 1) < len(_bilstm_preds) else None
                    _prev_pred_type = _pb["next_pitch"] if _pb else _predict_next(pitches, c_idx - 1)["next_pitch"]
                    _hit = _prev_pred_type == prev["pitch_type"]
                    _prev_pred = (f'<span style="font-size:.72rem;color:{"#34d399" if _hit else "#f87171"};'
                                  f'font-weight:700;margin-left:.4rem">{"✓ 예측 적중" if _hit else "✗ 빗나감"}</span>')

                st.markdown(
                    f'<div class="pitch-card" style="background:rgba(15,23,42,.6);border-color:{_m["color"]}44">'
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
                    '<span style="color:#334155;font-size:.9rem">⚾ 경기 로드 후 재생</span></div>',
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
                st.markdown(
                    f'<div class="card-badge card-badge-pred">🔮 예측 · {_pred_basis}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pitch-card pred-hero" style="background:linear-gradient(135deg,rgba(15,23,42,.8),'
                    f'rgba(46,27,75,.4));border-color:rgba(167,139,250,.3)">'
                    f'<div style="display:flex;align-items:center;gap:.8rem">'
                    f'<div style="font-size:2.2rem;line-height:1">{_nm["emoji"]}</div>'
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
                labels = [f'{c}<br><span style="font-size:7px">{PITCH_META.get(c,PITCH_META["OTHER"])["name"]}</span>' for c in codes]

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
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0), height=140, showlegend=False,
                    xaxis=dict(tickfont=dict(size=7, color="#64748b"),
                               gridcolor="rgba(0,0,0,0)", zeroline=False),
                    yaxis=dict(tickformat=".0%", tickfont=dict(size=7, color="#475569"),
                               gridcolor="rgba(148,163,184,.04)", zeroline=False,
                               range=[0, max(vals) * 1.4 if vals else 1]),
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
    for _col, _lbl, _val, _sub in zip(
        st.columns(4),
        ["분析 투구", "패스트볼", "변화구", "평균 구속"],
        [str(_tot), f"{_ff/max(_tot,1):.0%}", f"{(_br+_os)/max(_tot,1):.0%}",
         f"{_avgspd:.1f}" if _avgspd else "—"],
        ["구", f"{_ff}구", f"{_br+_os}구", "mph"],
    ):
        with _col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.28rem">{_lbl}</div>'
                f'<div style="font-size:1.55rem;font-weight:800;color:#e2e8f0;line-height:1">{_val}</div>'
                f'<div style="font-size:.7rem;color:#64748b;margin-top:.1rem">{_sub}</div>'
                f'</div>', unsafe_allow_html=True)

# 오프라인 스캔 폴링 (2초마다 완료 여부 확인)
_poll_stid = st.session_state.get("_scan_task_id")
if _poll_stid and _scan_tasks.get(_poll_stid, {}).get("status") == "scanning":
    time.sleep(2.0)
    st.rerun()

# 포즈 감지 태스크 폴링 (실시간 폴백 사용 시)
_active_tid = st.session_state.get("_pose_task_id")
if _active_tid and _pose_tasks.get(_active_tid, {}).get("status") == "processing":
    time.sleep(0.4)
    st.rerun()

# ── 타임스탬프 싱크 폴링 ──
# _vid_pl 불문 — 영상 시각을 한 번이라도 받으면 계속 polling해서 싱크 놓치지 않음
_spoll_vtimes  = st.session_state.get("video_pitch_times", [])
_spoll_nsi     = st.session_state.get("_next_scan_idx", 0)
_spoll_vid_t   = st.session_state.get("_vid_t")
_spoll_t_wall  = st.session_state.get("_vid_t_wall", 0.0)
_spoll_staleness = time.time() - _spoll_t_wall if _spoll_t_wall else 999
if (_spoll_vtimes
        and _spoll_nsi < len(_spoll_vtimes)
        and _spoll_vid_t is not None
        and _spoll_staleness < 300.0):
    # JS가 pitch_times를 직접 감지(200ms) → 폴링은 폴백용으로만 사용
    time.sleep(1.0)
    st.rerun()
