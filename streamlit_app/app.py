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

/* 밑바탕을 한 단 올렸다. 거의 검정(#080e1a)이라 회색 글자가 전부 묻혔고 화면이
   무거웠다. 대각선 두 줄로 만들던 X자 무늬는 알파를 절반으로 낮췄다 — 영상 뒤에서
   교차하며 시선을 끌어 어수선했다. */
[data-testid="stAppViewContainer"]{
  min-height:100vh;
  background:
    linear-gradient(115deg,transparent 49.7%,rgba(96,165,250,.022) 49.7% 50%,transparent 50%),
    linear-gradient(65deg,transparent 49.7%,rgba(96,165,250,.022) 49.7% 50%,transparent 50%),
    radial-gradient(ellipse 80% 50% at 50% -10%,rgba(59,130,246,.07),transparent),
    linear-gradient(180deg,#1c2745 0%,#18213a 45%,#141c2f 100%);
  background-attachment:fixed}
[data-testid="stSidebar"]{background:rgba(30,41,68,.97)!important;border-right:1px solid rgba(96,165,250,.2)}

/* 가로 스크롤을 원천 차단한다.
   실측 — 사용자 창이 약 1000 CSS px였는데 우측 패널 안에 min-width와 nowrap이 걸린
   요소가 있어 본문이 뷰포트보다 넓어졌다. 가로 스크롤이 생기자 왼쪽으로 스크롤된
   상태가 되어 사이드바의 "PitchIQ"와 "MLB"가 잘려 보였다. 넘칠 수 있는 곳을
   먼저 없앴지만(아래 min-width:0 / word-break), 새 요소가 또 넘치더라도 화면이
   깨지지 않도록 문서 레벨에서 한 번 더 막는다. */
html,body{max-width:100%;overflow-x:hidden}
[data-testid="stAppViewContainer"]{overflow-x:hidden}

/* 사이드바를 좁힌다. 300px 고정이라 1000px 창에서는 본문에 700px밖에 안 남고,
   그 700을 영상과 우측 패널이 나눠 가지면 패널이 200px로 짓눌린다. */
section[data-testid="stSidebar"]{width:clamp(206px,16vw,272px)!important;
  min-width:clamp(206px,16vw,272px)!important;max-width:clamp(206px,16vw,272px)!important}

/* 글씨가 작고 다닥다닥 붙어 보였다. 크기가 거의 rem이라 루트를 키우면 글자와 여백이
   같이 늘어난다 — 인라인 style이 수십 군데라 개별로 고치는 것보다 이쪽이 확실하다.

   같은 이유로 좁은 창에서는 루트를 줄인다. 17.5px 고정이었을 때 1000px 창에도
   1600px과 똑같은 크기의 글자·여백이 들어가 카드가 서로 붙고 글자가 두 줄로 접혔다
   — "겹치고 다닥다닥"의 실제 원인이다. 폭에 비례시키면 어느 창에서든 같은 비율로
   보인다. 1000px에서 14.5px, 1600px 이상에서 17.5px. */
html{font-size:clamp(14px,.5vw + 9.5px,17.5px)}
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;color:#eef2f8;
  line-height:1.65;letter-spacing:.005em}
#MainMenu,footer,header{visibility:hidden}

/* 스코어보드 타이포 — 큰 숫자/코드 요소는 콘덴스드 디스플레이 폰트 */
.team-score,.pitch-code,.conf-gauge-code,.hero-title,.intro-logo,.inning-box{
  font-family:'Oswald',sans-serif}

/* rerun 깜빡임 방지 */
[data-stale="true"]{opacity:1!important;transition:none!important}
.element-container{transition:none!important}
iframe{transition:none!important}

/* 스코어보드 */
.scoreboard{background:linear-gradient(135deg,rgba(30,41,66,.95),rgba(20,30,55,.95));
  border:1px solid rgba(59,130,246,.22);border-radius:14px;padding:.7rem 1.2rem;
  margin-bottom:.8rem;backdrop-filter:blur(16px)}
.team-score{font-size:1.7rem;font-weight:900;letter-spacing:-.03em;line-height:1}
.team-name{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#a6b3c6;margin-top:.05rem}
.inning-box{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);
  border-radius:8px;padding:.3rem .65rem;font-size:.8rem;font-weight:700;color:#93c5fd;text-align:center}
.count-dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 2px}

/* 패널 카드 — 타이포 확대 + hover glow */
.panel{background:rgba(30,41,66,.7);border:1px solid rgba(59,130,246,.12);
  border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.85rem;backdrop-filter:blur(8px);
  transition:border-color .2s,transform .2s}
.panel:hover{border-color:rgba(59,130,246,.35);transform:translateY(-2px)}
.panel-title{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#a6b3c6;margin-bottom:.5rem}
.player-name{font-size:1.15rem;font-weight:800;color:#e2e8f0;line-height:1.15}
.player-sub{font-size:.8rem;color:#a6b3c6;margin-top:.15rem}

/* 구종 카드 */
.pitch-card{border-radius:10px;padding:.95rem 1.25rem;margin-bottom:.6rem;
  border:1px solid rgba(255,255,255,.06);transition:border-color .2s,transform .2s}
.pitch-card:hover{transform:translateY(-2px)}
.pitch-code{font-size:2.4rem;font-weight:900;letter-spacing:-.02em;line-height:1}
.pitch-name{font-size:.85rem;color:#cbd5e1;margin-top:.1rem}
.pitch-speed{font-size:.9rem;font-weight:700;color:#a6b3c6;margin-top:.18rem}

/* 타임라인 행 — hover 시 상세정보 노출 */
.pitch-row{display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;
  border-radius:7px;margin-bottom:.22rem;font-size:.82rem;transition:background .15s}
.pitch-row-detail{max-height:0;opacity:0;overflow:hidden;font-size:.68rem;color:#a6b3c6;
  transition:max-height .2s ease,opacity .2s ease}
.pitch-row:hover{background:rgba(59,130,246,.08)!important}
.pitch-row:hover .pitch-row-detail{max-height:40px;opacity:1;margin-top:.15rem}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;
  font-size:.7rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.badge-actual{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.badge-pred{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}
.badge-sim{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.25)}

/* 하단 통계 카드 — hover lift */
.stat-card{background:rgba(30,41,66,.6);border:1px solid rgba(59,130,246,.1);
  border-radius:10px;padding:.9rem 1.1rem;text-align:center;margin-bottom:.5rem;
  transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:rgba(59,130,246,.4);transform:translateY(-3px)}

/* 랜딩 히어로 */
.hero-wrap{padding:2.4rem 1rem 1.6rem;text-align:center}
.hero-title{font-size:2.6rem;font-weight:900;letter-spacing:-.02em;line-height:1.1;
  background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text}
.hero-tagline{font-size:1.05rem;color:#cbd5e1;margin-top:.6rem}
.hero-badge{display:inline-block;margin-top:1rem;padding:.4rem 1rem;border-radius:999px;
  background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.35);color:#34d399;
  font-size:.85rem;font-weight:700}
.feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:1rem;margin-top:2.2rem}
.feature-card{background:rgba(30,41,66,.7);border:1px solid rgba(59,130,246,.14);
  border-radius:14px;padding:1.4rem 1.2rem;text-align:left;transition:border-color .2s,transform .2s;
  cursor:default}
.feature-card:hover{border-color:rgba(167,139,250,.5);transform:translateY(-4px)}
.feature-icon{font-size:1.8rem;margin-bottom:.5rem}
.feature-title{font-size:1.02rem;font-weight:800;color:#e2e8f0}
.feature-card-desc{max-height:0;opacity:0;overflow:hidden;font-size:.82rem;color:#cbd5e1;
  line-height:1.5;margin-top:0;transition:max-height .25s ease,opacity .25s ease,margin-top .25s ease}
.feature-card:hover .feature-card-desc{max-height:120px;opacity:1;margin-top:.55rem}

div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#1d4ed8,#6d28d9)!important;color:#fff!important;
  border:none!important;border-radius:9px!important;font-weight:600!important;
  padding:.45rem 1.1rem!important;transition:opacity .18s!important}
div[data-testid="stButton"]>button:hover{opacity:.82!important}
[data-testid="stFileUploader"]{border:1.5px dashed rgba(59,130,246,.3)!important;
  border-radius:10px!important;background:rgba(30,41,66,.4)!important}

/* 랜딩 스플래시 */
.intro-splash{position:fixed;inset:0;background:#131c30;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:1rem;z-index:9999}
.intro-splash .intro-logo{font-size:3.4rem;font-weight:900;letter-spacing:-.02em;
  background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;animation:introPulse 1.4s ease-in-out infinite}
.intro-splash .intro-tagline{font-size:1rem;color:#cbd5e1}
.intro-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.35rem .9rem;
  border-radius:999px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);
  color:#f87171;font-size:.78rem;font-weight:700;letter-spacing:.04em}
.intro-badge::before{content:"";width:7px;height:7px;border-radius:50%;background:#ef4444;
  animation:introPulse 1.1s ease-in-out infinite}
@keyframes introPulse{0%,100%{opacity:1}50%{opacity:.45}}

/* 스코어보드 강화 */
.seg-dot{display:inline-block;width:13px;height:13px;border-radius:3px;margin:0 2px}
.team-score{font-size:2.1rem}

/* ── 위계 정리 ────────────────────────────────────────────────
   화면이 밋밋했던 건 색이 아니라 위계 때문이다. 카드가 전부 같은 배경·같은 테두리·
   비슷한 글자 크기라 눈이 어디부터 볼지 정하지 못한다. 라벨은 더 죽이고 값은 더
   키운다. 요소는 하나도 늘리지 않는다 — 영상 컴포넌트 위의 요소 수가 바뀌면
   iframe이 리마운트된다 (TS-034). */
.block-container{padding-top:1.1rem!important;padding-bottom:2rem!important}

/* 스코어보드 위쪽에 중계 그래픽처럼 팀 색 띠를 깐다. 두 팀 색을 가운데서 가른다. */
.scoreboard{position:relative;overflow:hidden;padding:.9rem 1.2rem .8rem}
.scoreboard::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--away,#cbd5e1),var(--away,#cbd5e1) 46%,
    rgba(148,163,184,.12) 50%,var(--home,#cbd5e1) 54%,var(--home,#cbd5e1))}
.team-score{font-size:2.6rem;letter-spacing:-.045em}

/* 라벨 계열은 전부 한 단 낮춘다 */
.panel-title{font-size:.62rem;letter-spacing:.14em;color:#7b8aa1}
.player-name{font-size:1.25rem;letter-spacing:-.01em;color:#e8edf5}
.player-sub{font-size:.74rem;color:#93a1b6}

/* 방금 던진 구종은 오른쪽 열의 주인공이다. 왼쪽에 구종 색 띠를 세운다. */
.pitch-card{border-left-width:3px;padding:.85rem 1rem}
.pitch-code{font-size:2.1rem}
.pitch-name{font-size:.8rem;color:#b3bdcd}

/* 통계 카드는 숫자만 남긴다 */
.stat-card{padding:1rem .9rem}

/* 사이드바 카드 사이 리듬 */
.sidebar-card{margin-bottom:.85rem;padding:1rem 1.05rem}

/* ── 숨 쉴 자리 ───────────────────────────────────────────────
   "다닥다닥 붙어 있다"는 지적. 줄 간격만으로는 안 되고 요소 사이 여백을 벌려야 한다.
   작은 라벨들은 아예 한 단 키운다 — .58~.62rem은 읽으라고 만든 크기가 아니다. */
.panel{padding:1.25rem 1.4rem;margin-bottom:1.05rem}

/* 우측 열이 다닥다닥 붙어 있다는 지적으로 한 번 벌렸다가, "예측 카드까지 스크롤
   없이 보이게"로 조였더니 이번엔 겹쳐 보였다. 세로를 아끼는 방법을 바꿨다 —
   간격을 깎는 대신 영상 열을 좁혀(아래 st.columns) 영상 높이를 줄였다. 16:9라
   폭 100px을 양보하면 세로 56px이 그냥 생긴다. 간격은 읽을 수 있는 값으로 되돌린다. */
.panel-secondary{padding:1.2rem 1.3rem;margin-bottom:1.1rem;border-radius:12px}
[data-testid="stVerticalBlock"]{gap:.85rem}
.pitch-card{margin-bottom:.85rem}
.panel-title{font-size:.74rem;letter-spacing:.12em;color:#93a1b6;margin-bottom:.7rem}
.player-sub{font-size:.82rem}
.pitch-row{padding:.55rem .8rem;margin-bottom:.34rem;font-size:.88rem}
.pitch-name{font-size:.88rem}
.stat-card{padding:1.15rem 1rem}
.pl-row{padding:.3rem .35rem}
.pl-head{gap:.5rem;font-size:.87rem}
.pl-code{font-size:.78rem;width:2.1rem}
.pl-note{font-size:.76rem;line-height:1.65}
.pl-en{font-size:.68rem}
.pl-speed{font-size:1rem}
.scoreboard{padding:.7rem 1.3rem .6rem;margin-bottom:.6rem}
.stitch-divider{margin:1rem 0!important}

/* 보조 카드 / 카드 배지 */
.panel-secondary{background:rgba(30,41,66,.5);border:1px solid rgba(59,130,246,.08);
  border-radius:10px;padding:.8rem 1rem;margin-bottom:.65rem}
.panel-secondary .panel-title{font-size:.66rem}
.card-badge{display:inline-flex;align-items:center;gap:.3rem;font-size:.66rem;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;padding:.15rem .5rem;border-radius:999px;
  margin-bottom:.85rem;max-width:100%;white-space:normal;line-height:1.35;text-align:left}
.card-badge-pred{background:rgba(167,139,250,.14);color:#c4b5fd;border:1px solid rgba(167,139,250,.35)}
.card-badge-actual{background:rgba(52,211,153,.12);color:#6ee7b7;border:1px solid rgba(52,211,153,.3)}
.pred-hero{border-width:1.5px!important;padding:.85rem 1.1rem!important}
.pred-hero .pitch-code{font-size:2.2rem!important}

/* 게임 HUD — 원형 신뢰도 게이지 */
/* px가 아니라 rem이다. px로 두면 루트가 줄어드는 좁은 창에서 이것만 안 줄어들어
   예측 카드의 글자 칸을 잡아먹는다(구종명이 "포심 패스트 / 볼"로 접혔다). */
.conf-gauge{width:4rem;height:4rem;border-radius:50%;position:relative;flex-shrink:0;
  background:conic-gradient(var(--gauge-color) calc(var(--pct) * 3.6deg), rgba(255,255,255,.08) 0deg);
  display:flex;flex-direction:column;align-items:center;justify-content:center}
.conf-gauge::before{content:"";position:absolute;inset:.45rem;border-radius:50%;background:#1b2540}
.conf-gauge>*{position:relative;z-index:1}
.conf-gauge-code{font-size:1.25rem;font-weight:900;line-height:1}
.conf-gauge-pct{font-size:.6rem;font-weight:700;color:#cbd5e1;margin-top:.1rem}

/* ── 투수·타자·주자 카드 ──────────────────────────────────────────
   좁은 열에 세 정보를 밀어넣으니 글자를 줄일 수밖에 없어 다닥다닥해 보였다.
   그래서 두 층으로 나눈다 — 평소엔 읽을 수 있는 최소한만 크게 보여주고, 마우스를
   올리면 화면 가운데에 큰 카드로 펼친다. 구종 범례가 이미 같은 방식이라 조작이
   따로 배울 것 없이 일관된다. */
/* 팝업은 position:fixed지만 조상에 overflow:hidden이 걸리면 그래도 잘린다.
   본문 가로 스크롤을 막으려고 넣은 overflow-x:hidden이 그 역할을 하므로,
   이 카드가 들어앉는 열 컨테이너에는 clip을 걸지 않는다. */
[data-testid="stVerticalBlock"],[data-testid="column"]{overflow:visible}
.mu-wrap{cursor:default}
/* 힌트는 절대배치하면 안 된다 — right/bottom에 띄웠더니 내야 그림 밑의
   "루상 비어 있음" 글자 위에 겹쳐 앉았다. 카드 안쪽 마지막 줄로 흘려보낸다. */
.mu-hint{margin-top:.7rem;padding-top:.55rem;border-top:1px solid rgba(148,163,184,.1);
  font-size:.62rem;color:#6b7a91;text-align:right}
.mu-wrap:hover .panel-secondary{border-color:rgba(96,165,250,.4)}
.mu-wrap:hover .mu-hint{color:#93a1b6}

/* 오버레이 자신이 딤이고, 그 안에서 카드만 확대된다. 우측 열 폭에 안 갇힌다.
   딤을 자식 div로 두면 안 된다 — 부모에 transform이 걸려 있으면 자식
   position:fixed의 기준이 뷰포트가 아니라 그 부모 박스가 되어, 딤이 팝업 크기만
   덮고 화면 전체는 그대로 밝았다(실측). 그래서 fixed 요소에는 transform을 걸지
   않고, 확대는 안쪽 카드에만 준다. */
.mu-pop{position:fixed;inset:0;z-index:9998;
  display:flex;align-items:center;justify-content:center;
  background:rgba(8,13,24,.62);backdrop-filter:blur(3px);
  opacity:0;pointer-events:none;visibility:hidden;
  transition:opacity .18s ease,visibility 0s linear .2s}
.mu-wrap:hover .mu-pop{opacity:1;visibility:visible;transition-delay:0s}
.mu-inner{width:min(540px,86vw);max-height:88vh;overflow:auto;
  background:linear-gradient(180deg,#26324f,#1d2740);
  border:1px solid rgba(148,163,184,.22);border-radius:16px;
  padding:1.5rem 1.7rem 1.6rem;box-shadow:0 28px 70px rgba(0,0,0,.55);
  transform:scale(.965);transition:transform .22s cubic-bezier(.2,.8,.3,1)}
.mu-wrap:hover .mu-inner{transform:scale(1)}
.mu-hd{font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:#7b8aa1;margin-bottom:1rem}
.mu-grid{display:flex;align-items:flex-start;gap:1.4rem;flex-wrap:wrap}
.mu-side{flex:1 1 220px;min-width:0;display:flex;flex-direction:column;gap:1rem}
.mu-lbl{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#93a1b6;
  margin-bottom:.2rem}
.mu-nm{font-size:1.5rem;font-weight:800;line-height:1.2;overflow-wrap:anywhere}
.mu-sub{font-size:.86rem;color:#a6b3c6;margin-top:.15rem}
.mu-runner{flex:0 0 auto;text-align:center}
.mu-foot{margin-top:1.15rem;padding-top:.95rem;border-top:1px solid rgba(148,163,184,.14);
  display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.84rem;color:#cbd5e1}
.mu-foot b{color:#f1f5f9;font-weight:800}

/* 게임 HUD — COMBO 스트릭 배지 */
.combo-badge{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .6rem;border-radius:999px;
  background:rgba(251,191,36,.16);border:1px solid rgba(251,191,36,.4);color:#fbbf24;
  font-size:.72rem;font-weight:800;letter-spacing:.03em;margin-bottom:.7rem;
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
.sidebar-card{background:rgba(30,41,66,.55);border:1px solid rgba(59,130,246,.1);
  border-radius:10px;padding:.75rem .9rem;margin-bottom:.7rem}
</style>""", unsafe_allow_html=True)

# ══ 고정 데모 설정 ════════════════════════════════════════════════
FIXED_DEMO_GAME_PK   = 775300
FIXED_DEMO_VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
TEAM_COLORS = {"NYY": "#0C2340", "LAD": "#005A9C"}  # 고정 데모 게임은 이 두 팀만 등장
# 어두운 배경(#17203a)에 쓸 수 있게 밝기를 올린 변형. 구단 원색을 그대로 쓰면
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
#
# emoji를 뺐다. 색 점 옆에 같은 색 원 이모지를 두는 건 정보가 없고, 🔮·🐢 같은 건
# 분석 도구로 안 보인다. 대신 shape로 궤적 모양을 그린다 — 구종의 정체는 결국
# "어떻게 휘는가"라서 그림 하나가 이름보다 빠르다.
#
# speed는 MLB 일반 구간이고 이 경기 값이 아니다. 카드에 그렇게 적어둔다.
PITCH_META = {
    "FF": {"name": "포심 패스트볼", "en": "Four-Seam Fastball", "color": "#ef4444",
           "shape": "straight", "speed": "92~100",
           "simple": "가장 빠르고 가장 곧게 오는 공",
           "note": "투수가 던질 수 있는 가장 빠른 공이다. 공이 뒤로 회전하면서 날아와 "
                   "중력에 지는 정도가 덜하다. 그래서 타자 눈에는 예상보다 덜 떨어져 "
                   "보이고, 마치 떠오르는 것처럼 느껴진다. 다른 모든 구종은 이 공을 "
                   "기준으로 '얼마나 다르게 오는가'로 설명된다."},
    "FA": {"name": "패스트볼", "en": "Fastball", "color": "#f87171",
           "shape": "straight", "speed": "90~98",
           "simple": "종류를 특정 못 한 빠른 공",
           "note": "기록 시스템이 세부 종류를 못 붙인 빠른 공이다. 포심과 같은 계열로 "
                   "보면 된다."},
    "SI": {"name": "싱커", "en": "Sinker", "color": "#f97316",
           "shape": "sink", "speed": "90~97",
           "simple": "빠르면서 아래로 가라앉는 공",
           "note": "속구만큼 빠른데 홈플레이트 근처에서 아래로 처진다. 타자가 공 윗부분을 "
                   "때리게 되어 공이 뜨지 않고 땅으로 굴러간다. 홈런을 맞기 싫을 때 "
                   "던지는 공이다."},
    "FC": {"name": "커터", "en": "Cutter", "color": "#eab308",
           "shape": "cut", "speed": "86~94",
           "simple": "속구인 줄 알았는데 마지막에 살짝 비껴가는 공",
           "note": "거의 끝까지 속구처럼 오다가 마지막 순간 옆으로 조금 미끄러진다. "
                   "타자는 이미 속구라 판단하고 휘두른 뒤라 배트 중심을 살짝 벗어나 "
                   "맞는다. 배트가 부러지는 장면이 이 공에서 많이 나온다."},
    "SL": {"name": "슬라이더", "en": "Slider", "color": "#3b82f6",
           "shape": "slide", "speed": "80~89",
           "simple": "옆으로 미끄러지며 떨어지는 공",
           "note": "속구보다 조금 느리면서 옆으로 미끄러지듯 휘고 동시에 떨어진다. "
                   "타자가 스트라이크인 줄 알고 휘두르면 공은 이미 존 밖으로 나가 있다. "
                   "삼진을 잡으려 할 때 가장 많이 쓰는 공이다."},
    "ST": {"name": "스위퍼", "en": "Sweeper", "color": "#06b6d4",
           "shape": "sweep", "speed": "78~86",
           "simple": "옆으로 아주 크게 쓸려 나가는 공",
           "note": "슬라이더의 사촌인데 옆으로 휘는 양이 훨씬 크다. 떨어지는 건 적고 "
                   "빗자루로 쓸듯 옆으로 크게 빠져나간다. 최근 몇 년 사이 유행한 구종이다."},
    "CU": {"name": "커브", "en": "Curveball", "color": "#8b5cf6",
           "shape": "drop", "speed": "74~83",
           "simple": "위에서 아래로 뚝 떨어지는 느린 공",
           "note": "속구와 속도 차이가 가장 큰 공이다. 공이 앞으로 회전해서 무지개처럼 "
                   "큰 곡선을 그리며 뚝 떨어진다. 빠른 공을 기다리던 타자는 타이밍이 "
                   "완전히 어긋난다."},
    "KC": {"name": "너클커브", "en": "Knuckle Curve", "color": "#a855f7",
           "shape": "drop", "speed": "76~84",
           "simple": "더 급하게 떨어지는 커브",
           "note": "손가락 하나를 세워 쥐는 커브다. 일반 커브보다 떨어지는 각이 가파르다."},
    "CH": {"name": "체인지업", "en": "Changeup", "color": "#10b981",
           "shape": "sink", "speed": "79~89",
           "simple": "속구인 척하다 느리게 오는 공",
           "note": "던지는 팔 동작이 속구와 똑같은데 실제로는 10~15km/h 느리게 온다. "
                   "타자는 빠른 공이 올 줄 알고 미리 휘두르기 시작했다가 공이 아직 "
                   "도착하지 않아 헛스윙한다. 속이는 것이 목적인 공이다."},
    "FS": {"name": "스플리터", "en": "Splitter", "color": "#0891b2",
           "shape": "dive", "speed": "83~90",
           "simple": "타자 앞에서 갑자기 꺼지는 공",
           "note": "손가락 둘을 벌려 쥐고 던진다. 속구처럼 오다가 홈플레이트 바로 앞에서 "
                   "갑자기 뚝 꺼진다. 타자가 반응할 시간이 없어 헛스윙이나 땅볼이 나온다."},
    "FO": {"name": "포크볼", "en": "Forkball", "color": "#0e7490",
           "shape": "dive", "speed": "75~85",
           "simple": "스플리터보다 느리고 더 떨어지는 공",
           "note": "손가락을 더 깊이 벌려 쥐어서 더 느리고 낙차가 크다."},
    "KN": {"name": "너클볼", "en": "Knuckleball", "color": "#78716c",
           "shape": "wobble", "speed": "60~75",
           "simple": "회전이 없어 제멋대로 흔들리는 공",
           "note": "공을 거의 회전시키지 않고 던진다. 회전이 없으면 공기 흐름에 따라 "
                   "제멋대로 흔들려서 어디로 갈지 던진 투수도 모른다. 받아야 하는 포수도 "
                   "특수 글러브를 쓴다."},
    "EP": {"name": "이피어스", "en": "Eephus", "color": "#6b7280",
           "shape": "lob", "speed": "50~65",
           "simple": "산처럼 높이 띄우는 아주 느린 공",
           "note": "일부러 아주 느리게, 포물선을 그리도록 높이 띄워 던진다. 빠른 공에 "
                   "맞춰진 타자의 감각을 통째로 흔들려는 의도다. 아주 가끔 나온다."},
    "CS": {"name": "슬로커브", "en": "Slow Curve", "color": "#7c3aed",
           "shape": "drop", "speed": "65~75",
           "simple": "더 느린 커브",
           "note": "커브 중에서도 특히 느리다. 같은 커브를 두 가지 속도로 나눠 던지면 "
                   "타자가 더 헷갈린다."},
    "OTHER": {"name": "기타", "en": "Other", "color": "#8a99b0",
              "shape": "straight", "speed": "—",
              "simple": "분류되지 않은 투구",
              "note": "기록 시스템이 종류를 판정하지 못한 투구다."},
}

# 구종별 실제 무브먼트. Statcast 2024-06-03~09, **우완 투수** 18,944구 실측이다
# (scripts/measure_pitch_movement.py). 단위는 인치, 무회전 궤적 대비 변화량이다.
#
#   pfx_x  횡변화. 우완에서 **음수가 팔 쪽(3루 방향)**이다. 문서를 믿지 않고 데이터로
#          확인했다 — 싱커 -14.75, 슬라이더 +4.79로 반대 방향이고 우완 싱커는 팔 쪽이다.
#   pfx_z  종변화. 클수록 덜 떨어진다. 포심이 15.66으로 가장 크다.
#
# 표본이 작은 것(KN 73구)은 그대로 쓰되 화면에 n을 적는다. Statcast에 거의 안 잡히는
# 구종(FA·FO·EP·CS)은 실측이 없어 이웃 구종에서 추정했고 그렇다고 표시한다.
PITCH_MOVEMENT = {                      # code: (pfx_x, pfx_z, n)
    "FF": (-7.43, 15.66, 6311),
    "SL": (4.79, 1.64, 2966),
    "SI": (-14.75, 7.27, 2700),
    "FC": (2.58, 8.45, 1697),
    "CH": (-14.26, 4.85, 1557),
    "ST": (13.83, 0.95, 1309),
    "CU": (9.63, -9.96, 1232),
    "FS": (-10.87, 2.26, 743),
    "KC": (7.82, -10.20, 235),
    "KN": (-6.39, -6.17, 73),
    "FA": (-7.43, 15.66, 0),            # 포심과 같은 계열로 본다
    "FO": (-10.87, -2.00, 0),           # 스플리터보다 더 떨어진다
    "CS": (9.63, -14.00, 0),            # 커브보다 느리고 더 떨어진다
    "EP": (0.00, -30.00, 0),            # 산처럼 띄우는 공
}

# 인치 -> 그림 좌표. 존이 68x62이고 실제 스트라이크존이 약 17x25인치니 참값은 축마다
# 다르지만, 두 축에 같은 배율을 쓴다 — 횡·종을 다른 배율로 그리면 "슬라이더가 옆으로
# 더 가는지 아래로 더 가는지"가 배율 때문에 뒤바뀐다.
MOVE_SCALE = 1.6

ZONE = (66, 40, 68, 62)          # 스트라이크존 x, y, w, h
RELEASE = (100, 6)               # 릴리스 지점 (멀어서 화면 위쪽 가운데)
FASTBALL_END = (100, 60)         # 포심이 도달하는 지점 — 모든 변화의 기준


def _break_offset(code: str) -> tuple[float, float]:
    """포심 대비 변화량 (인치). 그림도 설명도 이 값 하나에서 나온다."""
    fx, fz, _ = PITCH_MOVEMENT["FF"]
    px, pz, _ = PITCH_MOVEMENT.get(code, PITCH_MOVEMENT["FF"])
    return px - fx, fz - pz          # (횡: 글러브 쪽 +, 종: 더 떨어질수록 +)


def _break_path(code: str) -> tuple[str, tuple[float, float]]:
    """궤적 경로와 도착점. 앞은 거의 직선이고 뒤에서 꺾인다 — 실제로 변화는 늦게 온다."""
    ox, oy = _break_offset(code)
    dx, dy = ox * MOVE_SCALE, oy * MOVE_SCALE
    x0, y0 = RELEASE
    x3, y3 = FASTBALL_END[0] + dx, FASTBALL_END[1] + dy
    x1, y1 = x0 + dx * 0.05, y0 + (y3 - y0) * 0.40
    x2, y2 = x0 + dx * 0.38, y0 + (y3 - y0) * 0.74
    return (f"M{x0} {y0} C{x1:.1f} {y1:.1f}, {x2:.1f} {y2:.1f}, {x3:.1f} {y3:.1f}",
            (x3, y3))


def _baseball(radius: float = 10.0) -> str:
    """실제 야구공. 흰 가죽에 붉은 실밥 두 줄."""
    r = radius
    return (
        f'<circle r="{r}" fill="url(#ballSkin)"/>'
        f'<path d="M{-r*0.55} {-r*0.83} C{-r*0.1} {-r*0.4}, {-r*0.1} {r*0.4}, '
        f'{-r*0.55} {r*0.83}" fill="none" stroke="#d64545" stroke-width="{r*0.16}" '
        f'stroke-linecap="round"/>'
        f'<path d="M{r*0.55} {-r*0.83} C{r*0.1} {-r*0.4}, {r*0.1} {r*0.4}, '
        f'{r*0.55} {r*0.83}" fill="none" stroke="#d64545" stroke-width="{r*0.16}" '
        f'stroke-linecap="round"/>'
        f'<circle r="{r}" fill="none" stroke="rgba(0,0,0,.22)" stroke-width="{r*0.08}"/>'
    )


def _pitch_arc(code: str, width: int = 32) -> str:
    """목록·카드에 쓰는 작은 아이콘 — 측면도.

    큰 그림과 같은 실측값에서 뽑되 시점만 다르다. 포수 시점 경로를 그대로 줄이면
    거의 수직이라 26px에서 전부 비슷한 사선이 되어 구분이 안 됐다. 옆에서 보면
    낙차가 길이로 펴지고, 횡변화는 늦게 꺾이는 정도로 드러난다.
    """
    meta = PITCH_META.get(code, PITCH_META["OTHER"])
    ox, oy = _break_offset(code)
    x0, y0 = 6, 10
    x3, y3 = 58, 10 + 5 + oy * 0.95          # 포심도 조금은 떨어진다
    x1, y1 = x0 + 20, y0 + (y3 - y0) * 0.10
    x2, y2 = x0 + 38, y0 + (y3 - y0) * (0.55 + abs(ox) * 0.004)
    return (
        f'<svg viewBox="0 0 64 52" width="{width}" height="{int(width * 0.81)}" '
        f'style="overflow:visible;flex-shrink:0">'
        f'<path d="M{x0} {y0} C{x1:.1f} {y1:.1f}, {x2:.1f} {y2:.1f}, {x3:.1f} {y3:.1f}" '
        f'fill="none" stroke="{meta["color"]}" stroke-width="3.4" '
        f'stroke-linecap="round" opacity=".85"/>'
        f'<circle cx="{x3:.1f}" cy="{y3:.1f}" r="4.6" fill="{meta["color"]}"/></svg>'
    )


def _pitch_arc_big(code: str) -> str:
    """호버 카드용 큰 그림 — 포수 시점, 실제 야구공이 날아온다.

    궤적은 Statcast 실측 무브먼트로 그린다. 정지 그림으로는 "휜다"를 글로 설명해야
    하는데 움직이면 설명이 필요 없다.

      - 잔상 셋이 뒤따른다. 하나만 움직이면 점이 미끄러지는 것처럼 보인다.
      - keySplines로 가까울수록 빠르게. 원근이 그렇다.
      - 공이 회전한다. 실밥이 도는 게 보여야 야구공으로 읽힌다.
      - 도착 순간 파동이 퍼진다. 어디에 꽂혔는지가 남는다.
      - 한 바퀴 돌고 잠깐 쉰다. 쉼 없이 반복하면 눈이 궤적을 못 따라간다.
    """
    meta = PITCH_META.get(code, PITCH_META["OTHER"])
    color = meta["color"]
    path, (ex, ey) = _break_path(code)
    ref_path, (rx, ry) = _break_path("FF")
    ox, oy = _break_offset(code)
    zx, zy, zw, zh = ZONE
    uid = f"p{code}"
    is_ff = code in ("FF", "FA")

    DUR = "2.6s"                       # 비행 1.9초 + 여운 0.7초
    FLY = 0.73
    SPLINE = ".42 0 .78 .45"           # 앞은 천천히, 뒤로 갈수록 빠르게

    def motion(delay: float) -> str:
        return (f'<animateMotion dur="{DUR}" repeatCount="indefinite" '
                f'keyPoints="0;1;1" keyTimes="0;{FLY};1" calcMode="spline" '
                f'keySplines="{SPLINE};0 0 1 1" begin="{delay}s">'
                f'<mpath href="#{uid}"/></animateMotion>')

    trails = ""
    for delay, r, op in ((-0.05, 4.0, .26), (-0.10, 3.2, .17), (-0.16, 2.4, .10)):
        trails += (f'<circle r="{r}" fill="{color}" opacity="{op}">{motion(delay)}'
                   f'<animate attributeName="opacity" values="0;{op};{op};0;0" '
                   f'keyTimes="0;.08;{FLY - 0.02};{FLY};1" dur="{DUR}" '
                   f'repeatCount="indefinite" begin="{delay}s"/></circle>')

    reference = "" if is_ff else (
        f'<path d="{ref_path}" fill="none" stroke="#93a1b6" stroke-width="1.5" '
        f'stroke-dasharray="4 5" opacity=".45"/>'
        f'<circle cx="{rx}" cy="{ry}" r="7.5" fill="none" stroke="#93a1b6" '
        f'stroke-width="1.5" stroke-dasharray="3 3" opacity=".5"/>')

    return (
        f'<svg viewBox="26 -14 148 128" width="100%" style="display:block">'
        f'<defs>'
        f'<radialGradient id="ballSkin" cx=".36" cy=".3" r=".78">'
        f'<stop offset="0" stop-color="#ffffff"/>'
        f'<stop offset=".6" stop-color="#f3f4f6"/>'
        f'<stop offset="1" stop-color="#c7ccd4"/>'
        f'</radialGradient>'
        f'<filter id="f{uid}" x="-80%" y="-80%" width="260%" height="260%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="2.2" flood-color="{color}" '
        f'flood-opacity=".85"/></filter>'
        f'</defs>'
        # 스트라이크존 — 3x3 격자라 도착 위치가 어디쯤인지 읽힌다
        f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" fill="rgba(96,165,250,.06)" '
        f'stroke="#93a1b6" stroke-width="1.5" opacity=".65"/>'
        f'<path d="M{zx+zw/3} {zy} V{zy+zh} M{zx+2*zw/3} {zy} V{zy+zh} '
        f'M{zx} {zy+zh/3} H{zx+zw} M{zx} {zy+2*zh/3} H{zx+zw}" '
        f'stroke="#93a1b6" stroke-width=".8" opacity=".22"/>'
        f'<path d="M{zx+6} {zy+zh+11} L{zx+zw-6} {zy+zh+11} L{zx+zw-6} {zy+zh+17} '
        f'L{zx+zw/2} {zy+zh+25} L{zx+6} {zy+zh+17} Z" fill="#cbd5e1" opacity=".45"/>'
        + reference
        + f'<path id="{uid}" d="{path}" fill="none" stroke="{color}" '
          f'stroke-width="1.8" opacity=".3"/>'
        + trails
        # 도착 파동
        + f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="none" stroke="{color}" '
          f'stroke-width="2" opacity="0">'
          f'<animate attributeName="r" values="4;4;20" keyTimes="0;{FLY};1" '
          f'dur="{DUR}" repeatCount="indefinite" calcMode="spline" '
          f'keySplines="0 0 1 1;.2 .7 .4 1"/>'
          f'<animate attributeName="opacity" values="0;0;.6;0" '
          f'keyTimes="0;{FLY};{FLY + 0.05};1" dur="{DUR}" repeatCount="indefinite"/>'
          f'</circle>'
        # 야구공 — 이동(g) > 크기(g) > 회전(g) 순으로 감싼다. 한 요소에 여러 변환을
        # 동시에 애니메이션할 수 없어서 이렇게 나눈다.
        + f'<g filter="url(#f{uid})">{motion(0)}'
          f'<g transform="scale(.26)">'
          f'<animateTransform attributeName="transform" type="scale" '
          f'values=".26;.36;.6;.95;.95" keyTimes="0;.32;.56;{FLY};1" dur="{DUR}" '
          f'repeatCount="indefinite" calcMode="spline" '
          f'keySplines="{SPLINE};{SPLINE};{SPLINE};0 0 1 1"/>'
          f'<g><animateTransform attributeName="transform" type="rotate" '
          f'from="0" to="360" dur="1.1s" repeatCount="indefinite"/>'
          + _baseball(10.0) +
          f'</g></g>'
          f'<animate attributeName="opacity" values="0;1;1;1;0;0" '
          f'keyTimes="0;.05;.5;{FLY};{FLY + 0.12};1" dur="{DUR}" repeatCount="indefinite"/>'
          f'</g>'
        + f'<text x="{RELEASE[0] - 16}" y="-3" font-size="8.5" fill="#93a1b6">릴리스</text>'
        + ('' if is_ff else
           f'<text x="{rx + 11}" y="{ry + 3}" font-size="8" fill="#93a1b6">포심</text>')
        # 실측 수치는 SVG 밖 HTML로 뺐다. 그림이 폭에 맞춰 3배로 늘어나면 글자도 같이
        # 커져 잘린다. 밖에 두면 글자 크기가 그림 배율과 무관해지고 줄바꿈도 된다.
        + '</svg>'
    )


def _pitch_stats_line(code: str) -> str:
    """그림 아래에 붙는 실측 수치. 그림은 느낌이고 숫자가 근거다."""
    ox, oy = _break_offset(code)
    n = PITCH_MOVEMENT.get(code, (0, 0, 0))[2]
    source = (f'Statcast 실측 n={n:,}' if n else '실측값 없어 이웃 구종에서 추정')
    if code in ("FF", "FA"):
        body = '모든 변화의 기준이 되는 공'
    else:
        body = (f'포심 대비 옆으로 <b>{abs(ox):.1f}인치</b> '
                f'{"글러브 쪽" if ox > 0 else "팔 쪽"}, '
                f'아래로 <b>{oy:.1f}인치</b> 더 떨어진다')
    return (f'<div class="pl-stat">{body}</div>'
            f'<div class="pl-src">포수 시점 · 우완 투수 기준 · {source}</div>')


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
    # 학습된 모델의 seq_input이 (None, 3, 6)이다. 5로 두면 shape 오류로 예측이 통째로
    # 죽는데, _run_bilstm_bg가 예외를 dict에만 담고 화면엔 안 올려서 "BiLSTM 계산 중"이
    # 영원히 남고 통계 폴백(신뢰도 20%)만 보였다.
    _SEQ_LEN    = 3
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
        # model.predict()가 아니라 모델을 직접 호출한다.
        #
        # predict()는 입력을 tf.data 파이프라인으로 감싸는데, 그 prefetch 스레드가
        # absl Mutex에서 영구히 잠들었다 — 스택을 떠보면 absl 심볼이 TensorFlow가 아니라
        # pyarrow가 싣고 온 libarrow에서 해결되고 있다(pandas 3.x가 pyarrow를 끌고 온다).
        # 서로 다른 absl 두 벌이 한 프로세스에 올라와 동기화 primitive가 어긋난 것이다.
        # 실측으로 155샘플 예측이 10분을 넘겨도 CPU 0.1%로 멈춰 있었다.
        #
        # 직접 호출은 tf.data를 아예 안 거친다. 155샘플이면 한 번에 올라가므로 배치를
        # 나눌 이유도 없다.
        probs_all = np.asarray(model(
            {
                "seq_input":     np.array(X_seq_list),
                "ctx_input":     np.array(X_ctx_list),
                "pitcher_input": np.array(X_pit_list),
                "batter_input":  np.array(X_bat_list),
            },
            training=False,
        ))
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
def _diamond_svg(on_1b: bool, on_2b: bool, on_3b: bool, size: int = 118) -> str:
    """실제 내야처럼 그린 주자 표시.

    전에는 마름모 네 개를 띄엄띄엄 찍어둬서 어디가 몇 루인지 안 보였다. 내야 흙과
    잔디, 베이스 패스를 같이 그리면 각 점이 무엇인지 설명이 필요 없다. 주자가 있는
    베이스는 노란색으로 채우고 링을 둘러 눈에 띄게 한다.
    """
    HOME, FIRST, SECOND, THIRD = (60, 104), (104, 60), (60, 16), (16, 60)

    def base(pos, on, rotate=True):
        x, y = pos
        fill = "#fbbf24" if on else "rgba(203,213,225,.28)"
        ring = (f'<rect x="{x-13}" y="{y-13}" width="26" height="26" rx="4" fill="none" '
                f'stroke="#fbbf24" stroke-width="2" opacity=".45" '
                f'transform="rotate(45,{x},{y})"/>') if on else ""
        return (ring + f'<rect x="{x-8}" y="{y-8}" width="16" height="16" rx="2.5" '
                f'fill="{fill}" transform="rotate(45,{x},{y})"/>')

    return (
        # width/height는 px로 두되 CSS로 상한만 건다. 좁은 창에서 이 그림이 안 줄어들면
        # 옆 이름 칸이 짓눌려 글자가 세로로 서거나 패널이 뷰포트를 넘긴다.
        f'<svg width="{size}" height="{size}" viewBox="0 0 120 120" '
        f'style="flex-shrink:1;max-width:{size}px;width:100%;height:auto">'
        # 잔디 -> 내야 흙 -> 베이스 패스 순으로 깐다
        '<path d="M60 8 L112 60 L60 112 L8 60 Z" fill="rgba(52,211,153,.07)"/>'
        '<path d="M60 20 L100 60 L60 100 L20 60 Z" fill="rgba(180,120,70,.16)"/>'
        '<path d="M60 104 L104 60 L60 16 L16 60 Z" fill="none" '
        'stroke="rgba(203,213,225,.3)" stroke-width="2.5"/>'
        # 마운드
        '<circle cx="60" cy="60" r="9" fill="rgba(180,120,70,.35)"/>'
        + base(SECOND, on_2b) + base(THIRD, on_3b) + base(FIRST, on_1b)
        # 홈플레이트 — 오각형이라 베이스와 구분된다
        + f'<path d="M{HOME[0]-8} {HOME[1]-6} L{HOME[0]+8} {HOME[1]-6} '
          f'L{HOME[0]+8} {HOME[1]+2} L{HOME[0]} {HOME[1]+9} L{HOME[0]-8} {HOME[1]+2} Z" '
          'fill="rgba(241,245,249,.65)"/>'
        + '<text x="104" y="78" font-size="11" fill="#a6b3c6" font-weight="700">1</text>'
        + '<text x="56" y="10" font-size="11" fill="#a6b3c6" font-weight="700">2</text>'
        + '<text x="6" y="78" font-size="11" fill="#a6b3c6" font-weight="700">3</text>'
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
    return "#cbd5e1"


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
@st.cache_data(show_spinner=False)
def _load_cv_verdicts() -> dict:
    """미리 계산해 둔 영상 판정. scripts/batch_cv_verdicts.py가 만든다.

    재생 중에 투구마다 YOLO를 돌리던 경로를 걷어냈다. 두 가지가 걸렸다 — 재생이
    무거워지고, 시연 중 실제로 본 투구만 채점되므로 표본이 수십 개에 머물렀다(n=43).
    미리 돌려두면 부하가 0이고 표본이 그대로 남는다.
    """
    path = os.path.join(ROOT, "streamlit_app", "fixed_demo_cv.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fp:
            raw = json.load(fp)
    except Exception:
        return {}
    return {int(k): v for k, v in raw.get("verdicts", {}).items()}



def _cv_accuracy(verdicts: dict, pitches: list[dict]) -> dict:
    """미리 계산한 판정을 Statcast 정답으로 채점한다.

    두 가지를 빼고 센다. 궤적을 못 잡은 건 판정이 아니고, 정답이 OFFSPEED면 2분류
    모델이 낼 수 없는 클래스라 무조건 오답이 된다 — 그건 모델이 아니라 범위의
    문제다 (ADR-0012). 대신 뺀 개수를 같이 돌려줘 화면에 적는다.
    """
    decided = scored = hits = fastballs = 0
    for idx, row in verdicts.items():
        group = row.get("group")
        if not group:
            continue
        decided += 1
        truth = _statcast_to_two_class(pitches[idx]["pitch_type"]) if idx < len(pitches) else None
        if truth:
            scored += 1
            hits += int(truth == group)
            fastballs += int(truth == "FASTBALL")
    # 기준선은 이 채점 표본의 다수 클래스 비율이다. 예전엔 58.1%를 적어뒀는데 그건
    # 실험실 테스트셋 값이라 표본 구성이 다르다. 같은 표본에서 계산해야 "아무것도
    # 안 하고 한쪽으로만 찍었을 때"와 정직하게 비교된다.
    base = max(fastballs, scored - fastballs) / scored if scored else None
    return {"attempted": len(verdicts), "decided": decided,
            "scored": scored, "hits": hits, "baseline": base,
            "acc": hits / scored if scored else None}



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
        # 여기서 삼키면 화면은 "BiLSTM 계산 중"에 영원히 멈춘다. 실제로 shape 오류를
        # 이 자리에 가둬두는 바람에 원인을 찾는 데 한참 걸렸다 — 콘솔에는 남긴다.
        import traceback
        traceback.print_exc()
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
    elif _bt.get("status") == "error":
        # 실패를 "계산 중"으로 남겨두면 영원히 기다리는 화면이 된다.
        st.session_state.bilstm_status = "error"

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
        '<div style="font-size:.78rem;color:#8a99b0;margin-top:.05rem">MLB 투구 예측 시스템</div>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitch-divider"></div>', unsafe_allow_html=True)


    # 경기 진행 요약 (로드된 경우)
    if loaded:
        cur = pitches[c_idx]
        half = "▲" if cur["inning_topbot"] == "Top" else "▼"
        _away_s = cur["away_score"]
        _home_s = cur["home_score"]
        st.markdown(
            '<div class="sidebar-card">'
            '<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#8a99b0;margin-bottom:.4rem">경기 현황</p>'
            f'<div style="font-size:.85rem;line-height:2.1;color:#a6b3c6">'
            f'<div>{cur["away_team"]} <span style="color:#e2e8f0;font-weight:700">{_away_s}</span>'
            f' : <span style="color:#e2e8f0;font-weight:700">{_home_s}</span> {cur["home_team"]}</div>'
            f'<div>이닝 <span style="color:#93c5fd;font-weight:700">{half} {cur["inning"]}회</span></div>'
            f'<div>카운트 <span style="color:#e2e8f0;font-weight:700">{cur["balls"]}-{cur["strikes"]}</span>'
            f'  아웃 <span style="color:#fbbf24;font-weight:700">{cur["outs"]}</span></div>'
            f'<div>진행 <span style="color:#a78bfa;font-weight:700">{c_idx}/{len(pitches)}구</span></div>'
            f'</div></div>', unsafe_allow_html=True)

    # 구종 범례 — 마우스를 올리면 화면 가운데 큰 카드가 뜬다.
    #
    # 처음엔 자리에서 아래로 펼쳤다. 사이드바가 좁아 그림이 200px도 안 됐고 목록이
    # 밀려 어디를 보고 있었는지 잃어버렸다. position:fixed로 뷰포트 기준에 띄우면
    # 사이드바 폭과 무관하게 크게 그릴 수 있다.
    _seen_types = {p["pitch_type"] for p in pitches} if loaded else set()
    _legend_rows = "".join(
        f'<div class="pl-row">'
        f'<div class="pl-head">'
        f'{_pitch_arc(_c, 26)}'
        f'<span class="pl-code" style="color:{_m["color"]}">{_c}</span>'
        f'<span class="pl-name">{_m["name"]}</span>'
        f'{"<span class=pl-dot></span>" if _c in _seen_types else ""}'
        f'</div>'
        f'<div class="pl-card">'
        f'<div class="pl-scrim"></div>'
        f'<div class="pl-inner" style="border-top:3px solid {_m["color"]}">'
        f'<div class="pl-hd">'
        f'<span class="pl-badge" style="background:{_m["color"]}1f;color:{_m["color"]};'
        f'border:1px solid {_m["color"]}55">{_c}</span>'
        f'<div style="flex:1">'
        f'<div class="pl-kr">{_m["name"]}</div>'
        f'<div class="pl-en">{_m["en"]}</div>'
        f'</div>'
        f'<div class="pl-spd">{_m["speed"]}<span>mph</span></div>'
        f'</div>'
        f'<div class="pl-simple">{_m["simple"]}</div>'
        f'<div class="pl-fig">{_pitch_arc_big(_c)}</div>'
        f'{_pitch_stats_line(_c)}'
        f'<div class="pl-note">{_m["note"]}</div>'
        f'</div></div></div>'
        for _c, _m in list(PITCH_META.items())[:10] if _c != "OTHER"
    )
    st.markdown(
        '<style>'
        # 사이드바 안쪽 블록이 내용을 잘라내면 팝업이 안 보인다
        '[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{overflow:visible!important}'
        '.pl-row{border-radius:8px;padding:.3rem .35rem;transition:background .18s ease}'
        '.pl-row:hover{background:rgba(96,165,250,.1)}'
        '.pl-head{display:flex;align-items:center;gap:.5rem;font-size:.87rem}'
        '.pl-code{font-weight:700;width:2.1rem;font-size:.78rem;letter-spacing:.02em}'
        '.pl-name{color:#a6b3c6}'
        '.pl-dot{width:5px;height:5px;border-radius:50%;background:#4dbd8a;'
        'margin-left:auto;flex-shrink:0}'
        # 뷰포트 한가운데 고정. 사이드바 폭에 안 갇힌다.
        '.pl-card{position:fixed;left:calc(50% + 4rem);top:50%;z-index:9998;'
        'opacity:0;pointer-events:none;visibility:hidden;'
        'transform:translate(-50%,-50%) scale(.965);'
        'transition:opacity .18s ease,transform .22s cubic-bezier(.2,.8,.3,1),'
        'visibility 0s linear .2s}'
        '.pl-row:hover .pl-card{opacity:1;visibility:visible;'
        'transform:translate(-50%,-50%) scale(1);transition-delay:0s}'
        '.pl-scrim{position:fixed;inset:0;z-index:-1;background:rgba(8,13,24,.62);'
        'backdrop-filter:blur(3px)}'
        '.pl-inner{width:min(560px,84vw);max-height:86vh;overflow:auto;'
        'background:linear-gradient(180deg,#26324f,#1d2740);'
        'border:1px solid rgba(148,163,184,.22);border-radius:16px;'
        'padding:1.5rem 1.7rem 1.6rem;box-shadow:0 28px 70px rgba(0,0,0,.55)}'
        '.pl-hd{display:flex;align-items:center;gap:.85rem;margin-bottom:.9rem}'
        '.pl-badge{font-size:1rem;font-weight:800;padding:.3rem .6rem;border-radius:8px;'
        'letter-spacing:.02em}'
        '.pl-kr{font-size:1.45rem;font-weight:800;color:#f1f5f9;line-height:1.2}'
        '.pl-en{font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:#93a1b6}'
        '.pl-spd{font-size:1.5rem;font-weight:800;color:#e2e8f0;white-space:nowrap}'
        '.pl-spd span{font-size:.72rem;font-weight:600;color:#93a1b6;margin-left:.25rem}'
        '.pl-simple{font-size:1.02rem;font-weight:700;color:#cbd5e1;line-height:1.5;'
        'margin-bottom:1rem;word-break:keep-all}'
        '.pl-fig{padding:.7rem .4rem .3rem;border-radius:12px;'
        'background:rgba(10,16,30,.5);border:1px solid rgba(148,163,184,.1)}'
        '.pl-stat{margin-top:.7rem;font-size:.88rem;color:#cbd5e1;line-height:1.6;'
        'word-break:keep-all}'
        '.pl-stat b{color:#f1f5f9;font-weight:800}'
        '.pl-src{margin-top:.2rem;font-size:.72rem;color:#7b8aa1}'
        '.pl-note{margin-top:.9rem;padding-top:.9rem;'
        'border-top:1px solid rgba(148,163,184,.14);'
        'font-size:.92rem;line-height:1.8;color:#a6b3c6;word-break:keep-all}'
        '</style>'
        '<div class="sidebar-card">'
        '<p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#93a1b6;margin-bottom:.3rem">구종 범례</p>'
        f'{_legend_rows}'
        '<p style="margin-top:.4rem;font-size:.66rem;color:#7b8aa1;line-height:1.5">'
        '마우스를 올리면 설명이 나온다 · '
        '<span style="display:inline-block;width:5px;height:5px;border-radius:50%;'
        'background:#4dbd8a;vertical-align:middle"></span> 이 경기에 나온 구종</p>'
        '</div>', unsafe_allow_html=True)


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
    _aw_color    = TEAM_ACCENTS.get(aw, "#cbd5e1")
    _hw_color    = TEAM_ACCENTS.get(hw, "#cbd5e1")

    st.markdown(
        f'<div class="scoreboard" style="--away:{_aw_color};--home:{_hw_color}">'
        # 폭을 제한하고 가운데 정렬한다. space-between으로 풀어두면 와이드 화면에서
        # 두 팀 점수가 양 끝으로 밀려나고 가운데 900px가 빈 영역이 된다.
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'gap:clamp(1.5rem,6vw,5rem);max-width:760px;margin:0 auto">'
        # 원정팀
        f'<div style="text-align:center;min-width:72px">'
        f'<div style="font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#a6b3c6">원정</div>'
        f'<div style="font-size:.8rem;font-weight:800;color:{_aw_color};letter-spacing:.08em;'
        f'border-bottom:2px solid {_aw_color};padding-bottom:.15rem;display:inline-block">{aw}</div>'
        f'<div class="team-score" style="color:{"#f1f5f9" if aws >= hws else "#8a99b0"}">{aws}</div>'
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
        f'<div style="font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#a6b3c6">홈</div>'
        f'<div style="font-size:.8rem;font-weight:800;color:{_hw_color};letter-spacing:.08em;'
        f'border-bottom:2px solid {_hw_color};padding-bottom:.15rem;display:inline-block">{hw}</div>'
        f'<div class="team-score" style="color:{"#f1f5f9" if hws >= aws else "#8a99b0"}">{hws}</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True)
else:
    _render_landing_hero()

st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

# 6회초 알림
#
# 자리를 먼저 잡고 내용만 채운다. 조건부로 st.warning()을 부르면 배너가 뜨는 순간
# 아래 영상 컴포넌트 위의 요소 수가 바뀐다. Streamlit은 요소의 위치로 컴포넌트를
# 식별하므로 iframe이 리마운트되고 영상이 처음으로 돌아간다 — 실측으로 6회에
# 진입한 직후 앱은 162구를 가리키는데 영상만 경기 시작(FLAHERTY P:3, 1회)으로
# 되돌아가 둘이 분리됐다. 자동 싱크는 전진만 하므로 스스로 복구되지도 않는다.
_sixth_inning_slot = st.empty()
if st.session_state.get("_sixth_inning_alert"):
    _sixth_inning_slot.warning("🔔 6회초 시작 — 1~5이닝 분석 구간 완료!", icon="⚾")

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
    # 영상 열을 좁히는 게 세로를 버는 가장 싼 방법이다. 영상은 16:9라 폭이 줄면
    # 높이가 그 56%만큼 같이 준다 — 우측 패널은 그만큼 여유가 생기고, 간격을 깎아
    # 다닥다닥해 보이게 만들 필요가 없어진다. 1000px 창 실측 기준으로 잡았다.
    col_video, col_panel = st.columns([2.5, 1.5], gap="medium")
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

            # 이동 직후 영상이 아직 옛 시각을 보고하는 구간을 건너뛴다. 목표 근처에
            # 닿으면 그때부터 다시 자동 싱크를 받는다.
            _pending_seek = st.session_state.get("_pending_seek_t")
            if _pending_seek is not None:
                if _vid_t is not None and abs(_vid_t - _pending_seek) <= 5.0:
                    st.session_state._pending_seek_t = None
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
                '<div style="background:rgba(23,32,54,.9);border:1.5px dashed rgba(59,130,246,.2);'
                'border-radius:12px;height:300px;display:flex;flex-direction:column;'
                'align-items:center;justify-content:center;gap:.6rem">'
                '<div style="font-size:2.8rem">🎬</div>'
                '<div style="color:#6b7a91;font-size:.85rem">사이드바에서 YouTube URL을 입력하거나 영상을 업로드하세요</div>'
                '<div style="color:#46536b;font-size:.72rem">YOLO가 자동으로 투구를 감지합니다</div>'
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
            # c_idx는 대기 중인 투구다. 사용자가 보는 기준은 방송 스코어버그의
            # 투구수이므로 "지금까지 던진 개수"(= c_idx)를 그대로 쓴다. 슬라이더
            # 손잡이 숫자도 같은 값이라 둘이 어긋나지 않는다.
            _pitch_label = f"투구 {c_idx} / {len(pitches)} | "
            if _sync_note:
                st.caption(f"{_pitch_label}{_sync_note}")

            # 슬라이더 (투구 타임라인)
            # 아래 여백이 .1rem이었을 때 슬라이더가 손잡이 위에 띄우는 현재값("0")이
            # 이 제목 글자를 그대로 덮었다. 값 라벨은 트랙 위쪽 바깥에 절대배치되므로
            # 제목이 그만큼 비켜줘야 한다.
            st.markdown('<p style="font-size:.72rem;font-weight:700;color:#8a99b0;letter-spacing:.09em;text-transform:uppercase;margin:.5rem 0 1.15rem">투구 타임라인</p>', unsafe_allow_html=True)
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
                    # 영상이 실제로 옮겨가기 전 한동안 옛 시각을 계속 보고한다. 그동안
                    # 자동 싱크가 그 옛 시각으로 인덱스를 정하면, 뒤로 이동한 순간
                    # 곧바로 원래 자리로 끌려간다(자동 싱크는 전진만 하므로). 도착을
                    # 확인할 때까지 보고를 무시하려고 목표 시각을 남긴다.
                    st.session_state._pending_seek_t = st.session_state.seek_to

            # 슬라이더 손잡이를 현재 인덱스에 맞춘다. 위젯을 만들기 **전에** 키를 쓰는 것이
            # 정해진 방법이다 — 콜백 안에서 자기 위젯 키를 건드리면 on_change와 얽혀
            # 방금 누른 값이 되돌아온다(실측: 버튼을 눌러도 seek_to가 0.0으로 잡혔다).
            # **무조건 덮어쓰면 안 된다.** 슬라이더를 끌면 재실행이 걸리는데, 그 재실행에서
            # 이 줄이 사용자가 끈 값을 위젯이 만들어지기 전에 c_idx로 되돌린다. 그러면
            # 아래 sel != c_idx 가 영원히 성립하지 않는다 — 실측으로 161까지 끌어도
            # 손잡이가 3으로 돌아오고 영상도 안 움직였다.
            #
            # 그래서 "인덱스가 다른 이유로 바뀌었을 때"만 손잡이를 옮긴다. 마지막으로
            # 밀어넣은 값을 따로 기억해두면 자동 싱크·버튼(인덱스가 먼저 바뀐다)과
            # 슬라이더 조작(위젯 값이 먼저 바뀐다)을 구분할 수 있다.
            if st.session_state.get("_slider_synced_idx") != c_idx:
                st.session_state.pitch_slider = c_idx
                st.session_state._slider_synced_idx = c_idx

            # 버튼과 같은 이유로 on_change 콜백을 쓴다. 반환값으로 처리하면 재실행
            # 경쟁에 밀린다 (아래 버튼 주석 참고).
            def _slider_moved_cb() -> None:
                _goto_pitch_cb(st.session_state.pitch_slider)

            st.slider("투구 선택", 0, max(len(pitches)-1, 0),
                      key="pitch_slider", label_visibility="collapsed",
                      on_change=_slider_moved_cb)

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
            st.markdown('<p style="font-size:.72rem;font-weight:700;color:#8a99b0;letter-spacing:.09em;text-transform:uppercase;margin:.4rem 0 .15rem">최근 투구</p>', unsafe_allow_html=True)
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
                _bg  = "rgba(59,130,246,.1)" if _is_c else "rgba(23,32,54,.6)"
                _bdr = "1px solid rgba(59,130,246,.3)" if _is_c else "1px solid rgba(148,163,184,.07)"
                _ev  = _r["events"] if _r["events"] and _r["events"] not in ("nan", "None", "") else ""
                _ev_html = f' <span style="color:#34d399;font-size:.65rem">[{_ev}]</span>' if _ev else ""
                _row_batter = _r["batter_name"].split(",")[0] if _r.get("batter_name") else "—"
                _row_desc   = _row_desc_map.get(_r.get("description"), _r.get("description") or "—")
                st.markdown(
                    f'<div class="pitch-row" style="background:{_bg};border:{_bdr}">'
                    f'<div style="display:flex;align-items:center;gap:.5rem;width:100%">'
                    f'<span style="color:#8a99b0;width:1.4rem;font-size:.68rem;font-weight:{"700" if _is_c else "400"}">'
                    f'#{_r["pitch_idx"]+1}</span>'
                    f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
                    f'font-weight:700;color:{_m["color"]};width:3.4rem">'
                    f'{_pitch_arc(_pt, 22)}{_pt}</span>'
                    f'<span style="color:#a6b3c6;font-size:.68rem">{_spd} mph</span>'
                    f'<span style="color:#a6b3c6;font-size:.65rem;margin-left:auto">'
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
                'color:#6b7a91;font-size:.85rem;border:1px dashed rgba(59,130,246,.1);border-radius:12px">'
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
                    f'<span style="color:#cbd5e1;font-size:.9rem;text-decoration:line-through">{_old_p}</span>'
                    f'<span style="color:#8a99b0;font-size:1rem;font-weight:300">→</span>'
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
                    f'<div style="background:rgba(30,41,66,.8);border:1px solid rgba(52,211,153,.35);'
                    f'border-radius:12px;padding:.65rem 1rem;margin-bottom:.55rem">'
                    f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
                    f'color:#34d399;margin-bottom:.25rem">⚾ 타자 교체</div>'
                    f'<div style="font-size:1.05rem;font-weight:800;color:#e2e8f0">{_new_b}</div>'
                    f'<div style="font-size:.72rem;color:#a6b3c6;margin-top:.1rem">{_bat_team} · {_new_bh}타</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── 투수 / 타자 정보 ──
            batting_team = cur["away_team"] if cur["inning_topbot"] == "Top" else cur["home_team"]
            # 이름 두 줄(좌) + 내야 그림(우)을 한 행에 놓는다. 예전엔 이름 아래 주자
            # 행을 따로 둬서 카드가 ~275px였고, 그만큼 예측 카드가 화면 밖으로 밀렸다 —
            # "예측 구종까지 스크롤 없이"가 요구라 이 카드가 제일 먼저 줄어야 한다.
            _runner_cnt = sum((cur["on_1b"], cur["on_2b"], cur["on_3b"]))
            _runner_txt = (" · ".join(n for n, on in (("1루", cur["on_1b"]), ("2루", cur["on_2b"]),
                                                      ("3루", cur["on_3b"])) if on)
                           or "루상 비어 있음")
            _half_kor = "초" if cur["inning_topbot"] == "Top" else "말"
            st.markdown(
                f'<div class="mu-wrap">'
                f'<div class="panel-secondary">'
                # flex-wrap — 열이 좁으면 그림이 이름 밑으로 내려간다. 안 그러면 그림이
                # flex-shrink:0이라 이름 칸이 몇 px로 짓눌려 글자가 세로로 선다(실측).
                #
                # min-width는 0이어야 한다. 150px을 주면 좁은 창에서 이 칸이 안 줄어들어
                # 패널 전체가 뷰포트보다 넓어지고, 문서에 가로 스크롤이 생겨 왼쪽이
                # 잘려 보였다. 줄바꿈으로 흡수시키는 게 맞다.
                f'<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">'
                f'<div style="flex:1 1 140px;display:flex;flex-direction:column;gap:.75rem;min-width:0">'
                f'<div style="border-left:3px solid #60a5fa;padding-left:.7rem">'
                f'<div style="font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;'
                f'color:#93a1b6">투수</div>'
                f'<div style="font-size:1.22rem;font-weight:800;color:#60a5fa;line-height:1.25;'
                f'overflow-wrap:anywhere">'
                f'{cur["pitcher_name"]} <span style="font-size:.78rem;font-weight:600;'
                f'color:#a6b3c6;white-space:nowrap">{cur["pitcher_hand"]}구</span></div>'
                f'</div>'
                f'<div style="border-left:3px solid #cbd5e1;padding-left:.7rem">'
                f'<div style="font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;'
                f'color:#93a1b6">타자 · {batting_team}</div>'
                f'<div style="font-size:1.22rem;font-weight:800;color:#f1f5f9;line-height:1.25;'
                f'overflow-wrap:anywhere">'
                f'{cur["batter_name"]} <span style="font-size:.78rem;font-weight:600;'
                f'color:#a6b3c6;white-space:nowrap">{cur["batter_hand"]}타</span></div>'
                f'</div></div>'
                # 주자 — 내야 그림 하나로 충분하다. 텍스트는 그림 밑에 한 줄만.
                f'<div style="text-align:center;flex:0 1 96px;min-width:64px">'
                f'{_diamond_svg(cur["on_1b"], cur["on_2b"], cur["on_3b"], size=96)}'
                f'<div style="font-size:.72rem;color:#a6b3c6;margin-top:.15rem">'
                f'주자 <span style="color:#fbbf24;font-weight:800">{_runner_cnt or "없음"}'
                f'{"명" if _runner_cnt else ""}</span></div>'
                f'<div style="font-size:.64rem;color:#8a99b0">{_runner_txt}</div>'
                f'</div></div>'
                f'<div class="mu-hint">마우스를 올리면 크게 보인다</div>'
                f'</div>'  # /panel-secondary
                # ── 호버 팝업 ──
                f'<div class="mu-pop"><div class="mu-inner">'
                f'<div class="mu-hd">{cur["inning"]}회 {_half_kor} · '
                f'{cur["away_team"]} {cur["away_score"]} : {cur["home_score"]} {cur["home_team"]}</div>'
                f'<div class="mu-grid">'
                f'<div class="mu-side">'
                f'<div style="border-left:3px solid #60a5fa;padding-left:.85rem">'
                f'<div class="mu-lbl">투수</div>'
                f'<div class="mu-nm" style="color:#60a5fa">{cur["pitcher_name"]}</div>'
                f'<div class="mu-sub">{cur["pitcher_hand"]}구</div></div>'
                f'<div style="border-left:3px solid #cbd5e1;padding-left:.85rem">'
                f'<div class="mu-lbl">타자 · {batting_team}</div>'
                f'<div class="mu-nm" style="color:#f1f5f9">{cur["batter_name"]}</div>'
                f'<div class="mu-sub">{cur["batter_hand"]}타</div></div>'
                f'</div>'
                f'<div class="mu-runner">'
                f'{_diamond_svg(cur["on_1b"], cur["on_2b"], cur["on_3b"], size=168)}'
                f'<div style="font-size:.98rem;color:#a6b3c6;margin-top:.4rem">주자 '
                f'<span style="color:#fbbf24;font-weight:800">{_runner_cnt or "없음"}'
                f'{"명" if _runner_cnt else ""}</span></div>'
                f'<div style="font-size:.82rem;color:#8a99b0">{_runner_txt}</div>'
                f'</div></div>'
                f'<div class="mu-foot">'
                f'<span>볼카운트 <b>{cur["balls"]}-{cur["strikes"]}</b></span>'
                f'<span>아웃 <b>{cur["outs"]}</b></span>'
                f'<span>이 경기 <b>{c_idx}/{len(pitches)}구</b></span>'
                f'</div>'
                f'</div></div>'  # /mu-inner /mu-pop
                f'</div>',       # /mu-wrap
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
                    _dinfo = _desc_map.get(_ap["description"], ("", "#8a99b0"))
                    _dot_c = _dinfo[1]
                    _dots_html += (
                        f'<span title="{_dinfo[0]}" style="display:inline-block;width:9px;height:9px;'
                        f'border-radius:50%;background:{_dot_c};margin-right:3px"></span>'
                    )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0 .7rem">'
                    f'<span style="font-size:.72rem;color:#8a99b0;white-space:nowrap">타석 {_ab_pitch_n}구</span>'
                    f'<div>{_dots_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── 방금 던진 구종 ──
            # 배지와 제목을 한 줄에 — 따로 두면 st.markdown 두 개가 세로로 쌓여 ~25px를
            # 더 먹고, 그만큼 아래 예측 카드가 화면 밖으로 밀린다.
            # 아래 여백은 이 줄에서 준다. .1rem이었을 때 "실측" 알약의 밑변이 바로 아래
            # 카드의 윗선에 닿아 겹쳐 보였다. Streamlit 컨테이너 gap은 요소 사이에
            # 항상 걸리는 값이 아니라 여기서 직접 벌리는 게 확실하다.
            st.markdown(
                '<div style="display:flex;align-items:center;gap:.45rem;margin:.1rem 0 .85rem">'
                '<span class="card-badge card-badge-actual" style="margin-bottom:0">실측</span>'
                '<span class="panel-title" style="font-size:.72rem;font-weight:700;letter-spacing:.1em;'
                'text-transform:uppercase;color:#8a99b0;margin-bottom:0">방금 던진 구종</span>'
                '</div>', unsafe_allow_html=True)

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
                _desc_info = _desc_map.get(_desc_raw, (_desc_raw, "#a6b3c6"))
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
                    f'<div class="pitch-card {_reveal_cls}" style="background:rgba(30,41,66,.6);border-color:{_m["color"]}44">'
                    f'<div style="display:flex;align-items:center;gap:.8rem">'
                    f'<div style="display:flex;align-items:center;justify-content:center;'
                    f'width:3.1rem;height:2.6rem;border-radius:10px;'
                    f'background:{_m["color"]}14;border:1px solid {_m["color"]}33">'
                    f'{_pitch_arc(_display_code, 34)}</div>'
                    f'<div style="flex:1">'
                    f'<div class="pitch-code" style="color:{_m["color"]}">{_display_code}</div>'
                    f'<div class="pitch-name">{_ocr_type_str or _m["name"]}</div>'
                    f'<div style="margin-top:.2rem;display:flex;align-items:center;flex-wrap:wrap;gap:.3rem">'
                    f'<span style="color:#a6b3c6;font-size:.85rem;font-weight:600">{_spd}</span>'
                    f'<span style="background:{_desc_col}22;color:{_desc_col};border:1px solid {_desc_col}44;'
                    f'border-radius:999px;padding:.08rem .4rem;font-size:.68rem;font-weight:700">{_desc_kor}</span>'
                    + (f'<span style="color:#34d399;font-size:.72rem;font-weight:700">[{_ev_kor}]</span>' if _ev_kor else "")
                    + (_prev_pred)
                    + f'</div></div></div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="pitch-card" style="min-height:70px;display:flex;align-items:center;'
                    'justify-content:center;background:rgba(23,32,54,.5);border-color:rgba(59,130,246,.1)">'
                    '<span style="color:#8a99b0;font-size:.9rem">경기 로드 후 재생</span></div>',
                    unsafe_allow_html=True)

            # ── 영상만으로 낸 구종 (CV 궤적 판정) ──
            # 위 실측은 Statcast API에서 왔다. 아래는 API 없이 영상 궤적만으로 낸 값이라
            # 둘을 나란히 두면 CV 적중률이 화면에서 바로 채점된다.
            #
            # 재생 중에 돌리지 않는다. scripts/batch_cv_verdicts.py가 미리 판정해 둔 것을
            # 읽기만 한다. 예전엔 투구마다 YOLO를 돌려 재생이 무거웠고, 시연 중 실제로 본
            # 투구만 채점되므로 표본이 n=43에 머물렀다. 사이드바 토글도 같이 없앴다 —
            # 부하가 없으면 끄고 켤 이유가 없다.
            _cv_all  = _load_cv_verdicts()
            _cv_vd   = _cv_all.get(_ocr_i)
            _cv_stat = _cv_accuracy(_cv_all, pitches) if _cv_all else None

            if _cv_vd and _cv_vd.get("group"):
                _cv_group = _cv_vd["group"]
                _cv_col   = "#60a5fa" if _cv_group == "FASTBALL" else "#c084fc"
                _cv_kor   = "속구 계열" if _cv_group == "FASTBALL" else "변화구 계열"
                _cv_truth = _statcast_to_two_class(prev["pitch_type"]) if prev else None
                _cv_mark  = ""
                if _cv_truth:
                    _cv_ok   = _cv_truth == _cv_group
                    _cv_mark = (f'<span style="font-size:.68rem;font-weight:700;margin-left:.35rem;'
                                f'color:{"#34d399" if _cv_ok else "#f87171"}">'
                                f'{"적중" if _cv_ok else "오답"}</span>')
                _cv_body = (
                    f'<span style="color:{_cv_col};font-weight:800;font-size:.9rem">{_cv_kor}</span>'
                    f'<span style="color:#a6b3c6;font-size:.72rem;margin-left:.4rem">'
                    f'{_cv_vd.get("confidence", 0):.0%} · 궤적 {_cv_vd.get("n_points", 0)}점</span>'
                    + _cv_mark)
            elif _cv_vd:
                # 궤적을 못 잡은 건 숨기지 않는다. 숨기면 정확도가 실제보다 좋아 보인다.
                _cv_why = {"no_detections": "공이 화면에서 안 잡힘", "no_trajectory": "궤적이 안 그려짐",
                           "too_few_points": "궤적이 너무 짧음"}.get(
                               _cv_vd.get("reason", ""), _cv_vd.get("reason", "") or "사유 없음")
                _cv_body = (f'<span style="color:#8a99b0;font-size:.76rem">이 공은 영상으로 못 맞힘 '
                            f'<span style="color:#6b7a91">· {_cv_why}</span></span>')
            elif not _cv_all:
                _cv_body = ('<span style="color:#6b7a91;font-size:.76rem">'
                            '사전 판정 파일이 없다 — scripts/batch_cv_verdicts.py</span>')
            else:
                _cv_body = ('<span style="color:#8a99b0;font-size:.76rem">'
                            '이 공은 영상 판정 대상이 아님</span>')

            # 위계를 뒤집는다. 예전에는 "판정 대상 아님" 안내가 카드에서 제일 크고 각주가
            # 세 줄이라, 정작 이 카드의 결론인 적중률이 구석에 밀려 있었다. 적중률을
            # 가장 크게 두고 근거는 한 줄로 줄인 뒤 나머지는 title 툴팁으로 넘긴다.
            if _cv_stat and _cv_stat["acc"] is not None:
                _cv_rate  = f'{_cv_stat["acc"]:.1%}'
                _cv_sub   = f'{_cv_stat["hits"]}/{_cv_stat["scored"]}구'
                _cv_foot  = (f'무작정 한쪽으로 찍으면 {_cv_stat["baseline"]:.1%} · '
                             f'이번 경기 {_cv_stat["decided"]}/{_cv_stat["attempted"]}구 판정')
                _cv_tip   = (f'Statcast API 없이 중계 영상 궤적만으로 속구/변화구를 가른 결과. '
                             f'대상 {_cv_stat["attempted"]}구 중 {_cv_stat["decided"]}구 판정, '
                             f'그중 {_cv_stat["scored"]}구 채점(OFFSPEED는 2분류로 표현 불가라 제외). '
                             f'기준선 {_cv_stat["baseline"]:.1%}')
            else:
                _cv_rate, _cv_sub = '—', ''
                _cv_foot = '속구 vs 변화구 2분류'
                _cv_tip  = 'Statcast API 없이 중계 영상 궤적만으로 구종을 가른다'

            st.markdown(
                f'<div title="{_cv_tip}" style="margin:.2rem 0 .85rem;padding:.7rem .8rem;'
                f'border-radius:10px;background:rgba(23,32,54,.5);'
                f'border:1px solid rgba(77,189,138,.18)">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap">'
                f'<span style="font-size:.62rem;font-weight:700;letter-spacing:.08em;'
                f'text-transform:uppercase;color:#8a99b0;white-space:nowrap">영상만으로 · CV'
                f'<span style="margin-left:.35rem;padding:.05rem .3rem;border-radius:4px;'
                f'background:rgba(77,189,138,.14);color:#4dbd8a;font-size:.58rem;'
                f'letter-spacing:0">사전 판정</span></span>'
                f'<span style="display:flex;align-items:baseline;gap:.28rem;white-space:nowrap">'
                f'<span style="font-size:1.05rem;font-weight:800;color:#4dbd8a;line-height:1">{_cv_rate}</span>'
                f'<span style="font-size:.6rem;color:#7b8aa1">{_cv_sub}</span></span>'
                f'</div>'
                f'<div style="margin-top:.3rem">{_cv_body}</div>'
                f'<div style="margin-top:.28rem;font-size:.58rem;color:#7b8aa1">{_cv_foot}</div>'
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

                if _bilstm_res:
                    _pred_basis = "BiLSTM 모델 예측 — 직전 투구 흐름 기반"
                elif st.session_state.get("bilstm_status") == "error":
                    _pred_basis = "통계 기반 예측 (BiLSTM 실패 — 콘솔 로그 확인)"
                elif st.session_state.get("bilstm_status") == "done":
                    # 모델이 다 돌았는데도 이 구에 값이 없는 경우다. 그 투수의 2025 학습
                    # 데이터가 없거나(교체 투수) 시퀀스 3구가 아직 안 쌓인 구간이다.
                    _pred_basis = "통계 기반 예측 (이 투수는 학습 이력 부족)"
                else:
                    _pred_basis = "통계 기반 예측 (BiLSTM 계산 중)"
                if st.session_state.pred_streak >= 2:
                    st.markdown(
                        f'<div class="combo-badge">🔥 COMBO x{st.session_state.pred_streak}</div>',
                        unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card-badge card-badge-pred">예측 · {_pred_basis}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pitch-card pred-hero card-reveal" style="background:linear-gradient(135deg,rgba(30,41,66,.8),'
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
                    f'<span style="color:#8a99b0;font-size:.72rem;margin-left:.2rem">신뢰도</span>'
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
                    textfont=dict(size=10, color="#cbd5e1"),
                    customdata=list(zip(codes, _pitch_kor_names)),
                    hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>확률: %{x:.1%}<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=8, t=6, b=6), showlegend=False,
                    height=max(120, 26 * len(codes) + 24),
                    bargap=0.28,
                    yaxis=dict(autorange="reversed",   # 확률 높은 구종이 위로
                               tickfont=dict(size=10, color="#cbd5e1"),
                               gridcolor="rgba(0,0,0,0)", zeroline=False),
                    xaxis=dict(tickformat=".0%", tickfont=dict(size=8, color="#8a99b0"),
                               gridcolor="rgba(148,163,184,.06)", zeroline=False,
                               range=[0, max(vals) * 1.25 if vals else 1]),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 투수 이번 경기 구종 분포 (누적)
            if c_idx >= 3:
                seen = [p for p in pitches[:c_idx]
                        if p["pitcher_id"] == cur["pitcher_id"] and p["pitch_type"]]
                if seen:
                    cnt   = Counter(p["pitch_type"] for p in seen)
                    _pc   = list(cnt.keys())
                    _pv   = [cnt[c] for c in _pc]
                    _pcol = [PITCH_META.get(c, PITCH_META["OTHER"])["color"] for c in _pc]
                    _plab = [f'{c} {PITCH_META.get(c,PITCH_META["OTHER"])["name"]}' for c in _pc]
                    fig2  = go.Figure(go.Pie(
                        labels=_plab, values=_pv,
                        marker=dict(colors=_pcol, line=dict(color="#131c30", width=2)),
                        hole=0.55, textinfo="label+percent",
                        textfont=dict(size=7.5, color="#cbd5e1"),
                        hovertemplate="%{label}: %{value}구 (%{percent})<extra></extra>",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=4, b=0), height=170, showlegend=False,
                        annotations=[dict(text=f"<b>{sum(_pv)}</b><br>구", x=0.5, y=0.5,
                                          font=dict(size=11, color="#e2e8f0"), showarrow=False)],
                    )
                    st.markdown(f'<p style="font-size:.62rem;font-weight:700;color:#8a99b0;letter-spacing:.09em;text-transform:uppercase;margin:.15rem 0 .1rem">{cur["pitcher_name"].split(",")[0]} 구종 분포</p>', unsafe_allow_html=True)
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
                f'<div style="font-size:.7rem;color:#8a99b0;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.28rem">{_icon}&nbsp;{_lbl}</div>'
                f'<div style="font-size:1.55rem;font-weight:800;color:{_vcol};line-height:1">{_val}</div>'
                f'<div style="font-size:.7rem;color:#a6b3c6;margin-top:.1rem">{_sub}</div>'
                f'</div>', unsafe_allow_html=True)

# 오프라인 스캔 폴링 (2초마다 완료 여부 확인)
_poll_stid = st.session_state.get("_scan_task_id")
if _poll_stid and _scan_tasks.get(_poll_stid, {}).get("status") == "scanning":
    time.sleep(2.0)
    st.rerun()

# CV 판정 폴링 — 백그라운드 결과를 화면에 올리려면 재렌더가 필요하다.

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
