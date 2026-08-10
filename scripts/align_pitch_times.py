"""
[폐기됨 — TS-032] 이 접근은 실패했다. 재사용하기 전에 아래를 읽어라.

마운드 프레임 차분에는 투구를 가릴 신호가 없었다. 홀드아웃 검증에서 아무것도 안 한
균등 분할보다 나빴다.

    앵커(적합, 순환)  n=33  표준편차  2.2s   창 안 70%
    홀드아웃          n=32  표준편차 24.9s   창 안  3%
    균등 분할(기준선) n=32  표준편차  7.1s   창 안  9%

기록으로 남긴다. 다음에 이 신호를 다시 쓰려는 사람이 같은 12분을 안 쓰게 하려는 것이고,
홀드아웃과 기준선을 같이 재야 폐기 판단이 나온다는 예시이기도 하다.

────────────────────────────────────────────────────────────────────────

마운드 모션 점수 시계열에서 투구 시각 320개를 뽑아 Statcast 투구 순서에 맞춘다.

목표: 타임라인에서 보간을 없앤다. 지금은 앵커가 320구 중 33개뿐이라 사이를 직선으로
때우고, 최대 간격이 1139초(약 44구)다. 투구 시각을 전부 알면 조회만 하면 된다.

왜 OCR이 아니라 모션인가: 방송 오버레이는 투구 +2.8~4.2초에야 뜬다(TS-031). 그리고
tesseract 1회가 1040ms라 8231초를 1fps로만 훑어도 143분이다(실측). 모션은 프록시에서
전체 스캔이 2분이면 끝난다.

검증 방법이 핵심이다. 절대 정답(각 투구의 진짜 영상 시각)은 없다. 대신 OCR로 확정한
앵커 33개가 있다 — "오버레이 시각 t에서 투구 인덱스 i". 뽑은 피크가 옳다면
`t - peak_time[i]`가 33개 전부에서 비슷한 값(오버레이 지연 ≈ 3초)이어야 한다.
흩어지면 정렬이 틀린 것이다.

실행:
  venv/bin/python3 scripts/align_pitch_times.py <scores.npy> <fps> [--write]
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from timeline_anchor import resolve_anchors  # noqa: E402

GAME_PK       = 775300
ANCHOR_PATH   = os.path.join(ROOT, "streamlit_app", "fixed_demo_anchors.json")
OUT_PATH      = os.path.join(ROOT, "streamlit_app", "fixed_demo_pitch_times.json")
VIDEO_DUR_SEC = 8231.0

# 스캔으로 확인된 실제 투구 간격은 최소 12초였다. 8초보다 가까운 두 피크는 같은 투구의
# 리플레이나 카메라 컷으로 본다.
MIN_GAP_SEC   = 8.0
SMOOTH_SEC    = 1.0     # 지속 모션 강조 — 카메라 컷은 단발이라 눌린다


def load_pitches() -> list[dict]:
    from pybaseball import statcast_single_game
    df = statcast_single_game(GAME_PK).sort_values(
        ["inning", "inning_topbot", "at_bat_number", "pitch_number"],
        ascending=[True, False, True, True],
    )
    return [{"pitcher_id": int(p)} for p in df["pitcher"]]


# 오버레이가 뜨는 지연의 중앙값. 스캔창 +2.8~4.2초의 가운데를 쓴다.
OVERLAY_LAG_SEC = 3.4


def smooth(scores: np.ndarray, fps: float) -> np.ndarray:
    k = max(1, int(fps * SMOOTH_SEC))
    return np.convolve(scores, np.ones(k) / k, mode="same")


def pick_in_window(sm: np.ndarray, fps: float, t0: float, t1: float,
                   n_want: int) -> list[int]:
    """[t0, t1) 구간에서 점수 큰 순으로 최소 간격을 지켜 정확히 n_want개를 고른다."""
    if n_want <= 0:
        return []
    a, b = max(0, int(t0 * fps)), min(len(sm), int(t1 * fps))
    if b <= a:
        return []

    seg   = sm[a:b]
    gap   = int(fps * MIN_GAP_SEC)
    taken: list[int] = []
    for i in np.argsort(-seg):
        if len(taken) >= n_want:
            break
        if all(abs(int(i) - j) > gap for j in taken):
            taken.append(int(i))
    # 간격 제약 때문에 모자라면 그만큼만 돌려준다 — 호출부가 균등 분할로 메운다.
    return sorted(a + i for i in taken)


def pick_peaks(sm: np.ndarray, fps: float, n_pitches: int,
               anchors: list[tuple[float, int]]) -> list[float]:
    """
    앵커 사이 구간마다 그 구간의 투구 수만큼만 피크를 고른다.

    전역 상위 N개로 뽑으면 실패한다(실측: 앵커 대조 지연 표준편차 198초). 모션 점수는
    액션이 많은 구간에서 통째로 높아서, 개수를 전역으로만 맞추면 그런 구간에 피크가
    몰리고 조용한 구간은 통째로 건너뛴다. 구간별 투구 수는 앵커가 확정해주므로
    그 제약을 그대로 쓴다.
    """
    # 앵커의 오버레이 시각을 투구 시각으로 되돌린다.
    pts: list[tuple[float, int]] = [(0.0, -1)]
    pts += [(t - OVERLAY_LAG_SEC, i) for t, i in anchors]
    pts += [(VIDEO_DUR_SEC, n_pitches)]

    times: dict[int, float] = {i: t for t, i in pts if 0 <= i < n_pitches}

    for (ta, ia), (tb, ib) in zip(pts, pts[1:]):
        need = ib - ia - 1
        if need <= 0:
            continue
        idxs = pick_in_window(sm, fps, ta, tb, need)
        for k, gi in enumerate(range(ia + 1, ib)):
            if k < len(idxs):
                times[gi] = idxs[k] / fps
            else:                       # 피크가 모자란 구간은 균등 분할로 메운다
                times[gi] = ta + (tb - ta) * (k + 1) / (need + 1)

    return [times[i] for i in range(n_pitches)]


def lag_stats(name: str, overlay_times: list[float], pitch_times: np.ndarray) -> None:
    """
    오버레이 시각마다 가장 가까운 예측 투구 시각과의 차이를 본다.

    옳다면 이 차이가 +2.8~4.2초 근처에 몰려야 한다. 다만 320구가 평균 25.7초 간격이라
    아무 시각이나 잡아도 가장 가까운 투구까지는 평균 6.4초밖에 안 된다 — 그래서 '가깝다'
    자체는 증거가 못 되고, **부호와 집중도**를 봐야 한다. 투구가 오버레이보다 먼저면
    차이가 항상 양수여야 한다.
    """
    if not overlay_times:
        return
    lags = []
    for t_ov in overlay_times:
        j = int(np.argmin(np.abs(pitch_times - t_ov)))
        lags.append(t_ov - pitch_times[j])
    lags = np.array(lags)
    in_win = int(((lags >= 2.0) & (lags <= 5.0)).sum())
    print(f"  {name:12s} n={len(lags):3d}  중앙값 {np.median(lags):+6.1f}s  "
          f"표준편차 {lags.std():5.1f}s  범위 {lags.min():+7.1f}~{lags.max():+7.1f}s  "
          f"창(+2~5s) 안 {in_win}/{len(lags)} = {in_win/len(lags):.0%}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    scores = np.load(sys.argv[1])
    fps    = float(sys.argv[2])
    write  = "--write" in sys.argv

    pitches  = load_pitches()
    counters = [(float(r["t"]), r.get("counter"))
                for r in json.load(open(ANCHOR_PATH))]
    anchors  = resolve_anchors(counters, pitches, VIDEO_DUR_SEC)
    all_ov   = [float(r["t"]) for r in json.load(open(ANCHOR_PATH))]
    fit_ov   = {t for t, _ in anchors}
    held_ov  = [t for t in all_ov if t not in fit_ov]

    print(f"투구 {len(pitches)}구 · OCR 앵커 {len(anchors)}개(맞추는 데 씀) · "
          f"홀드아웃 오버레이 {len(held_ov)}개 · 점수 {len(scores)}개 ({len(scores)/fps:.0f}초)\n")

    sm    = smooth(scores, fps)
    times = np.array(pick_peaks(sm, fps, len(pitches), anchors))
    gaps  = np.diff(times)
    print(f"투구 시각 {len(times)}개 · 간격 중앙값 {np.median(gaps):.1f}s "
          f"· 최소 {gaps.min():.1f}s · 최대 {gaps.max():.1f}s\n")

    print("오버레이 시각 - 가장 가까운 예측 투구 시각 (기대: +2.8~4.2초에 집중)")
    lag_stats("앵커(적합)", sorted(fit_ov), times)
    lag_stats("홀드아웃", held_ov, times)

    # 기준선: 예측을 균등 분할로 바꿨을 때 같은 지표가 얼마나 나오는지.
    uniform = np.arange(len(pitches)) / len(pitches) * VIDEO_DUR_SEC
    lag_stats("균등(기준선)", held_ov, uniform)

    if write:
        rows = [{"idx": i, "t": round(float(t), 3)} for i, t in enumerate(times)]
        json.dump(rows, open(OUT_PATH, "w"), indent=1)
        print(f"\n저장: {OUT_PATH} ({len(rows)}개)")
    else:
        print("\n(--write 없이 실행 — 파일 저장 안 함)")


if __name__ == "__main__":
    main()
