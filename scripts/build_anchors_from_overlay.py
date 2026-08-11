"""
오버레이 P:N 판독을 앵커로 바꾸고, 홀드아웃으로 균등 분할 기준선과 대조한다.

TS-032에서 같은 자리에서 한 번 속았다. 적합에 쓴 앵커로 재면 70%가 나오는데
홀드아웃에서는 3%였고 아무것도 안 한 균등 분할(9%)보다 나빴다. 그래서 여기서는
세 가지를 항상 같이 잰다.

    균등 분할        아무것도 안 한 기준선
    기존 앵커 33개   지금 앱에 들어 있는 것
    새 앵커          이번에 만든 것 (홀드아웃 제외하고 적합)

측정 단위는 초가 아니라 **구 수**다. 사용자가 보는 건 "몇 번째 투구로 표시되는가"지
초가 아니다. 초 오차도 같이 낸다.

실행:
  venv/bin/python3 scripts/build_anchors_from_overlay.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from timeline_anchor import (  # noqa: E402
    index_at_time, pitcher_pitch_counts, resolve_anchors, time_at_index,
)

COUNTERS = os.path.join(ROOT, "output", "timeline", "overlay_counters.jsonl")
PITCHES = os.path.join(ROOT, "output", "timeline", "pitches_775300.json")
SCAN = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
OLD_ANCHORS = os.path.join(ROOT, "streamlit_app", "fixed_demo_anchors.json")
OUT_JSON = os.path.join(ROOT, "output", "timeline", "anchors_new.json")
OUT_PNG = os.path.join(ROOT, "output", "timeline", "anchor_eval.png")

DURATION = 8231.0
FOLDS = 4
SEED = 20260811


def counter_runs(rows: list[dict]) -> list[tuple[float, int]]:
    """같은 P:N이 이어지는 구간을 하나로 접고 (처음 뜬 시각, 값)을 낸다.

    한 투구의 P:N은 다음 투구까지 10초 넘게 떠 있고 그 사이 리플레이로 바가
    사라졌다 다시 나타나면 구간이 쪼개진다. 값이 같으면 같은 투구다.
    """
    runs: list[tuple[float, int]] = []
    for row in sorted(rows, key=lambda r: r["start"]):
        value = row["value"]
        if value is None:
            continue
        if runs and runs[-1][1] == value:
            continue
        runs.append((float(row["start"]), int(value)))
    return runs


def calibrate_delay(runs: list[tuple[float, int]], truth: list[float]) -> float:
    """투구 시각과 'P:N이 뜬 시각'의 간격. 스코어버그는 구속을 먼저 3초쯤 보여준다."""
    starts = np.array([t for t, _ in runs])
    gaps = []
    for t in truth:
        after = starts[(starts >= t) & (starts <= t + 15)]
        if after.size:
            gaps.append(float(after[0] - t))
    return float(np.median(gaps)) if gaps else 0.0


def internal_consistency(anchors: list[tuple[float, int]],
                         observations: list[tuple[float, int]],
                         pitches: list[dict]) -> str:
    """적합에 안 쓰인 성질로 앵커를 검사한다.

    같은 투수 안에서 P:N이 1 늘면 인덱스도 정확히 1 늘어야 한다. resolve_anchors는
    '증가하는가'만 보고 '얼마나 증가하는가'는 안 본다. 그래서 이 일치율은 공짜로
    얻는 독립 검사다.
    """
    by_time = {round(t, 3): c for t, c in observations}
    counts = pitcher_pitch_counts(pitches)
    ok = bad = 0
    for (t0, i0), (t1, i1) in zip(anchors, anchors[1:]):
        c0, c1 = by_time.get(round(t0, 3)), by_time.get(round(t1, 3))
        if c0 is None or c1 is None:
            continue
        same_pitcher = pitches[i0]["pitcher_id"] == pitches[i1]["pitcher_id"]
        if not same_pitcher or c1 <= c0:
            continue
        if (i1 - i0) == (c1 - c0) and counts[i1] == c1:
            ok += 1
        else:
            bad += 1
    total = ok + bad
    return f"{ok}/{total} = {ok/total:.0%}" if total else "표본 없음"


def evaluate(name: str, fit_anchors: list[tuple[float, int]],
             holdout: list[tuple[float, int]]) -> dict:
    """홀드아웃 지점에서 예측 인덱스와 실제 인덱스의 차이."""
    errors = [index_at_time(t, fit_anchors, DURATION, 320) - idx for t, idx in holdout]
    secs = [time_at_index(idx, fit_anchors, DURATION, 320) - t for t, idx in holdout]
    e = np.array(errors, dtype=float)
    return {
        "name": name, "n": len(e),
        "median_abs": float(np.median(np.abs(e))),
        "mean_abs": float(np.abs(e).mean()),
        "std": float(e.std()),
        "p90_abs": float(np.percentile(np.abs(e), 90)),
        "within1": float((np.abs(e) <= 1).mean()),
        "within2": float((np.abs(e) <= 2).mean()),
        "sec_median_abs": float(np.median(np.abs(secs))),
    }


def main() -> None:
    rows = [json.loads(line) for line in open(COUNTERS) if line.strip()]
    pitches = json.load(open(PITCHES))
    truth = [float(t) for t in json.load(open(SCAN))["pitch_times"]]
    old_rows = json.load(open(OLD_ANCHORS))
    old_obs = [(float(r["t"]), r["counter"]) for r in old_rows]

    runs = counter_runs(rows)
    delay = calibrate_delay(runs, truth)
    print(f"구간 {len(rows)}개 · 판독 {sum(1 for r in rows if r['value'] is not None)}개 "
          f"· 서로 다른 P:N 런 {len(runs)}개")
    print(f"투구 -> P:N 표시 지연 중앙값 {delay:.2f}s (구속 표시 구간)\n")

    observations = [(t - delay, c) for t, c in runs]

    all_anchors = resolve_anchors(observations, pitches, DURATION)
    old_anchors = resolve_anchors(old_obs, pitches, DURATION)
    print(f"새 앵커 {len(all_anchors)}개 · 기존 앵커 {len(old_anchors)}개")
    print(f"내부 일관성(같은 투수 P:N 증가분 == 인덱스 증가분): "
          f"{internal_consistency(all_anchors, observations, pitches)}\n")

    if len(all_anchors) < 2:
        print("앵커가 너무 적다 — 중단")
        return

    # 홀드아웃: 관측을 4겹으로 나눠 한 겹씩 빼고 적합, 뺀 겹에서만 잰다.
    rng = np.random.default_rng(SEED)
    fold_of = rng.integers(0, FOLDS, size=len(observations))
    anchor_index = {round(t, 3): idx for t, idx in all_anchors}

    stats = {n: [] for n in ("새 앵커", "균등 분할", "기존 앵커 33개")}
    for fold in range(FOLDS):
        fit_obs = [o for o, f in zip(observations, fold_of) if f != fold]
        held = [(t, anchor_index[round(t, 3)])
                for (t, _), f in zip(observations, fold_of)
                if f == fold and round(t, 3) in anchor_index]
        if not held:
            continue
        fit_anchors = resolve_anchors(fit_obs, pitches, DURATION)
        stats["새 앵커"].append(evaluate("새 앵커", fit_anchors, held))
        stats["균등 분할"].append(evaluate("균등 분할", [], held))
        stats["기존 앵커 33개"].append(evaluate("기존 앵커 33개", old_anchors, held))

    print(f"{'방법':<16}{'n':>5}{'|오차| 중앙':>10}{'평균':>8}{'p90':>7}"
          f"{'±1구':>8}{'±2구':>8}{'초오차중앙':>11}")
    summary = {}
    for name, folds in stats.items():
        if not folds:
            continue
        agg = {k: float(np.mean([f[k] for f in folds]))
               for k in folds[0] if k != "name"}
        summary[name] = agg
        print(f"{name:<16}{agg['n']:5.0f}{agg['median_abs']:10.2f}"
              f"{agg['mean_abs']:8.2f}{agg['p90_abs']:7.1f}"
              f"{agg['within1']:8.0%}{agg['within2']:8.0%}{agg['sec_median_abs']:11.1f}")

    baseline = summary.get("균등 분할", {}).get("mean_abs")
    new = summary.get("새 앵커", {}).get("mean_abs")
    print()
    if baseline is None or new is None:
        print("판정 불가")
    elif new < baseline:
        print(f"판정: 균등 분할 대비 평균 오차 {baseline:.2f}구 -> {new:.2f}구 "
              f"({(1-new/baseline):.0%} 감소)")
    else:
        print(f"판정: 실패 — 균등 분할({baseline:.2f}구)을 못 이겼다({new:.2f}구). 폐기.")

    json.dump([{"t": t, "counter": c} for t, c in observations],
              open(OUT_JSON, "w"), indent=1)
    print(f"\n관측 저장: {OUT_JSON} ({len(observations)}개 -> 앵커 {len(all_anchors)}개)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    grid = np.linspace(0, DURATION, 1200)
    ax.plot(grid, [index_at_time(t, [], DURATION, 320) for t in grid],
            label="uniform split", lw=1.2, ls="--", color="gray")
    ax.plot(grid, [index_at_time(t, old_anchors, DURATION, 320) for t in grid],
            label=f"old anchors ({len(old_anchors)})", lw=1.2, color="tab:orange")
    ax.plot(grid, [index_at_time(t, all_anchors, DURATION, 320) for t in grid],
            label=f"new anchors ({len(all_anchors)})", lw=1.4, color="tab:blue")
    ax.scatter([t for t, _ in all_anchors], [i for _, i in all_anchors],
               s=6, color="tab:blue", zorder=3)
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("pitch index")
    ax.legend()
    ax.set_title("video time -> pitch index")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"그래프: {OUT_PNG}")


if __name__ == "__main__":
    main()
