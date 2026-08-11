"""
오버레이 프록시에서 프레임 간 변화점을 찾고, 이미 아는 투구 시각 65개로 검증한다.

스코어버그 카운터 박스는 투구가 있을 때만 내용이 바뀐다 — 구속이 떴다가 P:N으로
돌아간다. 박스 안 픽셀만 보므로 카메라 전환·리플레이에 흔들리지 않는다. TS-032의
마운드 모션이 실패한 이유가 정확히 그 흔들림이었다.

여기까지는 디코드가 끝난 배열만 훑으므로 비용이 없다. OCR은 여기서 고른 지점에만
건다 (tesseract 1회 1040ms라 조밀 OCR은 143분).

검증은 fixed_demo_scan.json의 투구 시각 65개로 한다. 이 시각들은 오버레이 OCR이
MPH를 본 지점이라 "변화점이 여기 있어야 한다"의 정답 노릇을 한다.

실행:
  venv/bin/python3 scripts/find_overlay_changes.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "output", "timeline", "overlay_frames.npy")
SCAN = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
OUT_DIFF = os.path.join(ROOT, "output", "timeline", "overlay_diff.npy")
OUT_PNG = os.path.join(ROOT, "output", "timeline", "overlay_diff.png")

FPS = 4


def frame_diff(frames: np.ndarray) -> np.ndarray:
    """인접 프레임 평균 절대차. 값이 클수록 박스 내용이 바뀐 것."""
    a = frames[:-1].astype(np.int16)
    b = frames[1:].astype(np.int16)
    return np.abs(b - a).mean(axis=(1, 2)).astype(np.float32)


def find_events(diff: np.ndarray, threshold: float, min_gap_sec: float) -> list[float]:
    """임계 초과 지점을 최소 간격으로 묶어 이벤트 1개씩 남긴다.

    구속 등장 -> P:N 복귀만 해도 변화가 두 번 이상 난다. 한 투구를 한 점으로
    접으려면 묶어야 한다. 묶음 안에서는 가장 이른 지점을 쓴다 — 구속이 뜨는 순간이
    투구에 가장 가깝다.
    """
    above = np.flatnonzero(diff > threshold)
    if above.size == 0:
        return []
    gap = int(min_gap_sec * FPS)
    events = [above[0]]
    for i in above[1:]:
        if i - events[-1] > gap:
            events.append(i)
    return [(i + 1) / FPS for i in events]      # diff[i]는 frame i -> i+1


def score(events: list[float], truth: list[float], lo: float, hi: float) -> dict:
    """정답 시각마다 [t+lo, t+hi] 안의 이벤트를 찾아 리콜과 오프셋을 잰다."""
    arr = np.array(events)
    offsets, hit = [], 0
    for t in truth:
        window = arr[(arr >= t + lo) & (arr <= t + hi)]
        if window.size:
            hit += 1
            offsets.append(float(window[0] - t))
    return {
        "n_events": len(events),
        "recall": hit / len(truth),
        "hit": hit,
        "median_offset": float(np.median(offsets)) if offsets else float("nan"),
        "std_offset": float(np.std(offsets)) if offsets else float("nan"),
    }


def main() -> None:
    frames = np.load(FRAMES)
    diff = frame_diff(frames)
    np.save(OUT_DIFF, diff)
    truth = [float(t) for t in json.load(open(SCAN))["pitch_times"]]

    print(f"프레임 {frames.shape} · 차분 {diff.shape[0]}점 · {diff.shape[0]/FPS:.0f}s")
    qs = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
    print("차분 분포: " + "  ".join(
        f"p{q}={np.percentile(diff, q):.2f}" for q in qs))
    print(f"최대 {diff.max():.2f} · 평균 {diff.mean():.2f}\n")

    print(f"{'thr':>6} {'gap':>5} {'이벤트':>7} {'리콜':>7} {'중앙오프셋':>10} {'표준편차':>8}")
    for threshold in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for min_gap in (6.0, 10.0):
            events = find_events(diff, threshold, min_gap)
            s = score(events, truth, -6.0, 12.0)
            print(f"{threshold:6.1f} {min_gap:5.1f} {s['n_events']:7d} "
                  f"{s['recall']:6.0%} {s['median_offset']:10.2f} {s['std_offset']:8.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib 없음 — 그래프 생략")
        return

    t = np.arange(diff.shape[0]) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(t, diff, lw=0.3)
    axes[0].set_title("overlay box frame diff (full)")
    axes[0].set_xlabel("video time (s)")
    lo, hi = 0, 400
    m = (t >= lo) & (t <= hi)
    axes[1].plot(t[m], diff[m], lw=0.8)
    for pt in [x for x in truth if lo <= x <= hi]:
        axes[1].axvline(pt, color="crimson", ls="--", lw=0.8)
    axes[1].set_title(f"{lo}-{hi}s zoom (dashed = known pitch times)")
    axes[1].set_xlabel("video time (s)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\n그래프: {OUT_PNG}")


if __name__ == "__main__":
    main()
