"""
앱이 뽑은 사슬이 학습 궤적과 같은 종류인지 진단한다.

배경: 같은 모델이 Savant 클립에서 정확도 0.783인데 전체 중계 영상에서 0.410이다
(TS-025). 배선 버그는 아니다 — 캐시된 학습 궤적을 같은 추론 경로로 통과시키면
0.942가 나온다. 그래서 남은 설명은 "찾아낸 궤적이 학습이 본 것과 다른 종류"다.

왜 다를 수 있는가: Savant 클립은 투구가 항상 2.8~4.2초에 오도록 잘려 있어 폭 1.4초
창의 '가장 긴 움직임'이 곧 투구다. 전체 중계에서는 투구 시점을 몰라 창을 4.5초로
넓혔고, 그 안에는 포수 송구·타자 움직임·카메라 팬·리플레이가 함께 들어온다.

이 스크립트가 답하는 것은 하나다:

  앱 사슬의 특징 분포가 학습 분포 안에 있는가?

  크게 벗어남  -> 잘못된 움직임을 잡고 있다. 투구 판별 게이트로 해결 가능 (싸다)
  비슷함       -> 궤적은 맞는데 모델이 못 맞히는 것. 재학습이 필요하다 (비싸다)

이 판단 없이 게이트부터 만들면, 틀렸을 때 그 시간이 통째로 날아간다.

실행:
  venv/bin/python3 scripts/diagnose_domain_gap.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_type_cv.live_classifier import (  # noqa: E402
    CHAIN_MAX_JUMP_PX,
    CHAIN_MIN_TOTAL_MOVE_PX,
    DEFAULT_LOOKBACK_END_SEC,
    DEFAULT_LOOKBACK_START_SEC,
    DETECT_TARGET_FPS,
    TWO_CLASSES,
    align_chain_frames,
)
from pitch_type_cv.pitch_group_map import ocr_pitch_name_to_group  # noqa: E402
from pitch_type_cv.trajectory_features import (  # noqa: E402
    box_sizes_for_chain,
    compute_trajectory_features,
    extract_trajectory_candidates,
    longest_moving_chain_frames,
)

OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
SCAN_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
VIDEO_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
TRAIN_PATH = os.path.join(OUT_DIR, "dataset_clips.csv")
APP_FEATURES_PATH = os.path.join(OUT_DIR, "app_chain_features.csv")
FIG_PATH = os.path.join(OUT_DIR, "domain_gap.png")

# 분포 비교에 쓸 특징. 중요도 상위이면서 평행이동에 영향받지 않는 것들로 고른다.
# end_frame은 앱에서 상수로 정렬되므로(TS-025) 비교 의미가 없어 뺀다.
COMPARE_COLUMNS = [
    "vertical_accel_px", "vertical_drop_px", "curvature_ratio",
    "speed_ratio_late_early", "apparent_speed_px_per_frame",
    "frame_span", "path_length_px", "horizontal_deviation_px",
    "box_growth_per_frame",
]


def extract_app_features() -> pd.DataFrame:
    """앱과 완전히 같은 경로로 사슬을 뽑아 특징을 남긴다."""
    from ultralytics import YOLO

    video = max(
        (os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
         if f.endswith((".mp4", ".mkv", ".webm"))),
        key=os.path.getsize,
    )
    scan = json.load(open(SCAN_PATH))
    det = YOLO(os.path.join(ROOT, "models", "ball_broadcast_v1.pt"))

    rows = []
    for t, d in zip(scan["pitch_times"], scan["pitch_data"]):
        truth = ocr_pitch_name_to_group(d.get("pitch_type"))
        if truth not in TWO_CLASSES:
            continue
        candidates = extract_trajectory_candidates(
            video, float(t), det,
            lookback_start_sec=DEFAULT_LOOKBACK_START_SEC,
            lookback_end_sec=DEFAULT_LOOKBACK_END_SEC,
            target_fps=DETECT_TARGET_FPS, imgsz=960, conf=0.05,
        )
        chain = longest_moving_chain_frames(
            candidates, CHAIN_MAX_JUMP_PX, CHAIN_MIN_TOTAL_MOVE_PX
        ) if candidates else []
        if not chain:
            continue
        sizes = box_sizes_for_chain(chain, candidates)
        framed = [(f, x, y, s) for (f, x, y), s in zip(align_chain_frames(chain), sizes)]
        features = compute_trajectory_features(
            [(x, y) for _f, x, y, _s in framed],
            frame_indices=[f for f, _x, _y, _s in framed],
            box_sizes=[s for _f, _x, _y, s in framed],
        )
        if features is None:
            continue
        rows.append({**features, "t": float(t), "truth": truth})
        print(f"  t={float(t):8.1f}s  {truth:8s}  점 {len(chain):2d}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(APP_FEATURES_PATH, index=False)
    print(f"\n앱 사슬 {len(df)}개 특징 저장: {APP_FEATURES_PATH}")
    return df


def mahalanobis(sample: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = sample - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    train = train[train["has_trajectory"] & train["group"].isin(TWO_CLASSES)].reset_index(drop=True)
    print(f"학습 궤적 {len(train)}개 (2분류)")

    if os.path.exists(APP_FEATURES_PATH):
        app = pd.read_csv(APP_FEATURES_PATH)
        print(f"앱 사슬 {len(app)}개 (캐시 사용: {APP_FEATURES_PATH})\n")
    else:
        print("앱 경로로 사슬 추출 중 (YOLO)...\n")
        app = extract_app_features()

    if app.empty:
        raise SystemExit("앱 사슬이 하나도 안 잡혔다")

    # 1) 특징별 비교 — 앱 평균이 학습 분포에서 몇 표준편차 떨어져 있는가
    print(f"{'특징':<28}{'학습 평균':>12}{'앱 평균':>12}{'차이(σ)':>10}")
    print("-" * 62)
    shifts = {}
    for col in COMPARE_COLUMNS:
        tm, ts = train[col].mean(), train[col].std()
        am = app[col].mean()
        z = (am - tm) / ts if ts else 0.0
        shifts[col] = z
        flag = " <<" if abs(z) >= 1.0 else ""
        print(f"{col:<28}{tm:>12.2f}{am:>12.2f}{z:>10.2f}{flag}")

    # 2) 마할라노비스 거리 — 앱 사슬이 학습 분포 안에 드는가
    mu = train[COMPARE_COLUMNS].mean().values
    cov = np.cov(train[COMPARE_COLUMNS].values, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    d_train = mahalanobis(train[COMPARE_COLUMNS].values, mu, inv_cov)
    d_app = mahalanobis(app[COMPARE_COLUMNS].values, mu, inv_cov)
    cutoff = np.percentile(d_train, 95)
    inside = float((d_app <= cutoff).mean())

    print("\n마할라노비스 거리 (학습 분포 중심 기준)")
    print(f"  학습  중앙값 {np.median(d_train):.2f}  95분위 {cutoff:.2f}")
    print(f"  앱    중앙값 {np.median(d_app):.2f}  최대 {d_app.max():.2f}")
    print(f"  앱 사슬 중 학습 95분위 안에 드는 비율: {inside:.1%}")

    # 3) 판정
    n_shifted = sum(1 for z in shifts.values() if abs(z) >= 1.0)
    print("\n[판정]")
    if inside < 0.60 or n_shifted >= 3:
        print(f"  분포가 벗어난다 (안쪽 {inside:.0%}, 1σ 이상 어긋난 특징 {n_shifted}개).")
        print("  -> 잘못된 움직임을 잡고 있을 가능성이 높다. 투구 판별 게이트 (갈래 A)로 간다.")
        print("     게이트 성공 기준: 통과분 정확도 0.70 이상 & 판정률 40% 이상.")
    else:
        print(f"  분포가 비슷하다 (안쪽 {inside:.0%}, 1σ 이상 어긋난 특징 {n_shifted}개).")
        print("  -> 궤적은 맞는데 모델이 못 맞히는 것이다. 게이트로는 안 되고 재학습이 필요하다 (갈래 C).")

    # 4) 그림
    plt.rcParams["font.family"] = "AppleGothic"    # 한글 tofu 방지 (TS-004)
    plt.rcParams["axes.unicode_minus"] = False
    ncol = 3
    nrow = (len(COMPARE_COLUMNS) + ncol) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.4 * nrow))
    axes = axes.ravel()
    for ax, col in zip(axes, COMPARE_COLUMNS):
        lo = min(train[col].quantile(.01), app[col].quantile(.01))
        hi = max(train[col].quantile(.99), app[col].quantile(.99))
        bins = np.linspace(lo, hi, 30)
        ax.hist(train[col], bins=bins, density=True, alpha=.55,
                color="#3b82f6", label=f"학습 (n={len(train)})")
        ax.hist(app[col], bins=bins, density=True, alpha=.55,
                color="#ef4444", label=f"앱 (n={len(app)})")
        ax.set_title(f"{col}  (Δ={shifts[col]:+.2f}σ)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    ax = axes[len(COMPARE_COLUMNS)]
    bins = np.linspace(0, max(np.percentile(d_train, 99), np.percentile(d_app, 99)), 30)
    ax.hist(d_train, bins=bins, density=True, alpha=.55, color="#3b82f6", label="학습")
    ax.hist(d_app, bins=bins, density=True, alpha=.55, color="#ef4444", label="앱")
    ax.axvline(cutoff, color="#111", linestyle="--", linewidth=1, label="학습 95분위")
    ax.set_title(f"마할라노비스 거리 — 앱의 {inside:.0%}가 안쪽", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    for ax in axes[len(COMPARE_COLUMNS) + 1:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)
    print(f"\n저장: {FIG_PATH}")


if __name__ == "__main__":
    main()
