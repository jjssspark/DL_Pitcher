"""
Savant 투구 클립에서 공 사슬이 **클립 시작 기준 몇 초**에 나타나는지 측정한다.

경로 (B)의 전제는 "릴리스가 클립 내 고정 오프셋"이다. 이게 성립해야 OCR 타임스탬프
없이 고정 윈도우로 자를 수 있고, 위상 정렬 문제(투구마다 비행의 다른 구간이 잡히는 현상)가
사라진다. 표본 9개로는 단정할 수 없어 그룹별로 충분히 뽑아 분포를 본다.

그룹별로 나눠 보는 이유: 오프셋이 구종마다 다르면 단일 윈도우가 특정 구종을 체계적으로
잘라내고, 그게 곧 선택 편향이 된다.

실행:
  venv/bin/python3 scripts/calibrate_clip_window.py --game-pk 775294 --per-group 20
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# 임의 위치 시크마다 H.264 디코더가 참조 프레임을 잃고 "mmco: unref short failure"를
# 수백 줄 뱉는다 — 디코딩 결과 자체는 정상이라 로그만 끈다. cv2 import 전에 설정해야 한다.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402

from pitch_type_cv.pitch_group_map import pitch_type_to_group  # noqa: E402
from pitch_type_cv.savant_clips import (  # noqa: E402
    extract_clip_url,
    parse_pitches,
    savant_clip_page_url,
    statsapi_feed_url,
)
from pitch_type_cv.trajectory_features import (  # noqa: E402
    extract_trajectory_candidates,
    longest_moving_chain,
)

CLIP_DIR = os.path.join(ROOT, "data", "raw", "savant_clips")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")

# 감지 설정은 OCR 경로(scripts/build_pitch_group_dataset.py)와 의도적으로 동일하게 맞춘다.
# 여기서 달라지면 캘리브레이션 결과를 본 파이프라인에 그대로 옮길 수 없다.
BALL_MODEL_PATH = os.path.join(ROOT, "models", "ball_broadcast_v1.pt")
DETECT_FPS = 30.0
DETECT_IMGSZ = 960
DETECT_CONF = 0.05
MAX_JUMP_PX = 60.0
MIN_TOTAL_MOVE_PX = 30.0

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
GROUPS = ("FASTBALL", "BREAKING", "OFFSPEED")


def fetch_clip(play_id: str) -> str | None:
    """클립을 받아 로컬 경로를 돌려준다. 페이지에 mp4가 없으면 None."""
    os.makedirs(CLIP_DIR, exist_ok=True)
    path = os.path.join(CLIP_DIR, f"{play_id}.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path

    headers = {"User-Agent": USER_AGENT}
    page = requests.get(savant_clip_page_url(play_id), headers=headers, timeout=60).text
    url = extract_clip_url(page)
    if url is None:
        return None

    response = requests.get(url, headers=headers, timeout=180)
    response.raise_for_status()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(response.content)
    os.replace(tmp_path, path)  # 중단돼도 반쪽 파일이 남지 않도록 원자적 교체
    return path


def chain_span_sec(path: str, model, max_scan_sec: float) -> tuple[float, float, float, int]:
    """
    클립 전체를 훑어 (클립길이, 사슬시작초, 사슬끝초, 사슬점수)를 돌려준다.
    사슬이 없으면 시작·끝은 -1.0.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    scan_sec = min(duration, max_scan_sec)
    # extract_trajectory_candidates는 (timestamp - lookback_start) ~ (timestamp - lookback_end)
    # 구간을 본다. 클립 앞에서부터 scan_sec까지를 보려면 timestamp를 scan_sec에 둔다.
    candidates = extract_trajectory_candidates(
        path, scan_sec, model,
        lookback_start_sec=scan_sec, lookback_end_sec=0.0,
        target_fps=DETECT_FPS, imgsz=DETECT_IMGSZ, conf=DETECT_CONF,
    )
    chain = longest_moving_chain(candidates, MAX_JUMP_PX, MIN_TOTAL_MOVE_PX)
    if len(chain) < 3:
        return duration, -1.0, -1.0, len(chain)

    # frames_in_window가 실제로 쓰는 샘플 간격. 59.94fps 영상은 step=2라 유효 fps가
    # 정확히 30이 아니다 — 3초 구간에서 20ms 어긋나므로 fps/step으로 되돌린다.
    step = max(1, round(fps / DETECT_FPS))
    effective_fps = fps / step

    # longest_moving_chain은 좌표만 돌려주므로 좌표 -> frame_idx 역인덱스를 만든다.
    # 같은 좌표가 여러 프레임에 나오면 가장 이른 프레임을 쓴다 (분포 관찰 목적이라 충분).
    frame_of: dict[tuple[float, float], int] = {}
    for frame_idx, detections in candidates:
        for x, y, _conf in detections:
            frame_of.setdefault((round(x, 3), round(y, 3)), frame_idx)

    indices = [
        frame_of[(round(x, 3), round(y, 3))]
        for x, y in chain
        if (round(x, 3), round(y, 3)) in frame_of
    ]
    if not indices:
        return duration, -1.0, -1.0, len(chain)
    return duration, min(indices) / effective_fps, max(indices) / effective_fps, len(chain)


def plot_distribution(rows: list[dict], out_path: str, game_pk: int) -> None:
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 5))

    found = [r for r in rows if r["chain_start_sec"] >= 0]
    for group in GROUPS:
        starts = [r["chain_start_sec"] for r in found if r["group"] == group]
        if starts:
            ax_hist.hist(starts, bins=30, alpha=0.6, label=f"{group} (n={len(starts)})")
    ax_hist.set_xlabel("chain start (sec from clip start)")
    ax_hist.set_ylabel("pitches")
    ax_hist.set_title(f"game_pk={game_pk} chain start distribution")
    ax_hist.legend()

    data = [[r["chain_start_sec"] for r in found if r["group"] == g] for g in GROUPS]
    labels = [f"{g}\n(n={len(d)})" for g, d in zip(GROUPS, data)]
    kept = [(lab, d) for lab, d in zip(labels, data) if d]
    if kept:
        ax_box.boxplot([d for _, d in kept], tick_labels=[lab for lab, _ in kept])
    ax_box.set_ylabel("chain start (sec)")
    ax_box.set_title("by group - no overlap means a single window biases")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-pk", type=int, required=True)
    parser.add_argument("--per-group", type=int, default=20,
                        help="그룹당 표본 수. 그룹 간 오프셋 차이를 보려면 균등하게 뽑는다")
    parser.add_argument("--max-scan-sec", type=float, default=12.0,
                        help="클립 앞에서부터 훑을 최대 길이. 릴리스 위치를 가정하지 않으려면 넉넉히 둔다")
    args = parser.parse_args()

    if not os.path.exists(BALL_MODEL_PATH):
        raise SystemExit(f"공 감지 모델이 없습니다: {BALL_MODEL_PATH}")

    headers = {"User-Agent": USER_AGENT}
    feed = requests.get(statsapi_feed_url(args.game_pk), headers=headers, timeout=60).json()
    clips = parse_pitches(feed, args.game_pk)

    picked: dict[str, list] = {g: [] for g in GROUPS}
    for clip in clips:
        group = pitch_type_to_group(clip.pitch_type)
        if group and len(picked[group]) < args.per_group:
            picked[group].append(clip)
    targets = [(g, c) for g in GROUPS for c in picked[g]]
    print(f"[1/3] 대상 {len(targets)}투구 "
          f"({', '.join(f'{g} {len(picked[g])}' for g in GROUPS)})")

    from yolo_detector import load_model
    model = load_model(BALL_MODEL_PATH)
    import torch
    if torch.backends.mps.is_available():
        model.to("mps")
        print("       MPS 가속 사용")

    print(f"[2/3] 클립 다운로드 + 전체 스캔 (최대 {args.max_scan_sec}초)...")
    rows = []
    for i, (group, clip) in enumerate(targets, 1):
        try:
            path = fetch_clip(clip.play_id)
        except Exception as exc:
            print(f"  [{i}/{len(targets)}] {clip.play_id[:8]} 다운로드 실패: {exc}")
            continue
        if path is None:
            print(f"  [{i}/{len(targets)}] {clip.play_id[:8]} mp4 없음")
            continue

        duration, start, end, points = chain_span_sec(path, model, args.max_scan_sec)
        rows.append({
            "play_id": clip.play_id,
            "pitch_type": clip.pitch_type,
            "group": group,
            "clip_duration_sec": round(duration, 3),
            "chain_start_sec": round(start, 3),
            "chain_end_sec": round(end, 3),
            "chain_points": points,
        })
        mark = f"{start:5.2f}~{end:5.2f}s" if start >= 0 else "사슬없음    "
        print(f"  [{i}/{len(targets)}] {group:9s} {clip.pitch_type:3s} "
              f"길이 {duration:5.2f}s  {mark}  {points}점")

    if not rows:
        raise SystemExit("수집된 클립이 없습니다.")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "clip_window_calibration.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = os.path.join(OUT_DIR, "clip_window_calibration.png")
    plot_distribution(rows, png_path, args.game_pk)

    found = [r for r in rows if r["chain_start_sec"] >= 0]
    print(f"[3/3] 사슬 확보 {len(found)}/{len(rows)} ({len(found)/len(rows):.1%})")
    for group in GROUPS:
        starts = sorted(r["chain_start_sec"] for r in found if r["group"] == group)
        if not starts:
            continue
        print(f"  {group:9s} n={len(starts):3d}  중앙값 {starts[len(starts) // 2]:5.2f}s  "
              f"범위 {starts[0]:5.2f}~{starts[-1]:5.2f}s")
    print(f"저장: {csv_path}\n      {png_path}")


if __name__ == "__main__":
    main()
