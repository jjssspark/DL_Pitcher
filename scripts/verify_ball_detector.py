"""
학습된 공 감지기를 실제 중계 영상에 적용해 눈으로 확인한다.
실행: venv/bin/python3 scripts/verify_ball_detector.py --weights runs/ball_broadcast/weights/best.pt

TS-014의 재발 방지 장치다. 그때 mAP50 0.852짜리 모델의 실전 감지율이 3%였고,
곡률비·정확도·중요도 어느 지표도 "글러브를 보고 있다"를 말해주지 못했다.
숫자가 아니라 박스를 그린 이미지가 판단 근거다.

통과 기준(사전 선언, 사후 조정 금지):
  - 투구당 감지 프레임 비율 >= 30%
  - 박스 최대변 중앙값 <= 25px  (TS-014의 글러브가 49px, 실제 공은 720p에서 5~15px)
  - 컨택트 시트에서 박스가 공 위에 있는 것이 육안으로 보일 것
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 임의 위치 시크마다 H.264 디코더가 참조 프레임을 잃고 로그를 수백 줄 뱉는다 (cv2 import 전에 설정).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEO = os.path.join(ROOT, "streamlit_app", ".yolo_cache", "pitchiq_hq_yt_gMm3EODDb6w.mp4")
OCR_CACHE = os.path.join(
    ROOT, "output", "pitch_type_cv", "ocr_cache", "pitchiq_hq_yt_gMm3EODDb6w.json"
)
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv", "detector_check")

DETECT_FPS = 30.0        # build_pitch_group_dataset.py와 동일
LOOKBACK_START = 3.0
LOOKBACK_END = 0.3

CROP_HALF = 80           # 감지 지점 주변 160x160을 잘라 2배 확대 -> 320x320 타일
TILES_PER_PITCH = 6

# 궤적 사슬 구성 파라미터
MAX_JUMP_PX = 60.0
MIN_TOTAL_MOVE_PX = 30.0

# --- 통과 기준 ---
# 초판의 "윈도우 프레임 대비 감지율 30%"는 폐기했다. 물리적으로 통과 불가능한
# 기준이었다 — 윈도우는 2.7초(82프레임)인데 공이 실제로 날아가는 시간은 0.45초
# (90mph로 18.44m)뿐이라, 완벽한 감지기도 상한이 약 17%다. 분모를 잘못 잡았다(TS-017).
#
# 대신 궤적 품질을 직접 측정한다: 실제로 이동하는 사슬이 몇 프레임이나 이어지는가.
# 5프레임은 compute_trajectory_features의 최소 요구(3점)에 여유를 둔 값이다.
PASS_CHAIN_FRAMES = 5
PASS_PITCH_RATIO = 0.75      # 8투구 중 6개
PASS_MEDIAN_SIDE_PX = 25.0


def sample_pitches(n: int, seed: int) -> list[tuple[int, float, str]]:
    """OCR 캐시에서 (원본 인덱스, 타임스탬프, 구종명)을 n개 무작위 추출한다."""
    with open(OCR_CACHE, encoding="utf-8") as f:
        cache = json.load(f)

    entries = [
        (i, t, (p.get("pitch_type") or "unknown"))
        for i, (t, p) in enumerate(zip(cache["timestamps"], cache["pitch_data"]))
    ]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(entries), size=min(n, len(entries)), replace=False)
    return [entries[i] for i in sorted(idx)]


def detect_window(model, timestamp_sec: float, imgsz: int) -> tuple[list[dict], int]:
    """
    윈도우 안의 프레임마다 **모든** 감지 후보를 모은다.
    돌려주는 값: (프레임별 감지 목록, 검사한 프레임 수)

    최고 신뢰도 1건만 남기면 안 된다 — 정지 오탐이 실제 공보다 신뢰도가 높은 경우가
    있어(0.71 vs 0.45) 그 프레임에서 공이 통째로 버려진다(TS-017).
    """
    from pitch_type_cv.trajectory_features import frames_in_window
    from yolo_detector import detect_ball_in_frame

    frames = frames_in_window(
        VIDEO, timestamp_sec, LOOKBACK_START, LOOKBACK_END, target_fps=DETECT_FPS
    )
    found = []
    for frame_idx, frame in enumerate(frames):
        detections = detect_ball_in_frame(model, frame, imgsz=imgsz)
        if not detections:
            continue
        found.append({
            "frame_idx": frame_idx,
            "frame": frame,
            "detections": [
                {
                    "bbox": d["bbox"],
                    "conf": d["conf"],
                    "cx": float(d["cx"]),
                    "cy": float(d["cy"]),
                    "max_side": float(max(
                        d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]
                    )),
                }
                for d in detections
            ],
        })
    return found, len(frames)


def chain_for_window(found: list[dict]) -> list[tuple[float, float]]:
    """감지 후보들에서 실제로 이동하는 최장 사슬을 뽑는다."""
    from pitch_type_cv.trajectory_features import longest_moving_chain

    candidates = [
        (f["frame_idx"], [(d["cx"], d["cy"], d["conf"]) for d in f["detections"]])
        for f in found
    ]
    return longest_moving_chain(candidates, MAX_JUMP_PX, MIN_TOTAL_MOVE_PX)


def chain_members(found: list[dict], chain: list[tuple[float, float]]) -> list[dict]:
    """사슬에 속한 감지만 (frame_idx, frame, bbox, conf, max_side) 형태로 되돌린다."""
    wanted = {(round(x), round(y)) for x, y in chain}
    members = []
    for f in found:
        for d in f["detections"]:
            if (round(d["cx"]), round(d["cy"])) in wanted:
                members.append({"frame_idx": f["frame_idx"], "frame": f["frame"], **d})
                break
    return members


def make_tile(detection: dict) -> np.ndarray:
    """감지 지점 주변을 잘라 2배 확대하고 박스를 그린 320x320 타일."""
    frame = detection["frame"]
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = detection["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 화면 가장자리에서도 항상 같은 크기가 나오도록 좌표를 먼저 클램프한다.
    left = max(0, min(cx - CROP_HALF, w - 2 * CROP_HALF))
    top = max(0, min(cy - CROP_HALF, h - 2 * CROP_HALF))
    crop = frame[top:top + 2 * CROP_HALF, left:left + 2 * CROP_HALF].copy()
    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)

    scale = 320 / (2 * CROP_HALF)
    cv2.rectangle(
        crop,
        (int((x1 - left) * scale), int((y1 - top) * scale)),
        (int((x2 - left) * scale), int((y2 - top) * scale)),
        (0, 0, 255), 2,
    )
    label = f"f{detection['frame_idx']} {detection['max_side']:.0f}px c{detection['conf']:.2f}"
    cv2.putText(crop, label, (6, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return crop


def render_pitch(detections: list[dict], out_path: str) -> None:
    """
    윈도우 전체에 고르게 퍼진 감지들의 컨택트 시트를 저장한다.

    신뢰도 상위 6개를 고르면 안 된다 (TS-015). 상위 신뢰도는 시간적으로 몰리는
    경향이 있어 배경이 서로 비슷해지고, 크롭이 감지점 중심이라 움직임까지 사라져
    "정지 물체를 잡고 있다"는 잘못된 인상을 준다. 시간축으로 균등 추출해야
    궤적의 시작·중간·끝을 함께 볼 수 있다.
    """
    idx = np.linspace(0, len(detections) - 1, min(TILES_PER_PITCH, len(detections)))
    picks = [detections[int(round(i))] for i in idx]

    tiles = [make_tile(d) for d in picks]
    while len(tiles) < TILES_PER_PITCH:
        tiles.append(np.zeros((320, 320, 3), dtype=np.uint8))

    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(out_path, sheet)


def format_trajectory(detections: list[dict]) -> list[str]:
    """
    프레임별 원좌표를 표로 만든다.

    크롭 타일은 "박스 안의 것이 공인가"만 답할 수 있고 "움직이는가"는 답할 수 없다
    (TS-015). 궤적 판단은 반드시 이 원좌표로 한다.
    """
    rows = ["  frame  t_rel     cx    cy  size  conf"]
    for d in detections:
        t_rel = -LOOKBACK_START + d["frame_idx"] / DETECT_FPS
        rows.append(
            f"   {d['frame_idx']:4d}  {t_rel:+.2f}s  {d['bbox'][0] + (d['bbox'][2] - d['bbox'][0]) // 2:4d} "
            f"{d['bbox'][1] + (d['bbox'][3] - d['bbox'][1]) // 2:5d}  {d['max_side']:4.0f}  {d['conf']:.2f}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="공 감지기 실전 검증")
    parser.add_argument("--weights", required=True, help="학습된 가중치 경로")
    parser.add_argument("--pitches", type=int, default=8, help="검사할 투구 수")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="", help="출력 하위 폴더 접미사 (조건 비교용)")
    args = parser.parse_args()

    import torch
    from yolo_detector import load_model

    out_dir = os.path.join(OUT_DIR, f"imgsz{args.imgsz}{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.weights)
    if torch.backends.mps.is_available():
        model.to("mps")
        print("MPS 가속 사용")

    all_sides: list[float] = []
    chain_lengths: list[int] = []
    lines = []

    for idx, timestamp, label in sample_pitches(args.pitches, args.seed):
        found, n_frames = detect_window(model, timestamp, args.imgsz)
        chain = chain_for_window(found)
        members = chain_members(found, chain)
        chain_lengths.append(len(chain))

        sides = [d["max_side"] for d in members]
        all_sides.extend(sides)
        median_side = float(np.median(sides)) if sides else float("nan")

        n_raw = sum(len(f["detections"]) for f in found)
        ok = "OK " if len(chain) >= PASS_CHAIN_FRAMES else "부족"
        line = (
            f"투구 #{idx:3d} t={timestamp:8.2f}s {label:12s} | "
            f"원시감지 {n_raw:3d} → 이동사슬 {len(chain):3d}프레임 {ok} | "
            f"최대변 중앙값 {median_side:5.1f}px"
        )
        print(line)
        lines.append(line)

        if members:
            safe_label = "".join(c for c in label if c.isalnum()) or "unknown"
            render_pitch(members, os.path.join(out_dir, f"pitch_{idx:03d}_{safe_label}.jpg"))
            lines.extend(format_trajectory(members))
            lines.append("")

    arr = np.array(all_sides)
    n_ok = sum(1 for n in chain_lengths if n >= PASS_CHAIN_FRAMES)
    pitch_ratio = n_ok / len(chain_lengths) if chain_lengths else 0.0
    median_side = float(np.median(arr)) if len(arr) else float("nan")

    summary = ["", "=== 종합 ==="]
    summary.append(
        f"이동사슬 {PASS_CHAIN_FRAMES}프레임 이상 : {n_ok}/{len(chain_lengths)}투구 "
        f"({pitch_ratio:.0%})  (기준 >= {PASS_PITCH_RATIO:.0%})"
    )
    summary.append(f"사슬 길이 중앙값        : {np.median(chain_lengths):.1f}프레임")
    summary.append(
        f"박스 최대변 중앙값      : {median_side:.1f}px (기준 <= {PASS_MEDIAN_SIDE_PX:.0f}px)"
    )
    if len(arr):
        q = np.percentile(arr, [0, 25, 50, 75, 100])
        summary.append(
            f"최대변 분포             : min {q[0]:.0f} | 25% {q[1]:.0f} | "
            f"50% {q[2]:.0f} | 75% {q[3]:.0f} | max {q[4]:.0f}"
        )

    passed = pitch_ratio >= PASS_PITCH_RATIO and median_side <= PASS_MEDIAN_SIDE_PX
    summary.append("")
    summary.append(
        f"정량 게이트: {'통과' if passed else '실패'} "
        f"— 육안 확인은 {out_dir} 의 이미지로 별도 판단"
    )
    print("\n".join(summary))

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines + summary) + "\n")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
