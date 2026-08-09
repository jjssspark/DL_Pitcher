"""
앱 경로의 판정률이 왜 실험실의 3분의 1인지 진단한다.

실측: Savant 클립에서 궤적 확보율 86.3%인데, 전체 경기 영상 + OCR 오버레이 시각에서는
26~36%다. 창 위치를 1.4초에서 3.0초까지 옮겨도 평평하다 — 창을 잘못 잡은 것이라면
어딘가에서 봉우리가 생겨야 한다.

해상도·fps는 학습 클립과 동일하다(1280x720, 59.94fps). 그래서 남은 가설은
"오버레이 시각과 실제 투구 시각의 간격이 일정하지 않다"다.

넓은 창(t-8 ~ t+2초)을 통째로 훑어 공이 실제로 어느 오프셋에 있는지 찍는다.
간격이 일정하면 분포가 한 점에 모이고, 일정하지 않으면 흩어진다.

실행:
  venv/bin/python3 scripts/diagnose_app_window.py --pitches 8
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pitch_type_cv.live_classifier import (  # noqa: E402
    CHAIN_MAX_JUMP_PX,
    CHAIN_MIN_TOTAL_MOVE_PX,
)
from pitch_type_cv.trajectory_features import (  # noqa: E402
    extract_trajectory_candidates,
    longest_moving_chain_frames,
)

SCAN_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
VIDEO_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
OUT_PATH = os.path.join(ROOT, "output", "pitch_type_cv", "app_window_diagnosis.json")

WIDE_START_SEC = 8.0   # t-8초부터
WIDE_END_SEC = -2.0    # t+2초까지 (음수 = 기준 시각 이후)
FPS = 59.94


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitches", type=int, default=8)
    args = parser.parse_args()

    from ultralytics import YOLO

    video = max(
        (os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
         if f.endswith((".mp4", ".mkv", ".webm"))),
        key=os.path.getsize,
    )
    times = json.load(open(SCAN_PATH))["pitch_times"][:args.pitches]
    det = YOLO(os.path.join(ROOT, "models", "ball_broadcast_v1.pt"))

    print(f"영상: {os.path.basename(video)}")
    print(f"넓은 창 t-{WIDE_START_SEC:.0f}s ~ t+{-WIDE_END_SEC:.0f}s, 투구 {len(times)}개\n")

    rows = []
    for t in times:
        candidates = extract_trajectory_candidates(
            video, t, det,
            lookback_start_sec=WIDE_START_SEC,
            lookback_end_sec=WIDE_END_SEC,
            imgsz=960, conf=0.05,
        )
        chain = longest_moving_chain_frames(
            candidates, CHAIN_MAX_JUMP_PX, CHAIN_MIN_TOTAL_MOVE_PX
        )
        if not chain:
            print(f"t={t:8.1f}s  넓은 창에서도 사슬 없음 (감지 프레임 {len(candidates)}개)")
            rows.append({"t": t, "found": False, "n_candidate_frames": len(candidates)})
            continue

        # 사슬의 프레임 인덱스는 창 시작 기준이다. 기준 시각 t로부터의 오프셋으로 되돌린다.
        first_off = WIDE_START_SEC - chain[0][0] / FPS
        last_off = WIDE_START_SEC - chain[-1][0] / FPS
        print(f"t={t:8.1f}s  공이 t-{first_off:.2f}s ~ t-{last_off:.2f}s 구간에 있음 "
              f"({len(chain)}점, 감지 프레임 {len(candidates)}개)")
        rows.append({
            "t": t, "found": True, "n_points": len(chain),
            "first_offset_sec": first_off, "last_offset_sec": last_off,
            "n_candidate_frames": len(candidates),
        })

    found = [r for r in rows if r["found"]]
    print(f"\n넓은 창 확보율 {len(found)}/{len(rows)}")
    if found:
        offs = sorted(r["first_offset_sec"] for r in found)
        print(f"공 시작 오프셋 분포: 최소 t-{max(offs):.2f}s / "
              f"중앙 t-{offs[len(offs) // 2]:.2f}s / 최대 t-{min(offs):.2f}s")
        print("→ 분포가 좁으면 창을 그 위치로 옮기면 되고, 넓으면 고정 창으로는 안 된다")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
