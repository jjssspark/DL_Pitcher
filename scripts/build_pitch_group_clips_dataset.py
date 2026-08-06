"""
투구별 Savant 클립으로 구종 그룹 데이터셋을 만든다 (경로 B).

OCR 경로(build_pitch_group_dataset.py)를 대체하는 수집 경로다. 차이:
  - 라벨을 StatsAPI에서 직접 받는다 — 판독률 100%, 순서 밀림 없음
  - 투구 1개 = 클립 1개라 릴리스가 클립 내 거의 고정 위치에 온다 → 고정 윈도우로 자른다
  - 경기가 YouTube 풀경기 업로드에 묶이지 않아 임의 game_pk를 쓸 수 있다

실행:
  venv/bin/python3 scripts/build_pitch_group_clips_dataset.py
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# 임의 위치 시크마다 H.264 디코더가 "mmco: unref short failure"를 수백 줄 뱉는다 —
# 디코딩 결과는 정상이라 로그만 끈다. cv2 import 전에 설정해야 적용된다.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import logging  # noqa: E402

import requests  # noqa: E402

from pitch_type_cv.clip_dataset import build_clip_dataset  # noqa: E402
from pitch_type_cv.savant_clips import (  # noqa: E402
    PitchClip,
    extract_clip_url,
    parse_pitches,
    savant_clip_page_url,
    statsapi_feed_url,
)
from pitch_type_cv.trajectory_features import (  # noqa: E402
    extract_trajectory_candidates,
    longest_moving_chain,
)

# 4경기. OFFSPEED 비율로 골랐다 — 기존 홀드아웃에서 OFFSPEED가 4개뿐이라 f1이 0.00이었다.
# 괄호 안은 StatsAPI 실측 (투구수, FAST/BREAK/OFF 비율, 2026-08-06 조사).
GAME_LIST: list[int] = [
    813026,  # 2025 WS G2 TOR@LAD (222구, 45.5/22.1/29.3%) — OFFSPEED 최다
    813024,  # 2025 WS G7 TOR@LAD (364구, 40.4/39.8/17.3%)
    813027,  # 2025 WS G1 TOR@LAD (327구, 48.0/39.8/12.2%)
    775294,  # 2024 WS G1 NYY@LAD (280구, 57.5/34.6/ 7.9%) — 기존 OCR 경로 기준선 경기
]

CLIP_DIR = os.path.join(ROOT, "data", "raw", "savant_clips")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
OUT_PATH = os.path.join(OUT_DIR, "dataset_clips.csv")
TRAJ_CACHE_DIR = os.path.join(OUT_DIR, "clip_trajectory_cache")

BALL_MODEL_PATH = os.path.join(ROOT, "models", "ball_broadcast_v1.pt")

# 감지 설정은 OCR 경로와 동일하다. imgsz 960 / conf 0.05는 둘 다 필수 (TS-018).
DETECT_FPS = 30.0
DETECT_IMGSZ = 960
DETECT_CONF = 0.05
MAX_JUMP_PX = 60.0
MIN_TOTAL_MOVE_PX = 30.0

# 클립 시작 기준 스캔 윈도우. scripts/calibrate_clip_window.py 실측(775294, 60투구)으로 정했다.
#
# 사슬 시작 시각의 5퍼센타일이 전 그룹 2.92초로 하한이 날카롭다 — 클립 프리롤이 고정이라는
# 뜻이다. 그룹별 중앙값도 3.03/3.22/3.24초로 0.21초 안에 모여, 구종별 위상 차이는 없다.
#
# 12초 전체 스캔의 사슬 확보율 93.3%는 타구 비행을 사슬로 잡은 건까지 센 값이라 부풀려져
# 있다. 이 윈도우로 좁히면 90.0%인데 이쪽이 실제 투구 확보율이다 (OCR 경로는 68%였다).
# 2.5~4.5초와 확보율이 54/60으로 같고 사슬 점수는 더 나으면서 감지 비용이 30% 적다.
DEFAULT_WINDOW_START = 2.8
DEFAULT_WINDOW_END = 4.2

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


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


def load_candidate_cache(cache_path: str) -> dict[str, list]:
    """JSONL 캐시를 {play_id: candidates}로 읽는다."""
    if not os.path.exists(cache_path):
        return {}
    cached = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # 중단으로 잘린 마지막 줄 — 그 투구만 다시 뽑으면 된다
            cached[entry["play_id"]] = entry["candidates"]
    return cached


def make_trajectory_extractor(model, window_start: float, window_end: float, keep_clips: bool):
    """
    클립을 받아 궤적을 뽑는 콜러블을 만든다.

    감지 후보 전체를 투구마다 즉시 JSONL에 append한다 — 중단돼도 진행분이 남고,
    사슬 임계값을 바꿔도 영상 재처리 없이 다시 계산할 수 있다. 프레임당 최고 신뢰도
    1건만 남기면 정지 오탐이 공보다 신뢰도가 높은 프레임에서 공을 잃는다 (TS-017).

    처리가 끝난 mp4는 기본적으로 지운다. 4경기 × 280투구 × 4.5MB = 5GB라
    캐시로 남길 이유가 없다 — 재취득은 play_id만 있으면 된다.
    """
    os.makedirs(TRAJ_CACHE_DIR, exist_ok=True)
    state: dict = {"game_pk": None, "cached": {}}

    def extract(clip: PitchClip) -> list[tuple[float, float]]:
        cache_path = os.path.join(TRAJ_CACHE_DIR, f"{clip.game_pk}.jsonl")
        if state["game_pk"] != clip.game_pk:
            state["game_pk"] = clip.game_pk
            state["cached"] = load_candidate_cache(cache_path)
            if state["cached"]:
                print(f"  [궤적] 캐시 사용: {len(state['cached'])}개")

        candidates = state["cached"].get(clip.play_id)
        if candidates is None:
            path = fetch_clip(clip.play_id)
            if path is None:
                return []
            # timestamp=window_end, lookback_start=(end-start) -> [start, end] 구간
            candidates = extract_trajectory_candidates(
                path, window_end, model,
                lookback_start_sec=window_end - window_start, lookback_end_sec=0.0,
                target_fps=DETECT_FPS, imgsz=DETECT_IMGSZ, conf=DETECT_CONF,
            )
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"play_id": clip.play_id, "candidates": candidates}) + "\n")
            state["cached"][clip.play_id] = candidates
            if not keep_clips:
                os.remove(path)

        # JSON 왕복 후에는 튜플이 리스트가 되므로 형태를 맞춰 넘긴다.
        normalized = [
            (frame_idx, [(float(x), float(y), float(c)) for x, y, c in detections])
            for frame_idx, detections in candidates
        ]
        return longest_moving_chain(normalized, MAX_JUMP_PX, MIN_TOTAL_MOVE_PX)

    return extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-start", type=float, default=DEFAULT_WINDOW_START)
    parser.add_argument("--window-end", type=float, default=DEFAULT_WINDOW_END)
    parser.add_argument("--keep-clips", action="store_true",
                        help="처리 후 mp4를 지우지 않는다 (4경기 기준 약 5GB)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.path.exists(BALL_MODEL_PATH):
        raise SystemExit(f"공 감지 모델이 없습니다: {BALL_MODEL_PATH}")

    from yolo_detector import load_model
    print("[1/3] YOLO 모델 로드...")
    model = load_model(BALL_MODEL_PATH)
    import torch
    if torch.backends.mps.is_available():
        model.to("mps")
        print("       MPS 가속 사용")

    extract_trajectory = make_trajectory_extractor(
        model, args.window_start, args.window_end, args.keep_clips
    )

    print(f"[2/3] {len(GAME_LIST)}개 경기 처리 "
          f"(윈도우 {args.window_start}~{args.window_end}초)...")
    headers = {"User-Agent": USER_AGENT}
    frames = []
    for game_pk in GAME_LIST:
        try:
            feed = requests.get(statsapi_feed_url(game_pk), headers=headers, timeout=60).json()
            clips = parse_pitches(feed, game_pk)
            print(f"  game_pk={game_pk}: {len(clips)}투구")
            frames.append(build_clip_dataset(clips, extract_trajectory))
        except Exception as exc:
            print(f"  game_pk={game_pk} 처리 실패, 건너뜁니다: {exc}")

    import pandas as pd
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        raise SystemExit("[중단] 생성된 데이터셋이 비어 있습니다.")
    dataset_df = pd.concat(non_empty, ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_df.to_csv(OUT_PATH, index=False)

    trained = dataset_df[dataset_df["has_trajectory"]]
    print(f"[3/3] 저장 완료: {OUT_PATH}")
    print(f"      전체 투구 {len(dataset_df)}개, 궤적 확보 {len(trained)}개 "
          f"({len(trained) / len(dataset_df):.1%})")
    print("\n그룹별 궤적 확보율 (선택 편향 확인용):")
    for group, sub in dataset_df.groupby("group"):
        kept = int(sub["has_trajectory"].sum())
        print(f"  {group:9s} {len(sub):4d} -> {kept:4d}  ({kept / len(sub):.1%})")
    print("\n경기별 확보 샘플:")
    print(trained.groupby(["game_pk", "group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
