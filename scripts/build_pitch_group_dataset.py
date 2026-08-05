"""
CV 구종 그룹 분류기 파일럿 — 학습용 데이터셋 생성.
실행: venv/bin/python3 scripts/build_pitch_group_dataset.py

GAME_LIST에 Fox 중계 + YouTube에 영상이 있는 game_pk 5-10개를 직접 채운 뒤 실행한다.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pitch_type_cv.dataset import GameSpec, build_dataset  # noqa: E402
from pitch_type_cv.trajectory_features import extract_trajectory_window  # noqa: E402

# 실제 파일럿 경기 목록 (game_pk, YouTube URL) — 5-10개를 채운 뒤 실행
GAME_LIST: list[GameSpec] = [
    # GameSpec(game_pk=775300, youtube_url="https://youtu.be/gMm3EODDb6w"),
]

CACHE_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
OUT_PATH = os.path.join(OUT_DIR, "dataset.csv")


def main() -> None:
    if not GAME_LIST:
        print(
            "[중단] GAME_LIST가 비어 있습니다. "
            "5-10개의 (game_pk, YouTube URL)을 직접 채운 뒤 다시 실행하세요."
        )
        return

    from pybaseball import statcast_single_game
    from pose_detector import scan_pitch_overlays
    from yolo_detector import load_model, resolve_video_path

    print("[1/3] YOLO 모델 로드...")
    yolo_model = load_model()

    def fetch_statcast(game_pk: int):
        df = statcast_single_game(game_pk)
        return df.sort_values(
            ["game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

    def resolve_video(url: str) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return resolve_video_path(url, download_dir=CACHE_DIR)

    def extract_trajectory(video_path: str, timestamp_sec: float):
        return extract_trajectory_window(video_path, timestamp_sec, yolo_model)

    print(f"[2/3] {len(GAME_LIST)}개 경기 데이터셋 조립 중...")
    dataset_df = build_dataset(
        games=GAME_LIST,
        fetch_statcast=fetch_statcast,
        resolve_video=resolve_video,
        scan_overlays=scan_pitch_overlays,
        extract_trajectory=extract_trajectory,
    )

    if dataset_df.empty:
        print("[중단] 생성된 데이터셋이 비어 있습니다. 위 경고 로그를 확인하세요.")
        return

    print(f"[3/3] 저장 중... ({len(dataset_df)}개 샘플)")
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_df.to_csv(OUT_PATH, index=False)
    print(f"저장 완료: {OUT_PATH}")
    print(dataset_df["group"].value_counts())


if __name__ == "__main__":
    main()
