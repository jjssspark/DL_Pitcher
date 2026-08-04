"""
고정 데모 영상(https://youtu.be/gMm3EODDb6w, game_pk=775300)의 방송 오버레이 OCR
스캔 결과를 streamlit_app/fixed_demo_scan.json에 정적으로 저장한다.
배포 환경은 이 파일만 읽고, 실시간 다운로드/OCR을 하지 않는다. 로컬에서 1회만 실행.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "streamlit_app"))

GAME_PK   = 775300
VIDEO_URL = "https://youtu.be/gMm3EODDb6w"
SCAN_VER  = "v5-ocr-overlay"  # app.py의 _SCAN_VER와 동일하게 유지
OUT_PATH  = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")


def main() -> None:
    from pybaseball import statcast_single_game
    from yolo_detector import resolve_video_path
    from pose_detector import scan_pitch_overlays

    print(f"[1/3] Statcast 투구 수 조회: game_pk={GAME_PK}")
    df = statcast_single_game(GAME_PK)
    n_pitches = len(df)
    print(f"  총 {n_pitches}구 (전체 경기 스캔 — 이닝 제한 없음)")

    print(f"[2/3] 영상 다운로드: {VIDEO_URL}")
    cache_dir  = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
    os.makedirs(cache_dir, exist_ok=True)
    video_path = resolve_video_path(VIDEO_URL, download_dir=cache_dir)
    print(f"  다운로드 완료: {video_path}")

    print("[3/3] 방송 오버레이 OCR 스캔 (전체 경기)...")
    # max_pitches=0 → 조기 종료 없이 영상 끝까지 스캔 (오프라인 1회 생성이라 시간 제약 없음)
    pitch_times, pitch_data = scan_pitch_overlays(
        video_path, expected_count=n_pitches,
        max_pitches=0, skip_start_sec=0.0,
    )
    print(f"  {len(pitch_times)}개 투구 타임스탬프 감지")

    with open(OUT_PATH, "w") as f:
        json.dump({"version": SCAN_VER, "pitch_times": pitch_times, "pitch_data": pitch_data}, f)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
