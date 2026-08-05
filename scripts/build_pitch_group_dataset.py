"""
CV 구종 그룹 분류기 파일럿 — 학습용 데이터셋 생성.
실행: venv/bin/python3 scripts/build_pitch_group_dataset.py

GAME_LIST에 Fox 중계 + YouTube에 영상이 있는 game_pk 5-10개를 직접 채운 뒤 실행한다.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# 임의 위치 시크마다 H.264 디코더가 참조 프레임을 잃고 "mmco: unref short failure"를
# 수백 줄 뱉는다 — 디코딩 결과 자체는 정상이라 로그만 끈다. cv2 import 전에 설정해야 적용된다.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from pitch_type_cv.dataset import GameSpec, build_dataset  # noqa: E402
from pitch_type_cv.trajectory_features import (  # noqa: E402
    extract_trajectory_points,
    longest_smooth_run,
)

# 실제 파일럿 경기 목록 (game_pk, YouTube URL) — 5-10개를 채운 뒤 실행
# 전부 공식 MLB YouTube 채널의 "FULL GAME"(하이라이트 아님) 업로드, 월드시리즈라 FOX 단독 중계 확정.
# game_pk는 pybaseball.statcast()로 실측 대조 완료 (2026-08-05).
GAME_LIST: list[GameSpec] = [
    GameSpec(game_pk=775294, youtube_url="https://youtu.be/gMm3EODDb6w"),  # 2024 WS G1 NYY@LAD
    # GameSpec(game_pk=775296, youtube_url="https://youtu.be/3Oc4S_1np98"),  # 2024 WS G5 LAD@NYY
    # GameSpec(game_pk=813027, youtube_url="https://youtu.be/6BrK9Ep5M6o"),  # 2025 WS G1 TOR@LAD
    # GameSpec(game_pk=813026, youtube_url="https://youtu.be/4qx-dm9VyVU"),  # 2025 WS G2 TOR@LAD
    # GameSpec(game_pk=813032, youtube_url="https://youtu.be/UEv1WTdLpQc"),  # 2025 WS G3 LAD@TOR
    # GameSpec(game_pk=813023, youtube_url="https://youtu.be/l6-5sUsj9J4"),  # 2025 WS G4 LAD@TOR
    # GameSpec(game_pk=813024, youtube_url="https://youtu.be/SU_bMo561b8"),  # 2025 WS G7 TOR@LAD
]

# 720p 다운로드는 59.94fps라 같은 2.7초 윈도우가 163프레임이 된다 (360p는 29.97fps → 81프레임).
# 원 설계가 30fps 기준(yolo_detector.DEFAULT_FPS)이므로 30fps로 샘플링해 되돌린다 — 궤적
# 시간해상도는 설계값 그대로이고 YOLO 호출만 절반이 된다.
DETECT_FPS = 30.0

CACHE_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
OUT_PATH = os.path.join(OUT_DIR, "dataset.csv")
OCR_CACHE_DIR = os.path.join(OUT_DIR, "ocr_cache")
TRAJ_CACHE_DIR = os.path.join(OUT_DIR, "trajectory_cache")

# COCO 'sports ball'은 중계 화면에서 오탐이 잦다. 첫 실행 결과 233개 샘플 중 65%가
# 곡률비 5 초과(실제 투구는 1~1.5)로, 프레임마다 다른 물체를 잡은 랜덤워크였다.
MIN_BALL_CONF = 0.15   # detect_ball_in_frame의 하한과 동일 — 캐시가 있어 사후 조정 가능
MAX_JUMP_PX = 60.0     # 720p·30fps 기준. 실제 공은 프레임 사이를 순간이동하지 않는다


def resolve_video_hq(url: str) -> str:
    """
    OCR용 720p 다운로드. yolo_detector.resolve_video_path는 기존 모션 감지
    데모용으로 360p를 하드코딩하고 있어 오버레이 텍스트(12x80px 수준)가
    OCR로 안정적으로 읽히지 않는다 — 캐시 파일명을 달리해 별도 보관한다.
    """
    import re as _re
    import yt_dlp

    os.makedirs(CACHE_DIR, exist_ok=True)
    match = _re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    video_id = match.group(1) if match else _re.sub(r"\W+", "_", url)
    out_path = os.path.join(CACHE_DIR, f"pitchiq_hq_yt_{video_id}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"[YouTube-HQ] 캐시 사용: {out_path}")
        return out_path

    ydl_opts = {
        "format": (
            "bestvideo[vcodec^=avc1][height<=720]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]"
            "/best[ext=mp4][height<=720]"
            "/best[height<=720]"
        ),
        "outtmpl": os.path.join(CACHE_DIR, f"pitchiq_hq_yt_{video_id}.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }
    print(f"[YouTube-HQ] 다운로드 중: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_path):
        raise RuntimeError(f"720p 다운로드 결과를 찾을 수 없습니다: {out_path}")
    print(f"[YouTube-HQ] 다운로드 완료 → {out_path}")
    return out_path


def scan_overlays_cached(video_path: str) -> tuple[list[float], list[dict]]:
    """
    OCR 스캔 결과를 캐시해 재실행 시 건너뛴다. 전체 영상 스캔이 궤적 추출보다 오래
    걸리는데 매 실행마다 처음부터 다시 돌기 때문이다.

    한계: 완료된 스캔만 저장한다. 스캔 도중 중단되면 그 경기 진행분은 남지 않는다.
    scan_pitch_overlays가 호출마다 해시 상태를 초기화해 첫 샘플 프레임에서 무조건
    OCR을 트리거하므로, skip_start_sec으로 구간을 쪼개 이어붙이면 경계마다 가짜 검출이
    생겨 Statcast 순서 페어링이 밀린다 — 부분 재개를 일부러 지원하지 않는 이유다.
    """
    from pose_detector import scan_pitch_overlays

    key = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(OCR_CACHE_DIR, f"{key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[OCR] 캐시 사용: {cache_path} ({len(cached['timestamps'])}개)")
        return cached["timestamps"], cached["pitch_data"]

    timestamps, pitch_data = scan_pitch_overlays(video_path)

    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"timestamps": timestamps, "pitch_data": pitch_data}, f, ensure_ascii=False)
    os.replace(tmp_path, cache_path)  # 중단돼도 반쪽 캐시가 남지 않도록 원자적 교체
    print(f"[OCR] 캐시 저장: {cache_path} ({len(timestamps)}개)")
    return timestamps, pitch_data


def _load_trajectory_cache(cache_path: str) -> dict[str, list]:
    """JSONL 캐시를 읽어 {타임스탬프 문자열: [[x, y, conf], ...]}로 돌려준다."""
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
                continue  # 중단으로 잘린 마지막 줄 — 해당 투구만 다시 뽑으면 된다
            cached[f"{entry['t']:.4f}"] = entry["points"]
    return cached


def make_trajectory_extractor(yolo_model):
    """
    원시 감지 결과를 JSONL로 증분 저장하면서 궤적을 뽑는 콜러블을 만든다.

    투구 한 건마다 즉시 append하므로 중간에 중단돼도 진행분이 남는다 (OCR 캐시는
    완주해야만 저장돼 2시간 진행분을 통째로 날린 적이 있다). 신뢰도·연속성 필터는
    캐시 위에서 적용하므로, 임계값을 바꿔도 영상 재처리 없이 다시 계산할 수 있다.
    """
    os.makedirs(TRAJ_CACHE_DIR, exist_ok=True)
    state: dict = {"path": None, "cached": {}}

    def extract(video_path: str, timestamp_sec: float) -> list[tuple[float, float]]:
        key = os.path.splitext(os.path.basename(video_path))[0]
        cache_path = os.path.join(TRAJ_CACHE_DIR, f"{key}.jsonl")
        if state["path"] != cache_path:
            state["path"] = cache_path
            state["cached"] = _load_trajectory_cache(cache_path)
            if state["cached"]:
                print(f"[궤적] 캐시 사용: {cache_path} ({len(state['cached'])}개)")

        ts_key = f"{timestamp_sec:.4f}"
        points = state["cached"].get(ts_key)
        if points is None:
            points = extract_trajectory_points(
                video_path, timestamp_sec, yolo_model, target_fps=DETECT_FPS
            )
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": timestamp_sec, "points": points}) + "\n")
            state["cached"][ts_key] = points

        confident = [(x, y) for x, y, conf in points if conf >= MIN_BALL_CONF]
        return longest_smooth_run(confident, MAX_JUMP_PX)

    return extract


def main() -> None:
    if not GAME_LIST:
        print(
            "[중단] GAME_LIST가 비어 있습니다. "
            "5-10개의 (game_pk, YouTube URL)을 직접 채운 뒤 다시 실행하세요."
        )
        return

    from pybaseball import statcast_single_game
    from yolo_detector import load_model

    print("[1/3] YOLO 모델 로드...")
    yolo_model = load_model()

    # Ultralytics는 macOS에서 MPS를 자동 선택하지 않아 CPU로 돈다 (125ms/frame → 48ms/frame)
    import torch
    if torch.backends.mps.is_available():
        yolo_model.to("mps")
        print("       MPS 가속 사용")

    def fetch_statcast(game_pk: int):
        df = statcast_single_game(game_pk)
        return df.sort_values(
            ["game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

    def resolve_video(url: str) -> str:
        return resolve_video_hq(url)

    extract_trajectory = make_trajectory_extractor(yolo_model)

    print(f"[2/3] {len(GAME_LIST)}개 경기 데이터셋 조립 중...")
    dataset_df = build_dataset(
        games=GAME_LIST,
        fetch_statcast=fetch_statcast,
        resolve_video=resolve_video,
        scan_overlays=scan_overlays_cached,
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
