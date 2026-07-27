"""
Phase 2: YOLOv8 기반 야구공 감지 및 투구 궤적 추적
실행: python src/yolo_detector.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import os
import argparse
from collections import deque

from ultralytics import YOLO


def resolve_video_path(source: str, download_dir: str = None) -> str:
    """
    YouTube URL이면 yt-dlp로 로컬에 H.264 MP4로 다운로드 후 경로 반환.
    로컬 파일이면 그대로 반환.

    pip OpenCV는 FFmpeg 없이 컴파일되어 HTTP 스트림/AV1/VP9를 디코딩 못하므로
    H.264 포맷을 명시적으로 요청한다.
    """
    if not (source.startswith("http://") or source.startswith("https://")):
        return source

    try:
        import yt_dlp
    except ImportError:
        raise ImportError("pip install yt-dlp 먼저 실행해주세요")

    import tempfile
    import glob as _glob
    import re as _re

    out_dir = download_dir or tempfile.gettempdir()

    # 0단계: URL에서 video_id 직접 추출 → 네트워크 요청 없이 캐시 즉시 확인
    _vid_match = _re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", source)
    if _vid_match:
        _quick_id  = _vid_match.group(1)
        _quick_mp4 = os.path.join(out_dir, f"pitchiq_yt_{_quick_id}.mp4")
        if os.path.exists(_quick_mp4) and os.path.getsize(_quick_mp4) > 1_000_000:
            print(f"[YouTube] 캐시 즉시 사용 (네트워크 없음): {_quick_mp4}")
            return _quick_mp4

    # 1단계: 영상 ID와 제목만 추출 (다운로드 X)
    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
        info     = ydl.extract_info(source, download=False)
        video_id = info.get("id", "unknown")
        title    = info.get("title", source)

    # 이미 다운로드된 파일이 있으면 재사용
    cached = os.path.join(out_dir, f"pitchiq_yt_{video_id}.mp4")
    if os.path.exists(cached) and os.path.getsize(cached) > 1_000_000:
        print(f"[YouTube] 캐시 사용: {cached}")
        return cached

    # 2단계: H.264(avc1) MP4로 다운로드
    # YouTube는 AV1/VP9가 기본이지만 pip OpenCV는 H.264만 안정 디코딩
    out_tmpl = os.path.join(out_dir, f"pitchiq_yt_{video_id}.%(ext)s")
    ydl_opts = {
        "format": (
            "bestvideo[vcodec^=avc1][height<=360]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]"
            "/best[ext=mp4][height<=360]"
            "/best[height<=360]"
        ),
        "outtmpl":              out_tmpl,
        "quiet":                False,
        "noplaylist":           True,
        "merge_output_format":  "mp4",
    }
    print(f"[YouTube] 다운로드 중: {title}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])

    # 실제 저장된 파일 탐색 (ext가 mp4이 아닐 수도 있음)
    mp4_path = os.path.join(out_dir, f"pitchiq_yt_{video_id}.mp4")
    if os.path.exists(mp4_path):
        print(f"[YouTube] 다운로드 완료 → {mp4_path}")
        return mp4_path

    matches = _glob.glob(os.path.join(out_dir, f"pitchiq_yt_{video_id}.*"))
    if matches:
        local_path = matches[0]
        print(f"[YouTube] 다운로드 완료 → {local_path}")
        return local_path

    raise FileNotFoundError(f"다운로드 실패 (파일 없음): video_id={video_id}")


# 감지 설정
# COCO 클래스명. 커스텀 모델은 클래스명이 숫자("0","1")일 수 있으므로
# 클래스 필터 없이 전체 허용 (커스텀 모델은 공만 학습됨)
BALL_CLASS_NAMES = None   # None = 클래스 필터 비활성화 (커스텀 모델용)
COCO_BALL_NAMES  = {"sports ball", "baseball"}   # COCO 모델용 fallback
CONF_THRESHOLD   = 0.15   # 방송 화면 소형 야구공 감지용 (기본 0.3은 너무 높음)
TRACK_BUFFER     = 30   # 궤적 표시용 최근 N프레임 보관

# 구속 추정 상수 (60.5ft = 18.44m 투구판-홈플레이트 거리)
MOUND_TO_PLATE_M = 18.44
DEFAULT_FPS      = 30


def load_model(model_path: str = None) -> YOLO:
    """
    YOLOv8 모델 로드
    model_path 없으면 COCO 사전학습 yolov8n 사용
    """
    if model_path and os.path.exists(model_path):
        print(f"[로드] 커스텀 모델: {model_path}")
        return YOLO(model_path)

    print("[로드] yolov8n.pt (COCO 사전학습) — 야구공 = 'sports ball'")
    return YOLO("yolov8n.pt")


def detect_ball_in_frame(model: YOLO, frame: np.ndarray, imgsz: int = 640) -> list[dict]:
    """
    단일 프레임에서 야구공 감지

    Returns:
        [{"bbox": [x1,y1,x2,y2], "conf": float, "cx": int, "cy": int}, ...]
    """
    results = model(frame, verbose=False, imgsz=imgsz)[0]
    detections = []

    for box in results.boxes:
        cls_name = results.names[int(box.cls[0])].lower()
        # COCO 모델일 때만 클래스명으로 필터 (커스텀 모델은 공만 학습됨)
        if cls_name not in COCO_BALL_NAMES and cls_name.isalpha():
            continue
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
        })

    return detections


def estimate_speed(trajectory: list[tuple], fps: float, px_per_meter: float) -> float | None:
    """
    궤적의 첫-마지막 프레임 픽셀 이동거리로 구속 추정 (mph)
    px_per_meter: 화면상 1m가 몇 픽셀인지 (캘리브레이션 필요)
    """
    if len(trajectory) < 2 or px_per_meter <= 0:
        return None

    x0, y0 = trajectory[0]
    x1, y1 = trajectory[-1]
    px_dist = np.hypot(x1 - x0, y1 - y0)
    meters  = px_dist / px_per_meter
    seconds = len(trajectory) / fps
    if seconds == 0:
        return None

    mps = meters / seconds
    mph = mps * 2.237
    return round(mph, 1)


def process_video(
    video_path: str,
    model: YOLO,
    output_path: str = None,
    px_per_meter: float = 0.0,
    show: bool = False,
    max_frames: int = 0,
    frame_skip: int = 3,
    imgsz: int = 320,
) -> tuple[list[dict], float]:
    """
    영상 전체 처리: 적응형 프레임 샘플링 + 궤적 추적

    frame_skip: 기본 N프레임마다 1회 추론 (공 감지 중엔 자동으로 매프레임)
    imgsz:      YOLO 입력 해상도 (320 권장 — 640 대비 ~3배 빠름)

    Returns:
        (pitch_events, fps)
    """
    video_path = resolve_video_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    est_min = total / frame_skip * 0.0077 / 60  # 7.7ms/frame 기준

    print(f"[영상] {width}x{height} @ {fps:.1f}fps  총 {total}프레임")
    print(f"[샘플링] skip={frame_skip}, imgsz={imgsz}  예상 처리시간 ~{est_min:.0f}분")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory_buf: deque = deque(maxlen=TRACK_BUFFER)
    pitch_events: list[dict] = []
    current_pitch: dict | None = None
    frame_idx     = 0
    no_ball_count = 0
    active_frames = 0                   # 공 감지 후 매프레임 추적 카운터
    ACTIVE_WINDOW   = int(fps * 1.5)    # 공 감지 후 1.5초간 매프레임 유지
    NO_BALL_TIMEOUT = int(fps * 0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 적응형 샘플링: 투구 추적 중엔 매프레임, 평상시엔 skip
        if active_frames == 0 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        dets = detect_ball_in_frame(model, frame, imgsz=imgsz)

        if dets:
            best = max(dets, key=lambda d: d["conf"])
            cx, cy = best["cx"], best["cy"]
            trajectory_buf.append((cx, cy))
            no_ball_count = 0
            active_frames = ACTIVE_WINDOW

            if current_pitch is None:
                current_pitch = {"start_frame": frame_idx, "trajectory": []}
            current_pitch["trajectory"].append((cx, cy))

            if writer or show:
                pts = list(trajectory_buf)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = (int(255 * alpha), int(100 * (1 - alpha)), 255)
                    cv2.line(frame, pts[i-1], pts[i], color, 2)
                x1, y1, x2, y2 = best["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            no_ball_count += 1
            if active_frames > 0:
                active_frames -= 1
            if current_pitch and no_ball_count >= NO_BALL_TIMEOUT:
                speed = estimate_speed(current_pitch["trajectory"], fps, px_per_meter)
                pitch_events.append({
                    "start_frame": current_pitch["start_frame"],
                    "end_frame":   frame_idx,
                    "trajectory":  current_pitch["trajectory"],
                    "estimated_speed_mph": speed,
                })
                print(f"  투구 {len(pitch_events):>3}: 프레임 {current_pitch['start_frame']}"
                      f"~{frame_idx}  구속 {speed} mph")
                current_pitch = None

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Baseball Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            print(f"[중단] max_frames={max_frames} 도달")
            break

    if current_pitch:
        speed = estimate_speed(current_pitch["trajectory"], fps, px_per_meter)
        pitch_events.append({
            "start_frame": current_pitch["start_frame"],
            "end_frame":   frame_idx,
            "trajectory":  current_pitch["trajectory"],
            "estimated_speed_mph": speed,
        })

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[완료] 총 {len(pitch_events)}개 투구 감지")
    return pitch_events, fps


def trajectory_to_lstm_features(event: dict) -> dict | None:
    """
    감지된 투구 궤적 → LSTM 입력 피처 추정값 반환
    (구속만 추정 가능; pfx/plate_x/z는 카메라 캘리브레이션 필요)

    Returns:
        {"release_speed": float|None, "note": str}
    """
    speed = event.get("estimated_speed_mph")
    return {
        "release_speed": speed,
        "note": "pfx_x/pfx_z/plate_x/plate_z requires camera calibration",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="야구공 감지 및 투구 궤적 추적")
    parser.add_argument("--video",  required=True, help="입력 영상 경로")
    parser.add_argument("--model",  default=None,  help="커스텀 YOLO 모델 경로 (.pt)")
    parser.add_argument("--output", default=None,  help="결과 영상 저장 경로")
    parser.add_argument("--px-per-meter", type=float, default=0.0,
                        help="구속 추정용: 화면 1m = N픽셀 (0이면 추정 안 함)")
    parser.add_argument("--show", action="store_true", help="실시간 화면 표시")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="처리할 최대 프레임 수 (0=전체)")
    args = parser.parse_args()

    model  = load_model(args.model)
    events = process_video(
        video_path=args.video,
        model=model,
        output_path=args.output,
        px_per_meter=args.px_per_meter,
        show=args.show,
        max_frames=args.max_frames,
    )

    print("\n=== 투구 이벤트 요약 ===")
    for i, e in enumerate(events):
        print(f"  [{i+1}] 프레임 {e['start_frame']}~{e['end_frame']}"
              f"  궤적 {len(e['trajectory'])}포인트"
              f"  구속 {e['estimated_speed_mph']} mph")
