"""
실시간 투구 감지 — OpenCV 프레임 차이 기반 모션 감지
YOLO 없이 순수 OpenCV로 투구 타이밍을 감지한다.
야구 중계에서 투수가 공을 던지면 마운드 영역에 큰 모션이 발생하는 원리.
"""
import cv2
import numpy as np


def load_pose_model(model_name: str = "motion") -> str:
    """인터페이스 호환용 — 모션 감지는 모델 불필요"""
    return "motion"


def extract_frames(
    video_path: str,
    center_sec: float,
    duration: float = 2.0,
    step: int = 3,
    max_frames: int = 15,
) -> list[np.ndarray]:
    """
    로컬 MP4에서 center_sec 기준 ±duration/2초 프레임 추출.
    max_frames 제한으로 대용량 파일에서 무한 대기 방지.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int((center_sec - duration / 2) * fps))
    end_frame   = int((center_sec + duration / 2) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames, frame_count = [], 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if cur_frame > end_frame:
            break
        if frame_count % step == 0:
            # 처리 속도용 리사이즈
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def detect_pitch_motion(
    model,          # 사용 안 함 (인터페이스 호환용)
    frames: list[np.ndarray],
    imgsz: int = 320,
) -> tuple[bool, float, float]:
    """
    프레임 차이 기반 투구 감지.

    Returns:
        (is_pitch, max_score, avg_score)
    """
    if len(frames) < 4:
        return False, 0.0, 0.0

    scores = []
    for i in range(1, len(frames)):
        h, w = frames[i].shape[:2]

        # 마운드 영역: 화면 중앙 (투수가 있는 곳)
        y1, y2 = h * 3 // 8, h * 7 // 8
        x1, x2 = w * 2 // 8, w * 6 // 8

        prev_crop = cv2.cvtColor(frames[i - 1][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        curr_crop = cv2.cvtColor(frames[i][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        diff  = cv2.absdiff(prev_crop, curr_crop).astype(np.float32)
        score = float(diff.mean()) / 255.0
        scores.append(score)

    if not scores:
        return False, 0.0, 0.0

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    THRESHOLD = 0.065       # 360p 압축 방송 기준 (원본 0.10은 너무 높음)
    SUB_THRESHOLD = 0.035   # 와인드업 구간 낮은 모션 기준

    # 조건 1: max가 임계값 초과
    # 조건 2: 카메라 전환(단발 고모션) 필터 — max가 avg의 2.0배 이상
    # 조건 3: SUB_THRESHOLD 넘는 프레임 2개 이상 (지속 모션 = 실제 투구)
    high_count = sum(1 for s in scores if s > SUB_THRESHOLD)
    is_pitch = (
        max_score > THRESHOLD
        and max_score > avg_score * 2.0
        and high_count >= 2
    )
    return is_pitch, max_score, avg_score


def scan_pitch_overlays(
    video_path: str,
    expected_count: int = 0,
    max_pitches: int = 0,
    skip_start_sec: float = 0.0,
) -> tuple[list[float], list[dict]]:
    """
    Fox 방송 오버레이 OCR 스캔.
    스코어박스 우측 'P:N → MPH' 전환으로 투구 타이밍 감지,
    구종 박스 OCR로 구종 추출.
    Returns: (timestamps, [{'pitch_type': str, 'speed': int|None}, ...])
    """
    try:
        import pytesseract
        import re as _re
    except ImportError:
        print("[OCR] pytesseract 없음 — scan_video_pitches로 폴백")
        return [], []

    PITCH_MAP = {
        "KNUCKLE CURVE": "Knuckle Curve", "MNUCKLE CURVE": "Knuckle Curve",
        "CURVEBALL": "Curveball",   "CURVE BALL": "Curveball",
        "CURVE": "Curveball",
        "4-SEAM": "4-Seam Fastball", "4 SEAM": "4-Seam Fastball",
        "FASTBALL": "Fastball",      "FOUR SEAM": "4-Seam Fastball",
        "SINKER": "Sinker",          "TWO SEAM": "2-Seam Fastball",
        "2-SEAM": "2-Seam Fastball",
        "SLIDER": "Slider",          "SWEEPER": "Sweeper",
        "CHANGEUP": "Changeup",      "CHANGE UP": "Changeup",
        "CUTTER": "Cutter",          "SPLITTER": "Splitter",
        "KNUCKLEBALL": "Knuckleball",
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step  = 15          # 0.5초 간격
    min_sep = 12.0

    skip_frame = int(skip_start_sec * fps)
    if skip_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)

    timestamps:  list[float] = []
    pitch_data:  list[dict]  = []
    prev_hash    = -1
    prev_conf_t  = -999.0
    frame_idx    = skip_frame
    ocr_cfg_type = "--psm 6"
    ocr_cfg_speed = "--psm 6"
    _pending_pcount_fill = False  # MPH 감지 후 다음 P:N 값으로 채워야 하는 상태

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t   = frame_idx / fps
            h, w = frame.shape[:2]

            # 스코어박스 첫줄 우측 — P:N vs MPH 영역
            sr = frame[int(h * 0.769):int(h * 0.800), int(w * 0.875):w]
            gr = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)

            # 빠른 해시: 8×4 썸네일 합
            thumb    = cv2.resize(gr, (8, 4), interpolation=cv2.INTER_AREA)
            curr_hash = int(thumb.sum())

            if abs(curr_hash - prev_hash) > 150:
                # 변화 감지 → OCR
                big = cv2.resize(gr, (gr.shape[1] * 6, gr.shape[0] * 6),
                                 interpolation=cv2.INTER_CUBIC)
                _, b = cv2.threshold(big, 100, 255, cv2.THRESH_BINARY)
                try:
                    raw_s = pytesseract.image_to_string(b, config=ocr_cfg_speed).upper().strip().replace("\n", " ")
                except Exception:
                    raw_s = ""

                if "MPH" in raw_s and t - prev_conf_t >= min_sep:
                    # 이전 pending은 P:N 못 읽은 것 → pitch_count=None 유지
                    _pending_pcount_fill = False

                    # 구속 파싱: "94 MPH" 또는 "94.3→943" OCR 오독 보정
                    speed = None
                    for m in _re.finditer(r"(\d+)", raw_s):
                        v = int(m.group(1))
                        if 50 <= v <= 115:
                            speed = v
                            break
                        elif 500 <= v <= 1150:  # 소수점 누락 오독 (943 → 94.3mph)
                            speed = round(v / 10)
                            break

                    # 구종 OCR: 어두운 박스 영역
                    tr = frame[int(h * 0.718):int(h * 0.762), int(w * 0.68):w]
                    gt = cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY)
                    bt = cv2.resize(gt, (gt.shape[1] * 5, gt.shape[0] * 5),
                                    interpolation=cv2.INTER_CUBIC)
                    _, bbt = cv2.threshold(bt, 130, 255, cv2.THRESH_BINARY)
                    try:
                        raw_t = pytesseract.image_to_string(bbt, config=ocr_cfg_type).upper().strip().replace("\n", " ")
                    except Exception:
                        raw_t = ""

                    pitch_type = None
                    for kw in sorted(PITCH_MAP.keys(), key=len, reverse=True):
                        if kw in raw_t:
                            pitch_type = PITCH_MAP[kw]
                            break

                    timestamps.append(t)
                    pitch_data.append({"pitch_type": pitch_type, "speed": speed, "pitch_count": None})
                    prev_conf_t = t
                    _pending_pcount_fill = True  # 다음 P:N 감지 시 채울 것
                    print(f"[OCR] {t:.1f}s → {pitch_type} {speed}mph")

                    if max_pitches > 0 and len(timestamps) >= max_pitches:
                        break

                elif ("P:" in raw_s or "P;" in raw_s) and _pending_pcount_fill and pitch_data:
                    # P:N 카운터 감지 → 직전 MPH의 pitch_count 채우기
                    _pm = _re.search(r'P\s*[;:]\s*(\d+)', raw_s)
                    if _pm:
                        pcount = int(_pm.group(1))
                        pitch_data[-1]["pitch_count"] = pcount
                        _pending_pcount_fill = False
                        print(f"[OCR] P:{pcount} 확인 → 타임스탬프 {timestamps[-1]:.1f}s 매핑")

            prev_hash = curr_hash

        frame_idx += 1

    cap.release()
    print(f"[OCR] 완료: {len(timestamps)}개 감지 (예상 {expected_count}개)")
    return timestamps, pitch_data


def ocr_check_pitch_overlay(
    video_path: str,
    check_time: float,
) -> tuple[bool, str | None, int | None]:
    """
    단일 프레임 OCR — 방송 오버레이 실시간 투구 감지.
    Returns: (is_pitch, pitch_type_name, speed_mph)
    """
    try:
        import pytesseract
        import re as _re
    except ImportError:
        return False, None, None

    PITCH_MAP = {
        "KNUCKLE CURVE": "Knuckle Curve", "MNUCKLE CURVE": "Knuckle Curve",
        "CURVEBALL": "Curveball", "CURVE BALL": "Curveball", "CURVE": "Curveball",
        "4-SEAM": "4-Seam Fastball", "4 SEAM": "4-Seam Fastball",
        "FASTBALL": "Fastball", "FOUR SEAM": "4-Seam Fastball",
        "SINKER": "Sinker", "TWO SEAM": "2-Seam Fastball", "2-SEAM": "2-Seam Fastball",
        "SLIDER": "Slider", "SWEEPER": "Sweeper",
        "CHANGEUP": "Changeup", "CHANGE UP": "Changeup",
        "CUTTER": "Cutter", "SPLITTER": "Splitter", "KNUCKLEBALL": "Knuckleball",
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(check_time * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False, None, None

    h, w = frame.shape[:2]

    # 구속 영역 OCR — 여러 임계값 + 반전으로 강건하게 시도
    sr = frame[int(h * 0.769):int(h * 0.800), int(w * 0.875):w]
    gr = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gr, (gr.shape[1] * 6, gr.shape[0] * 6), interpolation=cv2.INTER_CUBIC)

    raw_s = ""
    for _thr in [100, 128, 80, 150]:
        _, _b = cv2.threshold(big, _thr, 255, cv2.THRESH_BINARY)
        try:
            _t = pytesseract.image_to_string(_b, config="--psm 7 --oem 3").upper().strip()
            if "MPH" in _t:
                raw_s = _t.replace("\n", " ")
                break
            if not raw_s:
                raw_s = _t.replace("\n", " ")
        except Exception:
            pass
        _, _b_inv = cv2.threshold(big, _thr, 255, cv2.THRESH_BINARY_INV)
        try:
            _t2 = pytesseract.image_to_string(_b_inv, config="--psm 7 --oem 3").upper().strip()
            if "MPH" in _t2:
                raw_s = _t2.replace("\n", " ")
                break
        except Exception:
            pass

    if "MPH" not in raw_s:
        return False, None, None, None

    # 숫자 직접 추출 — OCR 오독 보정 (7→T/1, 9→I/g, 0→O)
    _norm = (raw_s.replace("T", "7").replace("I", "1").replace("l", "1")
                  .replace("O", "0").replace("G", "6").replace("S", "5")
                  .replace("|", "1").replace("g", "9"))
    speed = None
    for m in _re.finditer(r"(\d+)", _norm):
        v = int(m.group(1))
        if 50 <= v <= 115:
            speed = v
            break
        elif 500 <= v <= 1150:  # 소수점 누락 오독 (943 → 94.3mph)
            speed = round(v / 10)
            break

    # 구종 영역 OCR
    tr = frame[int(h * 0.718):int(h * 0.762), int(w * 0.68):w]
    gt = cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY)
    bt = cv2.resize(gt, (gt.shape[1] * 5, gt.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
    _, bbt = cv2.threshold(bt, 130, 255, cv2.THRESH_BINARY)

    try:
        raw_t = pytesseract.image_to_string(bbt, config="--psm 6").upper().strip().replace("\n", " ")
    except Exception:
        raw_t = ""

    pitch_type = None
    for kw in sorted(PITCH_MAP.keys(), key=len, reverse=True):
        if kw in raw_t:
            pitch_type = PITCH_MAP[kw]
            break

    # P:N 카운터 읽기 — MPH 이후 2~4초 뒤 프레임에서 P:N 출현
    pitch_count = None
    for _ahead in [2.0, 3.0, 1.5, 4.0]:
        _cap2 = cv2.VideoCapture(video_path)
        _cap2.set(cv2.CAP_PROP_POS_FRAMES, int((check_time + _ahead) * fps))
        _ret2, _fr2 = _cap2.read()
        _cap2.release()
        if not _ret2:
            continue
        _h2, _w2 = _fr2.shape[:2]
        _sr2 = _fr2[int(_h2 * 0.769):int(_h2 * 0.800), int(_w2 * 0.875):_w2]
        _gr2 = cv2.cvtColor(_sr2, cv2.COLOR_BGR2GRAY)
        _big2 = cv2.resize(_gr2, (_gr2.shape[1] * 6, _gr2.shape[0] * 6), interpolation=cv2.INTER_CUBIC)
        _, _b2 = cv2.threshold(_big2, 100, 255, cv2.THRESH_BINARY)
        try:
            _raw2 = pytesseract.image_to_string(_b2, config="--psm 7 --oem 3").upper().strip()
        except Exception:
            _raw2 = ""
        _pm = _re.search(r'P\s*[;:]\s*(\d+)', _raw2)
        if _pm:
            pitch_count = int(_pm.group(1))
            break

    return True, pitch_type, speed, pitch_count


def scan_video_pitches(
    video_path: str,
    expected_count: int = 0,
    step: int = 10,
    min_sep: float = 15.0,
    threshold: float = 0.09,
    skip_start_sec: float = 0.0,
    max_pitches: int = 0,
) -> list[float]:
    """
    영상 오프라인 스캔 → 투구 타임스탬프 목록 반환.

    max_pitches > 0: 해당 수 찾으면 조기 종료 (이닝 제한용).
    ratio > 1.5: 화면 중앙 집중 모션만 투구로 인정 (카메라컷·리플레이 제거).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec    = total_frames / fps

    # skip_start_sec가 영상 절반 이상이면 0으로 폴백
    if skip_start_sec >= total_sec * 0.5:
        skip_start_sec = 0.0

    skip_start_frame = int(skip_start_sec * fps)
    if skip_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_start_frame)

    time_scores: list[tuple[float, float]] = []
    prev_gray     = None
    frame_idx     = skip_start_frame
    last_greedy_t = -9999.0
    greedy_count  = 0
    stop_after_t  = None  # max_pitches 도달 후 조기 종료 시각

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
                h, w  = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                ph, pw = prev_gray.shape[:2]
                if (ph, pw) != (h, w):
                    prev_gray = cv2.resize(prev_gray, (w, h))

                y1, y2 = h * 3 // 8, h * 7 // 8
                x1, x2 = w * 2 // 8, w * 6 // 8
                region_diff  = cv2.absdiff(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
                full_diff    = cv2.absdiff(prev_gray, gray)
                region_score = float(region_diff.mean()) / 255.0
                full_score   = float(full_diff.mean())   / 255.0

                ratio = region_score / max(full_score, 0.01)
                # 1.5 이상: 중앙에 집중된 모션 = 투구. 미만: 카메라컷·전체 흔들림 = 페널티
                score = region_score * ratio if ratio > 1.5 else region_score * 0.2
                time_scores.append((frame_idx / fps, score))

                # 조기 종료: max_pitches 개 발견 후 min_sep 더 스캔하고 중단
                if max_pitches > 0 and stop_after_t is None:
                    t_now = frame_idx / fps
                    if score > threshold and t_now - last_greedy_t >= min_sep:
                        greedy_count += 1
                        last_greedy_t = t_now
                        if greedy_count >= max_pitches:
                            stop_after_t = t_now + min_sep

            prev_gray = gray

        if stop_after_t is not None and frame_idx / fps > stop_after_t:
            break

        frame_idx += 1

    cap.release()

    if not time_scores:
        return []

    candidates = [(t, s) for t, s in time_scores if s > threshold]
    candidates.sort(key=lambda x: -x[1])

    selected: list[float] = []
    for t, s in candidates:
        if not any(abs(t - sel_t) < min_sep for sel_t in selected):
            selected.append(t)

    selected.sort()

    # expected_count 기준으로 너무 많으면 상위 점수 순으로 trim
    if expected_count > 0 and len(selected) > int(expected_count * 1.2):
        score_map = {t: s for t, s in time_scores}
        scored = sorted(selected, key=lambda t: -score_map.get(t, 0))
        selected = sorted(scored[:int(expected_count * 1.1)])

    early = f", 조기종료(max={max_pitches})" if stop_after_t else ""
    print(f"[VideoScan] 완료: {len(selected)}개 투구 감지 (예상: {expected_count}개{early})")
    return selected
