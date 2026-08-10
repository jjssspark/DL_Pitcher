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
    read_counter: bool = False,
) -> tuple[bool, str | None, int | None, int | None]:
    """
    단일 프레임 OCR — 방송 오버레이 투구 감지.
    Returns: (is_pitch, pitch_type_name, speed_mph, pitch_count)

    read_counter=True면 투수 투구수(P:N)까지 읽는다. 기본은 끔 — 실측으로 이 판독이
    한 번에 4.7~9.7초 걸려(감지 자체는 2.4초) 재생 중 호출하면 감지 주기가 그만큼
    벌어진다. 오프라인 사전 스캔처럼 시간을 쓸 수 있는 경로에서만 켠다.
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

    pitch_count = _read_pitch_counter(video_path, check_time, fps) if read_counter else None
    return True, pitch_type, speed, pitch_count


# 스코어버그의 투수 투구수(P:N)를 읽는다. 이 값이 있으면 "지금 몇 번째 투구인가"를
# 절대값으로 알 수 있어 타임라인을 실제 중계에 맞출 수 있다.
#
# 이전 구현은 실측 판독률이 0%였다. 원인이 둘이었다.
#
#   (1) 시점 — MPH 이후 1.5~4.0초만 훑었는데 그 구간엔 아직 구속이 떠 있다.
#       스코어버그는 두 상태를 오간다: 투구 직후 "FLAHERTY 79MPH",
#       그다음 "FLAHERTY P: 6". 실측으로 P:N은 +4~+12초에 나온다.
#   (2) 정규식 — r'P\s*[;:]\s*(\d+)' 가 콜론을 요구했는데 OCR은 이 폰트를
#       'P- 6', 'P. 15', 'PR: 16'으로 읽는다. 그래서 전부 버려졌다.
#
# 한 프레임만으로는 여전히 못 믿는다 — 임계값별로 [25,29] [16,18]처럼 갈렸다.
# P:N은 투구 사이 내내 고정돼 있으므로 여러 프레임 x 여러 임계값의 표를 모아
# 다수결한다. 표가 갈리면 None을 돌려준다. 틀린 앵커는 없느니만 못하다.
#
# 남은 오독 양상 하나를 호출측이 알아야 한다: 크롭 아래에 타자 번호 행
# ("2. SOTO")이 있어 가끔 작은 수가 잡힌다. 실측에서 오독은 전부 감소 방향이었다
# (16 -> 3, 16 -> 2). 같은 투수의 P:N은 줄지 않으므로 호출측에서 단조성으로 거른다.
_PN_OFFSETS = (4.5, 6.0, 7.5, 9.0, 10.5)
_PN_THRESHOLDS = (90, 110, 130, 150, 170)
_PN_MIN_VOTES = 2


def _read_pitch_counter(video_path: str, check_time: float, fps: float) -> int | None:
    """투구 직후 구간에서 P:N을 여러 프레임 다수결로 읽는다. 확신 없으면 None."""
    try:
        import pytesseract
        import re as _re
    except ImportError:
        return None

    from collections import Counter

    pattern = _re.compile(r'P[^0-9]{0,3}(\d{1,3})')
    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=P:0123456789"

    cap = cv2.VideoCapture(video_path)      # 오프셋마다 열지 않고 한 번만 연다
    if not cap.isOpened():
        return None

    votes: list[int] = []
    try:
        for offset in _PN_OFFSETS:
            # 이미 같은 값이 두 표 모였으면 더 읽지 않는다. 프레임 5개 x 임계값 5개를
            # 다 돌면 한 번에 4.7초(최대 15초)까지 걸려 실시간 감지가 오히려 막힌다.
            if votes and Counter(votes).most_common(1)[0][1] >= _PN_MIN_VOTES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((check_time + offset) * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            region = frame[int(h * 0.766):int(h * 0.803), int(w * 0.875):w]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            big = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6),
                             interpolation=cv2.INTER_CUBIC)
            for thr in _PN_THRESHOLDS:
                _, binary = cv2.threshold(big, thr, 255, cv2.THRESH_BINARY)
                try:
                    text = pytesseract.image_to_string(binary, config=config).upper().strip()
                except Exception:
                    continue
                if "MPH" in text:      # 아직 구속 표시 상태 — 카운터가 아니다
                    continue
                found = pattern.search(text.replace("\n", " "))
                if found:
                    value = int(found.group(1))
                    if 1 <= value <= 200:
                        votes.append(value)
    finally:
        cap.release()

    if not votes:
        return None
    top, count = Counter(votes).most_common(1)[0]
    return top if count >= _PN_MIN_VOTES else None


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
