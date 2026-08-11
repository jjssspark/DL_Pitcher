"""
카운터 판독에 실패한 구간에서 구속(MPH)을 읽는다.

스코어버그의 그 칸은 두 상태를 오간다 — 투구 직후 "87MPH"가 3초쯤 떴다가 "P: 4"로
돌아간다. read_overlay_counters.py는 MPH 상태를 그냥 버렸다(value=None). 그런데
버려진 쪽이 더 중요한 신호였다.

카운터는 인플레이 타구가 나오면 못 쓴다. 방송이 리플레이로 넘어가면서 스코어버그가
사라지고, 다음 타자가 들어설 때까지 P:N이 안 보인다. 실측(TS-034):

    49.75 -> 60.75  P:3
    61.50 -> 63.25  (판독 실패)   <- 사실은 87MPH, 4구째
    63.25 -> 89.00  (바 없음)     <- 리플레이
    89.00 -> 90.25  P:4           <- 앵커가 여기 박혔다. 26초 늦다.

구속 표시는 인플레이 타구 뒤에도 뜬다. 그래서 카운터가 못 보는 지점의 투구 시각을
구속 표시가 알려준다.

새로 영상을 훑지 않는다. 이미 만들어둔 프록시(overlay_frames.npy)에 다 들어 있고,
구간도 overlay_counters.jsonl의 것을 그대로 쓴다 — 두 판독이 같은 시간축에 놓여야
나중에 섞을 수 있다.

실행:
  nohup venv/bin/python3 -u scripts/read_overlay_speeds.py > /tmp/speeds.log 2>&1 &
"""
import json
import os
import re
import time
from collections import Counter

import cv2
import numpy as np
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "output", "timeline", "overlay_frames.npy")
SEGMENTS = os.path.join(ROOT, "output", "timeline", "overlay_counters.jsonl")
OUT = os.path.join(ROOT, "output", "timeline", "overlay_speeds.jsonl")

FPS = 4
THRESHOLDS = (90, 110, 130, 150, 170)
MIN_VOTES = 2

# MPH 앞의 숫자. 글자 화이트리스트를 못 쓴다 — "MPH"를 읽어야 구속 상태인 걸 안다.
PATTERN = re.compile(r"(\d{2,3})\s*MPH")
CONFIG = "--psm 7 --oem 3"

# OCR 오독 보정. pose_detector.ocr_check_pitch_overlay와 같은 치환이다.
CONFUSIONS = str.maketrans({"T": "7", "I": "1", "l": "1", "O": "0",
                            "G": "6", "S": "5", "|": "1", "g": "9"})


def read_speed(frames: np.ndarray, start: float, end: float) -> tuple[int, int | None]:
    """(MPH를 본 횟수, 구속) — 구간 안 프레임을 임계값별로 읽어 다수결.

    두 값을 따로 낸다. 쓰려는 건 "구속 표시가 언제 떴는가"지 구속 자체가 아니다.
    자릿수는 실측에서 95 -> 55처럼 어긋나는데(9를 S로 읽고 치환표가 5로 바꾼다),
    표시가 떴다는 사실은 그 프레임들에서 일관되게 읽힌다. 값 파싱 실패로 구간을
    통째로 버리면 정작 필요한 시각을 잃는다.
    """
    seen = 0
    votes: list[int] = []
    for ratio in (0.35, 0.6, 0.85):
        idx = int((start + (end - start) * ratio) * FPS)
        if idx >= len(frames):
            continue
        big = cv2.resize(frames[idx], (104 * 6, 24 * 6), interpolation=cv2.INTER_CUBIC)
        for threshold in THRESHOLDS:
            _, binary = cv2.threshold(big, threshold, 255, cv2.THRESH_BINARY)
            try:
                text = pytesseract.image_to_string(binary, config=CONFIG).upper().strip()
            except Exception:
                continue
            flat = text.replace("\n", " ")
            if "MPH" not in flat:
                continue
            seen += 1
            found = PATTERN.search(flat.translate(CONFUSIONS))
            if found:
                value = int(found.group(1))
                if 55 <= value <= 105:     # 이 범위 밖은 오독이다
                    votes.append(value)
    speed = None
    if votes:
        value, count = Counter(votes).most_common(1)[0]
        speed = value if count >= MIN_VOTES else None
    return seen, speed


def main() -> None:
    frames = np.load(FRAMES)
    rows = [json.loads(line) for line in open(SEGMENTS) if line.strip()]
    targets = [r for r in rows if r["value"] is None]
    print(f"프레임 {frames.shape} · 전체 구간 {len(rows)}개 "
          f"· 카운터 실패 {len(targets)}개", flush=True)

    started = time.time()
    got = 0
    with open(OUT, "w") as fp:
        for i, row in enumerate(targets, 1):
            seen, speed = read_speed(frames, row["start"], row["end"])
            fp.write(json.dumps({"start": row["start"], "end": row["end"],
                                 "mph": speed, "mph_votes": seen}) + "\n")
            fp.flush()
            if seen >= MIN_VOTES:
                got += 1
            if i % 25 == 0 or i == len(targets):
                rate = (time.time() - started) / i
                print(f"[{i:4d}/{len(targets)}] {row['start']:7.1f}s "
                      f"votes={seen} mph={speed} · 구속구간 {got} ({got/i:.0%}) "
                      f"· {rate:.2f}s/건 · 남은 {rate*(len(targets)-i)/60:.0f}분",
                      flush=True)

    print(f"\n구속 표시로 판정된 구간 {got}/{len(targets)} = {got/len(targets):.0%} "
          f"· 총 {(time.time()-started)/60:.1f}분", flush=True)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
