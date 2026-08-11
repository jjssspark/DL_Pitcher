"""
오버레이 프록시의 안정 구간마다 P:N을 읽는다.

스코어버그 카운터 박스는 두 상태를 오간다: 투구 직후 "79MPH"가 3초쯤 떴다가
"P: 6"으로 돌아가 다음 투구까지 10초 이상 고정된다. 고정돼 있는 동안 픽셀은 완전히
같으므로, 변하는 지점으로 잘라 구간을 만들고 구간마다 한 번만 읽으면 된다.

왜 원본이 아니라 프록시를 읽는가 — 실측 대조(같은 20지점):

    경로                        속도       판독률
    영상 seek + OCR (기존)      8s/건      11/20
    프록시 프레임 직접 OCR      1.0s/건    16/20

기존 판독값과 겹치는 11건 중 10건이 일치했다. 어긋난 1건(t=866.9s)은
build_timeline_anchors.py가 주석에 "불가능한 값"이라고 적어둔 바로 그 지점이고,
프록시 쪽(28 -> 33, 122초에 5구)이 물리적으로 맞다.

빠른 이유는 디코드를 안 하기 때문이다. 기존 경로는 1.1GB 60fps 원본을 매번 seek
한다. 프록시는 이미 메모리에 있는 104x24 배열이라 tesseract 시간만 든다.

판독 규칙은 pose_detector._read_pitch_counter와 같다 — 임계값을 바꿔가며 읽고 같은
값이 두 번 나와야 채택한다. 한 프레임 한 임계값으로는 [25,29]처럼 갈린다.

실행:
  nohup venv/bin/python3 -u scripts/read_overlay_counters.py > /tmp/counters.log 2>&1 &
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
OUT = os.path.join(ROOT, "output", "timeline", "overlay_counters.jsonl")

FPS = 4
INK_THRESHOLD = 190       # 글자만 남기는 이진화 값
XOR_STABLE = 0.006        # 이 밑이면 내용이 안 바뀐 것으로 본다
MIN_SEGMENT_SEC = 1.0     # 이보다 짧으면 전환 중이거나 노이즈다
THRESHOLDS = (90, 110, 130, 150, 170)
MIN_VOTES = 2

PATTERN = re.compile(r"P[^0-9]{0,3}(\d{1,3})")
CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=P:0123456789"


def stable_segments(frames: np.ndarray) -> list[tuple[float, float]]:
    """스코어버그가 떠 있고 내용이 안 바뀌는 구간 [시작, 끝) 목록.

    배경 밝기가 프레임마다 흔들려(반투명 바) 원시 픽셀 차분으로는 글자 변화가
    묻힌다. 밝은 글자만 이진화한 뒤 XOR 해야 신호가 남는다.
    """
    mask = frames > INK_THRESHOLD
    median = np.median(frames, axis=(1, 2))
    # 바가 사라지면 크롭 전체가 중계 화면이라 밝아진다. 글자가 아예 없는 구간도 뺀다.
    present = (median < 100) & (mask.mean(axis=(1, 2)) > 0.005)

    xor = (mask[1:] ^ mask[:-1]).mean(axis=(1, 2))
    stable = (xor < XOR_STABLE) & present[1:] & present[:-1]

    segments, i, n = [], 0, len(stable)
    while i < n:
        if not stable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and stable[j + 1]:
            j += 1
        start, end = i / FPS, (j + 2) / FPS
        if end - start >= MIN_SEGMENT_SEC:
            segments.append((start, end))
        i = j + 1
    return segments


def read_counter(frames: np.ndarray, start: float, end: float) -> int | None:
    """구간 안 프레임을 임계값별로 읽어 다수결. 두 표를 못 모으면 None."""
    votes: list[int] = []
    for ratio in (0.35, 0.6, 0.85):
        if votes and Counter(votes).most_common(1)[0][1] >= MIN_VOTES:
            break
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
            if "MPH" in text:          # 구속 표시 상태 — 카운터가 아니다
                continue
            found = PATTERN.search(text.replace("\n", " "))
            if found:
                value = int(found.group(1))
                if 1 <= value <= 140:  # 한 투수가 한 경기에 140구를 넘지 않는다
                    votes.append(value)
    if not votes:
        return None
    value, count = Counter(votes).most_common(1)[0]
    return value if count >= MIN_VOTES else None


def main() -> None:
    frames = np.load(FRAMES)
    segments = stable_segments(frames)
    print(f"프레임 {frames.shape} · 안정 구간 {len(segments)}개 "
          f"(>= {MIN_SEGMENT_SEC}s)", flush=True)

    started = time.time()
    got = 0
    with open(OUT, "w") as fp:
        for i, (start, end) in enumerate(segments, 1):
            value = read_counter(frames, start, end)
            fp.write(json.dumps({"start": start, "end": end, "value": value}) + "\n")
            fp.flush()
            if value is not None:
                got += 1
            if i % 50 == 0 or i == len(segments):
                rate = (time.time() - started) / i
                print(f"[{i:4d}/{len(segments)}] {start:7.1f}s P={value} "
                      f"· 판독 {got} ({got/i:.0%}) · {rate:.2f}s/건 "
                      f"· 남은 {rate*(len(segments)-i)/60:.0f}분", flush=True)

    print(f"\n판독 {got}/{len(segments)} = {got/len(segments):.0%} "
          f"· 총 {(time.time()-started)/60:.1f}분", flush=True)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
