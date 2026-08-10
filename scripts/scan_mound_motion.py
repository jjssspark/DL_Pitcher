"""
마운드 프록시 영상에서 모션 점수 시계열을 뽑는다.

주의: 이 시계열로 투구 시각을 찾으려던 시도는 실패했다 (TS-032, scripts/align_pitch_times.py).
스캔 자체는 정상 동작하고 8231초를 36초에 처리한다 — 못 쓰는 건 신호지 이 코드가 아니다.
다른 용도로 쓰기 전에 TS-032의 홀드아웃 숫자를 먼저 보라.

왜 프록시인가: 원본(1.1GB, 8231초, 59.94fps)을 순차 디코드하면 117.8분이다(실측).
ffmpeg으로 마운드 영역만 잘라 240x134 / 15fps로 줄이면 8배속으로 17분이면 끝나고,
그 결과를 스캔하는 데는 수초밖에 안 걸린다.

크롭 좌표는 pose_detector.detect_pitch_motion이 쓰던 것과 같다
(y: 3/8~7/8, x: 2/8~6/8). 즉 프록시는 이미 마운드만 담고 있어 여기서는 프레임 전체를
그대로 차분하면 된다.

detect_pitch_motion을 그대로 못 쓰는 이유: 그 함수는 후보 시각 주변 15~20프레임을 받아
예/아니오만 돌려주는 창 단위 판정기다. 8231초를 연속 점수로 만들려면 시계열이 필요하다.

실행:
  venv/bin/python3 scripts/scan_mound_motion.py <proxy.mp4> <out.npy>
"""
import sys
import time

import cv2
import numpy as np


def motion_series(proxy_path: str) -> tuple[np.ndarray, float]:
    """
    프레임 간 절대차의 평균을 시계열로 뽑는다.

    Returns: (점수 배열, fps). 점수[i]는 프레임 i+1과 i 사이의 차이이므로
    길이가 프레임 수보다 1 작고, 시각은 t = i / fps 로 환산한다.
    """
    cap = cv2.VideoCapture(proxy_path)
    if not cap.isOpened():
        raise SystemExit(f"프록시를 열 수 없다: {proxy_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    scores: list[float] = []
    prev = None
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            scores.append(float(cv2.absdiff(prev, gray).mean()) / 255.0)
        prev = gray
        if scores and len(scores) % 30000 == 0:
            print(f"  {len(scores)/fps:7.0f}초 처리 ({time.time()-started:.0f}s)", flush=True)

    cap.release()
    return np.asarray(scores, dtype=np.float32), fps


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    proxy, out = sys.argv[1], sys.argv[2]

    t0 = time.time()
    scores, fps = motion_series(proxy)
    np.save(out, scores)

    print(f"프레임 {len(scores)+1}개 · fps {fps:.2f} · {len(scores)/fps:.1f}초 분량")
    print(f"점수 평균 {scores.mean():.4f} · 중앙값 {np.median(scores):.4f} "
          f"· 90분위 {np.percentile(scores, 90):.4f} · 최대 {scores.max():.4f}")
    print(f"소요 {time.time()-t0:.1f}s → 저장 {out}")


if __name__ == "__main__":
    main()
