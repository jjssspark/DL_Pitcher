"""
방송 스코어버그 박스만 잘라낸 그레이 프레임 배열을 만든다.

왜 프록시인가: 원본(1280x720, 59.94fps, 8230.7초)을 순차 디코드하면 실측 117.8분이다.
스코어버그 박스는 104x24픽셀이라 여기만 뽑으면 배열 전체가 수십 MB에 들어간다.
한 번 만들어두면 변화점 임계값을 바꿔가며 다시 훑는 게 공짜다 (TS-032에서 마운드
프록시를 매번 다시 만든 것과 반대로).

크롭 좌표는 _read_pitch_counter가 쓰는 것(h 0.766~0.803, w 0.875~1.0)에서 오른쪽 끝을
잘라냈다. 원래 폭(160px)은 오른쪽 55px이 스코어버그 밖 중계 화면이라, 카메라가
움직일 때마다 프레임 차분이 통째로 튄다. 박스 안만 남긴다.

중간 mp4를 거치지 않고 rawvideo를 파이프로 받는다. h264로 한 번 굽으면 24픽셀짜리
글자가 압축으로 뭉개져 차분 신호가 그만큼 죽는다.

실행:
  nohup venv/bin/python3 -u scripts/scan_overlay_proxy.py > /tmp/proxy.log 2>&1 &
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, "streamlit_app", ".yolo_cache",
                     "pitchiq_hq_yt_gMm3EODDb6w.mp4")
OUT = os.path.join(ROOT, "output", "timeline", "overlay_frames.npy")

CROP = "104:24:1120:551"     # w:h:x:y — 스코어버그 우측 카운터 박스
FPS = 4                      # 0.25초 해상도. 오버레이 상태는 수 초씩 유지된다
W, H = 104, 24


def main() -> None:
    started = time.time()
    cmd = [
        "ffmpeg", "-v", "error", "-i", VIDEO,
        "-vf", f"crop={CROP},fps={FPS},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    print(" ".join(cmd), flush=True)

    frame_bytes = W * H
    frames = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_bytes * 64)
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(H, W))
        if len(frames) % 4000 == 0:
            print(f"{len(frames)} 프레임 · {len(frames)/FPS:.0f}s "
                  f"· 경과 {time.time()-started:.0f}s", flush=True)

    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg 실패 rc={proc.returncode}\n{err}", file=sys.stderr, flush=True)
        sys.exit(1)

    arr = np.stack(frames)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, arr)
    print(f"\n{arr.shape} · {arr.nbytes/1e6:.0f}MB · {arr.shape[0]/FPS:.1f}s "
          f"· 총 {time.time()-started:.0f}초", flush=True)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
