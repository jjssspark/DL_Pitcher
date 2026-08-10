"""
이미 감지된 투구 시각에서 방송 스코어버그의 투수 투구수(P:N)를 다시 읽는다.

왜 필요한가: 앱은 영상 8231초를 320투구로 균등 분할해 25.72초마다 인덱스를 넘긴다.
그런데 스캔된 실제 투구 간격은 중앙값 96.6초, 최소 12초, 최대 626초다. 광고·리플레이·
타자 교체 때문이다. 균등 가정으로는 구조적으로 어긋나고, 사용자에게는 "타임라인이
공 한 개 느리다"로 보인다 (TS-007, TS-008).

P:N을 읽으면 "이 시각 = 그 투수의 N번째 투구"를 알 수 있고, Statcast 투구 목록의
투수별 누적 카운트와 맞추면 절대 인덱스가 나온다. 그 지점들을 앵커로 삼아 사이를
보간하면 균등 가정을 버릴 수 있다.

fixed_demo_scan.json에 이미 pitch_count가 43개 있지만 쓸 수 없다. 옛 판독 로직으로
읽혀서 42번 전이 중 17번이 감소이고, t=866s P=9 -> t=988s P=33처럼 120초에 24구가
늘어난 불가능한 값이 섞여 있다.

전체 재스캔(8231초)은 필요 없다. 투구 시각 65개는 이미 알고 있으므로 그 지점만
다시 읽는다.

실행:
  nohup venv/bin/python3 -u scripts/build_timeline_anchors.py > /tmp/anchors.log 2>&1 &
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pose_detector import ocr_check_pitch_overlay  # noqa: E402

SCAN_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
VIDEO_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
# 분석 산출물이 아니라 앱이 실행 중에 읽는 데이터라 streamlit_app/에 둔다.
# output/은 .gitignore 대상이라 여기에 두면 다른 환경에서 파일이 없어 조용히
# 균등 분할로 되돌아간다. 같은 성격인 fixed_demo_scan.json도 이 자리에 있다.
OUT_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_anchors.json")


def find_video() -> str:
    """캐시에서 가장 큰 영상 = 데모용 원본."""
    return max(
        (os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
         if f.endswith((".mp4", ".mkv", ".webm"))),
        key=os.path.getsize,
    )


def main() -> None:
    video = find_video()
    scan = json.load(open(SCAN_PATH))
    times = [float(t) for t in scan["pitch_times"]]
    print(f"영상 {os.path.basename(video)} · 투구 시각 {len(times)}개", flush=True)

    rows = []
    started = time.time()
    for i, t in enumerate(times, 1):
        t0 = time.time()
        _is_pitch, _ptype, _speed, counter = ocr_check_pitch_overlay(
            video, t, read_counter=True
        )
        rows.append({"t": t, "counter": counter})
        print(f"[{i:2d}/{len(times)}] t={t:7.1f}s  P={counter}  ({time.time()-t0:.1f}s)",
              flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(rows, open(OUT_PATH, "w"), indent=1)

    got = sum(1 for r in rows if r["counter"] is not None)
    print(f"\n판독 {got}/{len(rows)} = {got/len(rows):.0%}"
          f" · 총 {time.time()-started:.0f}초", flush=True)
    print(f"저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
