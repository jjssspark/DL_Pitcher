"""
데모 경기 전 투구를 영상 궤적만으로 미리 판정해 둔다.

앱은 지금 재생 중에 투구마다 YOLO를 돌린다. 그래서 두 가지가 걸린다 — 재생이
무거워지고, 시연 중 실제로 본 투구만 채점되므로 표본이 수십 개에 머문다(실측 n=43).
미리 돌려두면 재생 부하가 0이 되고 표본이 한 자리 늘어난다.

입력 시각은 **구속 표시가 뜬 시각**이다. classify_video_pitch의 창이 t-4.0 ~ t+0.5초로
뒤를 돌아보게 돼 있어서(TS-025), 오버레이가 뜬 시각을 넣어야 그 앞의 투구를 잡는다.
투구 시각을 넣으면 창이 어긋난다.

TS-035에서 뽑은 구속 구간 250개를 쓴다. 기존 스캔 65개보다 낫다.

감지 설정(imgsz 960, conf 0.05)과 창 폭은 기본값 그대로 둔다 — ADR-0010.

실행:
  nohup venv/bin/python3 -u scripts/batch_cv_verdicts.py > /tmp/cv_batch.log 2>&1 &
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from timeline_anchor import index_at_time, resolve_anchors  # noqa: E402

SPEEDS = os.path.join(ROOT, "output", "timeline", "overlay_speeds.jsonl")
ANCHORS = os.path.join(ROOT, "streamlit_app", "fixed_demo_anchors.json")
PITCHES = os.path.join(ROOT, "output", "timeline", "pitches_775300.json")
VIDEO = os.path.join(ROOT, "streamlit_app", ".yolo_cache",
                     "pitchiq_hq_yt_gMm3EODDb6w.mp4")
OUT = os.path.join(ROOT, "streamlit_app", "fixed_demo_cv.json")

DURATION = 8231.0
N_PITCHES = 320
MIN_VOTES = 2

# 창을 두 번 놓는다. 구속 표시가 뜨기 시작한 시각을 그대로 넣으면 창(t-4.0 ~ t+0.5)이
# 앞으로 밀려 궤적 끝이 잘리는 경우가 있다. 전체해상도 스캔이 MPH를 읽어낸 시각은
# 구간 시작보다 중앙 +0.91초 늦다. 앞쪽 30구 실측:
#
#   변형          판정   적중   정확도
#   offset 0       21     19    90.5%
#   +0.9s          24     21    87.5%
#
# 정확도가 내린 건 분모가 늘어서다. 새로 판정된 3구가 전부 맞았고 전체 30구 중 맞힌
# 개수는 19 -> 21로 늘었다. 그래서 첫 창이 실패한 것만 두 번째 창으로 다시 본다 —
# 정답을 보고 고르는 게 아니라 "창 A 실패 시 창 B"라 실제 파이프라인 전략 그대로다.
WINDOW_OFFSETS = (0.0, 0.9)


def overlay_times_by_pitch() -> dict[int, float]:
    """구속 표시가 뜬 시각 -> 그 표시가 가리키는 투구 인덱스.

    앵커 인덱스는 '그 시각까지 던진 개수'다(TS-034). 구속 표시가 뜬 시점에는 그 공이
    이미 던져졌으므로, 방금 던진 공은 하나 앞이다.

    한 표시가 구간 분할로 쪼개져 같은 인덱스가 두 번 나오면 앞선 것을 쓴다.
    """
    speeds = [json.loads(line) for line in open(SPEEDS) if line.strip()]
    counters = [(float(r["t"]), r["counter"]) for r in json.load(open(ANCHORS))]
    pitches = json.load(open(PITCHES))
    anchors = resolve_anchors(counters, pitches, DURATION)

    by_pitch: dict[int, float] = {}
    for row in sorted(speeds, key=lambda r: r["start"]):
        if row.get("mph_votes", 0) < MIN_VOTES:
            continue
        t = float(row["start"])
        idx = index_at_time(t, anchors, DURATION, N_PITCHES) - 1
        if 0 <= idx < N_PITCHES and idx not in by_pitch:
            by_pitch[idx] = t
    return by_pitch


def _save(verdicts: dict, attempted: int) -> int:
    """지금까지 결과를 파일에 쓴다. 판정된 개수를 돌려준다."""
    decided = sum(1 for r in verdicts.values() if r.get("group"))
    payload = {
        "version": "v2-two-class-2window",
        "video": os.path.basename(VIDEO),
        "verdicts": verdicts,
        "summary": {"attempted": attempted, "decided": decided,
                    "total_pitches": N_PITCHES},
    }
    with open(OUT, "w") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    return decided


def main() -> None:
    from pitch_type_cv.group_classifier import load_classifier
    from pitch_type_cv.live_classifier import (
        TWO_CLASS_MODEL_PATH, classify_video_pitch,
    )
    from ultralytics import YOLO

    targets = overlay_times_by_pitch()
    print(f"판정 대상 {len(targets)}구 / {N_PITCHES} "
          f"({len(targets)/N_PITCHES:.0%})", flush=True)

    # 이어달리기. 이미 판정이 난 투구는 건드리지 않고, 아직 안 써본 창만 시도한다.
    # 전체 재실행이 5.8시간이라 실패분만 다시 보는 쪽이 훨씬 싸다.
    verdicts: dict[str, dict] = {}
    if os.path.exists(OUT):
        with open(OUT) as fp:
            verdicts = json.load(fp).get("verdicts", {})
        # v1은 창을 하나만 썼다. 기록이 없으면 offset 0을 이미 해봤다는 뜻이다 —
        # 안 그러면 실패분 55구에 offset 0을 한 번 더 돌려 1.8시간이 3.5시간이 된다.
        for row in verdicts.values():
            row.setdefault("tried", [0.0])
    todo = [(idx, t) for idx, t in sorted(targets.items())
            if not verdicts.get(str(idx), {}).get("group")
            and set(verdicts.get(str(idx), {}).get("tried", [])) != set(WINDOW_OFFSETS)]
    print(f"기존 판정 {sum(1 for r in verdicts.values() if r.get('group'))}구 "
          f"· 이번에 볼 것 {len(todo)}구", flush=True)
    if not todo:
        print("할 일 없음", flush=True)
        return

    classifier = load_classifier(TWO_CLASS_MODEL_PATH)
    detector = YOLO(os.path.join(ROOT, "models", "ball_broadcast_v1.pt"))

    started = time.time()
    gained = 0
    for i, (idx, t) in enumerate(todo, 1):
        prev = verdicts.get(str(idx), {})
        tried = list(prev.get("tried", []))
        row = prev or {"t": t, "group": None, "confidence": 0.0,
                       "n_points": 0, "reason": "not_attempted"}
        for off in WINDOW_OFFSETS:
            if off in tried:
                continue
            try:
                v = classify_video_pitch(classifier, detector, VIDEO, t + off)
                row = {"t": t, "group": v.group, "confidence": v.confidence,
                       "n_points": v.n_points, "reason": v.reason, "offset": off}
            except Exception as exc:                  # 실패는 실패로 남긴다
                row = {"t": t, "group": None, "confidence": 0.0,
                       "n_points": 0, "reason": f"error: {exc}", "offset": off}
            tried.append(off)
            if row["group"]:
                break
        row["tried"] = tried
        verdicts[str(idx)] = row
        if row["group"]:
            gained += 1
        if i % 5 == 0 or i == len(todo):
            _save(verdicts, len(targets))       # 중간에 죽어도 이어달릴 수 있게
            rate = (time.time() - started) / i
            print(f"[{i:3d}/{len(todo)}] idx={idx} t={t:7.1f}s "
                  f"{row['group'] or '판정불가(' + str(row['reason']) + ')'} "
                  f"· 새로 판정 {gained} ({gained/i:.0%}) · {rate:.1f}s/건 "
                  f"· 남은 {rate*(len(todo)-i)/60:.0f}분", flush=True)

    decided = _save(verdicts, len(targets))
    print(f"\n판정 {decided}/{len(targets)} = {decided/len(targets):.0%} "
          f"· 총 {(time.time()-started)/60:.1f}분", flush=True)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
