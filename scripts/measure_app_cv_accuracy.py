"""
앱 경로의 CV 구종 판정을 실제 중계 영상으로 채점한다.

실험실 수치(LOGO 0.783)는 Savant 투구별 클립에서 잰 것이다. 앱은 전체 경기 영상을
쓰고 창 위치도 다르므로, 그 수치가 앱에서 재현된다는 보장이 없다. 여기서 그걸 잰다.

정답은 방송 오버레이 OCR이 읽은 구종을 쓴다. Statcast API가 아니라 OCR인 이유는
스캔 타임스탬프와 이미 정렬돼 있기 때문이다 — API 라벨을 쓰려면 65개 스캔 시각을
320개 투구 인덱스에 맞추는 별도 문제가 생긴다. OCR은 완벽하지 않지만(판독률 상한이
있다) 같은 프레임에서 읽은 값이라 정렬 오차가 없다.

이 영상의 경기(775300)는 학습에 쓴 4경기(775294, 813024, 813026, 813027)에
포함되지 않는다. 진짜 홀드아웃이다.

창 위치 보정용 --sweep을 붙였다. 학습 창 폭 1.4초는 고정하고 위치만 옮긴다.

실행:
  venv/bin/python3 scripts/measure_app_cv_accuracy.py
  venv/bin/python3 scripts/measure_app_cv_accuracy.py --sweep
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pitch_type_cv.group_classifier import load_classifier  # noqa: E402
from pitch_type_cv.live_classifier import (  # noqa: E402
    DEFAULT_LOOKBACK_END_SEC,
    DEFAULT_LOOKBACK_START_SEC,
    TWO_CLASS_MODEL_PATH,
    TWO_CLASSES,
    classify_video_pitch,
)
from pitch_type_cv.pitch_group_map import ocr_pitch_name_to_group  # noqa: E402

SCAN_PATH = os.path.join(ROOT, "streamlit_app", "fixed_demo_scan.json")
VIDEO_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
OUT_PATH = os.path.join(ROOT, "output", "pitch_type_cv", "app_cv_accuracy.json")

WINDOW_WIDTH_SEC = 1.4  # 학습 창 폭. 바꾸면 frame_span 계열이 분포를 벗어난다.
SWEEP_STARTS = [3.0, 2.6, 2.2, 1.8, 1.4]


def find_video() -> str:
    files = sorted(
        (f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".mkv", ".webm"))),
        key=lambda f: os.path.getsize(os.path.join(VIDEO_DIR, f)),
        reverse=True,
    )
    if not files:
        raise SystemExit(f"{VIDEO_DIR}에 영상이 없다")
    return os.path.join(VIDEO_DIR, files[0])


def load_labeled_pitches() -> list[tuple[float, str]]:
    """(영상 시각, 2분류 정답). OCR이 구종을 못 읽었거나 OFFSPEED면 뺀다."""
    scan = json.load(open(SCAN_PATH))
    pairs = []
    for t, d in zip(scan["pitch_times"], scan["pitch_data"]):
        group = ocr_pitch_name_to_group(d.get("pitch_type"))
        if group in TWO_CLASSES:
            pairs.append((float(t), group))
    return pairs


def measure(clf, det, video: str, pairs: list, start: float, end: float, verbose: bool) -> dict:
    hits = scored = unavailable = 0
    for t, truth in pairs:
        v = classify_video_pitch(clf, det, video, t,
                                 lookback_start_sec=start, lookback_end_sec=end)
        if not v:
            unavailable += 1
            if verbose:
                print(f"  t={t:8.1f}s  판정불가({v.reason})")
            continue
        scored += 1
        hit = v.group == truth
        hits += hit
        if verbose:
            print(f"  t={t:8.1f}s  정답 {truth:8s} → CV {v.group:8s} "
                  f"conf={v.confidence:.2f} pts={v.n_points} {'O' if hit else 'X'}")

    total = len(pairs)
    return {
        "lookback_start_sec": start,
        "lookback_end_sec": end,
        "n_labeled": total,
        "n_scored": scored,
        "n_unavailable": unavailable,
        "coverage": scored / total if total else 0.0,
        "accuracy": hits / scored if scored else 0.0,
        "hits": hits,
    }


def print_row(r: dict) -> None:
    print(f"  창 {r['lookback_start_sec']:.1f}~{r['lookback_end_sec']:.1f}s  "
          f"판정률 {r['coverage']:5.1%} ({r['n_scored']}/{r['n_labeled']})  "
          f"정확도 {r['accuracy']:5.1%} ({r['hits']}/{r['n_scored']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="창 위치를 훑어 최적값을 찾는다 (오래 걸린다)")
    parser.add_argument("--limit", type=int, default=0, help="투구 수 제한 (빠른 확인용)")
    args = parser.parse_args()

    from ultralytics import YOLO

    video = find_video()
    pairs = load_labeled_pitches()
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"영상: {os.path.basename(video)}")
    print(f"라벨된 투구 {len(pairs)}개 (OCR 판독 성공 & 2분류에 해당)")
    print(f"모델: {os.path.basename(TWO_CLASS_MODEL_PATH)}\n")

    clf = load_classifier(TWO_CLASS_MODEL_PATH)
    det = YOLO(os.path.join(ROOT, "models", "ball_broadcast_v1.pt"))

    if args.sweep:
        print("창 위치 훑기 (폭 1.4초 고정)")
        results = []
        for start in SWEEP_STARTS:
            r = measure(clf, det, video, pairs, start, start - WINDOW_WIDTH_SEC, verbose=False)
            results.append(r)
            print_row(r)
        best = max(results, key=lambda r: r["accuracy"] * r["coverage"])
        print(f"\n최적(정확도×판정률): 창 "
              f"{best['lookback_start_sec']:.1f}~{best['lookback_end_sec']:.1f}s")
        payload = {"sweep": results, "best": best}
    else:
        r = measure(clf, det, video, pairs,
                    DEFAULT_LOOKBACK_START_SEC, DEFAULT_LOOKBACK_END_SEC, verbose=True)
        print()
        print_row(r)
        payload = {"single": r}

    print("\n기준: 실험실 LOGO 정확도 0.783 (Savant 클립, 학습 4경기 교차검증)")
    print("      최빈값 기준선 0.543")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
