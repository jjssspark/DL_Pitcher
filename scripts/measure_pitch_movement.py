"""
구종별 실제 무브먼트를 Statcast에서 잰다. 범례 궤적 그림의 근거값이다.

그림에 쓰던 (dx, dy)는 눈대중으로 넣은 값이었다. Statcast는 무회전 궤적 대비 얼마나
휘었는지를 pfx_x(횡), pfx_z(종)로 피트 단위로 준다 — 그걸 그대로 쓰면 "정확도"라는
말을 붙일 수 있다.

우완 투수만 쓴다. 그림이 우완 기준이고, 좌완은 횡변화 부호가 뒤집혀 섞으면 상쇄된다.

부호 규약은 문서를 믿지 않고 데이터로 확인한다 — 싱커와 슬라이더는 반드시 반대
방향이어야 하고, 우완 싱커는 팔 쪽(3루 방향)이다. 출력에서 그 관계를 같이 찍는다.

실행:
  nohup venv/bin/python3 -u scripts/measure_pitch_movement.py > /tmp/movement.log 2>&1 &
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "pitch_movement.json")

# 정규시즌 한 주. 희귀 구종(너클볼·이피어스·포크볼)까지 잡으려면 이 정도는 필요하다.
START, END = "2024-06-03", "2024-06-09"
MIN_SAMPLE = 30


def main() -> None:
    from pybaseball import statcast

    print(f"Statcast {START} ~ {END} 수집 중...", flush=True)
    df = statcast(start_dt=START, end_dt=END)
    print(f"전체 {len(df):,}구", flush=True)

    df = df[(df["p_throws"] == "R")
            & df["pitch_type"].notna()
            & df["pfx_x"].notna() & df["pfx_z"].notna()]
    print(f"우완 · 무브먼트 값 있음 {len(df):,}구\n", flush=True)

    rows = {}
    for code, g in df.groupby("pitch_type"):
        if len(g) < MIN_SAMPLE:
            continue
        rows[str(code)] = {
            "n": int(len(g)),
            # 피트 -> 인치. 중계·분석에서 쓰는 단위가 인치다.
            "pfx_x": round(float(g["pfx_x"].mean()) * 12, 2),
            "pfx_z": round(float(g["pfx_z"].mean()) * 12, 2),
            "speed": round(float(g["release_speed"].mean()), 1),
            "speed_p10": round(float(np.percentile(g["release_speed"].dropna(), 10)), 1),
            "speed_p90": round(float(np.percentile(g["release_speed"].dropna(), 90)), 1),
        }

    print(f"{'구종':<6}{'n':>8}{'횡 pfx_x':>10}{'종 pfx_z':>10}"
          f"{'평균구속':>9}{'p10~p90':>13}")
    for code, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{code:<6}{r['n']:8,}{r['pfx_x']:10.2f}{r['pfx_z']:10.2f}"
              f"{r['speed']:9.1f}   {r['speed_p10']:.0f}~{r['speed_p90']:.0f}")

    # 부호 규약 확인 — 싱커와 슬라이더가 반대여야 한다
    si, sl = rows.get("SI"), rows.get("SL")
    if si and sl:
        print(f"\n부호 확인: SI {si['pfx_x']:+.2f} · SL {sl['pfx_x']:+.2f} -> "
              f"{'반대 방향 OK' if si['pfx_x'] * sl['pfx_x'] < 0 else '같은 방향?? 규약 재확인 필요'}")
        print(f"우완 싱커가 양수이면 pfx_x 양수 = 팔 쪽(3루 방향): "
              f"{'그렇다' if si['pfx_x'] > 0 else '아니다 — 부호를 뒤집어야 한다'}")

    ff = rows.get("FF")
    if ff:
        print(f"\n포심 대비 종변화 차이(인치, 클수록 더 떨어진다)")
        for code, r in sorted(rows.items(), key=lambda kv: kv[1]["pfx_z"], reverse=True):
            print(f"  {code:<5}{ff['pfx_z'] - r['pfx_z']:+7.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fp:
        json.dump({"window": [START, END], "throws": "R", "types": rows}, fp,
                  ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
