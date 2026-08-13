"""데모 영상 편집 — 구간 컷 + 크로스페이드 + 한글 자막 태워넣기.

원본은 2분 27초 화면 녹화(3420x1948 60fps, 무음)다. 여기서 볼 만한 구간만 뽑아
65초로 줄이고, 무음이라 설명이 안 되므로 자막을 얹는다.

자막은 파일로 분리하지 않고 영상에 태운다(hardsub). 링크로 공유할 때 플레이어가
자막 파일을 읽어주는지에 기대지 않으려는 것이다. 대신 유튜브가 자막을 텍스트로
읽지 못하므로, 검색 유입이 필요하면 설명란에 문구를 따로 넣어야 한다.

자막 문구를 고치려면 SEGMENTS의 자막 줄을 바꾸고 다시 돌린다.

실행:
  venv/bin/python3 scripts/build_demo.py [원본.mov]
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "demo")
TMP = os.path.join(OUT_DIR, ".build")

DEFAULT_SRC = os.path.expanduser("~/Desktop/baseball-pitch-predictor_demo.mov")
FONT = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
FONT_NAME = "AppleGothic"

W, H, FPS = 1920, 1080, 30
XF = 0.4  # 크로스페이드 길이

# 우측 패널을 읽히게 하려고 쓰는 펀치인.
# 원본 3420x1948에서 우측 정렬로 1.4배 당긴 16:9 영역이다. 실측 좌표 기준으로
# 실측/CV/예측 카드(y 806~1855)가 전부 들어온다.
PUNCH = "crop=2443:1374:977:500"

# (원본 시작초, 길이, 펀치인 여부, [자막 줄])
#
# 펀치인 구간은 자막을 위로 올린다. 아래로 두면 우측 패널의 예측 카드를 그대로
# 덮는다 — 크게 당겨 보여주려던 것을 자막이 가리면 펀치인을 한 의미가 없다.
SEGMENTS = [
    (0.5,   3.5, False, ["Statcast 데이터 · 중계 영상 · 예측을 한 화면에"]),
    (3.5,   8.0, False, ["구종 10종을 Statcast 실측 무브먼트로 표현",
                         "pfx_x · pfx_z 기반 · 포수 시점"]),
    (10.5,  6.5, False, ["방금 던진 공: 실측 FF 96mph",
                         "직전 예측이 맞았는지 그 자리에서 채점"]),
    (26.0,  8.0, False, ["예측이 연속으로 맞으면 COMBO 누적"]),
    (41.0,  9.0, True,  ["영상만으로 낸 판정: 변화구 계열 (적중)",
                         "Statcast API 없이 YOLOv8 궤적만 사용"]),
    # 이 구간은 실측이 너클커브인데 CV가 속구 계열로 오답을 냈고 BiLSTM 예측도
    # 빗나갔다. 화면에 그렇게 적혀 있으므로 자막도 그렇게 쓴다.
    (101.0, 8.5, True,  ["틀린 판정도 숨기지 않는다 — 이 공은 오답",
                         "채점 125구 기준 66.4% · 기준선 53.6%"]),
    (129.5, 5.0, False, ["타자가 바뀌면 다음 구종을 다시 계산"]),
    (134.0, 5.5, False, ["매치업 상세는 마우스를 올리면 크게"]),
    (139.5, 7.5, False, ["투수별 구종 분포와 누적 통계"]),
]

TITLE = ["PitchIQ", "MLB 투구 예측 · 중계 영상 구종 판정"]
TITLE_DUR = 3.5
END = [
    "다음 구종 예측 (BiLSTM · 8분류)   48.5%",
    "영상만으로 구종 판정 (2분류)   66.4%",
    "기준선 53.6%",
    "github.com/jjssspark/DL_Pitcher",
]
END_DUR = 4.0


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("실패:", " ".join(cmd[:8]), "...", file=sys.stderr)
        print(r.stderr[-2500:], file=sys.stderr)
        sys.exit(1)


def card(path, lines, dur, big_first):
    """타이틀/엔드 카드를 색 소스 위에 글자만 얹어 만든다.

    expansion=none이 반드시 있어야 한다. drawtext는 기본적으로 문구 안의 %를 확장
    문법으로 해석해서, '48.5%'가 들어간 줄이 통째로 렌더되지 않고 사라졌다
    (실측: 엔드 카드 4줄 중 %가 없는 마지막 줄만 보였다). textfile로 넘겨도
    같은 확장을 거치므로 소용이 없었고, expansion=none이라야 %가 글자로 찍힌다.
    """
    draws = []
    n = len(lines)
    total_h = sum(96 if (big_first and i == 0) else 52 for i in range(n)) + 26 * (n - 1)
    y = (H - total_h) // 2
    for i, line in enumerate(lines):
        size = 96 if (big_first and i == 0) else 52
        color = "0x60a5fa" if (big_first and i == 0) else "0xe2e8f0"
        safe = line.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
        draws.append(
            f"drawtext=fontfile={FONT}:text='{safe}':expansion=none:"
            f"fontsize={size}:fontcolor={color}:x=(w-text_w)/2:y={y}"
        )
        y += size + 26
    vf = ",".join(draws)
    run(["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=0x141c2f:s={W}x{H}:r={FPS}:d={dur}",
         "-vf", vf, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", path])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.exists(src):
        print(f"원본 영상이 없다: {src}", file=sys.stderr)
        print("경로를 인자로 넘겨라: scripts/build_demo.py <원본.mov>", file=sys.stderr)
        sys.exit(1)

    os.makedirs(TMP, exist_ok=True)

    clips = []

    print("[1/5] 타이틀 카드")
    p = f"{TMP}/00_title.mp4"
    card(p, TITLE, TITLE_DUR, True)
    clips.append((p, TITLE_DUR, None))

    print("[2/5] 구간 컷")
    for i, (start, dur, punch, caps) in enumerate(SEGMENTS):
        p = f"{TMP}/{i+1:02d}_seg.mp4"
        vf = (f"{PUNCH}," if punch else "") + \
             f"scale={W}:{H}:force_original_aspect_ratio=decrease," \
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:0x141c2f,fps={FPS},setsar=1"
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
             "-i", src, "-an", "-vf", vf, "-pix_fmt", "yuv420p",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", p])
        clips.append((p, dur, caps))
        print(f"      seg{i+1} {start}s +{dur}s {'(punch-in)' if punch else ''}")

    print("[3/5] 엔드 카드")
    p = f"{TMP}/99_end.mp4"
    card(p, END, END_DUR, False)
    clips.append((p, END_DUR, None))

    # ── 크로스페이드 체인 ──
    # xfade는 앞 클립 끝과 뒤 클립 앞을 겹친다. 겹치는 만큼 전체가 짧아지므로
    # offset을 누적 길이에서 XF만큼 뺀 값으로 준다.
    print("[4/5] 이어붙이기")
    inputs = []
    for p, _, _ in clips:
        inputs += ["-i", p]

    starts = [0.0]       # 최종 타임라인에서 각 클립이 시작하는 시각
    acc = clips[0][1]
    parts = []
    cur = "0:v"
    for i in range(1, len(clips)):
        offset = acc - XF
        starts.append(offset)
        out = f"v{i}"
        parts.append(f"[{cur}][{i}:v]xfade=transition=fade:duration={XF}:"
                     f"offset={offset:.3f}[{out}]")
        cur = out
        acc = acc + clips[i][1] - XF
    filt = ";".join(parts)

    joined = f"{TMP}/joined.mp4"
    run(["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", filt, "-map", f"[{cur}]",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", joined])
    print(f"      총 길이 {acc:.1f}초")

    # ── 자막 ──
    print("[5/5] 자막")

    def ts(t):
        h = int(t // 3600)
        m = int(t % 3600 // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    # 클립 인덱스 → 펀치인 여부 (타이틀 카드가 0번이라 1칸 밀린다)
    punch_of = {i + 1: seg[2] for i, seg in enumerate(SEGMENTS)}

    events = []
    for idx, ((p, dur, caps), st) in enumerate(zip(clips, starts)):
        if not caps:
            continue
        # 클립이 화면에 온전히 보이는 구간에만 띄운다. 앞뒤 크로스페이드 동안은 비운다.
        a, b = st + XF + 0.15, st + dur - XF - 0.15
        if b - a < 0.6:
            continue
        text = "\\N".join(caps)
        style = "SubTop" if punch_of.get(idx) else "Sub"
        events.append(f"Dialogue: 0,{ts(a)},{ts(b)},{style},,0,0,0,,{text}")

    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{FONT_NAME},46,&H00F1F5F9,&H00000000,&HB4120A08,0,0,0,0,100,100,0,0,3,12,0,2,90,90,52,1
Style: SubTop,{FONT_NAME},46,&H00F1F5F9,&H00000000,&HB4120A08,0,0,0,0,100,100,0,0,3,12,0,8,90,90,44,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(events) + "\n"

    ass_path = f"{OUT_DIR}/demo_subtitles.ass"
    with open(ass_path, "w") as f:
        f.write(ass)

    final = f"{OUT_DIR}/pitchiq_demo.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", joined,
         "-vf", f"subtitles={ass_path}:fontsdir=/System/Library/Fonts/Supplemental",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "slow",
         "-crf", "23", "-movflags", "+faststart", final])

    print(f"\n완료: {final}")
    print(f"자막:  {ass_path}")


if __name__ == "__main__":
    main()
