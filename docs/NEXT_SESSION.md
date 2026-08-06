# 다음 세션 착수 문서 — CV 구종 분류 파일럿

마지막 갱신: 2026-08-06 · 마지막 커밋: `515efb6`

---

## 붙여넣을 프롬프트

```
CV 구종 분류 파일럿을 이어서 한다. docs/NEXT_SESSION.md를 먼저 읽고,
docs/TROUBLESHOOTING.md의 TS-018(감지 임계값 무효)과 TS-019(정지 오탐 4번째 재발)를
확인해라.

지난 세션 결론: 감지기 문제를 해결했다. Roboflow 중계 도메인 데이터셋으로
파인튜닝(models/ball_broadcast_v1.pt)하고, imgsz 960 + conf 0.05로 추론하며,
longest_moving_chain으로 정지 오탐을 걸러낸다. 감지 게이트 7/8(88%) 통과.
분류 성능은 0.612 vs 최빈값 기준선 0.510, 순열검정 p=0.017로 유의하다.

남은 한계 3개가 전부 "1경기뿐"에서 온다:
  (1) game_pk 홀드아웃 불가 -> 현재 수치는 누수 허용 상한
  (2) 선택 편향 - 궤적 확보율이 FASTBALL 63% / BREAKING 72% / OFFSPEED 87%
  (3) 홀드아웃 49개, OFFSPEED는 4개뿐

오늘 할 일: 경기 수를 늘려 위 3개를 동시에 푼다.
1. 데이터 수집 경로를 정한다 — OCR 방식 확장이냐, MLB StatsAPI + Baseball Savant
   투구별 클립으로 전환이냐. 아래 "두 갈래" 절을 읽고 나에게 추천안을 말해라.
2. 3경기 이상으로 dataset.csv 재생성
3. game_pk 단위 홀드아웃으로 재측정. 특징 중요도와 순열검정을 반드시 함께 본다
   — 정확도만 보면 안 된다 (TS-014)
4. 선택 편향이 경기가 늘어도 남는지 확인

플랜 먼저 세우고 확인받은 뒤 실행해라.
```

---

## 지금 상태

| 항목 | 상태 |
|---|---|
| 공 감지 | **해결** — 게이트 7/8(88%), 사슬 중앙값 7프레임, 박스 21px |
| 궤적 품질 | **해결** — 너클커브 낙차 가속, 싱커 단조 낙하 (물리적으로 타당) |
| 분류 성능 | 0.612 vs 기준선 0.510 (+10.2%p), 5-fold CV 0.664 ± 0.077 |
| 유의성 | 순열검정 1000회 **p = 0.017** |
| 특징 중요도 | 최대/최소 비 **4.95** (TS-014 당시 1.23 = 평평 = 무신호) |
| 데이터 규모 | **1경기 161샘플** ← 남은 병목 |

## 감지 파이프라인 (확정 설정)

```
모델      models/ball_broadcast_v1.pt
          Roboflow pitchtracking/baseball-detection-2 v4, yolov8n 33에폭, mAP50 0.960
          가중치는 .gitignore의 *.pt에 걸려 리포에 없다. 재학습:
            venv/bin/python3 src/train_yolo.py --workspace pitchtracking \
              --project baseball-detection-2 --version 4 \
              --data-dir data/raw/baseball-detection-2 \
              --project-dir runs --name ball_broadcast --epochs 33

추론      imgsz=960, conf=0.05   <- 둘 다 필수. 이유는 TS-018
궤적      longest_moving_chain(candidates, max_jump_px=60, min_total_move_px=30)
윈도우    t-3.0s ~ t-0.3s, 30fps 샘플링
```

**imgsz와 conf를 바꾸면 안 되는 이유**: 640은 공이 너무 작고 1280은 학습 시점(640 stretch)
대비 과대하다. conf는 Ultralytics 기본값이 0.25인데 이 도메인의 공은 그 아래로 내려간다.
`yolo_detector.CONF_THRESHOLD`는 무효이므로 **반드시 모델에 직접 넘겨야 한다**.

## 남은 한계 — 전부 "1경기"에서 온다

1. **누수 허용 상한**. `game_pk`가 하나뿐이라 경기 단위 홀드아웃이 불가능하고, 노트북이
   투구 단위 층화 분할로 폴백한다. 같은 투수의 투구가 학습·홀드아웃 양쪽에 들어간다.
2. **선택 편향**. 궤적 확보율이 균일하지 않다.

   ```
   FASTBALL  103 -> 65  (63.1%)
   BREAKING  116 -> 83  (71.6%)
   OFFSPEED   15 -> 13  (86.7%)
   ```

   빠른 공일수록 프레임 간 이동이 커서 사슬이 끊긴다. 현재 수치는 전체 투구가 아니라
   "궤적이 잡히는 68%"를 대표한다.
3. **표본 부족**. 홀드아웃 49개, OFFSPEED는 4개라 f1이 0.00이다.

## 두 갈래 — 데이터 수집 경로

### (A) 기존 OCR 방식 확장

`scripts/build_pitch_group_dataset.py`의 `GAME_LIST`에 주석 처리된 6경기가 이미 들어 있다.
주석만 풀면 된다. 전부 공식 MLB 채널 FULL GAME 업로드이고 game_pk는 실측 대조를 마쳤다.

- 장점: 코드 변경 없음. 파이프라인이 이미 검증됨
- 단점: 경기당 720p 다운로드 + OCR 전체 스캔이 오래 걸린다 (실측 2시간대). 6경기면 반나절.
  OCR 판독률 85%가 상한이라 나머지 15%는 라벨을 못 얻는다

### (B) MLB StatsAPI + Baseball Savant 투구별 클립

지난 세션에 검증만 하고 적용하지 않은 경로다. **OCR·타임스탬프·윈도우 캘리브레이션이 전부 사라진다.**

```python
# 280/280 투구에 playId + 구종명. OCR 불필요.
https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
  -> liveData.plays.allPlays[].playEvents[] 에서 isPitch=True, playId 보유

# 투구 1개 = 클립 1개. 1280x720, 59.6fps, 6.8초, 5MB. 릴리스가 항상 3.4초 부근.
https://baseballsavant.mlb.com/sporty-videos?playId={playId}
  -> 페이지의 .mp4 링크를 html.unescape 후 GET (Referer 헤더 필수, 없으면 403)
```

- 장점: 라벨 100%(OCR 85% 한계 없음). 클립마다 릴리스 시점이 3.4초로 고정이라 **위상 정렬 문제도
  해결**된다. 경기당 다운로드가 5MB × 투구수라 720p 풀경기(1.2GB)보다 가볍다
- 단점: 다운로더를 새로 만들어야 한다. `resolve_video`/`scan_overlays` 인터페이스를 갈아끼워야 함

**추천**: (B). (A)는 반나절을 써도 OCR 85% 한계와 위상 misalignment가 그대로 남는다. (B)는
구현에 시간이 들지만 세 문제를 한 번에 없앤다. 다만 파일럿을 빨리 닫는 게 목적이면 (A)로
3경기만 돌려도 `game_pk` 홀드아웃은 성립한다.

## 위상 정렬 문제 (미해결, (B) 택하면 자동 해결)

OCR 타임스탬프와 실제 투구 순간의 간격이 투구마다 다르다. 같은 f45 프레임인데 어떤 투구는
투수 딜리버리 중이고 어떤 투구는 타자가 이미 스윙 중이다. 그래서 감지된 구간이 투구마다
비행의 서로 다른 부분(초반/중반/종반)에 해당한다.

`apparent_speed_px_per_frame`이 최상위 특징(0.301)인데, 원근 압축 때문에 릴리스 직후 구간이
잡히면 빠르게, 플레이트 근처가 잡히면 느리게 나온다. **노이즈가 아니라 체계적 왜곡이다.**
지금 성능이 유의하게 나온 건 이 왜곡에도 불구하고 신호가 있다는 뜻이므로, 정렬하면 더 오를
여지가 있다.

## 재사용 가능한 것

- `output/pitch_type_cv/ocr_cache/` — OCR 스캔 결과 (재스캔 2시간 절약)
- `output/pitch_type_cv/trajectory_cache/*.jsonl` — 프레임별 감지 후보 전체.
  사슬 임계값을 바꿔도 영상 재처리 없이 재계산 가능
- `*.coco.bak.*` — TS-014 이전 COCO 감지기 산출물. 비교용으로만 남겨둠, 쓰면 안 됨
- `scripts/verify_ball_detector.py` — 감지기를 바꿀 때마다 돌린다.
  `--weights`, `--imgsz`, `--conf`, `--tag`로 조건 비교
- 노트북의 특징 중요도 / 순열검정 / 선택 편향 셀 — 경기가 늘어도 그대로 쓴다

## 판정 규칙 (바꾸지 말 것)

- **기준선은 최빈값이다.** 33% 랜덤 아님. 클래스 불균형 때문에 아무것도 학습하지 않아도
  50%가 나온다
- **정확도만 보지 않는다.** 특징 중요도가 평평하면(최대/최소 비가 1에 가까우면) 정확도와
  무관하게 무신호다. TS-014에서 이 지표만이 유일한 단서였다
- **게이트 통과 후에도 궤적 좌표를 눈으로 본다.** TS-019에서 7/8 통과한 사슬의 절반이
  정지 오탐이었다. 실패는 알아서 눈에 띄지만 성공은 그렇지 않다

## 건드리면 안 되는 것

- `runs/` — YOLO 자동 생성
- `models/`, `*.pt` — 재학습 비용이 크다. 삭제 전 반드시 확인
- `src/pose_detector.py`, `src/feature_engineering.py` — 설계 문서상 비범위
  (기존 데모 파이프라인에 영향). 수정은 `src/pitch_type_cv/` 안에서만
- `src/yolo_detector.py` — 원칙적으로 비범위. TS-018 때 동작 불변인 선택적 인자(`conf`)만
  추가했다. 추가 수정이 필요하면 데모 경로 영향을 먼저 확인할 것
