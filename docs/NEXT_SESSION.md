# 다음 세션 착수 문서 — CV 구종 분류 파일럿

마지막 갱신: 2026-08-05 · 마지막 커밋: `f2c1a62`

---

## 붙여넣을 프롬프트

```
CV 구종 분류 파일럿을 이어서 한다. docs/NEXT_SESSION.md와 docs/TROUBLESHOOTING.md의
TS-013·TS-014를 먼저 읽어라.

지난 세션 결론: YOLO가 야구공을 한 번도 잡은 적이 없었다(감지 713건 중 공 크기 0건,
전부 투수 글러브). 해결책으로 Roboflow의 pitchtracking/baseball-detection-2를 쓰기로
확정했다 — MLB 중계 센터필드 앵글, 클래스 baseball 1개, CC BY 4.0, v4 4,153장.

오늘 할 일:
1. .env의 ROBOFLOW_API_KEY로 데이터셋 v4 다운로드 (없으면 나에게 요청)
2. yolov8n에서 파인튜닝. 학습 산출물은 runs/ 아래 자동 생성되므로 손대지 말 것
3. 학습된 모델을 실제 경기 영상에 적용해 감지 박스를 렌더링하고 눈으로 확인.
   이게 통과 기준이다 — mAP 숫자만 보고 넘어가지 말 것 (TS-014에서 mAP 0.852짜리
   모델이 실전 감지율 3%였다)
4. 통과하면 dataset.csv 재생성 후 분류 성능 재측정

플랜 먼저 세우고 확인받은 뒤 실행해라.
```

---

## 지금 상태

| 항목 | 상태 |
|---|---|
| OCR 오버레이 판독 | 동작 (237/280 = 85%), 캐시됨 |
| 라벨 출처 | OCR 판독 구종 (Statcast 페어링 폐기) |
| 궤적 윈도우 `t-3.0s ~ t-0.3s` | **검증 완료** — 실제 투구 순간을 담고 있음 |
| 공 감지 | **실패** — 이것만 남았다 |
| 분류 성능 | RF 5-fold 0.517 vs 최빈값 기준선 0.496 (사실상 무신호) |

## 왜 막혔나 (요약, 상세는 TS-014)

```
COCO yolov8n                  감지 713건 · 공 크기(<=12px) 0건 · 중앙값 49px(글러브)
models/baseball_detector.pt   감지율 3% · 학습셋이 사인볼·리틀리그 사진 (도메인 불일치)
imgsz 640 -> 1920             감지 건수만 증가, 공은 여전히 0건
프레임 차분                    프레임당 후보 132개 (선수 몸이 blob으로 쪼개짐)
```

## 정한 방향

**Roboflow `pitchtracking/baseball-detection-2`로 파인튜닝.**
https://universe.roboflow.com/pitchtracking/baseball-detection-2

- 라벨된 이미지를 육안 확인함 — MLB 중계 센터필드 앵글, 포수 미트 도착 시점의 공에 소형 박스
- 클래스 `baseball` 1개 · CC BY 4.0 · v4 = 4,153장 (train 3,418 / valid 487 / test 248)
- 전처리: Auto-Orient + **640x640 stretch** + Adaptive Equalization

**주의**: 640x640 stretch는 1280x720 원본의 가로를 눌러 공을 더 작게 만든다. 학습과
추론에 같은 전처리를 적용하면 일관되지만 이게 성능 상한이 된다. 부족하면 Fork Dataset으로
1280 유지 버전을 다시 생성한다.

**막힌 것**: 다운로드에 Roboflow API 키 필요 → 계정 가입은 사용자가 직접 해야 한다.
발급 후 `.env`에 `ROBOFLOW_API_KEY=`로 넣는다.

## 감지기가 해결되면 바로 쓸 수 있는 것

**MLB StatsAPI + Baseball Savant 투구별 클립** (검증 완료, 미적용)

```python
# 280/280 투구에 playId + 구종명. OCR 불필요.
https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
  -> liveData.plays.allPlays[].playEvents[] 에서 isPitch=True, playId 보유

# 투구 1개 = 클립 1개. 1280x720, 59.6fps, 6.8초, 5MB. 릴리스가 항상 3.4초 부근.
https://baseballsavant.mlb.com/sporty-videos?playId={playId}
  -> 페이지의 .mp4 링크를 html.unescape 후 GET (Referer 헤더 필수, 없으면 403)
```

이걸 쓰면 OCR 판독(85% 한계) · Statcast 순서 페어링 · 윈도우 캘리브레이션이 전부 사라진다.

## 재사용 가능한 것

- `output/pitch_type_cv/ocr_cache/` — OCR 스캔 결과 (재스캔 2시간 절약)
- `output/pitch_type_cv/trajectory_cache/*.jsonl` — 원시 `(x, y, conf)`. 감지기를 바꾸면
  무효화되지만 필터 임계값 재실험에는 그대로 쓸 수 있다
- `trajectory_features.longest_smooth_run()` — 감지기가 정확해지면 여전히 유효
- `detect_ball_in_frame`은 이미 커스텀 모델을 받도록 준비됨 (`CONF_THRESHOLD=0.15`,
  `BALL_CLASS_NAMES=None`) — 호출부에 경로만 넘기면 된다

## 고쳐야 할 것

`notebooks/01_pitch_group_classifier.ipynb`의 성공 기준이 "33% 랜덤 베이스라인 상회"인데,
클래스가 115/103/14로 불균형이라 아무것도 배우지 않아도 49.6%가 나온다. **최빈값 기준선으로
교체해야 한다.** 안 그러면 51.7%를 성공으로 오판한다.

또 1경기만으로는 `game_pk` 단위 홀드아웃이 불가능해 노트북이 그대로는 돌지 않는다.

## 건드리면 안 되는 것

- `runs/` — YOLO 자동 생성
- `models/`, `*.pt` — 재학습 비용이 크다. 삭제 전 반드시 확인
- `src/yolo_detector.py`, `src/pose_detector.py`, `src/feature_engineering.py` —
  설계 문서상 비범위 (기존 데모 파이프라인에 영향). 수정은 `src/pitch_type_cv/` 안에서만
