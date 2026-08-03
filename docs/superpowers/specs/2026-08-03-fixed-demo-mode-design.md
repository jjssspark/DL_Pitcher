# PitchIQ 고정 데모 모드 전환 — 설계

## 배경

지금 앱은 방문자가 game_pk와 YouTube URL을 직접 입력해야 동작하는 범용 도구다. 실제로는
`game_pk=775300`(2024 월드시리즈 1차전, 2024-10-25, LAD 홈 vs NYY 원정, 320개 투구) +
`https://youtu.be/gMm3EODDb6w` 조합만 검증됐다. 포트폴리오 방문자가 아무 입력 없이 바로
결과를 보게 하기 위해, 이 조합을 하드코딩하고 페이지 로드 시 자동으로 재생+예측이 시작되게 바꾼다.

## 배포 환경 제약 (중요)

Streamlit Community Cloud는 앱의 기본 포트만 외부에 노출한다. 지금 `local_video_player` +
자체 `http.server`(포트 8510~8530) 조합은 `localhost:{port}`를 가리키는데, 이 `localhost`는
**방문자 브라우저 기준**이라 배포 환경에서는 항상 깨진다(방문자 자신의 컴퓨터를 찾으려다 실패).
따라서 배포판은 반드시 `youtube_player`(iframe 임베드, YouTube CDN에서 직접 재생)만 써야 한다.

같은 이유로 yt-dlp 서버측 다운로드 + 실시간 OCR 스캔도 배포 환경에서 하지 않는다 — 클라우드
IP는 YouTube가 봇으로 간주해 차단하는 경우가 잦아 신뢰성이 낮다. 대신 로컬에서 미리 스캔한
결과를 정적 파일로 커밋해 배포판은 그 파일만 읽는다.

## 목표

1. 페이지 로드 시 자동으로 game_pk=775300 로드 + BiLSTM 계산 시작
2. 영상은 항상 YouTube iframe(`youtube_player`)으로 재생 — 로컬 다운로드/서빙 경로 배제
3. 투구 타이밍 동기화는 로컬에서 미리 만든 정적 캐시 파일로 대체 (실시간 OCR/다운로드 없음)
4. 기존 "다른 경기 수동 입력" 경로는 유지하되 기본 화면에서는 숨김(고급 옵션으로 격하)

## 비범위

- BiLSTM/YOLO 모델 로직 변경 없음
- 랜딩 페이지 문구/디자인 개편은 이번 스펙에서 제외 (다음 라운드에서 "UI가 별로다" 피드백과
  함께 별도로 다룸)
- 기존 수동 game_pk/영상 URL 입력 기능 완전 삭제 안 함 (숨김 처리만)

## A. 고정 데모 데이터 주입

`_DEFAULTS` 딕셔너리(app.py) 초기화 이후, 세션이 아직 아무 게임도 로드하지 않은 최초 접속
시점에 다음을 자동 수행한다 (기존 "경기 로드" 버튼 클릭 시 로직과 동일한 절차를 최초 1회
자동으로 트리거):

- `game_pk = "775300"` 설정 → `fetch_game_pitches(775300)` 호출 → `game_pitches`/`game_meta` 채움
- `video_src = "https://youtu.be/gMm3EODDb6w"` 설정
- BiLSTM 백그라운드 계산 트리거 (기존 `_run_bilstm_bg`와 동일)
- 트리거 조건: `st.session_state.game_pk == ""` (한 번도 로드된 적 없음) 이고 session 최초 진입일 때만 —
  이미 로드된 세션이나 사용자가 "초기화"를 누른 뒤에는 재자동로드하지 않는다 (초기화는 여전히 빈 랜딩으로 돌아감).

## B. 재생 경로를 iframe으로 고정

현재 `_use_local_player = bool(_local_play_path and os.path.exists(_local_play_path))` 분기가
로컬 파일이 있으면 무조건 `local_video_player`를 우선 사용하게 되어 있다. 배포 환경에서는
애초에 로컬 파일 다운로드 자체를 시도하지 않으므로(아래 C), `_local_video_path`가 항상 비어있어
자연히 `youtube_player` 경로로 빠진다. 별도의 배포/로컬 분기 플래그는 추가하지 않고, "C. 정적
캐시 사용 시 다운로드를 트리거하지 않는다"는 조건만으로 자연스럽게 iframe 경로가 선택되게 한다.

## C. 정적 스캔 캐시

1. 로컬에서 1회: `https://youtu.be/gMm3EODDb6w` 다운로드 → 기존 `scan_pitch_overlays()`
   (1~5이닝 범위, 기존 관례 그대로) 실행 → `pitch_times`, `pitch_data` 획득
2. 결과를 `streamlit_app/fixed_demo_scan.json`으로 저장 (형식: 기존 `_scan_cache_path`가 쓰는
   `{"version": ..., "pitch_times": [...], "pitch_data": [...]}`와 동일한 스키마)
3. 이 파일은 `.gitignore`의 `streamlit_app/.scan_cache/` 규칙과 무관한 별도 경로이므로 정상적으로
   커밋된다.
4. 앱 시작 시 (game_pk=775300이 방금 자동 로드된 직후): `_local_video_path`를 설정하지 않고
   대신 `video_pitch_times`/`_scan_raw_data`를 `fixed_demo_scan.json`에서 직접 읽어 채운다.
   `_scan_status = "done"`으로 표시해 기존 "타임스탬프 기반 자동싱크" 로직(app.py, `_vid_t` 기반
   비교 블록)이 그대로 작동하게 한다.

## D. 수동 입력 경로 유지

사이드바의 game_pk/영상 URL 입력 필드와 "초기화" 버튼은 그대로 둔다. 자동 주입은 최초 진입
시에만 일어나고, 사용자가 직접 다른 game_pk를 입력하면 기존 흐름(수동 다운로드+실시간 스캔)이
정상 동작한다 — 이건 로컬 개발/테스트용으로는 여전히 유효하다.

## 검증

- 로컬: 세션을 완전히 새로 시작했을 때 자동으로 775300 로드 + 영상 iframe 재생 + 예측 갱신 확인
- 배포: `fixed_demo_scan.json` 커밋 후 재배포, 방문자 링크 클릭 시 입력 없이 바로 재생+예측되는지 확인
- 회귀: 사이드바에서 다른 game_pk를 수동 입력하는 기존 흐름이 깨지지 않았는지 확인
