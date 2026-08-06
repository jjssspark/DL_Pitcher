"""
MLB StatsAPI 라이브 피드와 Baseball Savant 클립 페이지 파싱.

OCR 경로(dataset.py)와의 차이: 라벨을 중계 오버레이가 아니라 StatsAPI에서 직접 받는다.
OCR 판독률 85% 상한이 사라지고, 투구와 라벨이 playId로 1:1 묶여 순서 밀림이 없다.

네트워크 IO는 여기 두지 않는다 — 호출하는 스크립트가 담당한다.
"""
import html
import re
from dataclasses import dataclass

# Savant 페이지는 mp4 URL을 <source src>, JSON 블록 등 여러 형태로 싣는다.
# 경로가 base64 유사 문자열이라 '='가 들어가므로 문자 클래스에서 배제하면 안 된다.
_MP4_PATTERN = re.compile(r'https?://[^\s"\'<>]+?\.mp4[^\s"\'<>]*')


@dataclass(frozen=True)
class PitchClip:
    """한 투구 = 한 클립. play_id로 Savant 클립을, pitch_type으로 라벨을 얻는다."""

    play_id: str
    pitch_type: str  # Statcast 코드 (FF, SL, CH, ...)
    game_pk: int


def parse_pitches(feed: dict, game_pk: int) -> list[PitchClip]:
    """
    StatsAPI feed/live 응답에서 투구 이벤트를 순서대로 뽑는다.

    playId나 구종 코드가 없는 이벤트는 제외한다 — 전자는 클립을 특정할 수 없고
    후자는 라벨이 없다. 실측(후보 7경기)에서는 투구 이벤트 전건이 둘 다 갖고 있었지만,
    자동 볼 판정이나 데이터 지연 시 비는 경우가 있어 방어한다.
    """
    clips = []
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            play_id = event.get("playId")
            pitch_type = event.get("details", {}).get("type", {}).get("code")
            if not play_id or not pitch_type:
                continue
            clips.append(PitchClip(play_id=play_id, pitch_type=pitch_type, game_pk=game_pk))
    return clips


def extract_clip_url(page_html: str) -> str | None:
    """
    Savant sporty-videos 페이지에서 mp4 URL을 찾는다. 없으면 None.

    HTML 엔티티를 먼저 풀어야 쿼리스트링의 &amp;가 그대로 남지 않는다.
    """
    match = _MP4_PATTERN.search(html.unescape(page_html))
    return match.group(0) if match else None


def statsapi_feed_url(game_pk: int) -> str:
    return f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def savant_clip_page_url(play_id: str) -> str:
    return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
