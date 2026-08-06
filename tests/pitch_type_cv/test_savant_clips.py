from pitch_type_cv.savant_clips import PitchClip, extract_clip_url, parse_pitches


def _feed(play_events: list[dict]) -> dict:
    return {"liveData": {"plays": {"allPlays": [{"playEvents": play_events}]}}}


def _pitch_event(play_id: str | None = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                 code: str | None = "FF") -> dict:
    event: dict = {"isPitch": True, "details": {"type": {"code": code} if code else {}}}
    if play_id is not None:
        event["playId"] = play_id
    return event


def test_parses_pitch_events_into_clips():
    clips = parse_pitches(_feed([_pitch_event(code="SL")]), game_pk=999999)
    assert clips == [
        PitchClip(
            play_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            pitch_type="SL",
            game_pk=999999,
        )
    ]


def test_skips_non_pitch_events():
    """마운드 방문·투수 교체 같은 이벤트에는 isPitch가 False다."""
    mound_visit = {"isPitch": False, "playId": "ffffffff-0000-0000-0000-000000000000"}
    clips = parse_pitches(_feed([mound_visit, _pitch_event()]), game_pk=1)
    assert [c.pitch_type for c in clips] == ["FF"]


def test_skips_pitches_without_play_id():
    """playId가 없으면 Savant 클립을 특정할 수 없어 쓸 수 없다."""
    clips = parse_pitches(_feed([_pitch_event(play_id=None)]), game_pk=1)
    assert clips == []


def test_skips_pitches_without_type_code():
    """구종 코드가 없으면 라벨이 없다 — 자동 볼 판정 등에서 발생한다."""
    clips = parse_pitches(_feed([_pitch_event(code=None)]), game_pk=1)
    assert clips == []


def test_preserves_pitch_order_across_plays():
    feed = {"liveData": {"plays": {"allPlays": [
        {"playEvents": [_pitch_event(play_id="p1", code="FF"),
                        _pitch_event(play_id="p2", code="CH")]},
        {"playEvents": [_pitch_event(play_id="p3", code="CU")]},
    ]}}}
    assert [c.play_id for c in parse_pitches(feed, game_pk=1)] == ["p1", "p2", "p3"]


def test_extracts_mp4_url_from_page():
    page = '<video><source src="https://sporty-clips.mlb.com/abc123==.mp4" type="video/mp4"></video>'
    assert extract_clip_url(page) == "https://sporty-clips.mlb.com/abc123==.mp4"


def test_extracts_mp4_url_from_html_escaped_page():
    """Savant는 URL을 HTML 이스케이프해 내보낸다 — 언이스케이프하지 않으면 &amp;가 남는다."""
    page = '<a href="https://sporty-clips.mlb.com/x.mp4?a=1&amp;b=2">clip</a>'
    assert extract_clip_url(page) == "https://sporty-clips.mlb.com/x.mp4?a=1&b=2"


def test_returns_none_when_page_has_no_mp4():
    """페이지 구조가 바뀌면 조용히 빈 궤적을 만드는 대신 None으로 드러나야 한다."""
    assert extract_clip_url("<html><body>Video not available</body></html>") is None
