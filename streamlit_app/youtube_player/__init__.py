import os
import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")
_component = components.declare_component("youtube_player", path=_FRONTEND)


def youtube_player(
    video_id: str,
    start_sec: float = 0,
    seek_to: float = None,
    is_playing: bool = False,
    pitch_times: list = None,
    key: str = None,
) -> dict | None:
    """YouTube 플레이어. {time: float, playing: bool, pitch_idx: int} 반환."""
    return _component(
        video_id=video_id,
        start_sec=start_sec,
        seek_to=seek_to,
        is_playing=is_playing,
        pitch_times=pitch_times or [],
        key=key,
        default=None,
    )
