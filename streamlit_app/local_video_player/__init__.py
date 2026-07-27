import os
import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")
_component = components.declare_component("local_video_player", path=_FRONTEND)


def local_video_player(
    video_url: str,
    seek_to: float = None,
    is_playing: bool = False,
    key: str = None,
) -> dict | None:
    """{time: float, playing: bool} 반환."""
    return _component(
        video_url=video_url,
        seek_to=seek_to,
        is_playing=is_playing,
        key=key,
        default=None,
    )
