"""Stage 2 pipeline package.

Keep package import side effects minimal so lightweight unit tests can import
submodules without pulling in database/runtime dependencies.
"""

__all__ = [
    "tiler",
    "change_filter",
    "scene_processor",
    "event_engine",
    "stac_watcher",
]
