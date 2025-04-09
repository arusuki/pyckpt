import sys

if sys.platform == "darwin":
    from .darwin import *  # noqa: F403
else:
    raise ImportError(f"unsupported platform: {sys.platform}")
