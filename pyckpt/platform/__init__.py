import sys

if sys.platform == "darwin":
    from .darwin import *  # noqa: F403
elif sys.platform == "linux":
    pass
else:
    raise ImportError(f"unsupported platform: {sys.platform}")
