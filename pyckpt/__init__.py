import logging
import sys
from typing import IO, Optional

logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())


def configure_logging(
    level=logging.INFO,
    format_str: Optional[str] = None,
    stream: IO = sys.stderr,
):
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(
        format_str or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
