import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)


def configure_logging(
    level=logging.INFO,
    format_str: Optional[str] = None,
):
    formatter = logging.Formatter(
        format_str or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(level)
