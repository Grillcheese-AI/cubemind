"""GRL logging configuration via loguru.

Usage in experiments:
    from sandbox.grl_lib.log import logger, configure_experiment_log

    configure_experiment_log("he_moe", level="DEBUG")
    logger.info("H1 | step={} loss={:.4f}", step, loss)
    logger.success("H1 PASSED — kernel self-similarity = 1.0")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Remove default stderr handler — we'll configure our own
logger.remove()


def configure_experiment_log(
    experiment_name: str,
    level: str = "INFO",
    log_dir: str | Path | None = None,
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure loguru for an experiment.

    Args:
        experiment_name: Name for log file and format prefix.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. Defaults to sandbox/<experiment>/logs/.
        console: Enable console output.
        file: Enable file output.
    """
    logger.remove()  # Clear any prior config

    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{experiment_name}</cyan> | "
        "<level>{message}</level>"
    )

    if console:
        logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    if file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / experiment_name / "logs"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "{time:YYYY-MM-DD}.log",
            format=fmt,
            level=level,
            rotation="10 MB",
            retention="30 days",
        )
