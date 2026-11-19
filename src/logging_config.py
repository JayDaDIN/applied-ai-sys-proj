"""Logging setup for the recommender CLI.

Two destinations:

* **Console (stderr):** WARNING and above by default, so recommendations on
  stdout stay readable. ``--verbose`` lowers it to INFO and ``--debug`` to
  DEBUG.
* **Log file:** ``logs/recommender.log`` records INFO and above for every run
  (DEBUG with ``--debug``), including full tracebacks for unexpected errors.
  It rotates at about 1 MB and keeps 3 old files.

Modules get their logger with ``logging.getLogger(__name__)`` and never
configure handlers themselves; only the CLI entry point calls ``setup_logging``.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(levelname)s: %(message)s"
DEFAULT_LOG_FILE = Path("logs") / "recommender.log"


class ShortConsoleFormatter(logging.Formatter):
    """Console formatter that leaves out tracebacks (they go to the log file)."""

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info or record.exc_text:
            record = logging.makeLogRecord(record.__dict__)
            record.exc_info = None
            record.exc_text = None
        return super().format(record)


def setup_logging(
    console_level: int = logging.WARNING,
    log_file: Optional[Path] = DEFAULT_LOG_FILE,
    file_level: int = logging.INFO,
) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(min(console_level, file_level))

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(ShortConsoleFormatter(CONSOLE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
        except OSError as exc:
            # Logging must never stop the program; keep the console handler.
            root.warning("Could not open log file %s (%s); logging to console only.", log_file, exc)
        else:
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(file_handler)

    # The HTTP stack is noisy at INFO; keep it to warnings unless debugging.
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if console_level <= logging.DEBUG else logging.WARNING
        )
