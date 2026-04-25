"""
utils/logger.py — Structured Logging for SentinelFraud
Provides a single, consistently-configured logger used across all modules.
Writes human-readable logs to stdout and machine-readable JSON lines to disk
(mimicking what a real bank would ship to a SIEM like Splunk or Datadog).
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import LOGS_DIR, LOG_FILE, FLAGGED_LOG

# Ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class _JSONLineFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON object on its own line.
    This format is trivially ingestible by any log aggregation system.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "module":    record.module,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "sentinel") -> logging.Logger:
    """
    Returns a configured logger.  Call this once per module:
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Rotating file handler
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JSONLineFormatter())

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


# Flagged-transaction audit trail
def log_flagged_transaction(record: dict) -> None:
    """
    Appends a flagged/blocked transaction to the immutable audit log (JSONL).
    In production this would write to an append-only Kafka topic or S3 bucket.
    """
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(FLAGGED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
