"""
Logging utility for biomed-kg-agent.
"""

import logging
from pathlib import Path

from .config import settings


def setup_logger(
    name: str = "biomed_kg_agent",
    level: int = logging.INFO,
    log_dir: Path = settings.LOG_DIR,  # Use settings.LOG_DIR
    log_file_name: str = "biomed_kg_agent.log",
) -> logging.Logger:
    """
    Set up comprehensive logging for all biomed_kg_agent modules.

    Configures both console and file logging for the entire package,
    ensuring progress from all modules (NLP, entity linking, etc.) is visible.
    """
    # Configure the root biomed_kg_agent logger to capture all submodules
    # Python's logging hierarchy: "biomed_kg_agent.cli" automatically
    # propagates to "biomed_kg_agent"
    root_logger = logging.getLogger("biomed_kg_agent")
    root_logger.setLevel(level)

    # Get the specific logger requested
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only configure handlers once for the root logger
    if not root_logger.handlers:
        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler()
        console_fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        console_formatter = logging.Formatter(console_fmt)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

        # File handler (same format)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / log_file_name
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(console_formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)

            # Log setup completion
            root_logger.info("Logging configured for all biomed_kg_agent modules")
            root_logger.info(f"Log file: {log_file_path}")
            root_logger.info(f"Console logging: {level}")

    # Ensure all biomed_kg_agent submodule logs propagate to root
    if name.startswith("biomed_kg_agent"):
        logger.propagate = True

    return logger
