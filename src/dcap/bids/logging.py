# =============================================================================
#                              DCAP: Logging
# =============================================================================

import logging


def configure_logging(log_level: str) -> None:
    """
    Configure root logging.

    Usage example
        configure_logging("INFO")
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a module logger.

    Usage example
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
