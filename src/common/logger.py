"""Common Logging functions wrapper module"""
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Configurations
console = Console(width=140)
install()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True, console=console)],
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance with specified name.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Logger with specified name
    """
    return logging.getLogger(name)
