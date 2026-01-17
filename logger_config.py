import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """Configures the shared logger for the project.
    
    Args:
        level: The logging level to use (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance for a specific module.
    
    Args:
        name: The name of the module, usually __name__.
        
    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)
