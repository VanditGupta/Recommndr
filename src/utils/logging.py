"""Logging configuration for the Recommndr project."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

from config.settings import settings


def setup_logging(
    log_level: str = None,
    log_format: str = None,
    log_file: Path = None,
) -> None:
    """Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
        log_file: Optional log file path
    """
    # Use settings defaults if not provided
    log_level = log_level or settings.LOG_LEVEL
    log_format = log_format or settings.LOG_FORMAT
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Setup structlog configuration
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Rich text formatting for console
        console = Console()
        processors.append(structlog.dev.ConsoleRenderer(console=console))
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        if log_format == "json":
            file_handler.setFormatter(
                logging.Formatter('%(message)s')
            )
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    
    # Log the setup
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging setup complete",
        log_level=log_level,
        log_format=log_format,
        log_file=str(log_file) if log_file else None,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


def log_function_call(
    func_name: str,
    args: tuple = None,
    kwargs: dict = None,
    logger: structlog.BoundLogger = None,
) -> None:
    """Log function call details for debugging.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.debug(
        "Function call",
        function=func_name,
        args=args,
        kwargs=kwargs,
    )


def log_data_info(
    table_name: str,
    row_count: int,
    columns: list = None,
    sample_data: Any = None,
    logger: structlog.BoundLogger = None,
) -> None:
    """Log information about a dataset.
    
    Args:
        table_name: Name of the table/dataset
        row_count: Number of rows in the dataset
        columns: List of column names
        sample_data: Sample of the data for inspection
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(
        "Dataset information",
        table_name=table_name,
        row_count=row_count,
        columns=columns,
        sample_data=sample_data,
    )


def log_validation_result(
    validation_result: Dict[str, Any],
    logger: structlog.BoundLogger = None,
) -> None:
    """Log data validation results.
    
    Args:
        validation_result: Validation result dictionary
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(
        "Data validation completed",
        table_name=validation_result.get("table_name"),
        total_rows=validation_result.get("total_rows"),
        valid_rows=validation_result.get("valid_rows"),
        invalid_rows=validation_result.get("invalid_rows"),
        quality_score=validation_result.get("quality_score"),
        validation_errors=validation_result.get("validation_errors", []),
    )


def log_performance_metrics(
    operation: str,
    duration: float,
    row_count: int = None,
    memory_usage: float = None,
    logger: structlog.BoundLogger = None,
) -> None:
    """Log performance metrics for operations.
    
    Args:
        operation: Name of the operation performed
        duration: Duration in seconds
        row_count: Number of rows processed
        memory_usage: Memory usage in MB
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(
        "Performance metrics",
        operation=operation,
        duration_seconds=duration,
        rows_per_second=row_count / duration if row_count and duration > 0 else None,
        memory_usage_mb=memory_usage,
    )


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()
