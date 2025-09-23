"""
Global Logger Module for iree-kernel-benchmark

This module provides a centralized logging system that can replace print statements
across all modules with configurable handlers. It allows for dynamic handler changes
that affect all future logging calls globally.

Usage:
    from kernel_bench.utils.logger import get_logger, set_global_handler

    # Get logger instance
    logger = get_logger(__name__)

    # Use like print
    logger.info("This is an info message")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error message")

    # Change global handler (affects all loggers)
    set_global_handler(my_custom_handler)
"""

import sys
import threading
from typing import Callable, Optional, Any
from datetime import datetime

from tqdm import tqdm


class GlobalLoggerManager:
    """
    Manages global logging configuration and handlers.

    This class maintains a single global handler that can be dynamically changed
    and will affect all logger instances across all modules.
    """

    def __init__(self):
        self._global_handler: Optional[Callable] = None
        self._lock = threading.Lock()
        self._original_print = print
        self._loggers = set()

        # Default handler that mimics print behavior
        self.set_handler(self._default_handler)

    def _default_handler(self, level: str, name: str, message: str, *args, **kwargs):
        """Default handler that mimics print behavior."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the message like print would
        if args:
            try:
                formatted_message = message % args
            except (TypeError, ValueError):
                formatted_message = f"{message} {' '.join(map(str, args))}"
        else:
            formatted_message = str(message)

        # Color coding for different levels
        color_codes = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset_code = "\033[0m"

        # Check if output supports colors (terminal)
        use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        if use_colors and level in color_codes:
            level_str = f"{color_codes[level]}{level}{reset_code}"
        else:
            level_str = level

        # Format: [TIMESTAMP] [LEVEL] [MODULE] MESSAGE
        header = f"[{level_str}] "
        whitespace = " " * len(header)
        message_lines = formatted_message.split("\n")
        aligned_message = "\n".join(
            [message_lines[0]] + [f"{whitespace}{line}" for line in message_lines[1:]]
        )
        output = f"{header}{aligned_message}"

        # Use original print to avoid recursion
        self._original_print(output, **kwargs)

    def set_handler(self, handler: Callable[[str, str, str], None]):
        """
        Set a global handler function that will be used by all loggers.

        Args:
            handler: A function that takes (level, module_name, message, *args, **kwargs)
        """
        with self._lock:
            self._global_handler = handler

    def get_handler(self) -> Callable:
        """Get the current global handler."""
        with self._lock:
            return self._global_handler

    def register_logger(self, logger):
        """Register a logger instance for management."""
        with self._lock:
            self._loggers.add(logger)

    def unregister_logger(self, logger):
        """Unregister a logger instance."""
        with self._lock:
            self._loggers.discard(logger)


# Global instance
_global_manager = GlobalLoggerManager()


class GlobalLogger:
    """
    A logger class that uses the global handler system.

    This logger provides standard logging levels (debug, info, warning, error, critical)
    and can be used as a drop-in replacement for print statements.
    """

    def __init__(self, name: str):
        self.name = name
        _global_manager.register_logger(self)

    def _log(self, level: str, message: Any, *args, **kwargs):
        """Internal logging method that delegates to the global handler."""
        handler = _global_manager.get_handler()
        if handler:
            handler(level, self.name, message, *args, **kwargs)

    def debug(self, message: Any, *args, **kwargs):
        """Log a debug message."""
        self._log("DEBUG", message, *args, **kwargs)

    def info(self, message: Any, *args, **kwargs):
        """Log an info message."""
        self._log("INFO", message, *args, **kwargs)

    def warning(self, message: Any, *args, **kwargs):
        """Log a warning message."""
        self._log("WARNING", message, *args, **kwargs)

    def warn(self, message: Any, *args, **kwargs):
        """Alias for warning."""
        self.warning(message, *args, **kwargs)

    def error(self, message: Any, *args, **kwargs):
        """Log an error message."""
        self._log("ERROR", message, *args, **kwargs)

    def critical(self, message: Any, *args, **kwargs):
        """Log a critical message."""
        self._log("CRITICAL", message, *args, **kwargs)

    def log(self, message: Any, *args, **kwargs):
        """General log method (defaults to INFO level)."""
        self.info(message, *args, **kwargs)

    # Convenience methods that act like print
    def print(self, *args, **kwargs):
        """Print-like method that logs at INFO level."""
        if args:
            message = " ".join(str(arg) for arg in args)
        else:
            message = ""
        self.info(message, **kwargs)

    def __call__(self, *args, **kwargs):
        """Make the logger callable like print."""
        self.print(*args, **kwargs)


def get_logger(name: str = __name__) -> GlobalLogger:
    """
    Get a logger instance for the given module name.

    Args:
        name: The name of the module/logger (typically __name__)

    Returns:
        A GlobalLogger instance
    """
    return GlobalLogger(name)


def set_global_handler(handler: Callable[[str, str, str], None]):
    """
    Set a global handler that will be used by all logger instances.

    This function allows you to globally change how all logging is handled
    across all modules in your application.

    Args:
        handler: A function that takes (level, module_name, message, *args, **kwargs)

    Example:
        def custom_handler(level, module_name, message, *args, **kwargs):
            with open('app.log', 'a') as f:
                f.write(f"{level}: {module_name}: {message}\n")

        set_global_handler(custom_handler)
    """
    _global_manager.set_handler(handler)


def set_tqdm_handler():
    def parallel_log_handler(level, module_name, message, *args, **kwargs):
        tqdm.write(f"[{level}] {module_name}: {message}")

    set_global_handler(parallel_log_handler)


def get_global_handler() -> Callable:
    """Get the current global handler function."""
    return _global_manager.get_handler()


def reset_to_default_handler():
    """Reset the global handler to the default handler."""
    _global_manager.set_handler(_global_manager._default_handler)


# Convenience function for quick access
def create_file_handler(filename: str) -> Callable:
    """
    Create a file handler that writes logs to a file.

    Args:
        filename: Path to the log file

    Returns:
        A handler function that can be used with set_global_handler
    """

    def file_handler(level: str, name: str, message: str, *args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if args:
            try:
                formatted_message = message % args
            except (TypeError, ValueError):
                formatted_message = f"{message} {' '.join(map(str, args))}"
        else:
            formatted_message = str(message)

        log_line = f"[{timestamp}] [{level}] [{name}] {formatted_message}\n"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(log_line)

    return file_handler


def create_combined_handler(*handlers: Callable) -> Callable:
    """
    Create a handler that calls multiple handlers.

    Args:
        *handlers: Multiple handler functions

    Returns:
        A combined handler function
    """

    def combined_handler(level: str, name: str, message: str, *args, **kwargs):
        for handler in handlers:
            try:
                handler(level, name, message, *args, **kwargs)
            except Exception as e:
                # Fallback to print if handler fails
                print(f"Handler error: {e}")
                print(f"[{level}] [{name}] {message}")

    return combined_handler


# Module-level convenience logger
_module_logger = get_logger(__name__)

# Export the main interface
__all__ = [
    "get_logger",
    "set_global_handler",
    "set_tqdm_handler",
    "get_global_handler",
    "reset_to_default_handler",
    "create_file_handler",
    "create_combined_handler",
    "GlobalLogger",
]
