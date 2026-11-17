"""
    Code to handle the log and different errors that may appear.
    This version is adapted to create unique logger instances for specific output directories,
    forming the basis of an automated reporting pipeline.

    Created by: Eric March Vila (eric.march@upf.edu)
    On: 25/11/2022, 12:06 PM
    Updated: 2025-10-10
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

class QuietLogger:
    """A logger that does absolutely nothing."""
    def info(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): pass
    def critical(self, msg, *args, **kwargs): pass
    def debug(self, msg, *args, **kwargs): pass
    def exception(self, msg, *args, **kwargs): pass

def LOG(name: str, log_file_path: str, quiet: bool = False) -> logging.Logger:
    """
    Creates or gets a logger instance configured to write to a specific file.

    This function is designed to be called at the start of a process for a given
    dataset. It resets any existing handlers on the logger to ensure that
    each run logs to its own dedicated file.

    :param name: Unique name for the logger (e.g., the dataset name).
    :param log_file_path: The full path to the log file where records will be saved.
    :param quiet: If True, returns a QuietLogger instance that does nothing.
    :return logger: A configured logger instance or QuietLogger.
    """

    if quiet:
        return QuietLogger()

    # Get the directory part of the log file path.
    output_dir = os.path.dirname(log_file_path)
    # If output_dir is not an empty string, a directory was specified.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # If the logger already has handlers, remove them to reconfigure it for the new file path
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # --- File Handler ---
    # More detailed logs saved to the specific output file
    file_formatter = logging.Formatter(
        '%(levelname)-8s [%(asctime)s] %(thread)d - %(name)s.%(funcName)s - %(message)s'
    )
    fh = RotatingFileHandler(log_file_path, maxBytes=2000000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # --- Console (Stream) Handler ---
    # Less verbose logs for the user to see on the console
    stdout_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(stdout_formatter)
    logger.addHandler(ch)

    logger.info(f"Logger for '{name}' initialized. Logging to: {log_file_path}")
    return logger