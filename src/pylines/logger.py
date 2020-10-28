# Imports

import threading
import os
import sys
import logging
from typing import Optional

_lock = threading.Lock()
_pylines_handler: Optional[logging.Handler] = None

class LogFormatter(logging.Formatter):

    COLOR_CODES = {
        logging.CRITICAL: "\033[38;5;196m", # bright/bold magenta
        logging.ERROR:    "\033[38;5;9m", # bright/bold red
        logging.WARNING:  "\033[38;5;11m", # bright/bold yellow
        logging.INFO:     "\033[38;5;111m", # white / light gray
        logging.DEBUG:    "\033[1;30m"  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"
    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if (self.color == True and record.levelno in self.COLOR_CODES):
            record.color_on  = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on  = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)

class PylinesLogger:
    def __init__(self, config):
        self.config = config
        self.logger = self.setup_logging()
    
    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        if self.config["console_log_output"] == "stdout":
            console_log_output = sys.stdout
        else:
            console_log_output = sys.stderr
        
        console_handler = logging.StreamHandler(console_log_output)
        console_handler.setLevel(self.config["console_log_level"].upper())
        console_formatter = LogFormatter(fmt=self.config["log_line_template"], color=self.config["console_log_color"])
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        return logger

    def get_logger(self):
        return self.logger

def _setup_library_root_logger(name):
    logger_config = {
        'console_log_output': "stdout", 
        'console_log_level': "info",
        'console_log_color': True,
        'logfile_file': None,
        'logfile_log_level': "debug",
        'logfile_log_color': False,
        'log_line_template': f"%(color_on)s[{name}] %(funcName)-5s%(color_off)s: %(message)s"
    }
    logger = PylinesLogger(logger_config)
    return logger.get_logger()


def _configure_library_root_logger(name="Pylines") -> None:
    global _pylines_handler
    with _lock:
        if _pylines_handler:
            return
        _pylines_handler = _setup_library_root_logger(name)
        _pylines_handler.propagate = False


def get_logger(name: Optional[str] = "Pylines") -> logging.Logger:
    """
    Return a logger with the specified name.
    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """
    if name is None:
        name = "Pylines"
    _configure_library_root_logger(name)
    return _pylines_handler