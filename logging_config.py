
import logging
from json_log_formatter import JSONFormatter


def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


llm_logger = setup_logger('''llm_logger''', '''llm_calls.log''')
