import logging
from json_log_formatter import JSONFormatter
from opentelemetry import trace


class OtelSpanEventHandler(logging.Handler):
    """Log handler that emits log records as events on the current OTEL span."""
    def emit(self, record: logging.LogRecord) -> None:
        span = trace.get_current_span()
        if not span or not span.is_recording():
            return
        attributes = {
            "log.level": record.levelname,
            "log.message": self.format(record),
            "logger.name": record.name,
        }
        if record.exc_info:
            attributes["log.exception"] = self.formatException(record.exc_info)
        # Propagate any extra fields that were added to the record
        for key, value in record.__dict__.items():
            if key.startswith("_" ) or key in attributes:
                continue
            try:
                attributes[f"log.extra.{key}"] = value
            except Exception:
                pass
        span.add_event("log", attributes=attributes)


def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Mirror log records into the active OTEL span so they reach Langfuse
    otel_handler = OtelSpanEventHandler()
    logger.addHandler(otel_handler)

    return logger


llm_logger = setup_logger('''llm_logger''', '''llm_calls.log''')
