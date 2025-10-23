import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from logging import Formatter, Logger, StreamHandler, getLogger
from typing import Any, Dict, Optional, List
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(logging.Formatter):
    AVAILABLE_FIELDS = ["timestamp", "level", "logger", "message", "file", "line", "function", "process", "thread", "exception"]
    INTERNAL_FIELDS = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
    }

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        include_extra: bool = True,
    ):
        super().__init__()
        self.fields: List[str] = fields if fields else self.AVAILABLE_FIELDS.copy()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        field_mapping = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
            "process": record.process,
            "process_name": record.processName,
            "thread": record.thread,
            "thread_name": record.threadName,
            "exception": self.formatException(record.exc_info) if record.exc_info else None,
        }

        log_data = self.filtrer_fields_from_log_record(field_mapping, record)

        return json.dumps(log_data, ensure_ascii=False)

    def filtrer_fields_from_log_record(self, field_mapping: dict[str, Any], record: logging.LogRecord) -> dict[str, Any]:
        available_data: Dict[str, Any] = {}
        record_as_dict: Dict[str, Any] = record.__dict__

        for field in self.fields:
            if field in field_mapping.keys():
                available_data[field] = field_mapping[field]
            else:
                if field in record_as_dict.keys():
                    available_data[field] = record_as_dict[field]
        fields = list(available_data.keys()) + [field for field in self.INTERNAL_FIELDS if field not in list(available_data.keys())]
        extra_fields = [extra_field for extra_field in record_as_dict.keys() if extra_field not in fields]
        for extra_field in extra_fields:
            available_data[extra_field] = record_as_dict[extra_field]
        return available_data


def init_json_logger(
    name: str,
    level: str = "INFO",
    fields: Optional[List[str]] = None,
    include_extra: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(
            fields=fields,
            include_extra=include_extra,
        )
    )
    logger.addHandler(handler)

    logger.propagate = False

    return logger


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        start_time = time.time()

        logger = logging.getLogger("api")
        logger.info(
            "Requête entrante",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            },
        )

        try:
            response = await call_next(request)

            duration = time.time() - start_time

            logger.info(
                "Requête complétée",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )

            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.exception(
                "Erreur lors du traitement de la requête",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            raise


class ColoredFormatter(Formatter):
    """Custom formatter with colors for different log levels"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def init_logger(name: str, level: str = "INFO") -> Logger:
    """Initialize a logger with colored output"""
    logger = getLogger(name=name)
    logger.setLevel(level=level)
    handler = StreamHandler(stream=sys.stdout)
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Prevent propagation to root logger

    return logger
