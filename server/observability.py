from __future__ import annotations

import atexit
import io
import logging
import re
import sys
import threading
from typing import TextIO

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\x1b[0m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"


class CollapsingConsole:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        self._last_key: object | None = None
        self._last_message: str | None = None
        self._repeat_count = 0
        self._last_render_width = 0
        self._line_open = False
        atexit.register(self.close)

    def write(self, message: str, *, collapse_key: object | None = None) -> None:
        with self._lock:
            key = message if collapse_key is None else collapse_key
            if self._last_key == key:
                self._last_message = message
                self._repeat_count += 1
                self._render_current()
                return

            self._finish_current_line()
            self._last_key = key
            self._last_message = message
            self._repeat_count = 1
            self._render_current()

    def close(self) -> None:
        with self._lock:
            self._finish_current_line()

    def _render_current(self) -> None:
        assert self._last_message is not None
        rendered = self._last_message
        if self._repeat_count > 1:
            suffix = f"(x{self._repeat_count})"
            if self._supports_color():
                suffix = f"{GREEN}{suffix}{RESET}"
            rendered = f"{rendered} {suffix}"

        visible_width = len(ANSI_RE.sub("", rendered))
        clear_padding = max(0, self._last_render_width - visible_width)
        self._stream.write("\r" + rendered + (" " * clear_padding))
        self._stream.flush()
        self._last_render_width = visible_width
        self._line_open = True

    def _finish_current_line(self) -> None:
        if not self._line_open:
            return

        self._stream.write("\n")
        self._stream.flush()
        self._line_open = False
        self._last_render_width = 0

    def _supports_color(self) -> bool:
        is_tty = getattr(self._stream, "isatty", lambda: False)()
        return bool(is_tty)


class CollapsingLogHandler(logging.Handler):
    def __init__(self, console: CollapsingConsole | None = None) -> None:
        super().__init__()
        self._console = console or CollapsingConsole()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._console.write(
                self._render_record(record),
                collapse_key=self._collapse_key(record),
            )
        except Exception:
            self.handleError(record)

    def _render_record(self, record: logging.LogRecord) -> str:
        formatter = self.formatter or logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        timestamp = formatter.formatTime(record, formatter.datefmt)
        message = record.getMessage()
        rendered = self._compose_line(timestamp, record.levelname, record.name, message)

        if record.exc_info:
            rendered = f"{rendered}\n{formatter.formatException(record.exc_info)}"
        elif record.stack_info:
            rendered = f"{rendered}\n{formatter.formatStack(record.stack_info)}"

        return rendered

    def _compose_line(
        self, timestamp: str, levelname: str, logger_name: str, message: str
    ) -> str:
        if not self._supports_color():
            return f"{timestamp} {levelname} {logger_name} {message}"

        level_color = {
            "DEBUG": CYAN,
            "INFO": GREEN,
            "WARNING": YELLOW,
            "ERROR": RED,
            "CRITICAL": MAGENTA,
        }.get(levelname, RESET)
        return (
            f"{DIM}{timestamp}{RESET} "
            f"{level_color}{levelname}{RESET} "
            f"{CYAN}{logger_name}{RESET} "
            f"{message}"
        )

    def _collapse_key(
        self, record: logging.LogRecord
    ) -> tuple[str, str, str, str | None, str | None]:
        exc_text = None
        stack_text = None
        if record.exc_info:
            formatter = self.formatter or logging.Formatter()
            exc_text = formatter.formatException(record.exc_info)
        if record.stack_info:
            formatter = self.formatter or logging.Formatter()
            stack_text = formatter.formatStack(record.stack_info)

        return (
            record.levelname,
            record.name,
            record.getMessage(),
            exc_text,
            stack_text,
        )

    def _supports_color(self) -> bool:
        stream = self._console._stream
        is_tty = getattr(stream, "isatty", lambda: False)()
        return bool(is_tty)


def configure_logging(log_level: str, stream: TextIO | None = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handler = CollapsingLogHandler(CollapsingConsole(stream))
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)


def capture_console_output(messages: list[str]) -> str:
    stream = io.StringIO()
    console = CollapsingConsole(stream)
    for message in messages:
        console.write(message)
    console.close()
    return stream.getvalue()


def capture_collapsed_output(entries: list[tuple[object, str]]) -> str:
    stream = io.StringIO()
    console = CollapsingConsole(stream)
    for collapse_key, rendered in entries:
        console.write(rendered, collapse_key=collapse_key)
    console.close()
    return stream.getvalue()
