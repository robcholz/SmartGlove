from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            value.strip()
            for value in os.getenv("SMARTGLOVE_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        )
        return cls(
            host=os.getenv("SMARTGLOVE_SERVER_HOST", cls.host),
            port=_parse_int(os.getenv("SMARTGLOVE_SERVER_PORT"), cls.port),
            reload=_parse_bool(os.getenv("SMARTGLOVE_SERVER_RELOAD"), cls.reload),
            log_level=os.getenv("SMARTGLOVE_SERVER_LOG_LEVEL", cls.log_level).lower(),
            allowed_origins=origins,
        )

    def with_overrides(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        reload: bool | None = None,
        log_level: str | None = None,
    ) -> "Settings":
        return replace(
            self,
            host=self.host if host is None else host,
            port=self.port if port is None else port,
            reload=self.reload if reload is None else reload,
            log_level=self.log_level if log_level is None else log_level,
        )
