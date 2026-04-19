from __future__ import annotations

import argparse

import uvicorn

from server.app import create_app
from server.config import Settings
from server.observability import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SmartGlove network API server."
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--log-level", choices=["critical", "error", "warning", "info", "debug"]
    )
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env().with_overrides(
        host=args.host,
        port=args.port,
        reload=True if args.reload else None,
        log_level=args.log_level,
    )
    configure_logging(settings.log_level)

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        log_config=None,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
