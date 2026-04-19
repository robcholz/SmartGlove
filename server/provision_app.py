from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4173


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the SmartGlove provisioning web app."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def get_app_directory() -> Path:
    app_directory = Path(__file__).resolve().parent.parent / "provision-app"
    if not app_directory.is_dir():
        raise FileNotFoundError(f"Provision app directory not found: {app_directory}")
    return app_directory


def main() -> None:
    args = build_parser().parse_args()
    app_directory = get_app_directory()
    handler = partial(SimpleHTTPRequestHandler, directory=str(app_directory))

    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {app_directory} at http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping provision app server.")


if __name__ == "__main__":
    main()
