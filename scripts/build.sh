#!/bin/bash

set -e

which idf.py >/dev/null || {
    source ~/export-esp.sh >/dev/null 2>&1
}

BUILD_MODE="${1:-release}"
BIN_NAME="${2:-smart-glove}"

case "$BUILD_MODE" in
"release")
    cargo build --release --bin "$BIN_NAME"
    ;;
"debug")
    cargo build --bin "$BIN_NAME"
    ;;
*)
    echo "Wrong argument. Only \"debug\"/\"release\" arguments are supported"
    exit 1
    ;;
esac
