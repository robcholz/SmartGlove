#!/usr/bin/env bash

set -e

BUILD_MODE=""
BIN_NAME="${2:-smart-glove}"
case "$1" in
"" | "release")
    bash scripts/build.sh release "$BIN_NAME"
    BUILD_MODE="release"
    ;;
"debug")
    bash scripts/build.sh debug "$BIN_NAME"
    BUILD_MODE="debug"
    ;;
*)
    echo "Wrong argument. Only \"debug\"/\"release\" arguments are supported"
    exit 1
    ;;
esac

web-flash --chip esp32s3 "target/xtensa-esp32s3-espidf/${BUILD_MODE}/${BIN_NAME}"
