#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_build_dir() {
    if [ -d "${SCRIPT_DIR}/build" ]; then
        find "${SCRIPT_DIR}/build" -maxdepth 1 -type d -name 'cp*' -print -quit 2>/dev/null
    fi
}

BUILD_DIR=$(find_build_dir || true)

if [ -z "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "ERROR: No valid Ninja build directory found."
    echo "Run:  ./rebuild.sh --clean"
    exit 1
fi

echo "=== Building tryout in $BUILD_DIR ==="
ninja -C "$BUILD_DIR" tryout

echo "=== Running tryout ==="
"$BUILD_DIR/tryout"
