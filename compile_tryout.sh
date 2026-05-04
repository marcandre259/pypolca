#!/usr/bin/env bash
set -e

# Auto-detect scikit-build-core build directory
WHEEL_TAG="cp313-cp313-macosx_26_0_arm64"
BUILD_DIR="build/$WHEEL_TAG"

if [ ! -d "$BUILD_DIR" ]; then
    BUILD_DIR=$(find build -maxdepth 1 -type d -name 'cp313*' | head -n1)
fi

if [ -z "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "ERROR: No valid Ninja build directory found."
    echo "Run:  uv pip install -e . --no-build-isolation"
    exit 1
fi

echo "=== Building tryout in $BUILD_DIR ==="
ninja -C "$BUILD_DIR" tryout

echo "=== Running tryout ==="
"$BUILD_DIR/tryout"
