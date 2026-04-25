#!/bin/bash
# Build script for C++ SpeedFlow GStreamer plugin

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo "Building C++ SpeedFlow GStreamer Plugin"
echo "========================================"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "[1/3] Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}"

# Build
echo "[2/3] Building..."
make -j$(nproc)

# Check result
if [ -f "libgstspeedflow.so" ]; then
    echo "[3/3] Build successful!"
    echo ""
    echo "Plugin location: ${BUILD_DIR}/libgstspeedflow.so"
    echo ""
    echo "To use the C++ backend:"
    echo "  python3 main.py --backend cpp --source video.mp4 --mode display"
else
    echo "ERROR: Build failed - libgstspeedflow.so not found"
    exit 1
fi

echo "========================================"
echo "Build Complete!"
echo "========================================"
