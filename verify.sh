#!/usr/bin/env sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "./run.sh" ]; then
    chmod +x ./run.sh 2>/dev/null || true
    exec sh ./run.sh --verify "$@"
fi

if [ -f "../run.sh" ]; then
    chmod +x ../run.sh 2>/dev/null || true
    exec sh ../run.sh --verify "$@"
fi

if [ -d "bin" ]; then
    :
elif [ -d "../bin" ]; then
    cd ..
elif [ -d "../../bin" ]; then
    cd ../..
fi

TARGET_BIN=""
if command -v nvidia-smi >/dev/null 2>&1 && [ -f "bin/build_cuda" ]; then
    TARGET_BIN="bin/build_cuda"
elif [ -f "bin/build-linux-x86_64" ]; then
    TARGET_BIN="bin/build-linux-x86_64"
elif [ -f "bin/build" ]; then
    TARGET_BIN="bin/build"
fi

if [ -n "$TARGET_BIN" ]; then
    chmod +x "$TARGET_BIN" 2>/dev/null || true
    echo "[RUN]: $TARGET_BIN"
    exec "./$TARGET_BIN" --verify "$@"
fi

echo "[ERROR] Neither run.sh nor binary found in bin/." >&2
exit 1
