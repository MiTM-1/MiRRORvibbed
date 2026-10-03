#!/usr/bin/env bash
set -euo pipefail

ROOT="${FREEVIDEO_HOME:-/runpod-volume/freevideo-h3}"
SOURCE="${FREEVIDEO_SOURCE:-/opt/freevideo}"

if [ ! -x "$SOURCE/freevideo" ]; then
  echo "ERROR: FreeVideo is not installed in this worker image."
  exit 1
fi

mkdir -p "$ROOT"

echo "=================================================="
echo " MiRRORvidgen — MiniMax H3 / FreeVideo setup"
echo " Persistent root: $ROOT"
echo "=================================================="
echo
echo "Running read-only hardware/model plan first..."
"$SOURCE/freevideo" --root "$ROOT" setup --plan --json --plain

echo
if [ "${1:-}" != "--accept-model-license" ]; then
  echo "SETUP NOT STARTED."
  echo "FreeVideo requires explicit acceptance of the MiniMax H3 model license."
  echo "After reviewing it, rerun:"
  echo "  /h3-setup.sh --accept-model-license"
  exit 2
fi

echo "Explicit model-license acceptance flag received."
echo "Installing the isolated FreeVideo runtime and prepared H3 model cache..."
"$SOURCE/freevideo" --root "$ROOT" setup \
  --plain \
  --yes \
  --accept-model-license \
  --storage compact

echo
echo "Verifying installation..."
"$SOURCE/freevideo" --root "$ROOT" doctor --require-paths

echo
echo "H3_READY"
