#!/usr/bin/env bash
set -euo pipefail

ROOT="${FREEVIDEO_HOME:-/runpod-volume/freevideo-h3}"
BUNDLED_SOURCE="${FREEVIDEO_SOURCE:-/opt/freevideo}"
SOURCE="$ROOT/source"

if [ ! -x "$BUNDLED_SOURCE/freevideo" ]; then
  echo "ERROR: FreeVideo is not installed in this worker image."
  exit 1
fi

mkdir -p "$ROOT"
if [ ! -x "$SOURCE/freevideo" ]; then
  echo "Copying the pinned FreeVideo source to persistent storage..."
  rm -rf "$SOURCE"
  cp -a "$BUNDLED_SOURCE" "$SOURCE"
fi

echo "=================================================="
echo " MiRRORvidgen — MiniMax H3 / FreeVideo setup"
echo " Persistent root: $ROOT"
echo "=================================================="
echo
echo "Applying RunPod network-volume executable compatibility fix..."
/opt/venv/bin/python - "$SOURCE/freevideo_engine/bootstrap.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = """        uv = (self.root / 'tools' / uv_spec['executable'] if self.system == 'Windows' else
              self.root / 'tools' / 'uv-x86_64-unknown-linux-gnu' / 'uv')
        self.env['FREEVIDEO_UV'] = str(uv)
"""
replacement = """        uv = (self.root / 'tools' / uv_spec['executable'] if self.system == 'Windows' else
              self.root / 'tools' / 'uv-x86_64-unknown-linux-gnu' / 'uv')
        if self.system != 'Windows':
            # RunPod network volumes can preserve extracted files without the
            # executable bit FreeVideo expects. Restore it after extraction,
            # before supervised child-process validation.
            try:
                uv.chmod(uv.stat().st_mode | 0o111)
            except OSError as error:
                raise RuntimeError('Could not mark FreeVideo uv executable on RunPod volume: %s' % error)
            if not os.access(uv, os.X_OK):
                raise RuntimeError('FreeVideo uv exists but the RunPod volume does not allow execution: %s' % uv)
        self.env['FREEVIDEO_UV'] = str(uv)
"""
if needle in text:
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
elif "RunPod network volumes can preserve extracted files" not in text:
    raise SystemExit("Pinned FreeVideo bootstrap layout changed; refusing an unsafe patch")
PY

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
