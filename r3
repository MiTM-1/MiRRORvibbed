#!/usr/bin/env bash
set -euo pipefail

VOL="/workspace"
MODELS="$VOL/models"
STATE="$VOL/mirrorvidgen"
LOG="$STATE/ltx25-resume.log"

mkdir -p "$STATE" "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/unet" "$MODELS/clip"

export HF_HOME="$VOL/.cache/huggingface"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

HF=(/root/.local/bin/uvx --python 3.12 hf)

if ! "${HF[@]}" auth whoami >/dev/null 2>&1; then
  echo "ERROR: Hugging Face login is not available in $HF_HOME"
  exit 1
fi

cat > "$STATE/ltx25-resume-worker.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

VOL="/workspace"
MODELS="$VOL/models"
STATE="$VOL/mirrorvidgen"

export HF_HOME="$VOL/.cache/huggingface"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

HF=(/root/.local/bin/uvx --python 3.12 hf)

echo "[$(date -Is)] RESUME START"

echo "[$(date -Is)] 1/2 transformer"
"${HF[@]}" download Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors --local-dir "$MODELS" --max-workers 1

echo "[$(date -Is)] 2/2 text encoder"
"${HF[@]}" download Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors --local-dir "$MODELS" --max-workers 1

ln -sfn ../diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors "$MODELS/unet/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
ln -sfn ../text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors "$MODELS/clip/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"

touch "$STATE/LTX25_READY"
echo "[$(date -Is)] LTX-2.5 READY"
EOS

chmod +x "$STATE/ltx25-resume-worker.sh"
rm -f "$STATE/LTX25_READY"
: > "$LOG"
nohup setsid bash "$STATE/ltx25-resume-worker.sh" > "$LOG" 2>&1 < /dev/null &
echo $! > "$STATE/ltx25-resume.pid"
sleep 3
echo "MiRRORvidgen LTX-2.5 resume started"
echo
tail -n 20 "$LOG" || true
