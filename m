#!/usr/bin/env bash
set -euo pipefail

VOL="/workspace"
MODELS="$VOL/models"
STATE="$VOL/mirrorvidgen"
LOG="$STATE/ltx25-download.log"

echo "=============================================="
echo " MiRRORvidgen — LTX-2.5 installer"
echo "=============================================="

if [ ! -d "$VOL" ]; then
  echo "ERROR: /workspace is not available."
  exit 1
fi

mkdir -p "$STATE" \
  "$MODELS/diffusion_models" \
  "$MODELS/text_encoders" \
  "$MODELS/vae" \
  "$MODELS/latent_upscale_models" \
  "$MODELS/unet" \
  "$MODELS/clip" \
  "$STATE/workflows"

if ! command -v curl >/dev/null 2>&1; then
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates
fi

echo
echo "Installing the current Hugging Face CLI..."
if ! command -v hf >/dev/null 2>&1; then
  curl -LsSf https://hf.co/cli/install.sh | bash -s -- --exclude-skill
fi
export PATH="$HOME/.local/bin:$HOME/.local/share/hf/bin:$PATH"

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI install did not place 'hf' on PATH."
  echo "Open a new terminal and rerun this installer."
  exit 1
fi

echo
if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face login is required."
  echo "A SHORT DEVICE CODE will appear below."
  echo "Open the URL it gives you in Safari and approve the login."
  echo
  hf auth login --format agent
fi

echo
echo "Checking access to the official LTX-2.5 weights..."
if ! hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --dry-run >/tmp/mirrorvidgen-ltx25-dryrun.txt 2>&1; then
  cat /tmp/mirrorvidgen-ltx25-dryrun.txt || true
  echo
  echo "LTX-2.5 access has not been granted to this Hugging Face account yet."
  echo "Open this page in Safari while logged in and accept/grant access:"
  echo "https://huggingface.co/Lightricks/LTX-2.5"
  echo
  echo "Then run this MiRRORvidgen installer again."
  exit 2
fi

cat > "$STATE/ltx25-download-worker.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
VOL="/workspace"
MODELS="$VOL/models"
STATE="$VOL/mirrorvidgen"

echo "[$(date -Is)] Starting official LTX-2.5 model download..."

hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$MODELS"

ln -sfn ../diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  "$MODELS/unet/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"

ln -sfn ../text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  "$MODELS/clip/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"

curl -fL --retry 5 -o "$STATE/workflows/video_ltx2_5_t2v.json" \
  https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_t2v.json
curl -fL --retry 5 -o "$STATE/workflows/video_ltx2_5_i2v.json" \
  https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_i2v.json
curl -fL --retry 5 -o "$STATE/workflows/video_ltx2_5_flf2v.json" \
  https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_flf2v.json

cat > "$STATE/extra_model_paths_ltx25.yaml" <<'YAML'
mirrorvidgen_volume:
  base_path: /runpod-volume
  unet: models/unet/
  clip: models/clip/
  vae: models/vae/
  latent_upscale_models: models/latent_upscale_models/
  loras: models/loras/
  model_patches: models/model_patches/
YAML

touch "$STATE/LTX25_READY"
echo "[$(date -Is)] LTX-2.5 DOWNLOAD COMPLETE"
echo "LTX25_READY"
EOS

chmod +x "$STATE/ltx25-download-worker.sh"

cat > /usr/local/bin/mvlog <<EOF
#!/usr/bin/env bash
tail -n 30 "$LOG"
EOF
chmod +x /usr/local/bin/mvlog

cat > /usr/local/bin/mvstatus <<EOF
#!/usr/bin/env bash
if [ -f "$STATE/LTX25_READY" ]; then
  echo "MiRRORvidgen LTX-2.5: READY"
else
  echo "MiRRORvidgen LTX-2.5: downloading / not finished"
fi
echo
df -h "$VOL" | tail -n 1
echo
tail -n 12 "$LOG" 2>/dev/null || true
EOF
chmod +x /usr/local/bin/mvstatus

rm -f "$STATE/LTX25_READY"
nohup bash "$STATE/ltx25-download-worker.sh" > "$LOG" 2>&1 &
echo $! > "$STATE/ltx25-download.pid"

echo
echo "=============================================="
echo " LTX-2.5 DOWNLOAD STARTED IN BACKGROUND"
echo "=============================================="
echo "You can safely leave this terminal open or reconnect later."
echo "Type: mvstatus"
echo "to check progress."
