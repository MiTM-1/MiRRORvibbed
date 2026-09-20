FROM runpod/worker-comfyui:5.10.0-base

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir "kornia==0.8.2" "transformers>=4.50.0,<5"

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY ltx25_workflow.py /ltx25_workflow.py
COPY workflow_compiler.py /workflow_compiler.py
COPY tests/ltx25-t2v-smoke-request.json /ltx25-t2v-template.json

ENV NETWORK_VOLUME_DEBUG=false
ENV MIRRORVIDGEN_POLL_INTERVAL=1
ENV MIRRORVIDGEN_POLL_TIMEOUT=3500
ENV MIRRORVIDGEN_MAX_INLINE_VIDEO_BYTES=15000000
