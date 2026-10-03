FROM runpod/worker-comfyui:5.10.0-base

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ffmpeg ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir "kornia==0.8.2" "transformers>=4.50.0,<5"

# FreeVideo code is immutable in the image. Its heavy runtime, FP8 cache and
# H3 model files are deliberately kept on the persistent RunPod volume.
ARG FREEVIDEO_COMMIT=f7d117726a60567639dda7bba05ae353e58f7b97
RUN rm -rf /opt/freevideo \
    && git clone https://github.com/FlashML-org/FreeVideo.git /opt/freevideo \
    && git -C /opt/freevideo checkout "$FREEVIDEO_COMMIT" \
    && chmod +x /opt/freevideo/freevideo /opt/freevideo/setup.sh

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY freevideo_backend.py /freevideo_backend.py
COPY h3_ref2va_backend.py /h3_ref2va_backend.py
COPY h3-setup.sh /h3-setup.sh
COPY ltx25_workflow.py /ltx25_workflow.py
COPY workflow_compiler.py /workflow_compiler.py
COPY tests/ltx25-t2v-smoke-request.json /ltx25-t2v-template.json

# Fail the image build immediately if either backend routing module has a syntax error.
RUN chmod +x /h3-setup.sh \
    && PYTHONPATH=/ /opt/venv/bin/python -m py_compile \
    /handler.py /freevideo_backend.py /h3_ref2va_backend.py /ltx25_workflow.py /workflow_compiler.py

ENV NETWORK_VOLUME_DEBUG=false
ENV MIRRORVIDGEN_POLL_INTERVAL=1
ENV MIRRORVIDGEN_POLL_TIMEOUT=3500
ENV MIRRORVIDGEN_MAX_INLINE_VIDEO_BYTES=15000000

ENV FREEVIDEO_SOURCE=/opt/freevideo
ENV FREEVIDEO_HOME=/runpod-volume/freevideo-h3
ENV FREEVIDEO_HF_HOME=/runpod-volume/.cache/huggingface
ENV MIRRORVIDGEN_H3_RESULTS=/runpod-volume/results
ENV MIRRORVIDGEN_H3_TIMEOUT=3500
