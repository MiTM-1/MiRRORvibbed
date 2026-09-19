import base64
import json
import mimetypes
import os
import time
import traceback
from pathlib import Path

import requests
import runpod

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
POLL_INTERVAL = float(os.environ.get("MIRRORVIDGEN_POLL_INTERVAL", "1"))
POLL_TIMEOUT = int(os.environ.get("MIRRORVIDGEN_POLL_TIMEOUT", "3500"))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MIRRORVIDGEN_MAX_INLINE_VIDEO_BYTES", "15000000"))
NETWORK_RESULTS = Path("/runpod-volume/results")


def wait_for_comfy():
    deadline = time.time() + 300
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=5)
            if r.ok:
                return True
        except Exception as e:
            last_error = e
        time.sleep(1)
    raise RuntimeError(f"ComfyUI did not become ready: {last_error}")


def upload_image(name, encoded):
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    blob = base64.b64decode(encoded)
    files = {"image": (name, blob, "application/octet-stream")}
    data = {"overwrite": "true"}
    r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, data=data, timeout=120)
    r.raise_for_status()


def download_asset(name, url):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    target = Path("/comfyui/input") / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def queue_workflow(workflow):
    r = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": workflow},
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"ComfyUI workflow validation failed ({r.status_code}): {r.text[:8000]}")
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
    return prompt_id


def wait_for_history(prompt_id):
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data:
            entry = data[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            msgs = status.get("messages", [])
            for msg in msgs:
                if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                    raise RuntimeError(json.dumps(msg[1] if len(msg) > 1 else msg))
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Generation exceeded {POLL_TIMEOUT} seconds")


def iter_output_files(value):
    if isinstance(value, dict):
        if "filename" in value:
            yield value
        for v in value.values():
            yield from iter_output_files(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_output_files(v)


def fetch_output_file(info):
    filename = info.get("filename")
    if not filename:
        return None, None
    file_type = info.get("type", "output")
    if file_type == "temp":
        return None, None
    params = {
        "filename": filename,
        "subfolder": info.get("subfolder", ""),
        "type": file_type,
    }
    r = requests.get(f"http://{COMFY_HOST}/view", params=params, timeout=600)
    r.raise_for_status()
    return filename, r.content


def put_result(upload, blob, content_type):
    url = upload.get("url")
    if not url:
        return None
    headers = dict(upload.get("headers") or {})
    headers.setdefault("Content-Type", content_type)
    r = requests.put(url, data=blob, headers=headers, timeout=600)
    r.raise_for_status()
    return upload.get("public_url") or url.split("?", 1)[0]


def persist_large_video(job_id, filename, blob):
    folder = NETWORK_RESULTS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(filename).name
    target.write_bytes(blob)
    return str(target)


def handler(job):
    try:
        job_input = job.get("input") or {}
        workflow = job_input.get("workflow")
        if not isinstance(workflow, dict):
            return {"error": "Missing or invalid input.workflow"}

        wait_for_comfy()

        for image in job_input.get("images") or []:
            upload_image(image["name"], image["image"])

        for asset in job_input.get("assets") or []:
            if asset.get("url") and asset.get("name"):
                download_asset(asset["name"], asset["url"])

        prompt_id = queue_workflow(workflow)
        history = wait_for_history(prompt_id)
        outputs = history.get("outputs") or {}

        results = []
        seen = set()
        result_upload = job_input.get("result_upload") or {}

        for node_output in outputs.values():
            for info in iter_output_files(node_output):
                key = (info.get("subfolder", ""), info.get("filename", ""), info.get("type", "output"))
                if key in seen:
                    continue
                seen.add(key)

                filename, blob = fetch_output_file(info)
                if not filename or blob is None:
                    continue

                ext = Path(filename).suffix.lower()
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                is_video = ext in {".mp4", ".webm", ".mov", ".mkv", ".gif"}

                if is_video:
                    uploaded = None
                    if result_upload.get("url"):
                        uploaded = put_result(result_upload, blob, content_type)
                    if uploaded:
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "url",
                            "data": uploaded,
                            "bytes": len(blob),
                        })
                    elif len(blob) <= MAX_INLINE_VIDEO_BYTES:
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "base64",
                            "mime": content_type,
                            "data": base64.b64encode(blob).decode("utf-8"),
                            "bytes": len(blob),
                        })
                    else:
                        path = persist_large_video(job["id"], filename, blob)
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "network_path",
                            "data": path,
                            "bytes": len(blob),
                            "note": "Provide input.result_upload.url for a browser-accessible result URL.",
                        })
                else:
                    results.append({
                        "filename": filename,
                        "kind": "image",
                        "type": "base64",
                        "mime": content_type,
                        "data": base64.b64encode(blob).decode("utf-8"),
                        "bytes": len(blob),
                    })

        return {
            "status": "success",
            "prompt_id": prompt_id,
            "outputs": results,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


if __name__ == "__main__":
    print("MiRRORvidgen RunPod worker starting")
    runpod.serverless.start({"handler": handler})
