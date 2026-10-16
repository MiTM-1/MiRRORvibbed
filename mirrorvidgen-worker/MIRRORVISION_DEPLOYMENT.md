# MiRRORvision candidate — activation pending

## Production baseline
- Endpoint: r5xypdhn4sgccs / mirrorvidgen-h3
- Verified console image: ghcr.io/mitm-1/mirrorvidgen-worker:h3-addons-7f5a358e3b920bb91d477e4a88f271d5daf031e0
- Registry index digest: sha256:e575065ed15288a764e620c85a8ad9fa87493790daa78bb5b65812f8539a780d
- 48 GB Pro; one GPU per worker; mirrorvidgen-models volume US-IL-1.
- Active workers 0; maximum 2; idle timeout 5s; queue-delay scaling 4s.
- Execution timeout 3600s; FlashBoot enabled.
- Existing environment MIRRORVIDGEN_H3_ADDONS_TEST=true. Preserve it.

## Candidate
Dockerfile.vision inherits the immutable production baseline. It saves the inherited handler under h3_addon_handler_existing.py and delegates all old actions unchanged. Only mirrorvision_render is new. No model downloads, Python dependency upgrades, GPU changes or endpoint changes.

Native effects are CPU 2D using Pillow and FFmpeg. They are not a 3D engine. Semantic segmentation, OCR, transcription, subject verification and chained previous-frame conditioning are not implemented. Artwork is retained as original pixels; generated footage is quarantined when artwork exists. Full mode currently assembles H3 footage; its image-first reactive completion remains outstanding.

## Validation gates
1. Local H3 regression (15 tests), real MP4 fixtures (six mode/content combinations), original soundtrack frequency, frame count, duration, ordered two-section assembly, exact text raster OCR, measured bands/onsets/decay.
2. Container CPU check including an actual 5s image-only MP4, capability contract and inherited handler import.
3. Isolated scale-to-zero staging: verify startup, capabilities, exact media URLs, authenticated output callback, stored playable MP4, H3 regression/preflight.
4. Paid infrastructure cap $5 total, including startup. Do not proceed without bounded execution and cost visibility.
5. Production must remain unchanged until staging passes and production queue is empty.

## Rollback
Record the actual current console release immediately before activation. If H3 startup, capability or preflight fails, use New release to restore the baseline full tag above (or its verified immutable digest). Leave endpoint ID, GPU class, volume, environment and scaling untouched. Never purge queue or delete jobs. Site v125 is commit 885bde98f7799af0982ad47160600e72343e5ad4; v126 is 8c97b8e49bc0568be7a2e5aa65ad5592db4f85af. Site v126 remains capability-gated until a worker advertises rendererVersion 1.

## Failed candidate
a90530c3e19ef2874868d772d7065d6210dc99ec built successfully but failed its container check because mirrorvision_dispatch was installed only at /handler.py. Do not deploy that image. Commit 8e8235d1717230af7479d36d021a8e3a47a0bad8 installs both module and entrypoint paths.
