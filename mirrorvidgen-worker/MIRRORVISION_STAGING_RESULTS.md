# Actual staging results — 16 October 2026

Production endpoint r5xypdhn4sgccs has NOT been updated. The production H3 image, GPU, scaling, volume and user jobs remain unchanged.

Site v127: 4cad880e69efa89a40d3ff75ec9f7d4a95891aba.
Isolated staging endpoint: 5dczz4dix8iai0, mirrorvision-staging-v126; 48 GB Pro, max 1, active 0, execution timeout 600s, existing model volume. It scales to zero and is not an always-on worker.

## Deployed staging evidence
Image source 77cfd234cce519c098bff4b5a41cd4ef4196e796.
- H3 FL2VA and Ref2VA advertised ready.
- MiRRORvision rendererVersion 1, native2d true, serverAssembly true.
- advanced3d false; previousFrameConditioning false.
- Image-only Teaser request 92efd155-365f-4da5-8560-8d0c4e97b09f-u1: completed, 6009ms, uploaded true.
- Exact-text Teaser request 9c01c5db-5f41-44d6-8c6f-d07cad134ca5-u2: completed, 5860ms.
- Hybrid request 57323446-52df-460b-93e3-1d179c17be6e-u1: completed, 6474ms, uploaded true.
- Full assembly fixture also stored successfully. Its source footage was a local test clip, NOT a paid H3 generation.

All four files were downloaded from the secret-protected isolated staging namespace and fully decoded with FFmpeg:
- image-only Teaser: 10.000s, 240 H.264 frames, AAC, 168007 bytes
- text Teaser: 10.000s, H.264/AAC, 207330 bytes
- Hybrid: 15.000s, H.264/AAC, 255974 bytes
- Full assembly fixture: 15.000s, H.264/AAC, 97202 bytes

The Teaser soundtrack has 0.999922 correlation with the selected original PCM.
OCR on actual exported video confirmed MiTM — HOUSE MUSIC! and Coming Soon. with case, punctuation and glyph sequence preserved; line wraps were normalized for comparison.
Native image modes retain source pixel correlation above 0.998 at sampled frame times.

## Real bugs found
1. Initial image installed dispatcher only at /handler.py, so container selfcheck could not import the module. Repaired before staging.
2. Default Python urllib user agent received HTTP 403 from the app, while MiRRORvidgen-Worker/1.0 succeeded. Renderer downloads and callbacks now identify the worker client.
3. Actual exported frame brightness did not track bass positively because black overlay blending cancelled the base brightness response. This failed live visual synchronization acceptance despite successful MP4 generation. Additive light blending fixes the local exported correlation to 0.986531. Commit 78bb897f1e927a7c15804948c4fdcbf6cac85974 adds an exported-frame correlation check inside the container build. It must pass and be retested on staging before activation.

## Free verification
15 H3 regression tests; exact inherited dispatcher delegation; selected PCM silence/bands/onsets/decay; six real local render scenarios; output frame counts and durations; original audio frequency; ordered two-section assembly; exact typography raster OCR; orchestration/retry/recovery/storage mocks; production app build.

## Outstanding — not verified or complete
- Full H3 image-first reactivity and source/subject preservation.
- Previous-frame conditioning is not wired.
- Semantic segmentation, OCR/image understanding and lyrics transcription are not installed.
- Native effects are CPU 2D, not advanced 3D or volumetric rendering.
- Live Full H3 generation and multi-section H3 end-to-end tests were not run.
- Existing Library finalization and close/reopen behavior remain integration-test-only; the live staging files use a separate test callback, not user Library records.
- Real iPhone Safari interaction/animation testing has not been performed.
- Therefore production activation is NOT approved by these test results alone.

Observed account balance changed from $23.99 to $23.95 during staging tests, approximately $0.04; endpoint billing summaries lag. No $5 budget increase, new always-on worker or production GPU change was made.

Rollback: production has not changed. Staging can roll back to mirrorvision-77cfd234cce519c098bff4b5a41cd4ef4196e796 for comparison, but that version has the documented visual-sync defect. Do not promote it. Production baseline and exact configuration are recorded in MIRRORVISION_DEPLOYMENT.md.
