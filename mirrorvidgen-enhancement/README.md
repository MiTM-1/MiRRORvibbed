# MiRRORvidgen independent enhancement

Deploy this CPU service separately from the H3/LTX RunPod endpoint. Mount a
persistent volume at `/data`. Run one process; its queue has one processing slot.
Set `ENHANCEMENT_TOKEN` to a generated private bearer token. Set `SOURCE_HOST`
to `mirrorvidgen.mitm1.chatgpt.site`. Publish the service behind HTTPS.

Configure the existing Site's server-only `MIRRORVIDGEN_ENHANCEMENT_URL` and
`MIRRORVIDGEN_ENHANCEMENT_TOKEN` with that HTTPS origin and matching token.
The Site never forwards the token to browsers and never calls RunPod for enhancement.

Interpolation is FFmpeg motion-compensated `minterpolate` with MCI, AOBMC,
bidirectional motion estimation and variable block sizes. This creates actual
intermediate frames. It is not RIFE, frame duplication or naive blending.
Upscaling is genuine encoded Lanczos resampling, not AI detail reconstruction.
Results retain source lineage. Original audio is remuxed, not time-stretched.

Quality limitations: difficult occlusions/cuts may produce interpolation artefacts;
Lanczos cannot recover lost detail. CPU processing may be slow for long HD clips.
The UI stays unavailable until this independent service answers its health check.
