# H3 add-ons development preparation

## Status
Development-only candidate. NOT deployed, NOT GPU-validated and NOT functional in the live Site. No RunPod requests or model downloads were made.

Repository: MiTM-1/MiRRORvibbed
Branch: dev/h3-motion-turbo-preparation-20261008
Inspected head: 1980db951f86320d8ebdf20fab259f615509c443
Production revision: 7b21b2c015b21a512498cdcaa357da4720208cad
GitHub comparison: ten later commits, changing only three iptv files. All worker files and the production build workflow are unchanged.
Rollback image: ghcr.io/mitm-1/mirrorvidgen-worker:7b21b2c015b21a512498cdcaa357da4720208cad
Recorded digest: sha256:565f3dbf579c475aa8edc050b80757312e234adfdc5fd4c5d51abedcd1277068

## Actual backend inventory
- FL2VA: FreeVideo/VDN command-line runtime on /runpod-volume/freevideo-h3, not the ComfyUI FL2VA graph. Bundled FreeVideo revision f7d117726a60567639dda7bba05ae353e58f7b97. A persistent setup-matched source checkout takes precedence when present.
- Ref2VA: explicit ComfyUI core graph in h3_ref2va_generation.py.
- Ref2VA diffusion: minimax_h3_ref2va_pruned_int8_convrot.safetensors.
- Text encoder: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors.
- Video VAE: minimax_h3_video_vae_int8_convrot.safetensors.
- Audio VAE: minimax_h3_audio_vae_fp32.safetensors.
- Baseline Ref2VA sampler res_multistep, default beta scheduler, 20 steps, 24 fps. Native frame ladder 17*n+5.
- Optional Ref2VA Turbo file is already listed in model-management code but excluded from the default core installation; its installed presence is not established by reading source.
- Production base image runpod/worker-comfyui:5.10.0-base. Original Docker build cloned an unpinned custom-node repository and installed ranged dependencies, so rebuilding the old Dockerfile is not identical to the working binary image. Candidate extends the recorded image digest instead.
- Original worker polling timeout 3500s, FreeVideo timeout 3500s. Current observed provider timeout 3600s. None changed.
- Persistence: model directories, FreeVideo cache and results on /runpod-volume; image disk temporary. Candidate adds job-scoped paired latent checkpoints on that same volume.
- Existing legacy source/dependencies were not modified, installed or routed to by the add-on adapter.

## Candidate files
- Dockerfile.addons: extends production digest, pins Motion Context v0.6.2 revision 5335715abe54c1a9bfbe3494da29aae3e8635ce3; no dependency upgrades or weights download.
- h3_addons.py: copy-on-write Ref2VA graph adapter, duration/overlap planning, installed-node schema validation, workflow-specific Turbo checks.
- h3_context_store.py: durable, checksummed, atomic paired latent checkpoints; immutable job IDs; resolution/revision checks; staging under ComfyUI output for the upstream loader.
- h3_addon_handler.py: candidate wrapper; default requests delegate to original handler. Add-on generation disabled unless explicitly enabled for approved experiments; combined tests separately disabled. Preflight does not queue a prompt. Completed videos remain returned if checkpoint verification fails.
- tests/test_h3_addons.py: graph preservation, references/audio, overlap, normal fallback, missing LoRA, invalid context, durable integrity, interruption, actual baseline preflight no-queue, paid/combined gates.
- .github/workflows/h3-addons-compatibility.yml: unpaid Python checks only.
- .github/workflows/build-h3-addons-candidate.yml: manual candidate image build only, development refs only, unique h3-addons-SHA tag; no latest tag or endpoint deployment.
No original worker file or production workflow is edited. No live UI controls added before real validation.

## Supported preparation and limits
Motion Context graph prepared for current Ref2VA, carrying previous sampler output with 22 visual and 24 audio context frames. First clip saves context without overlap. Following clips load exact paired AV latent, retain existing references and trim both decoded streams with match_tail.
Existing completed MP4s do not contain saved sampler latents. This candidate intentionally rejects an absent checkpoint rather than pretending an MP4 is latent context. Decoded-video fallback is a future explicit path.
Only paired visual/audio latent mode is prepared. Visual-only continuation is blocked rather than silently ignoring a user's selection.
Master Music React mux is unchanged. Do not expose Motion Context for Music React until section timing/assembly is adapted for shortened delivered segments and exact master timing is tested.
Ref2VA Turbo Fast candidate uses the official 4-step LoRA, Euler/simple schedule and video/audio shifts 12/3. Official example uses BF16; this worker uses INT8 ConvRot. A real test is needed for LoRA application and output quality.
Ref2VA Turbo Balanced is blocked: no official verified 8-step Ref2VA model/configuration found.
Current FL2VA remains untouched. ComfyUI FL2VA add-ons cannot be attached directly to the FreeVideo CLI. Do not route FL2VA through Ref2VA or download another base checkpoint to simulate support.
Combined Motion+Turbo tests are independently gated and must wait until independent runs pass.
Turbo does not install, replace or select a new base model.

## Duration examples at 24 fps
| Request sampled by existing graph | Native frames | Motion overlap | Delivered new segment |
|---|---:|---:|---:|
| 5 seconds | 124 | 22 | 4.25 seconds |
| 10 seconds | 243 | 22 | 9.208333 seconds |
| 15 seconds | 362 | 22 | 14.166667 seconds |
These are graph-planned durations, not measured outputs. The candidate probes decoded output before accepting a persisted checkpoint. Joining/added-duration UI integration remains gated.

## Unpaid test results
15 candidate unit/contract tests passed.
6 existing workflow compiler tests passed.
Python compileall passed for worker source.
Pinned upstream Motion Context mock smoke suite passed (references, visual/audio alignment, resolution rejection, Save/Load and overlap trimming).
These use mocks/fixtures: no real H3 tensor layout, quantised LoRA output, GPU startup, quality, Safari or cost benchmark is proven.
Docker executable is unavailable in this workspace, so no local Docker build ran. The manual GitHub build definition is prepared but not dispatched. No candidate image has been published.

## Further validation sequence
1. Build candidate without replacing any live tag; inspect startup and ComfyUI node schemas/layout against the inherited image.
2. Read exact installed LoRA files and remaining volume capacity. Verify source/hash/size before any single necessary download; never overwrite existing files.
3. Explain expected paid GPU time/cost and obtain approval BEFORE any RunPod generation tests or production endpoint/image switch.
4. Test Motion Context on normal 20-step Ref2VA independently (initial + continuation), then Turbo Ref2VA separately. Compare same assets, prompt, seed, duration and resolution to the unchanged baseline.
5. Only after independent passes, approve and test combination. Keep Ref2VA 8-step and FreeVideo FL2VA add-ons disabled until separately established.
6. Before release add server capability gates, current-source checkpoint ownership/lifecycle, precise continuation display/join metadata and Music React master-timeline support; regress Library/queue/notifications/uploads/saving. Do not deploy merely because Python checks pass.
7. Record cold/warm setup, GPU processing, total billable time, output duration, likeness, motion and audio quality. Do not claim savings from 4-vs-20 steps alone.

Historical baseline log examples: 2703.653s and 3309.178s. App execution-only rate $0.00049/s gives $1.325 and $1.622, excluding startup/idle. No add-on benchmark measured yet. Current GPU console rate observed $1.75/hour; actual billing must be reconciled separately.

Storage impact: one optional LoRA plus per-job latent checkpoints; exact bytes and free volume space unmeasured. No volume enlargement, new endpoint, GPU upgrade, weight download or production change approved/applied.

Upstream sources:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/tree/5335715abe54c1a9bfbe3494da29aae3e8635ce3
https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225

