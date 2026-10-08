# Isolated H3 GPU test proposal — approval required

No paid GPU test, model download, production deployment or endpoint change has been performed.

## Resources

Use one isolated temporary test worker/pod with the same RTX 6000 Ada 48 GB class as production. Do not attach the candidate image to the live endpoint. Creating this temporary resource requires approval. Confirm its actual displayed rate before starting; the previously observed production rate was $1.75/hour. Stop if the chosen rate is higher or approval does not cover it.

Reuse existing base weights without overwriting them. Check whether the exact Ref2VA LoRA already exists and verify its SHA256. If absent, the only proposed weight download is `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` from the official lightx2v repository: published size 1.96 GB, SHA256 `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c`. Use a temporary staging name and promote only after verification. Do not download another base checkpoint. No download is authorized yet.

Prefer the existing regional persistent storage if it can be safely attached while production remains available. Use separate test output/context directories and read existing weights only. Confirm free space for the LoRA, checkpoints and MP4s. If a new paid volume is necessary, stop and quote it separately. No storage expansion is authorized.

## Proposed sequence: ten generations in two approval stages

Use fixed small reference assets, a fixed prompt, 480p, 24 fps and a requested five-second clip where the mode allows it. Keep seed 42 where supported. A Music React regression uses its existing minimum supported section duration. Save every result and probe it before judging success.

Stage 1: independent validation, five generations.

1. Normal Ref2VA baseline, with the same reference assets used for the comparisons.
2. Motion Context initial clip with the unchanged 20-step sampler; verify paired latent persistence.
3. Motion Context continuation of that clip; verify 22 video / 24 audio context defaults, boundary motion, trimming and measured duration.
4. Ref2VA Turbo Fast alone, correct 4-step LoRA, Euler/simple and 12/3 shifts; check applied LoRA keys, likeness, video and audio against baseline.
5. Normal FL2VA regression through the original FreeVideo path.

Stop on a compatibility failure. Do not combine add-ons until Motion and Turbo pass independently. Ref2VA Turbo Balanced remains disabled because no verified official eight-step Ref2VA LoRA/configuration was found. FL2VA add-ons remain disabled because the current runtime is FreeVideo rather than the ComfyUI graph.

Stage 2: five further generations, requiring separate approval after reviewing Stage 1.

6. Continue the Motion chain again after restarting the test worker; verify checkpoint recovery and resolution guards.
7. Repeat Turbo with the same inputs; measure warm loading and repeatability.
8. Motion Context plus Turbo Fast together; compare boundary motion/audio, likeness and actual duration.
9. Normal Continue Video regression using the current tail/final-frame reference route; validate joining, source preservation and a longer result.
10. Normal Music React regression with an uploaded master soundtrack; validate its original audio and timing. Motion Context for Music React remains unavailable until timeline integration has separate tests.

Read-only/no-generation checks additionally cover interrupted or corrupt checkpoint rejection, absent/wrong LoRA rejection, normal graph identity, model inventories/hashes, reference limits, startup, MP4 decoding, timestamps, library/result metadata, saving and playback of downloaded MP4s. A browser/iPhone check is required for Safari playback; CPU image startup does not establish it.

## Cost proposal

At $1.75/hour, five generations each consuming the current 3600-second provider execution allowance would be $8.75 in execution time. Propose a **$12 USD total compute budget for Stage 1**, including startup/loading/idle allowance, and another **$12 USD for Stage 2** only after review: **$24 USD for all ten tests**. This is a conservative approval budget, not a measured add-on price or guaranteed platform spending cap. Failed runs can also be billed. Monitor elapsed billable time, stop submitting jobs before the budget is exhausted, and stop the temporary worker promptly after tests. Do not silently retry or increase any timeout. If the approved budget cannot finish the planned tests, report the partial evidence and ask before further spend.

No Turbo speed or cost saving has been measured yet. Historical production 15-second Ref2VA processing examples were 2703.653 seconds and 3309.178 seconds, approximately $1.31 and $1.61 at $1.75/hour before startup/idle. Shorter test durations and Turbo may reduce time, but no proportional reduction is promised. Storage charges, if any new storage is required, must be quoted before approval rather than included as an unknown cost.

Record cold/warm startup, model loading, execution, total billed worker time and actual charge; video duration/frame count; reference likeness; motion; boundary and generated audio; latency; checkpoint/output storage bytes. Record billable time from provider evidence, not only the app timer.

## Rollback

Production remains on `ghcr.io/mitm-1/mirrorvidgen-worker:7b21b2c015b21a512498cdcaa357da4720208cad`, digest `sha256:565f3dbf579c475aa8edc050b80757312e234adfdc5fd4c5d51abedcd1277068`. Keep draft PR #1 unmerged. The test image has a unique `h3-addons-...` tag and add-on generation is disabled by default.

If testing fails, stop the temporary resource and keep evidence and completed MP4s. Leave production endpoint/configuration, original weights and original outputs untouched. No production rollback is necessary because it was never switched. Any later release needs separate approval; its rollback would restore the recorded image digest and disable add-ons while retaining Normal Quality and saved results.
