---
name: bitlaya-decision-metric
description: "BitLaya ablation decisions use rate-at-quality (bpp for 20/25/30/35 dB, paired bootstrap), not the ablate envelope mean"
metadata:
  node_type: memory
  type: project
  originSessionId: 19bb159a-1d2a-4a33-b022-fb65a372105a
  modified: 2026-09-25T18:56:22.918Z
---

Since the Milestone 1 position stage (2026-09-25), BitLaya variant decisions use `python scripts/summarize.py rate <ablation dir> [ref]`: complete-file bpp needed for pooled PSNR >= 20/25/30/35 dB, with paired image-bootstrap CIs, for raw and range-coded payloads.

**Why:** the `ablate` operational-envelope mean averages PSNR gain over 0.27-8.27 bpp, which is mostly 5-16 dB output; it contradicted itself between raw and range-coded payloads. The user's objective is file size at acceptable distortion.

**How to apply:** report the envelope too, but decide on rate at quality. For the lossless track, validation BCE equals the rate (cross-entropy x 8 = ideal bpp), so it is a valid selection metric there. See [[bitlaya-compute]].
