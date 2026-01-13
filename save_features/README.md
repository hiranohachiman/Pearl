## Overview
This directory houses the SPICA feature pipelines. Everything lives inside `save_features/` (code, Poetry envs, checkpoints). Once data and weights are staged, a single script fans out over every model and split.

## Quick Run
```bash
cd save_features
scripts/run_spica_feature_jobs.sh
```
That command sequentially runs BEiT3 → BLIP2 → CLIP++ → STELLA for `train/val/test`, writing results to `features/<model>/<split>/`. No extra options required.

## Prerequisites
- **Python / tooling**: Python 3.10, Poetry 1.8+, CUDA build of `torch`, and enough GPU VRAM/disk (tens of GB).
- **Data layout** (under the workspace root, sibling to `save_features/`):
  ```
  data/
    images/              # filenames match imgid
    spica_train.csv
    spica_val.csv
    spica_test.csv
  ```
- **Model checkpoints**: Store model checkpoints in `save_features/checkpoints/` using the following structure:
  ```
  checkpoints/
    beit3_base_itc_patch16_224.pth
    beit3.spm
    PAC++_clip_ViT-L-14.pth
    stella_en_400M_v5/
      <model files and configs>
  ```

Outputs land under `features/<model>/<split>/clip_*_features_*.npz`. Use the per-model Poetry envs for any follow-up inspection.

