# Docker deployment
Self-contained image for running `pipeline.py` (segmentation-tracking + kinematics + OSATS scoring).

## Build
```bash
docker build -t pitinst-pipeline docker/
```

## Run
Requires a CUDA-capable GPU on the host with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
docker run --rm --gpus all \
    -v /path/to/inputs:/app/inputs \
    -v /path/to/models:/app/models \
    -v /path/to/outputs:/app/outputs \
    pitinst-pipeline --str_video my_video
```

Any flag accepted by `pipeline.py` (see `python pipeline.py --help`) can be passed after the image name,
e.g. `--int_fps 25 --int_clip 5`.

## Notes
- `models/` (~2GB: `mask2former.onnx`, `sam2_encoder.onnx`, `sam2_decoder.onnx`, `moco_surg.torch`, `*.joblib`) 
  and `inputs/` (source videos) are **not** baked into the image — mount them as volumes at runtime so the
  image stays small and doesn't need rebuilding when models/videos change.
- Models can be downloaded here: https://huggingface.co/drdreets/PitInst
- If `--str_video` is omitted, the first `.mp4` file found under `/app/inputs` is used.
- Output CSVs and the annotated overlay video are written to `/app/outputs`.
