# Code-only repository manifest

This repository was generated from `mcoconv_code_export_test.tar.gz` on
2026-08-26. The original archive was preserved unchanged.

## Included

- DisCoNav/MCoCoNav Python entry points and experiment shell scripts.
- Project modules under `agents/`, `envs/`, `src/`, and `utils/`.
- RedNet inference source and its model-file placeholder.
- Project and Habitat task configuration files.
- The customized multi-robot Habitat Python package under
  `multi-robot-setting/`.
- The vendored YOLO/Ultralytics source required by the current import path
  under `detect/ultralytics/`, without weights, examples, tests, or media.
- Build metadata, dependency lists, documentation, and licenses that are part
  of the selected source tree.

## Excluded

- Dataset contents and generated episodes.
- Model weights, checkpoints, serialized arrays, and result files.
- Images, videos, meshes, archives, compiled libraries, and executables.
- Runtime outputs, logs, caches, and environment directories.
- Full upstream snapshots of `habitat-sim`, `habitat-lab`, and `detectron2`.
  These are external dependencies and should be installed at the revisions
  described by the project README. The customized multi-robot Habitat code is
  retained separately in `multi-robot-setting/`.
- YOLO documentation, examples, tests, and the bundled
  `detect/yolov10m/model.safetensors` weight file.

The `.gitignore` prevents the excluded artifact classes from being committed
later. Configuration files and dataset loader source code are intentionally
retained; they describe how to locate datasets but do not contain datasets.
