# GitHub Upload Checklist

This repository has been prepared to upload the source code while keeping large experiment outputs, datasets, model weights, and private review notes out of Git history.

## What Will Be Uploaded

- Python source files in the project root and subdirectories
- Configs and helper scripts
- `README.md`
- `requirements.txt`
- Lightweight assets such as `100x100.png`
- Project code under `agents/`, `configs/`, `detect/`, `envs/`, `multi-robot-setting/`, `RedNet/`, `src/`, and `utils/`

## What Will Be Ignored

- `VLM_EXP/`
- `VLM_EXP_bundle/`
- `TODO/`
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`
- `logs/`, `dump/`, `*.log`
- Downloaded datasets under `data/scene_datasets/`, `data/versioned_data/`, `data/objectgoal_hm3d_v2/`
- Model weights such as `*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.bin`

## Why These Are Ignored

- `VLM_EXP/` and `VLM_EXP_bundle/` currently account for most of the repository size and are experiment outputs rather than source code.
- `detect/yolov10m/model.safetensors` is about 63.57 MB. GitHub allows it, but large binary files are a poor fit for normal Git history and are better handled through a download link or Git LFS.
- `TODO/` contains review material and should not be published unless you explicitly want that.

## Recommended Publishing Structure

1. Keep this repository focused on code, configs, and documentation.
2. Provide dataset download instructions in `README.md` instead of uploading datasets.
3. Provide model download links in `README.md`, or move weights to Git LFS if you really need versioned binaries.
4. If you want to share representative results, publish a small curated subset separately using GitHub Releases or another storage service.

## Local Git Initialization

The repository can be initialized with:

```powershell
git init -b main
git add .
git commit -m "Initial commit"
```

## First Push Command Template

Create an empty GitHub repository first, then run:

```powershell
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

Or if you prefer SSH:

```powershell
git remote add origin git@github.com:<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

## Final Checks Before Pushing

1. Run `git status --ignored` and confirm ignored directories are not staged.
2. Review `git add .` carefully in case there are private notes or local-only scripts you still want excluded.
3. If this will be a public repository, double-check for institution-specific or unpublished material.
4. If you later decide to include model weights, use Git LFS before the first commit so the Git history stays clean.
