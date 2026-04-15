# MCoCoNav

MCoCoNav is a multi-robot object navigation framework that combines semantic mapping, classical local planning, visual-language reasoning, and multi-robot target coordination.

This README is intentionally minimal and focuses on the fastest path to reproduce the evaluation pipeline in this repository.

## 1. What You Need

- Python 3.10
- PyTorch 2.0.1
- CUDA 12.1
- `habitat-sim`
- `habitat-lab`
- `detectron2`
- an OpenAI-compatible multimodal VLM service

Install the lightweight Python dependencies in this repo:

```bash
pip install -r requirements.txt
```

## 2. Install Habitat

This codebase was developed on top of the `challenge-2022` versions of Habitat.

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout tags/challenge-2022
pip install -r requirements.txt
python setup.py install --headless
```

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/challenge-2022
pip install -e .
```

If you use the multi-robot setup in this repository, replace the corresponding Habitat package in `habitat-lab` with the modified version under `multi-robot-setting/`, or apply the same changes in your own Habitat installation.

## 3. Install Detectron2

Install `detectron2` using the official instructions for your PyTorch and CUDA versions.

## 4. Prepare Data

Download HM3D v0.2 / MP3D and the Habitat ObjectNav dataset separately, then organize the files as:

```text
MCoCoNav/
`-- data/
    |-- hm3d_annotated_basis.scene_dataset_config.json
    |-- hm3d_annotated_val_basis.scene_dataset_config.json
    |-- matterport_category_mappings.tsv
    |-- scene_datasets/
    |   |-- hm3d_v0.2/
    |   `-- mp3d/
    |-- datasets/
    |   `-- objectnav_hm3d_v2/
    |       |-- train/
    |       |-- val/
    |       `-- val_mini/
    `-- Rednetmodel/
        `-- rednet_semmap_mp3d_40.pth
```

The following lightweight files are already included in this repo:

- `data/hm3d_annotated_basis.scene_dataset_config.json`
- `data/hm3d_annotated_val_basis.scene_dataset_config.json`
- `data/matterport_category_mappings.tsv`

## 5. Prepare Model Weights

### RedNet

Place the RedNet checkpoint at:

```text
data/Rednetmodel/rednet_semmap_mp3d_40.pth
```

### YOLOv10

The current code imports YOLO from the bundled `detect/` directory.

The default weight path is:

```text
detect/yolov10m/model.safetensors
```

You need to download this file separately. The repository does not ship large weights.

## 6. Prepare the VLM Service

The code expects an external multimodal server with an OpenAI-style API:

```text
{base_url}/v1/chat/completions
```

Default:

```text
http://127.0.0.1:31511
```

The backend must support:

- image input
- standard chat completion requests
- the custom field `return_string_probabilities`

You can change the endpoint with `--base_url`.

## 7. Main Entry Point

Use:

```bash
python main.py
```

The main script:

- builds per-robot semantic maps
- extracts frontier candidates
- calls the VLM for scene and target reasoning
- coordinates multiple robots
- executes local navigation with FMM planning

## 8. Run Evaluation

### 2 robots on HM3D

```bash
python main.py \
  -d ./outputs \
  --exp_name hm3d_2agent \
  --num_agents 2 \
  --visualize 0 \
  --num_sem_categories 16 \
  --task_config tasks/multi_objectnav_hm3d.yaml \
  --base_url http://127.0.0.1:31511
```

### 4 robots on HM3D

```bash
python main.py \
  -d ./outputs \
  --exp_name hm3d_4agent \
  --num_agents 4 \
  --visualize 0 \
  --num_sem_categories 16 \
  --task_config tasks/multi_objectnav_hm3d_4agents.yaml \
  --base_url http://127.0.0.1:31511
```

## 9. Important Notes

- `--num_agents` should match the selected Habitat task config.
- The code writes logs and visualizations under `logs/` and `dump/`.
- The repository is evaluation-oriented; it does not provide a clean standalone training pipeline.
- The easiest reproduction path is to keep the current repo structure unchanged and only prepare the missing datasets, weights, and VLM service.

## 10. Core Files

For most users, the key files are:

- `main.py`
- `arguments.py`
- `agents/vlm_agents.py`
- `semantic_mapping.py`
- `src/SystemPrompt.py`
- `src/vlm.py`
- `src/GDP_agent.py`
- `envs/habitat/configs/tasks/`

