# MCoCoNav
## 📝 TODO List

 - [ ] Release demo video
 - [ ] Improve repository information
## Installation

The code has been tested with Python 3.10.8, CUDA 12.1.

### 1. Installing Dependencies
- We use adjusted versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below.

- Install habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
```

- Install habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
```

Back to the current repo, and replace the habitat folder in habitat-lab repo for the multi-robot setting: 

```
mv -r multi-robot-setting/habitat enter-your-path/habitat-lab
```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on torch v2.0.1, torchvision 0.15.2. 

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration.

### 2. Download HM3D_v0.2 and MP3D datasets

#### Habitat Matterport
Download [HM3D_v0.2](https://aihabitat.org/datasets/hm3d/) and [MP3D](https://niessner.github.io/Matterport/) datasets using the download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).

### 3. Download segmentation model

Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in RedNet/model path.

### 4. Install YOLOv10

Follow the [README](detect/README.md) to install YOLOv10.

### 5. Install VLM

We recommend recreating an environment to install VLM.

- Install CogVLM2:
```
git clone https://github.com/THUDM/CogVLM2.git
cd basic_demo
pip install -r requirements.txt
cd enter-your-path-of-MCoCoNav
mv VLM/glm4_openai_api_demo_1gpu.py CogVLM2/basic_demo/
```
- Download GLM-4V-9B:
[🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)
[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)
[💫 Wise Model](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B)

## Setup
Install other requirements:
```
cd MCoCoNav/
pip install -r requirements.txt
```

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
MCoCoNav/
  data/
    scene_datasets/
        hm3d_v0.2/
            val/
            hm3d_annotated_basis.scene_dataset_config.json
            hm3d_annotated_val_basis.scene_dataset_config.json
        mp3d/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectgoal_hm3d_v2/
        train/
        val/
        val_mini/
```

## Evaluation
### Start the VLM server:
```
python glm4_openai_api_demo_1gpu.py
```

### Eval 2-robot on HM3D_v0.2: 
```
python main.py -d ./VLM_EXP/multi_hm3d_2-robot/  --num_agents 2 --task_config tasks/multi_objectnav_hm3d.yaml
```

### Phase 0A: exploration/backtracking parameters

The current audited entrypoint is `review_main_final_260217.py`. It separates
the directional history evidence
`tau_es * ES + tau_js * JS` from the frontier/history switching rule, which
compares `JS` directly with `exploration_beta` after the warm-up period.

```bash
python review_main_final_260217.py \
  -d ./VLM_EXP/phase0a_default/ \
  --num_agents 2 \
  --task_config envs/habitat/configs/tasks/multi_objectnav_hm3d.yaml \
  --tau_es 2 --tau_js 1 \
  --exploration_beta 0.3 \
  --exploration_warmup_steps 125 \
  --history_heading_half_width 39
```

Use `scripts/run_fig8_exploration_grid.sh` for the traceable 3-by-7 grid. The
dated `main_tpami_1es*.py` programs are legacy snapshots and should not be used
for new Fig. 8 runs.

### R1-C4: exact VCG allocation with complete bundle execution

The new allocation implementation is integrated in `main_tpami_250903.py` and
`review_main_final_260217.py`. It is a code revision, not a reconstruction of the
original experiment. Older entrypoints do not implement the new bundle queue;
do not use them for validating this revision.

The Coordinator combines updated and initial scores using `consensus_lambda`,
then combines semantic scores and robot-specific inverse distances using
`voting_alpha`. `src/target_auction.py` maximizes total bundle valuation exactly
and computes Clarke-pivot payments using exact counterfactual allocations.
Payments and net utilities must pass settlement validation before the entire
allocation is committed to navigation. There is no payment-based reranking,
budget, or cross-round balance.

Each awarded bundle is saved as a queue of map coordinates and executed in its
canonical frontier-label order. FMM arrival advances the queue. Finished robots
scan in place while other bundles remain active. History selection, unavailable
frontiers, or a shared object goal explicitly cancel the affected queue; local
stagnation recovery pauses it and subsequently resumes it. A new auction cannot
overwrite an unfinished batch. Queue coordinates use a fixed shared full map,
so these entrypoints require `--global_downscaling 1`.

| CLI option | Default | Role |
|---|---:|---|
| `--consensus_lambda` | 0.3 | Weight of post-discussion decision scores |
| `--voting_alpha` | 0.7 | Weight of semantic scores versus inverse distance |
| `--auction_beta` | 0.8 | Valuation multiplier for every nonempty bundle |
| `--auction_reserve_price` | 0.0 | Seller value per retained, unallocated target |

The existing `--alpha` is an optimizer option, not the consensus voting weight.
The six `tpami_mrdn_lambda*.sh` / `tpami_mrdn_alpha*.sh` launchers now pass their
labelled consensus settings explicitly. With zero reserve, a common positive
`auction_beta` scales valuations, payments and utilities without changing the
welfare-maximizing assignment. It does not introduce bundle-dependent synergy.

Append these settings to the appropriate existing server launch command:

```bash
--consensus_lambda 0.3 --voting_alpha 0.7 --auction_beta 0.8 \
--auction_reserve_price 0 --global_downscaling 1
```

CPU tests (NumPy is required by the voting tests; Habitat and VLM services are not):

```bash
python -m unittest discover -s tests -v
```

The logs contain `VCGSettlement` (bundles, valuations, payments, utilities,
counterfactual welfare and unsold targets) and `BundleExecution` (installation,
completion, pause/resume and cancellation). Server navigation and performance
validation remain necessary. See the
[C4 implementation record](../reviews/issues/r1-c4-target-allocation/code-implementation.md)
for the change scope, numerical check and handoff checklist.
