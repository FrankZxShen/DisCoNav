#!/usr/bin/env bash
set -euo pipefail

# Phase 0A reproducibility entrypoint for the 3 x 7 Fig. 8 grid.
# Run from the repository root after the HM3D data and LVLM server are ready.

DISCONAV_PYTHON_BIN="${DISCONAV_PYTHON_BIN:-python}"
DISCONAV_VLM_URL="${DISCONAV_VLM_URL:-http://127.0.0.1:31511}"
DISCONAV_FIG8_ROOT="${DISCONAV_FIG8_ROOT:-./VLM_EXP/phase0a_fig8}"

weight_pairs=("2 1" "1 1" "1 2")
beta_values=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7")

for pair in "${weight_pairs[@]}"; do
    read -r tau_es tau_js <<< "${pair}"
    for beta in "${beta_values[@]}"; do
        run_dir="${DISCONAV_FIG8_ROOT}/tau_es_${tau_es}_tau_js_${tau_js}_beta_${beta}"
        "${DISCONAV_PYTHON_BIN}" review_main_final_260217.py \
            -d "${run_dir}" \
            --exp_name "seed_1" \
            --seed 1 \
            --num_agents 2 \
            --visualize 2 \
            --num_sem_categories 16 \
            --task_config envs/habitat/configs/tasks/multi_objectnav_hm3d.yaml \
            --base_url "${DISCONAV_VLM_URL}" \
            --tau_es "${tau_es}" \
            --tau_js "${tau_js}" \
            --exploration_beta "${beta}" \
            --exploration_warmup_steps 125 \
            --history_heading_half_width 39
    done
done
