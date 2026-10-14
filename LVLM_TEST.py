from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import sys
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import quaternion
import pickle
import io
import re
from copy import deepcopy

from skimage import measure
import skimage.morphology
from PIL import Image

import math
import cv2
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

# from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
# from constants import coco_categories, color_palette, category_to_id
from agents.vlm_agents import LLM_Agent
from agents.vlm_agents_gt import LLM_Agent_GT
# from agents.llm_agents import LLM_Agent
# from constants import (
#     color_palette, coco_categories, coco_categories_hm3d2mp3d,
#     hm3d_category, category_to_id, object_category
# )
from constants import *
from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env
# from envs.habitat.multi_agent_env import Multi_Agent_Env

# from src.habitat import (
#     make_simple_cfg,
#     pos_normal_to_habitat,
#     pos_habitat_to_normal,
#     pose_habitat_to_normal,
#     pose_normal_to_tsdf,
# )
# from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM, CogVLM2
from src.SystemPrompt import *
from src.GDP_agent import *
# from src.SystemPrompt import (
#     form_prompt_for_PerceptionVLM, 
#     form_prompt_for_FN,
#     form_prompt_for_DecisionVLM_Frontier,
#     form_prompt_for_DecisionVLM_History,
#     form_prompt_for_Frontier_Reason,

#     form_prompt_for_DecisionVLM_MetaPreprocess,
#     form_prompt_for_Module_Decision,
#     Perception_weight_decision,
#     Perception_weight_decision4,
#     Perception_weight_decision26,
#     extract_scene_image_description_results,
#     extract_scene_object_detection_results,
#     extract_scenario_exploration_analysis_results
# )
# from src.tsdf import TSDFPlanner
import utils.pose as pu

import utils.visualization as vu

from arguments import get_args
##### 如果是安装了yolov9环境，请将其注释取消
# from detect_yolov9 import Detect
from detect.ultralytics import YOLOv10


def optimal_weighted_combination(seq1, seq2, weights=(0.3, 0.7)):
    return [
        float(a)*weights[0] + float(b)*weights[1]
        for a, b in zip(seq1, seq2)
    ]
def calculate_R3_weighted_scores(decision_scores, frontier_distances, alpha=0.7):
    """
    返回格式: {agent_id: {target: weighted_score}}
    """
    weighted_results = {}
    
    for agent_id in decision_scores:
        # 获取有效目标（两者都存在的目标）
        targets = [
            chr(65+i) for i in range(len(decision_scores[agent_id])) 
            if chr(65+i) in frontier_distances[agent_id]
        ]
        
        # 提取对应数据
        decisions = np.array([
            decision_scores[agent_id][i] 
            for i, t in enumerate(targets)
        ])
        distances = np.array([
            frontier_distances[agent_id][t] 
            for t in targets
        ])
        
        # 计算加权分数
        norm_decisions = decisions/(decisions.sum()+1e-10)
        norm_distances = (1/(distances+1e-10))/(1/(distances+1e-10)).sum()
        weighted_scores = alpha*norm_decisions + (1-alpha)*norm_distances
        
        # 存储结果
        weighted_results[agent_id] = {
            t: float(weighted_scores[i]) 
            for i, t in enumerate(targets)
        }
    
    return weighted_results

# Devil's Advocate - Frontier
Devil_Prompt = '''
You are now operating as a strict frontier target selection skeptic. You are tasked with:** Presenting rebuttals** for other robots' reasons for choosing targets.

**INPUT ANALYSIS**
- YOUR PRIOR PROPOSAL:
  Target: {TARGET}  
  Confidence: {CONFIDENCE} 
  Reason: *"{MY_REASON}"*

- OTHER ROBOT PROPOSALS:
{OTHER_PROPOSAL}
- AVAILABLE TARGETS: {AVAILABLE_TARGETS}

**EXECUTION PROTOCOL**
1. INCONSISTENCY SCAN:
   - Verify claim inconsistency between YOUR PRIOR TARGET and OTHER PROPOSALS TARGET (if consistent, adjustments need to be made)
   - Identify conflicting accessibility evidence: *"Do clearance restrictions actually permit ALL proposals to reach target?"*

2. FORCE REBUTTAL REQUIREMENT:
   - Mandatory rebuttal of OTHER robots' TARGET
   - For each rebuttal:
        • Propose a **MANDATORY REBUTTAL**, taking into account the distribution of semantic information and frontier targets in the semantic map.
        • Modify Robot {AGENT_ID}'s target to new target [X] and give a recommendation.
        • Ensure that the modified new target **CANNOT be consistent** with YOUR PRIOR PROPOSAL's target. (If Your target is 'B', The proposed target cannot be B!)
**OUTPUT FORMAT (STRICT JSON)**
{{
  "rebuttal": {{
    "target": "A", 
    "rebuttal_reason": "<20-word excerpt>"
  }},
  "required_adjustment": {{
    "new_target": "D", 
    "recommendation": "<20-word excerpt>"
  }}
}}
'''


R2_Decision_Update_Prompt = '''
Given the following information: 
- Your Choice:
  Target: {MY_TARGET}
  Confidence: {MY_CONFIDENCE}
  Reason: **{MY_REASON}**

- ROBOT {AGENT_ID} REBUTTAL:
  Rebuttal: **{REBUTTAL_REASON}**
  Adjustment: **{NEW_TARGET}**
  Recommendation: **{RECOMMENDATION}**

After reviewing ROBOT {AGENT_ID} REBUTTAL to your initial frontier target choice, conduct a **balanced synthesis** of both perspectives to **re-evaluate all frontiers {AVAILABLE_TARGETS}**. Your reflection must: 

1. **Acknowledge key conflicts**:  
   - Contrast your original rationale for {MY_TARGET} (maintaining exploration direction/proximity avoidance)  
   - Against Robot {AGENT_ID}'s claim about target {MY_TARGET}'s rebuttal issues  
   - Explicitly identify where the evidence diverges  

2. **Re-assess ALL options**:  
   - Use this lens: *"Given Robot {AGENT_ID}'s rebuttal critique, is my original distance-based logic still sufficient? Conversely, does Robot 1 underweight exploration momentum by prioritizing proximity?"*  
   - Explicitly compare **{MY_TARGET}** (distance/non-switching) and **{NEW_TARGET}** (accessibility/proximity).  
   - Briefly check other target for hidden advantages if contradictions persist.  

3. **Determine the optimal choice through synthesis**:  
   - Only select Robot 1's recommendation if its evidence **materially invalidates** your original reasoning  
   - If evidence remains conflicting, prioritize **consensus-seeking strategies**  
   - Declare **one** target {AVAILABLE_TARGETS}.  

**Output Format:**
Your choice MUST in {AVAILABLE_TARGETS} **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''

# 恶魔代理人
def form_devils_prompt(
    reason: dict, 
    ) -> str:
    # 动态构建"其他代理提案"部分
    other_proposals_text = ""
    # 移除非必要空格和换行符
    my_reason = reason.replace('\n', ' ').strip()
    
    # for proposal in agent_data['other_agents_proposals']:
    other_proposals_text += (
            f"Robot: 2 | Target: A\n"
            f"Confidence: 0.3609 |"
            f"Reason: {my_reason}\n"
        )
    
    

    User_Prompt = Devil_Prompt.format(
        AGENT_ID = 1,
        TARGET = 'B',
        CONFIDENCE = 0.4834,
        OTHER_AGENTS = 2,
        OTHER_PROPOSAL = other_proposals_text.strip(),
        MY_REASON = "From the perspective of the scene information above, Frontier Point B is the correct choice due to its central location within the image, surrounded by various elements such as plants, tables, and appliances, indicating a versatile and accessible position within the space.",
        AVAILABLE_TARGETS = "A, B, C, D",
        # REMOVED_TARGET =  ', '.join(removed_target),
        
    )
    # 动态生成提示
    return User_Prompt


# R2反思
def form_decision_update_prompt(
    agent_id: int, 
    my_target: str,
    my_confidence: str,
    my_reason: str,
    rebuttal_reason: str, 
    new_target: str, 
    recommendation: str, 
    available_targets: str) -> str:

    
    User_Prompt = R2_Decision_Update_Prompt.format(
        AGENT_ID = agent_id,
        MY_TARGET = my_target,
        MY_CONFIDENCE = round(my_confidence,4),
        MY_REASON = my_reason,
        REBUTTAL_REASON = rebuttal_reason,
        NEW_TARGET = new_target,
        RECOMMENDATION = recommendation,
        AVAILABLE_TARGETS = available_targets
    )
    

    return User_Prompt


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:3" if args.cuda else "cpu")

    # logging.info(f"stride:{stride}")
    # logging.info(f"names:{names}")
    # logging.info(f"pt:{pt}")


    # Load VLM
    # vlm = VLM(args.vlm_model_id, args.hf_token, device)
    base_url = 'http://127.0.0.1:31517'
    cogvlm2 = CogVLM2(base_url) 
    # Load Yolo
    # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
    if args.yolo == 'yolov9':
        # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
        pass
    else:
        yolo = YOLOv10.from_pretrained(args.yolo_weights)
    # print(config_env)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    rgb = cv2.imread('/home/szx/project/MCoCoNav/VLM_EXP/multi_hm3d_2-robot_tpami_100steps_after_250811/dump/exp1/episodes/eps_76/Agents_vis/VisStep-149.png')

    img_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if img_rgb.dtype in [np.float32, np.float64]:
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
    
    # print(img_rgb.shape)

    Frontier_Choice_Prompt = form_prompt_for_Frontier_Reason('A')#这个可以换

    _, Choice_Reason = cogvlm2.simple_image_chat(User_Prompt=Frontier_Choice_Prompt,
                                    return_string_probabilities=None, img=img_rgb)

    # print("Choice_Reason:",Choice_Reason)

    # 需要被反驳的机器人是2，反驳者是1
    
    devil_prompt = form_devils_prompt(Choice_Reason)
    # print(devil_prompt)
    _, Devil_Advocate_Information = cogvlm2.COT2(User_Prompt1=Frontier_Choice_Prompt,
        User_Prompt2=devil_prompt,
        cot_pred1=Choice_Reason,
        return_string_probabilities=None,img=img_rgb)
    
    print("Devil_Prompt:",devil_prompt)
    
    print("Devil_Advocate_Information:",Devil_Advocate_Information)


    Devil_Advocate = {}
    frontier_distance = {}
    try:
        if bool(re.search(r'^\s*```json\b', Devil_Advocate_Information)):
            Devil_Advocate_Json = re.search(r'```(?:json)?\n([\s\S]*?)\n```', Devil_Advocate_Information)
            Devil_Advocate = json.loads(Devil_Advocate_Json.group(1))
        else:
            Devil_Advocate = json.loads(Devil_Advocate_Information)
    except Exception as e:
        print(e)
        Devil_Advocate = {}

    # print(Devil_Advocate)
    Frontier_Distances = {}
    Decision_Score = {}
    Decision_Score[1] = [0.0,0.63,0.11,0.26]
    if len(Devil_Advocate) != 0:
        # print("robot.frontier_coords:",robot.frontier_coords)
        all_frontier_letters = ['A', 'B', 'C', 'D']
        # Frontier_Distances[robot.id] = calculate_frontier_distance(robot.frontier_coords[robot.id-1], all_frontier_letters, self.cur_start)
        R2_decision_prompt = form_decision_update_prompt(1,
            rebuttal_reason=Devil_Advocate['rebuttal']['rebuttal_reason'],
            new_target=Devil_Advocate['required_adjustment']['new_target'],
            recommendation=Devil_Advocate['required_adjustment']['recommendation'],
            available_targets="A, B, C, D",
            my_target="A",
            my_confidence=0.3609,
            my_reason=Choice_Reason,
            )
        # print(R2_decision_prompt)
        # 3层思维链(决策)
        Last_Choice_Score, Last_Choice = cogvlm2.COT2(User_Prompt1=Frontier_Choice_Prompt,
                    User_Prompt2=R2_decision_prompt,
                    cot_pred1=Choice_Reason,
                    return_string_probabilities="[A, B, C, D]",img=img_rgb)
        # 获取origin Decision
        Origin_Decision_Score = [0.3609, 0.2157, 0.1843, 0.2391]
        print("Robot 2 - Last_Choice_Score: ",Last_Choice_Score)
        # print(Last_Choice)
        # print(Origin_Decision_Score)
        assert len(Last_Choice_Score) == len(Origin_Decision_Score)

        # DS^f
        Decision_Score[2] = optimal_weighted_combination(Last_Choice_Score, Origin_Decision_Score)
    else:
        Decision_Score[2] = [1,2,3,4]
    
    
    Frontier_Distances = {1: {'A': 75.17978451685, 'B': 93.08598175880189, 'C': 91.2414379544733, 'D': 78.64477096412705}, 2: {'A': 75.17978451685, 'B': 93.08598175880189, 'C': 91.2414379544733, 'D': 78.64477096412705}}
    #R3
    R3_vote = calculate_R3_weighted_scores(Decision_Score, Frontier_Distances)


    print("Decision_Score:",Decision_Score)
    print("R3_vote:",R3_vote)


    # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    # return avg_metrics
    

if __name__ == "__main__":
    main()