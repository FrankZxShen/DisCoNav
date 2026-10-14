import cv2
import ast
import re
import numpy as np
from src.SystemPrompt import *
import json
import logging
import random
from collections import defaultdict
from itertools import combinations
from math import isfinite

from src.target_auction import PriorityAuctionSystem

import ast
import numpy as np

def calculate_frontier_distance(frontiers_dict, frontier_keys, start):
    # 构建 letter_to_key 映射：'A' -> 'frontier_0', 'B' -> 'frontier_1', ...
    sorted_frontier_keys = sorted(
        frontiers_dict.keys(),
        key=lambda k: int(k.split('_')[1])  # 按编号排序
    )
    letter_to_key = {
        chr(65 + i): key for i, key in enumerate(sorted_frontier_keys)
    }

    def convert_entry(entry):
        # 提取坐标字符串
        return entry.split('centroid: ')[1].split(', number: ')[0]

    # 存储结果的距离字典
    distances = {}
    for letter in frontier_keys:
        if letter not in letter_to_key:
            raise KeyError(f"Letter '{letter}' does not have a corresponding frontier in the dictionary.")

        key_in_dict = letter_to_key[letter]
        entry = frontiers_dict[key_in_dict]

        coord_str = convert_entry(entry)
        x, y = ast.literal_eval(coord_str)

        # 计算欧氏距离
        dist = np.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2)

        # 保存进结果字典
        distances[letter] = dist

    return distances

def calculate_R3_weighted_scores(decision_scores, frontier_distances, alpha=0.7):
    """
    返回格式: {agent_id: {target: weighted_score}}
    """
    if not isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("voting alpha must lie in [0, 1]")
    weighted_results = {}
    
    for agent_id in decision_scores:
        # 获取有效目标（两者都存在的目标）
        targets = [
            chr(65+i) for i in range(len(decision_scores[agent_id])) 
            if chr(65+i) in frontier_distances.get(agent_id, {})
        ]
        if not targets:
            continue
        
        # 提取对应数据
        decisions = np.array([
            decision_scores[agent_id][ord(t) - ord('A')]
            for t in targets
        ], dtype=float)
        distances = np.array([
            frontier_distances[agent_id][t] 
            for t in targets
        ], dtype=float)
        if (not np.all(np.isfinite(decisions)) or np.any(decisions < 0)
                or not np.all(np.isfinite(distances)) or np.any(distances < 0)):
            raise ValueError("frontier scores and distances must be finite and nonnegative")
        
        # 计算加权分数
        norm_decisions = (decisions / decisions.sum() if decisions.sum() > 0
                          else np.full(len(targets), 1.0 / len(targets)))
        zero_distance = distances == 0
        if np.any(zero_distance):
            norm_distances = zero_distance.astype(float) / zero_distance.sum()
        else:
            inverse_distances = distances.min() / distances
            norm_distances = inverse_distances / inverse_distances.sum()
        weighted_scores = alpha*norm_decisions + (1-alpha)*norm_distances
        
        # 存储结果
        weighted_results[agent_id] = {
            t: float(weighted_scores[i]) 
            for i, t in enumerate(targets)
        }
    
    return weighted_results


def is_frontier_valid(image, frontiers_dict, frontier_key, threshold=10):
    # 解析指定 frontier 的坐标
    frontier_str = frontiers_dict[frontier_key]
    def convert_entry(entry):
        # 从字符串中提取坐标
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'{centroid_str}'

    # 遍历字典，并使用转换函数
    entry = frontier_str
    converted = convert_entry(entry)
    # print(converted)
    converted = ast.literal_eval(converted)
    # 获取该点的 RGB 值
    try:
        pixel_color = image[converted[1], converted[0]]  
    except IndexError:
        raise ValueError(f"Coordinates ({x}, {y}) are out of image bounds.")
    # 定义目标颜色（白色和灰色）
    white = np.array([255, 255, 255])
    gray = np.array([153, 153, 153])
    # 判断是否在阈值范围内
    # in_white = np.all(np.abs(pixel_color - white) <= threshold)
    in_gray = np.all(np.abs(pixel_color - gray) <= threshold)

    return not (in_gray), converted


def get_frontier_key_by_letter(full_Frontiers_dict, letter):
    # 获取当前存在的 frontier keys，并提取编号
    existing_indices = [int(k.split('_')[1]) for k in full_Frontiers_dict.keys()]
    # 排序后映射为字母顺序
    sorted_indices = sorted(existing_indices)
    # 构建字母到 frontier index 的映射
    letter_to_index = {chr(65 + i): idx for i, idx in enumerate(sorted_indices)}
    # 返回对应的 frontier key
    return f"frontier_{letter_to_index[letter]}"

def optimal_weighted_combination(seq1, seq2, weights=(0.3, 0.7)):
    if len(seq1) != len(seq2):
        raise ValueError("initial and updated score vectors must have the same length")
    if (len(weights) != 2 or any(not isfinite(w) or w < 0 for w in weights)
            or abs(sum(weights) - 1.0) > 1e-10):
        raise ValueError("decision weights must be nonnegative and sum to one")
    return [
        float(a)*weights[0] + float(b)*weights[1]
        for a, b in zip(seq1, seq2)
    ]

class Coordinator:
    def __init__(self, robots, vlm_models, *, decision_lambda=0.3,
                 voting_alpha=0.7, valuation_scale=0.8, reserve_price=0.0):
        """
        初始化协调器
        :param robots: 机器人代理列表 [RobotAgent]
        :param global_map: 全局语义地图对象 (含障碍物/可通行区域)
        :param vlm_models: 视觉语言模型字典 {'perception': VLM, 'decision': VLM}
        """
        self.robots = robots
        self.vlm_models = vlm_models
        for name, value in (("decision_lambda", decision_lambda),
                            ("voting_alpha", voting_alpha)):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isfinite(valuation_scale) or valuation_scale <= 0:
            raise ValueError("valuation_scale must be positive and finite")
        if not isfinite(reserve_price) or reserve_price < 0:
            raise ValueError("reserve_price must be nonnegative and finite")
        self.decision_lambda = decision_lambda
        self.voting_alpha = voting_alpha
        self.valuation_scale = valuation_scale
        self.reserve_price = reserve_price
        self.robot_positions = {}
        self.auction_result = None
        self.Final_Bundles = {}
        self.discussion_history = []  # 讨论历史记录
        self.round = 1  # 当前讨论轮次
        self.max_rounds = 1
        self.global_map = None
        self.available_targets = []
        self.r2_inputs = None
        self.r3_inputs = None
        self.secretary_mode = False  # 平票处理开关
        self.cur_start = None
        self.is_History = False
        
        self.Final_Decision = {}

    # R1
    def run_discussion(self,agent_choice_reasons):
        """执行完整讨论流程"""
        # Results belong to this round, even if discussion or voting is skipped.
        self.Final_Decision = {}
        self.Final_Bundles = {}
        self.auction_result = None
        self.r3_inputs = {}
        for round_num in range(1, self.max_rounds+1):
            # self._broadcast_messages(round_num)
            proposals = []
            num_robot = 0
            for robot in self.robots:
                # 获取多模态输入并生成提案
                # visual_input = self._generate_visual_prompt(robot)
                # text_input = self._generate_text_prompt(robot, round_num)
                # print(agent_choice_reasons)
                proposal = robot.propose(agent_choice_reasons[num_robot],)
                proposals.append(proposal)
                num_robot += 1
            self._update_discussion_history(proposals)
            self.round += 1
        # print(self.discussion_history)
        
        # # 投票决策
        self.r2_inputs = self._prepare_r2_inputs(self.discussion_history[-1])

    def _prepare_r2_inputs(self, last_round_proposals):
        """准备R2阶段输入"""
        r2_inputs = {}
        for robot in self.robots:
            # 构建R2输入包
            robot_proposal = next(p for p in last_round_proposals if p["agent"] == robot.id)
            other_proposals = [p for p in last_round_proposals if p["agent"] != robot.id]
            # print("robot_proposal:",robot_proposal)
            # print("other_proposals:",other_proposals)
            
            # 收集所有可用目标点（frontier points和history points）
            # available_targets = set()
            # for p in last_round_proposals:
            #     available_targets.add(p["proposal"]["target"])
            self.Final_Decision[robot.id] = robot_proposal["proposal"]["target"]
            
            r2_inputs[robot.id] = {
                "navigation_task": f"Navigate to find {robot.target_object}",
                "your_previous_proposal": {
                    "type": robot_proposal["proposal"]["type"],
                    "target": robot_proposal["proposal"]["target"],
                    "confidence": robot_proposal["proposal"]["confidence"],
                    "reason": robot_proposal["proposal"]["reason"]
                },
                "other_agents_proposals": [
                    {
                        "agent": p["agent"],
                        "type": p["proposal"]["type"],
                        "target": p["proposal"]["target"],
                        "confidence": p["proposal"]["confidence"],
                        "reason": p["proposal"]["reason"]
                    } for p in other_proposals
                ],
                "available_targets": self.available_targets,
                "critical_focus": ["accessibility", "target_consistency"],
                "discussion_rules": "Critically evaluate other robots' proposals. Identify error_type/state_type and evidence (e.g., if a target is in an obstacle zone). Update your proposal if new evidence emerges. In your critique, reference top-down semantic map."
            }
        
        return r2_inputs
            

    # R2 MCoCo_Decision_COT
    def run_r2_discussion(self, MCoCo_Decision_COTs):
        """执行R2阶段完整流程"""
        # 1. 初始化R2输入包
        r2_inputs = self.r2_inputs
        # print("r2_inputs:", r2_inputs)
        # 保存global_map
        # cv2.imwrite("self.Global_map.png",self.global_map)
        
        # 2. 执行机器人批判分析 O(n^2) -> 修改后需要保持复杂度为 O(n-1)
        Decision_Score = {}
        # for robot in self.robots:
        #     robot_input = r2_inputs[robot.id]
        #     raw_output = robot.evaluate_proposals(robot_input, r2_inputs[robot.id]['discussion_rules'], MCoCo_Decision_COTs)
        #     raw_outputs.append(raw_output)

        # 直接随机一个robot作为魔鬼代言人 目前是双机器人版本
        # print(len(self.robots))
        self.is_History = False
        devil_robot_number = random.randint(1, len(self.robots))
        for agent in r2_inputs:
            if r2_inputs[agent]['your_previous_proposal']['type'] == 'history':
                self.is_History = True
                break


        # targets_to_remove = {agent_data['your_previous_proposal']['target'] for agent_data in r2_inputs.values()} | \
        #             {proposal['target'] for agent_data in r2_inputs.values() for proposal in agent_data['other_agents_proposals']}
        # for agent_data in r2_inputs.values():
        #     removed_target = [t for t in agent_data['available_targets'] if t not in targets_to_remove]
        # devil_prompt = form_devils_advocate_prompt(devil_robot_number, r2_inputs[devil_robot_number], removed_target)
        R3_vote = {}
        if not self.is_History:
            devil_prompt = form_devils_advocate_prompt(devil_robot_number, r2_inputs[devil_robot_number])
            # print(devil_prompt)
            
            # print("=============================================================================")
            # 3层思维链
            _, Devil_Advocate_Information = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[devil_robot_number-1]['Meta_Prompt'],
                        User_Prompt2=MCoCo_Decision_COTs[devil_robot_number-1]['Frontier_Choice_Prompt'],
                        User_Prompt3=devil_prompt,
                        cot_pred1=MCoCo_Decision_COTs[devil_robot_number-1]['Meta_Choice'],
                        cot_pred2=MCoCo_Decision_COTs[devil_robot_number-1]['Choice_Reason'],
                        return_string_probabilities=None,img=MCoCo_Decision_COTs[devil_robot_number-1]['sem_map_frontier'])

            # _, Devil_Advocate_Information = self.vlm_models.simple_image_chat(User_Prompt=devil_prompt, 
            #                                                                 return_string_probabilities=None, 
            #                                                                 img=MCoCo_Decision_COTs[devil_robot_number-1]['sem_map_frontier'])
            # print(Devil_Advocate_Information)
            Devil_Advocate = {}
            frontier_distance = {}
            try:
                if bool(re.search(r'^\s*```json\b', Devil_Advocate_Information)):
                    Devil_Advocate_Json = re.search(r'```(?:json)?\n([\s\S]*?)\n```', Devil_Advocate_Information)
                    Devil_Advocate = json.loads(Devil_Advocate_Json.group(1))
                else:
                    Devil_Advocate = json.loads(Devil_Advocate_Information)
            except Exception as e:
                logging.error(f"Devil Advocate failed: {e}")
                Devil_Advocate = {}

            # print(Devil_Advocate)
            Frontier_Distances = {}
            if len(Devil_Advocate) != 0:
                for robot in self.robots:
                    # print("robot.frontier_coords:",robot.frontier_coords)
                    all_frontier_letters = ['A', 'B', 'C', 'D'][:len(robot.frontier_coords[robot.id-1])]
                    position = self.robot_positions.get(robot.id, self.cur_start)
                    if position is None:
                        raise ValueError(f"missing map position for robot {robot.id}")
                    Frontier_Distances[robot.id] = calculate_frontier_distance(robot.frontier_coords[robot.id-1], all_frontier_letters, position)
                    if robot.id != devil_robot_number:
                        R2_decision_prompt = form_decision_update_prompt(robot.id,
                            rebuttal_reason=Devil_Advocate['rebuttal']['rebuttal_reason'],
                            new_target=Devil_Advocate['required_adjustment']['new_target'],
                            recommendation=Devil_Advocate['required_adjustment']['recommendation'],
                            available_targets=self.available_targets,
                            agent_data=r2_inputs[robot.id])
                        # print(R2_decision_prompt)
                        # 3层思维链(决策)
                        Last_Choice_Score, Last_Choice = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[robot.id-1]['Meta_Prompt'],
                                    User_Prompt2=MCoCo_Decision_COTs[robot.id-1]['Frontier_Choice_Prompt'],
                                    User_Prompt3=R2_decision_prompt,
                                    cot_pred1=MCoCo_Decision_COTs[robot.id-1]['Meta_Choice'],
                                    cot_pred2=MCoCo_Decision_COTs[robot.id-1]['Choice_Reason'],
                                    return_string_probabilities="[A, B, C, D]",img=MCoCo_Decision_COTs[robot.id-1]['sem_map_frontier'])
                        # 获取origin Decision
                        Origin_Decision_Score = robot.origin_Decision_PR
                        logging.info(f"Robot {robot.id} - Last_Choice_Score: {Last_Choice_Score}")
                        # print(Last_Choice)
                        # print(Origin_Decision_Score)
                        assert len(Last_Choice_Score) == len(Origin_Decision_Score)

                        Decision_Score[robot.id] = optimal_weighted_combination(
                            Last_Choice_Score, Origin_Decision_Score,
                            weights=(self.decision_lambda, 1.0 - self.decision_lambda))
                    else:
                        Decision_Score[robot.id] = list(robot.origin_Decision_PR)

                # 投票
                R3_vote = calculate_R3_weighted_scores(
                    Decision_Score, Frontier_Distances, alpha=self.voting_alpha)
                logging.info(f"Frontier_Distances: {Frontier_Distances}") #{1: {'A': 32.28002478313795, 'B': 82.00609733428362, 'C': 75.96051605933177, 'D': 102.95630140987001}, 2: {'A': 32.28002478313795, 'B': 82.00609733428362, 'C': 75.96051605933177, 'D': 102.95630140987001}}
                logging.info(f"Decision_Score: {Decision_Score}") #{1: [0.277734375, 0.2745365853658536, 0.11404878048780487, 0.33411971227134146], 2: [0.0, 0.3473170731707317, 0.14536585365853658, 0.5073170731707317]}
                logging.info(f"Voting_Score: {R3_vote}")
                # print(self.available_targets) #['A', 'B', 'C', 'D']

        # 构建R2输出（投票固定值矩阵)
        ## 这个仅在消融中使用(消融-去掉R3)
        # if len(self.available_targets) > 0:
        #     try:
        #         self.Final_Decision = {agent: self.available_targets[score.index(max(score))] for agent, score in Decision_Score.items()}
        #     except Exception as e:
        #         logging.error(f"Final_Decision failed: {e}")
        #         logging.info(self.available_targets)
        #         logging.info(Decision_Score)
                
        
        self.r3_inputs = R3_vote
    

    # R3（拍卖）
    def run_r3_discussion(self):
        """执行R3阶段完整流程"""
        # 1. 初始化R3输入包
        r3_inputs = self.r3_inputs
        # print("r3_inputs:", r3_inputs)


        # 2. 开始拍卖
        ## 定义类
        self.auction_result = None
        self.Final_Bundles = {}
        if r3_inputs and not self.is_History:
            agents = []
            for robot in self.robots:
                agents.append(robot.id)
            auction = PriorityAuctionSystem(
                agents=agents, targets=self.available_targets, r3_inputs=r3_inputs,
                valuation_scale=self.valuation_scale, reserve_price=self.reserve_price)
            assignments = auction.auction_process()
            # Commit only a complete, validated VCG settlement. Payments are
            # consumed here with the resulting net utilities, not used to rerank.
            result = auction.get_settled_result()
            if not result.settled:
                raise ValueError("auction returned an unsettled allocation")
            self.auction_result = result
            self.Final_Bundles = dict(result.bundles)
            self.Final_Decision = assignments
            logging.info("VCGSettlement: %s", {
                "round_id": self.round,
                "consensus_lambda": self.decision_lambda,
                "voting_alpha": self.voting_alpha,
                "auction_beta": self.valuation_scale,
                "reserve_price": self.reserve_price,
                **result.as_dict(),
            })
        return self.auction_result
        

    def _prepare_r3_inputs(self, raw_outputs):
        """构建R3输入包"""
        r3_inputs = {}
        for robot in self.robots:
            r3_inputs[robot.id] = {
                "current_round": "R3",
                "task_context": f"Navigate to find {robot.target_object}",
                "your_previous_proposal": next(o for o in raw_outputs if o['agent'] == robot.id),
                "debate_history": self._flatten_discussion_history(),
                "other_agents_proposals": [p for p in raw_outputs 
                                         if p['agent'] != robot.id],
                "available_targets": self.available_targets,
                "discussion_rules": "Synthesize debate history to make final decision."
            }
        return r3_inputs
    def _flatten_discussion_history(self):
        """将讨论历史展平为单个列表"""
        flattened = []
        # print(self.discussion_history)
        for round in self.discussion_history:
            for proposal in round:
                record = {
                    "agent": proposal["agent"],
                    # "proposal": proposal["proposal"],
                    "critique": proposal.get("critique", {}),
                }
                flattened.append(record)
        return flattened
    

    def _broadcast_messages(self, current_round):
        """消息传递机制"""
        if current_round > 1:
            # 将上一轮所有提案同步到接收队列
            last_round_proposals = self.discussion_history[-1]
            for robot in self.robots:
                robot.receive_messages(last_round_proposals)

    # def _generate_visual_prompt(self, robot):
    #     """生成带标注的全局语义地图图像 (视觉输入)"""
    #     map_image = self.global_map.render()
    #     # 添加机器人专属标注
    #     map_image.annotate_position(robot.id, color='red')  # 当前位置
    #     map_image.annotate_frontiers(['A','B','C','D'], color='black')  # 前沿点
    #     map_image.annotate_history_nodes(['a','b','c'], color='gray')  # 历史节点
    #     return map_image

    def _generate_text_prompt(self, robot, round_num):
        """构建结构化文本输入 (参考设计文档输入格式)"""
        return {
            "current_round": f"R{round_num}",
            "task_context": f"Navigate to find {robot.target_object}.",
            "your_previous_proposal": robot.last_proposal if round_num > 1 else None,
            "other_agents_proposals": [p for p in self.discussion_history[-1] 
                                      if p['agent'] != robot.id] if self.discussion_history else [],
            "available_targets": {
                "frontier_points": robot.get_frontier_scores(),
                "history_nodes": robot.get_history_scores()
            },
            "discussion_rules": self._get_discussion_rules(round_num)
        }

    def _get_discussion_rules(self, round_num):
        """获取各轮讨论规则"""
        rules = {
            1: "R1: Share proposal with detailed reason",
            2: "R2: Critique others' proposals (e.g., unreachable, inconsistent)",
            3: "R3: Finalize proposal based on debate"
        }
        return rules[round_num]

    def _update_discussion_history(self, proposals):
        """更新讨论历史记录"""
        self.discussion_history.append(proposals)

    def _conduct_voting(self):
        """执行投票阶段"""
        votes = {}
        for robot in self.robots:
            final_proposal = self.discussion_history[-1][robot.id]
            target = final_proposal['proposal']['target']
            votes[target] = votes.get(target, 0) + 1
        return votes

    def _is_tie(self, votes):
        """检测是否平票"""
        max_votes = max(votes.values())
        return list(votes.values()).count(max_votes) > 1

    def _resolve_tie(self, votes):
        """启用秘书模式处理平票 (参考CMD阶段3)"""
        if self.secretary_mode:
            secretary = SecretaryAgent(self.vlm_models['decision'])
            return secretary.make_final_decision(votes, self.discussion_history[-1])
        else:
            # 默认返回最高置信度目标
            all_proposals = [p['proposal'] for p in self.discussion_history[-1]]
            return max(all_proposals, key=lambda x: x['confidence'])['target']


class RobotAgent:
    def __init__(self, robot_id, vlm_models):
        """
        初始化机器人代理
        :param robot_id: 机器人唯一标识
        :param target_object: 目标物体名称
        :param perception_module: 感知模块 (输出HFOVS/DS/HS/FN_PR)
        :param decision_module: 决策模块 (VLM驱动)
        """
        self.id = robot_id
        self.vlm_models = vlm_models
        self.global_map = None
        self.target_object = None
        self.ES = None
        self.DS = None
        self.last_proposal = None
        self.JS = None
        self.target_point = None
        self.frontier_coords = None #前沿点信息（每个智能体）
        self.history_coords = None #历史点坐标
        self.origin_Decision_PR = None
        self.navigation_mode = None

    def receive_messages(self, messages):
        """接收组内其他代理的消息"""
        self.decision.update_context(messages)

    def propose(self, vlm_output,):
        """生成导航提案 (参考多模态提示设计)"""
        # 1. 调用VLM生成理由和批判
        # vlm_output = self.decision.generate_response(
        #     visual_prompt=visual_input,
        #     text_prompt=text_input
        # )
        
        # 2. 构建结构化输出
        # Use the exact mode selected by the main navigation loop. The fallback
        # preserves compatibility with legacy callers that do not set it.
        proposal_type = self.navigation_mode
        if proposal_type is None:
            proposal_type = "frontier" if self.JS >= 0.3 else "history"
        # proposal_type = "history" # debug历史节点，记得该回去
        
        response = {
            "agent": self.id,
            "proposal": {
                "type": proposal_type,
                "target": self.target_point,
                "confidence": self.DS,
                "reason": vlm_output,
            },
            "critique": ''
        }
        
        # 3. 更新最后提案
        self.last_proposal = response
        return response

    # For R2
    def evaluate_proposals(self, r2_input, discussion_rules, MCoCo_Decision_COTs):
        """
        执行批判性分析并生成输出
        Args:
            r2_input: R2输入包(包含其他代理提案)
        Returns:
            结构化输出（含批判和更新提案）
        """
        # 1. 解析输入
        other_proposals = r2_input["other_agents_proposals"]
        my_proposal = r2_input["your_previous_proposal"]
        
        # 2. 执行批判分析
        critiques = []
        for proposal in other_proposals:
            critique = self._analyze_proposal(proposal, my_proposal["target"], discussion_rules, MCoCo_Decision_COTs)
            critiques.append(critique)
        
        # print(critiques)
        
        # 3. 生成更新提案（可选）
        # updated_proposal = self._update_proposal(my_proposal, other_proposals)
        
        # 4. 构建输出
        return {
            "agent": self.id,
            "critique": critiques[0] if critiques else {},  # 示例：仅批判第一个提案
            "available_targets": r2_input["available_targets"]
        }
    # For R2
    def _analyze_proposal(self, other_proposal, my_target, discussion_rules, MCoCo_Decision_COTs):
        """分析单个提案"""
        # print("my_target:",my_target)
        target = other_proposal["target"]
        # print(other_proposal)
        # print(self.frontier_coords)[n]
        # print(self.history_coords)[[x,y]]

        all_frontier_letters = f"[{'/'.join(['A', 'B', 'C', 'D'][:len(self.frontier_coords[0])])}]"
        frontier_key = get_frontier_key_by_letter(self.frontier_coords[other_proposal['agent']-1], target)

        # 1. 可达性检查
        # print(is_frontier_valid(self.global_map, self.frontier_coords[other_proposal['agent']-1], frontier_key))
        access, coords = is_frontier_valid(self.global_map, self.frontier_coords[other_proposal['agent']-1], frontier_key)
        # print(target)
        # print(all_frontier_letters)
        #### TEST!!!
        # print(MCoCo_Decision_COTs)
        # R2Prompt = form_R2Prompt_acc(
        #         robot=other_proposal['agent'],
        #         discussion_rules=discussion_rules,
        #         my_target = my_target,
        #         target=target,
        #         converted0 = coords[0],
        #         converted1 = coords[1],
        #         choice=all_frontier_letters,
        #     )
        # if other_proposal['type'] == 'frontier':
        #     _, Choice_Reason2 = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Prompt'],
        #         User_Prompt2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Frontier_Choice_Prompt'],
        #         User_Prompt3=R2Prompt,
        #         cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Choice'],
        #         cot_pred2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
        #         return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map_frontier'])
        # else: # history
        #     _, Choice_Reason2 = self.vlm_models.COT2(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['History_Choice_Prompt'],
        #         User_Prompt2=R2Prompt,
        #         cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
        #         return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map'])
        # print(R2Prompt)
        # try:
        #     Choice_Reason2 = json.loads(Choice_Reason2.strip())
        #     choice = Choice_Reason2["Choice"]
        #     reason = Choice_Reason2["Reason"]
        #     a = {
        #         "target": target,
        #         "error_type": "accessibility",
        #         "evidence": f"The target {target} is located in the obstacle zone (coordinates x={coords[0]},y={coords[1]})",
        #         "suggestion": f"Navigate to {choice}",
        #         "reason": reason
        #     }
        #     print(a)
        #     return {
        #         "target": target,
        #         "error_type": "accessibility",
        #         "evidence": f"The target {target} is located in the obstacle zone (coordinates x={coords[0]},y={coords[1]})",
        #         "suggestion": f"Navigate to {choice}",
        #         "reason": reason
        #     }
        # except Exception as ex:
        #     logging.info(f"[R2_acc VLM JSON ERROR]: {ex}")
        #     return {
        #         "target": target,
        #         "state_type": "valid",
        #         "suggestion": f"Robot {other_proposal['agent']} maintains target {target}."
        #     }
        #### TEST!!!
        if not access:
            # 构建R2提示(json)
            R2Prompt = form_R2Prompt_acc(
                robot=other_proposal['agent'],
                discussion_rules=discussion_rules,
                my_target = my_target,
                target=target,
                converted0 = coords[0],
                converted1 = coords[1],
                choice=all_frontier_letters,
            )
            if other_proposal['type'] == 'frontier':
                _, Choice_Reason2 = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Prompt'],
                    User_Prompt2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Frontier_Choice_Prompt'],
                    User_Prompt3=R2Prompt,
                    cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Choice'],
                    cot_pred2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
                    return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map_frontier'])
            else: # history
                _, Choice_Reason2 = self.vlm_models.COT2(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['History_Choice_Prompt'],
                    User_Prompt2=R2Prompt,
                    cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
                    return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map'])
            # print(R2Prompt)
            try:
                Choice_Reason2 = json.loads(Choice_Reason2.strip())
                choice = Choice_Reason2["Choice"]
                reason = Choice_Reason2["Reason"]
                return {
                    "target": target,
                    "error_type": "accessibility",
                    "evidence": f"The target {target} is located in the obstacle zone (coordinates x={coords[0]},y={coords[1]})",
                    "suggestion": f"Navigate to {choice}",
                    "reason": reason
                }
            except Exception as ex:
                logging.info(f"[R2_acc VLM JSON ERROR]: {ex}")
                return {
                    "target": target,
                    "state_type": "valid",
                    "suggestion": f"Robot {other_proposal['agent']} maintains target {self.target}."
                }
        
        # 2. 覆盖矛盾检查 由于之前已经修改过策略（强制选择不同的fronttier），此点可以忽略
        # overlap = self._calculate_coverage_overlap(my_target, target)
        # if overlap > 0.7:
        #     return {
        #         "target": target,
        #         "error_type": "coverage_conflict",
        #         "evidence": f"{target}与我的目标重叠率达{overlap*100:.0f}%",
        #         "suggestion": self._find_alternative_target(my_target)
        #     }


        
        # 3. 目标一致性检查(目前存在大量幻觉)
        # R2Prompt = form_R2Prompt_es(
        #         robot=other_proposal['agent'],
        #         discussion_rules=discussion_rules,
        #         goal = self.target_object,
        #         my_target = my_target,
        #         target=target,
        #         choice=all_frontier_letters,
        #         es = self.ES,
        #     )
        # if other_proposal['type'] == 'frontier':
        #     _, Choice_Reason2 = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Prompt'],
        #         User_Prompt2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Frontier_Choice_Prompt'],
        #         User_Prompt3=R2Prompt,
        #         cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Choice'],
        #         cot_pred2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
        #         return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map_frontier'])
        # else: # history
        #     _, Choice_Reason2 = self.vlm_models.COT2(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['History_Choice_Prompt'],
        #         User_Prompt2=R2Prompt,
        #         cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
        #         return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map'])
        # print(R2Prompt)
        # try:
        #     Choice_Reason2 = json.loads(Choice_Reason2.strip())
        #     choice = Choice_Reason2["Choice"]
        #     reason = Choice_Reason2["Reason"]
        #     a= {
        #         "target": target,
        #         "error_type": "target_consistency",
        #         "evidence": f"{target} location exists with {self.target_object} probability only {self.ES}",
        #         "suggestion": f"Navigate to {choice}",
        #         "reason": reason
        #     }
        #     print(a)

        #     return {
        #         "target": target,
        #         "error_type": "target_consistency",
        #         "evidence": f"{target} location exists with {self.target_object} probability only {self.ES}",
        #         "suggestion": f"Navigate to {choice}",
        #         "reason": reason
        #     }
        # except Exception as ex:
        #     logging.info(f"[R2_acc VLM JSON ERROR]: {ex}")
        #     return {
        #         "target": target,
        #         "state_type": "valid",
        #         "suggestion": f"Robot {other_proposal['agent']} maintains target {target}."
        #     }
        
        if self.ES < 0.1:
            # 构建R2提示(json)
            R2Prompt = form_R2Prompt(
                robot=other_proposal['agent'],
                discussion_rules=discussion_rules,
                goal = self.target_object, # 目标(bed\chair)
                my_target = my_target,
                target=target,
                choice=all_frontier_letters,
                es = self.ES,
            )
            if other_proposal['type'] == 'frontier':
                _, Choice_Reason2 = self.vlm_models.COT3(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Prompt'],
                    User_Prompt2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Frontier_Choice_Prompt'],
                    User_Prompt3=R2Prompt,
                    cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Meta_Choice'],
                    cot_pred2=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
                    return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map_frontier'])
            else: # history
                _, Choice_Reason2 = self.vlm_models.COT2(User_Prompt1=MCoCo_Decision_COTs[other_proposal['agent']-1]['History_Choice_Prompt'],
                    User_Prompt2=R2Prompt,
                    cot_pred1=MCoCo_Decision_COTs[other_proposal['agent']-1]['Choice_Reason'],
                    return_string_probabilities=None,img=MCoCo_Decision_COTs[other_proposal['agent']-1]['sem_map'])
            # print(R2Prompt)
            try:
                Choice_Reason2 = json.loads(Choice_Reason2.strip())
                choice = Choice_Reason2["Choice"]
                reason = Choice_Reason2["Reason"]

                return {
                    "target": target,
                    "error_type": "target_consistency",
                    "evidence": f"{target} location exists with {self.target_object} probability only {self.ES}",
                    "suggestion": f"Navigate to {choice}",
                    "reason": reason
                }
            except Exception as ex:
                logging.info(f"[R2_acc VLM JSON ERROR]: {ex}")
                return {
                    "target": target,
                    "state_type": "valid",
                    "suggestion": f"Robot {other_proposal['agent']} maintains target {self.target}."
                }

        
        # 4. 有效提案
        return {
            "target": target,
            "state_type": "valid",
            "suggestion": f"Robot {other_proposal['agent']} maintains target {target}."
        }

    # def _select_best_target(self, proposal_type):
    #     """根据置信度选择最优目标"""
    #     scores = self.get_frontier_scores() if proposal_type == "frontier" else self.get_history_scores()
    #     return max(scores, key=scores.get)

    # def _get_confidence(self, target):
    #     """获取固定置信度值 (DS或HS)"""
    #     if self.last_proposal and self.last_proposal['proposal']['target'] == target:
    #         return self.last_proposal['proposal']['confidence']
    #     # 首次选择时直接查询模块输出
    #     scores = self.get_frontier_scores() if self.FN_PR >= 0.5 else self.get_history_scores()
    #     return scores[target]

    # R3
    def generate_final_proposal(self, r3_input):
        """
        生成最终提案R3
        Args:
            r3_input: R3输入包
        Returns:
            结构化输出（含最终提案、共识、覆盖分析）
        """
        # 1. 解析输入
        debate_history = r3_input["debate_history"]
        my_last_proposal = r3_input["your_previous_proposal"]
        
        # 2. 分析辩论关键点
        supporting_evidences = self._find_supporting_points(debate_history)
        opposing_evidences = self._find_strong_oppositions(debate_history)
        complementary_targets = self._find_complementary_targets(debate_history)
        
        # 3. 决策树：是否切换目标？
        if opposing_evidences and not self._can_rebut(opposing_evidences):
            new_target = self._select_best_alternative(complementary_targets)
            updated_proposal = self._build_proposal(
                new_target, "接受反对证据，切换目标"
            )
        elif supporting_evidences and self._has_coverage_advantage():
            updated_proposal = self._build_proposal(
                my_last_proposal["target"], "强化支持证据"
            )
        else:
            updated_proposal = self._build_proposal(
                my_last_proposal["target"], "维持原方案，补充新见解"
            )
        
        # 4. 生成最终批判
        critique = self._generate_critique(debate_history)
        
        # 5. 生成覆盖分析
        coverage_analysis = self._generate_coverage_analysis(updated_proposal["target"])
        
        # 6. 构建输出
        return {
            "agent": self.id,
            "proposal": updated_proposal,
            "critique": critique,
            "coverage_analysis": coverage_analysis,
            "available_targets": r3_input["available_targets"]
        }



class SecretaryAgent:
    def __init__(self, vlm_model):
        """平票时的秘书代理"""
        self.vlm = vlm_model

    def make_final_decision(self, votes, discussion_data):
        """基于讨论历史做出最终决策"""
        # 构造秘书专用提示
        prompt = {
            "discussion_summary": discussion_data,
            "tie_votes": votes,
            "instruction": "Select the most plausible navigation target considering coverage and accessibility"
        }
        
        # 调用VLM进行最终判断
        decision = self.vlm.query(prompt)
        return decision['final_target']
