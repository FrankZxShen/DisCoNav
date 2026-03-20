import re

### 感知
# Caption_Prompt = '''
# You are a knowledgeable and skilled expert in indoor navigation planning for multiple robots. Imagine you are standing in the center of a room. Your task is to describe the entire scene around you with a focus on ​​passage area awareness—specifically detailing navigable corridors, doorways, passages, thresholds, and open floor space. Highlight clearances, obstructions, and access points that would impact robot movement.

# Explain each of the following criteria clearly and unambiguously:

# **Decision Criteria:**

# 1. **Passage Availability:**
#  - Identify doors, walkways, corridors, gaps.
# 2. **Clearance:**
#  - Note door, corridor, spatial buffers around obstacles.
# 3. **Transitions:**
#  - Mark connections to adjacent spaces (e.g., doors, hall entries, staircases).
# ------

# **Output Format:**
#  - [Passage Availability]
#  - [Clearance]
#  - [Transitions]
# '''

# Perception_System_Prompt = """
# Based on your description, you need to determine whether the current scene is worth exploring for the robot, based on the provided information. Follow the guidelines below to make your decision clearly and explicitly.

# **Information Format:**

# - **Target of Navigation:** Format - <Name>
# - **Scene Object (Object Detection):** Format - <Name, Confidence Score>

# **Decision Criteria:**

# 1. **Analyzing scene passability (Primary Criterion):**
#    - A scene is usually worth exploring if there are corridors and doors in the scene.
# 2. **High-Confidence Target Detection (Secondary Criterion):**
#    - If the object detection model predicts the presence of the target object with a high probability (e.g., >85%), determine if the scene is worth exploring.
# ------

# **Output Format:**

# Only return one of the following:
# Your recommendation should be a simple "[Yes, No]" statement:
# - **Yes:** The scene is worth exploring.
# - **No:** The scene is not worth exploring.
# ------
# """
Caption_Prompt = '''
You are an expert in indoor robot navigation and spatial affordance perception. Imagine you are a robot standing at the center of the scene shown in the image. Your task is to analyze and describe the space specifically from a **navigation and exploration** perspective.

Focus on identifying navigable paths, spatial connectivity, and structural features that indicate potential exploration opportunities. Avoid excessive attention to decorative or static object detail unless it affects navigation.

Please describe the scene using the following structure:

------
Exploration-Focused Scene Description:

1. Navigable Regions and Paths:
   - Describe all visible traversable regions that a robot could potentially navigate into (e.g., open hallways, visible doorways, connected floor spaces).
   - Indicate possible new areas to explore.

2. Obstacles and Blockages:
   - Identify any major obstructions (e.g., large furniture, walls, doors).
   - Describe how these elements limit navigation (e.g., fully block a hallway, partially occlude a doorway, block one side of the room).
   - Note whether blocked areas appear to have potential exploration value (e.g., a closed door that may lead to a new room).

3. Structural Layout Cues:  
   - Describe doors, doorways, arches, stairs, or large transitions that hint at spatial expansion.
   - Include flooring continuity, wall alignment that imply connectivity.

4. Goal Relevance Assessment
   - Evaluate whether the current scene is semantically related to the navigation target (e.g., if the target is a “bed”, but the current room appears to be a bathroom, exploration is likely unnecessary).
   - Consider the presence or absence of scene-relevant objects, furniture, and room layout.
   - Clearly state whether the scene seems promising for finding the target or should be deprioritized.

------

Be concise and precise. Focus on spatial affordances and traversability that a robot would use to plan its next move.
'''

Perception_System_Prompt = """
Based on your description, your task is to determine whether the current scene is worth exploring, based on a specified **navigation target** and a list of **detected scene objects**.

You must follow a clear and strict reasoning procedure based on the following rules:

------

**Input Format:**

- **Navigation Target:** <Name>  
- **Scene Objects (Detected):** <Object Name, Confidence Score>

------

**Decision Guidelines:**

1. **High-Confidence Target Detection (Primary Criterion):**
   - If the navigation target appears in the scene with a **confidence > 85%**, respond with **Yes** — the scene is worth exploring.
   - If not, proceed to contextual reasoning.

2. **Scene Contextual Reasoning (Secondary Criterion):**
   - Consider typical environments where the target object is usually found.
     - Example: **Bed** → bedroom, **Toilet** → bathroom, **TV** → living room.
   - If detected objects strongly suggest the correct scene type, exploration may still be worthwhile.

3. **Contextual Proximity Cues (Tertiary Clues):**
   - Consider indirect cues like **doors**, **walls**, or **room layout** that might suggest the target object is nearby.
   - Only support **Yes** if these cues align with typical co-location of the target and detected objects.

------

**Output Format:**

Only return one of the following:

- **[Yes]** - The scene is worth exploring.
- **[No]** - The scene is not worth exploring.

------
"""

### F/N
FN_System_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. Based on the current top-down semantic map of an indoor environment provided, you will see various points marked as either 'Frontier Points' or 'Historical Observation Points'. Frontier points represent unexplored areas that the robot has yet to navigate, while Historical Observation Points signify areas the robot has previously explored or observed.

**Information Format:**
- **Your Navigation Target:** Format - `<Name>`
- **Scene Objects:** Format - `<Name: [Coordinates: <(x1, y1), (x2, y2)...>]>`
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** Format - `<Name: [Coordinates: <(x1, y1), (x2, y2)...>]>`
- **Historical Observation Points (The green dots and corresponding green lowercase letters on the image):** Format - `<Name: [Coordinates: <[x1, y1], [x2, y2]...>]>`
- **Your location (The red arrow):** Format - `<(x, y)>`
- **Previous Movement:** Format - `<(x, y)>`

Your goal is to guide the robot for exploration purposes based on the relationship between different objects, the structure of the explored area, the robot's position, the proximity of exploration points, and the direction of previous movement. Consider the following factors when making your decision:

1. **Level of Exploration:**
   - If there are a very small number of Historical Observation Points and a large number of gaps in the semantic map, prefer exploring Frontier Points.

2. **Explorability Worthiness:**
   - Scenes that are worth exploring ({ISWORTH}) are usually more likely to be explored by choosing Frontier Points.

3. **Proximity and Accessibility:**
   - Evaluate how Your location relates to surrounding obstacles. Frontier Points or Historical Observation Points that are close and free of obstacles tend to have higher exploration priority.

4. **Relationship Between Location and Previous Movement:**
   - If Your location is too close to Previous Movement, it may indicate a collision trap. In such cases, prefer to explore Historical Observation Points that are close to Your location.

**Decision Format:**
Your recommendation should be a simple "[Yes, No]" statement:
- **Yes:** Explore a Frontier Point.
- **No:** Revisit a Historical Observation Point.

**Example:**

---

**Your Navigation Target**: `TargetObject`

**Scene Objects**:
- `Chair: [Coordinates: (3, 4), (2, 3)]`
- `Table: [Coordinates: (5, 6), (6, 7)]`

**Frontier Points (The black dots and corresponding black uppercase letters on the image)**:
- `A: [Coordinates: (8, 9)]`
- `B: [Coordinates: (10, 11)]`

**Historical Observation Points (The green dots and corresponding green lowercase letters on the image)**:
- `a: [Coordinates: (1, 2)]`
- `b: [Coordinates: (3, 5)]`

**Your location (The red dot)**: `(4, 4)`

**Previous Movement**: `(2, 2)`


**Decision:** Yes

---

Now, begin your analysis with the provided scene image information.

- **Your Navigation Target:** `{TARGET}`
- **Scene Objects:** `{SCENE_OBJECTS}`
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** `{FRONTIERS_RESULTS}`
- **Historical Observation Points (The green dots and corresponding green lowercase letters on the image):** `{HISTORY_NODES}`
- **Your location (The red arrow):** `{CUR_LOCATION}`
- **Previous Movement:** `{PRE}`

Would you recommend:
Yes) Exploring a frontier point? If so, answer ONLY Yes.
No) Revisiting a historical observation point? If so, answer ONLY No.

---
'''

### 决策COT1
Single_Agent_Decision_Prompt_Frontier1 = '''
You are a knowledgeable and skilled expert in indoor navigation planning. Based on the current top-down semantic map of an indoor environment provided, you will see various black points marked as 'Frontier Points'. Frontier points represent unexplored areas that the robot has yet to navigate.

Your task is to guide the robot for exploration purposes based on the relationship between different objects, the structure of the explored area, the robot's position, the proximity of the exploration point, and the direction of previous movement. Consider the following factors when making your decision:

1. **Proximity and Accessibility:**
   - Evaluate how Your location relates to surrounding obstacles. Frontier Points that are close and free of obstacles tend to have higher exploration priority.
2. **Relationship Between Location and Previous Movement:**
   - If Your location is too close to Previous Movement, it may indicate the robot is entering a collision trap. Prefer to explore Frontier Points that are farther from Your location.
3. **Exploration Consistency:**
   - Minimize frequent switches between Frontier Points. The robot should maintain its exploration direction unless an efficient switch is evident.
4. **Target-Oriented Exploration:**
   - If there are accessible Frontier Points in the direction of Your location (red arrow), and the scenario information includes Your Navigation Target, give the highest priority to exploring these Frontier Points without violating point (1).

**Your Recommendation:**
 Describe the information around each Frontier Point in the diagram and indicate whether you are likely to choose it for exploration based on the provided criteria.

------

**Example:**

**Your Navigation Target:** TargetObject

**Scene Objects:**

- Chair: [Coordinates: (3, 4), (2, 3)]
- Table: [Coordinates: (5, 6), (6, 7)]

**Frontier Points (The black dots and corresponding black uppercase letters on the image):**

- A: [Coordinates: (8, 9)]
- B: [Coordinates: (10, 11)]

**Your location (The red arrow):** (4, 4)

**Previous Movement:** (2, 2)

**Analysis:**

1. **Proximity and Accessibility:** Frontier Point A at (8, 9) is relatively close and free of obstacles.
2. **Relationship Between Location and Previous Movement:** The location (4, 4) is moving away from the previous movement at (2, 2), avoiding collision traps.
3. **Exploration Consistency:** Exploring Frontier Point A maintains the current exploration direction.
4. **Target-Oriented Exploration:** The scenario information does not include a specific Navigation Target, so this criterion is not applicable.

**Recommendation:** I am likely to choose Frontier Point A for exploration.

------

Now, begin your analysis with the provided scene information:

- **Your Navigation Target:** <{TARGET}>
- **Scene Objects:** <{SCENE_OBJECTS}>
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** <{FRONTIERS_RESULTS}>
- **Your location (The red arrow):** <{CUR_LOCATION}>
- **Previous Movement:** <{PRE}>
- **Your location facing (the direction indicated by the red arrow) scene information:** <{SCENE_INFORMATION}>

Please synthesize the above information to choose a Frontier Point in 'A', 'B', 'C', 'D' to explore.

------

Explanation Ends.
**Output Format:**
Your choice MUST in {AVAILABLE_TARGETS} **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''
### 决策COT2



### 决策2 暂时废弃
Single_Agent_Decision_Prompt_History = '''
Your choice is 'Historical Observation Points'. So you will now re-explore historical nodes in this indoor environment. 

The green dots and corresponding green lowercase letters on the left image represent the Historical Observation Points awaiting your exploration: 
{HISTORY_NODES}
where the confidence level following each green lowercase letter represents the exploration likelihood that this point was recorded. Point with higher confidence have higher exploration priority.
The Arrow is Your Location: {CUR_LOCATION}
Previous Movement: {PRE}

Your goal is to find the {TARGET}. You need to consider the following factors:
(1) Consider the proximity and accessibility of the points. You need to consider the proximity and accessibility of the front and your previous movement. Points that are closer and without barriers tend to have a higher exploration priority. 
(2) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(3) Minimize frequent switches between points. Use centroid for frontier selection. You should maintain its exploration direction unless an efficient switch is evident.
You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation point for the next timestep based on their relationship with your navigation goal. 

Your choice must be in 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' and 26 other lowercase letters representing historical observation points  **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''

### 小组讨论

# 生成选择Frontier的原因
# Frontier_Choice_Reason = '''
# Why did you choose Frontier Point {FRONTIER}? Please tell me the reason in two sentence based on the scene information and the top-down semantic map above.
# Your output should refer to the following format:
# - From the perspective of the scene information above: \n
# - From the perspective of the top-down semantic map:
# '''
Frontier_Choice_Reason = '''
Your choice is Frontier Point {FRONTIER}. Please tell me the reason in one sentence based on the scene information.
Your output should refer to the following format:
- From the perspective of the scene information above:
'''

History_Choice_Reason = '''
Your choice is Historical Observation Point. Please tell me in one sentence why you would choose Historical Observation Point over Frontier Point based on the top-down semantic map above.
Your output should refer to the following format:
- From the perspective of the top-down semantic map:
# '''

# Devil's Advocate - Frontier
Devil_Prompt = '''
You are now operating as a strict frontier target selection skeptic. You are tasked with:** Presenting rebuttals** for other robots' reasons for choosing targets.

**INPUT ANALYSIS**
- YOUR PRIOR PROPOSAL:
  Robot: {AGENT_ID}
  Target: {TARGET}  
  Confidence: {CONFIDENCE} 
  Reason: *"{MY_REASON}"*

- OTHER ROBOT PROPOSALS:
{OTHER_PROPOSAL}
- AVAILABLE TARGETS: {AVAILABLE_TARGETS}
- CRITICAL FOCUS: {CRITICAL_FOCUS}

**EXECUTION PROTOCOL**
1. INCONSISTENCY SCAN:
   - Verify claim inconsistency between YOUR PRIOR TARGET and OTHER PROPOSALS TARGET (if consistent, adjustments need to be made)
   - Identify conflicting accessibility evidence: *"Do clearance restrictions actually permit ALL proposals to reach target?"*

2. FORCE REBUTTAL REQUIREMENT:
   - Mandatory rebuttal of OTHER robots' TARGET
   - For each rebuttal:
        • Propose a **MANDATORY REBUTTAL**, taking into account the distribution of semantic information and frontier targets in the semantic map.
        • Modify Robot {AGENT_ID}'s target to new target [X] and give a recommendation.
        • Ensure that the modified new target **CANNOT be consistent** with YOUR PRIOR PROPOSAL's target.
**OUTPUT FORMAT (STRICT JSON)**
{{
  "rebuttal": {{
    "target": : "<letter>",
    "rebuttal_reason": "<20-word excerpt>"
  }},
  "required_adjustment": {{
    "new_target": "<letter>",
    "recommendation": "<20-word excerpt>"
  }}
}}
'''
# Devil's Advocate - HistoryXXXXX
Devil_Prompt_History = '''
You are now operating as a strict target selection skeptic. You are tasked with:** Presenting rebuttals** for other robots' reasons for choosing targets.

**INPUT ANALYSIS**
- YOUR PRIOR PROPOSAL:
  Robot: {AGENT_ID}
  Target: {TYPE}  
  Reason: *"{MY_REASON}"*

- OTHER ROBOT PROPOSALS:
{OTHER_PROPOSAL}
- AVAILABLE TARGETS: {AVAILABLE_TARGETS}
- CRITICAL FOCUS: {CRITICAL_FOCUS}

**EXECUTION PROTOCOL**
1. INCONSISTENCY SCAN:
   - Verify claim inconsistency between YOUR PRIOR TARGET and OTHER PROPOSALS TARGET (if consistent, adjustments need to be made)
   - Identify conflicting accessibility evidence: *"Do clearance restrictions actually permit ALL proposals to reach target?"*

2. FORCE REBUTTAL REQUIREMENT:
   - Mandatory rebuttal of OTHER robots' TARGET
   - For each rebuttal:
        • Propose a **MANDATORY REBUTTAL**, taking into account the distribution of semantic information and frontier targets in the semantic map.
        • Modify Robot {AGENT_ID}'s target to new target [X] and give a recommendation.
        • Ensure that the modified new target **CANNOT be consistent** with YOUR PRIOR PROPOSAL's target.
**OUTPUT FORMAT (STRICT JSON)**
{{
  "rebuttal": {{
    "target": : "{TARGET}",
    "rebuttal_reason": "<20-word excerpt>"
  }},
  "required_adjustment": {{
    "new_target": "<letter>",
    "recommendation": "<20-word excerpt>"
  }}
}}
'''

R2_Decision_Update_Prompt = '''
Given the following information: 
- Your Choice:
  Target: {MY_TARGET}
  Confidence: {MY_CONFIDENCE}
  Reason: **"{MY_REASON}"**

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



###XXXXXSX
R2Prompt_acc = '''
{discussion_rules}
{{
    "robot:" {ROBOT},
    "target": {TARGET},
    "error_type": "accessibility",
    "evidence": "The target {TARGET} is located in the obstacle zone (coordinates x={converted0},y={converted1})",
}}
Based on the detailed json description of the reachability check above, give your updated choice among the frontier points {CHOICE} and justify it. You need to make sure that your choice cannot be target {MY_TARGET}. Your output should conform to the following json format:
{{
Choice: {CHOICE}
Reason: [Your Reason]
}}
'''

R2Prompt_es = '''
{discussion_rules}
{{
    "robot": {ROBOT},
    "target": {TARGET},
    "error_type": "target_consistency",
    "evidence": "{TARGET} location exists with {GOAL} probability only {ES}",
}}
Based on the detailed json description of the target consistency check check above, give your updated choice among the frontier points {CHOICE} and justify it. You need to make sure that your choice cannot be target {MY_TARGET}. Your output should conform to the following json format:
{{
Choice: {CHOICE}
Reason: [Your Reason]
}}
'''


# If the current scene is worth exploring, you should prioritize exploring frontier points. If the current scene is not worth exploring, you should consider revisiting historical observation points.
# Your Answer MUST in 'Yes', 'No' **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
# '''


# 废弃
# Meta_Agent_Decision_Prompt2 = '''
# Based on the above decisions and analysis, please combine your knowledge to choose the navigation frontier for each robots.

# Your output format is as follows: 

# Agent0: <You need to select the Frontier corresponding to the black capital letter [A, B, C, D]>

# Agent1: <You need to select the Frontier corresponding to the black capital letter [A, B, C, D]>

# ...

# ''' 
# Example:

# Target of navigation: 'bed'

# Scene object (object detection): 
# bench: 0.45, <(186, 325,), (587, 479)>
# bed: 0.65, <(466, 448), (640, 480)>
# dining table: 0.77, <(125, 244), (201, 325)>
# chair: 0.77, <(65, 231), (146, 328)> 
# couch: 0.87, <(169, 247) (640, 475)>

# Scene object (semantic segmentation):
# sofa: <(294, 202), (267, 216), (284, 217), (288, 258), (307, 235)>
# table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

# Are you confident that the robot is worth exploring in this scenario? Please answer Yes or No.

def form_prompt_for_PerceptionVLM(target, objs, yolo) -> str:

    object_detection = ''
    if yolo == 'yolov9':
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for item in objs:
                name, confidence, coords = item
                # coord_pairs = coords[0].split(',')
                # coord1 = coord_pairs[0], coord_pairs[1]
                # coord2 = coord_pairs[2], coord_pairs[3]
                detection = f"  {name}, {confidence}\n"
                object_detection += detection
    else:
        # print(objs)
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for name, confidence in objs.items():
                detection = f"  {name}, {confidence}\n"
                object_detection += detection

    # semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
    #                         for j in range(len(value[i]))]) + f">"
    #                         for i in range(len(value))])
    #                         for key, value in agents_seg_list.items()]) + "\n"

    Perception_Template = """
- **Target of navigation:** {TARGET}

- Scene object (Object Detection): 
{OBJECT_DETECTION}

**Decision:**
"""
    User_Prompt = Perception_Template.format(
                        TARGET = target,
                        OBJECT_DETECTION = object_detection,
                    )
    User_Prompt = Perception_System_Prompt + User_Prompt



    return Caption_Prompt, User_Prompt

def form_prompt_for_FN(target, agents_seg_list, Perception_PR, pre_goal_point, Frontier_list, cur_location, History_nodes) -> str:

    def convert_entry(entry):
        # 从字符串中提取坐标
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'
    
    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"

    # 遍历字典，并使用转换函数
    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'
    
    if pre_goal_point == []:
        pre_goal_point = 'No Movements'

    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a")+i)}: [Coordinates: {History_nodes[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'
    

    if Perception_PR[0] >= 0.50:
        isworth = 'The current scene is worth exploring.'
    else:
        isworth = 'The current scene is not worth exploring.'
    User_Prompt1 = FN_System_Prompt.format(
                        SCENE_OBJECTS = semantic_segmentation,
                        TARGET = target,
                        ISWORTH = isworth,
                        FRONTIERS_RESULTS = Frontiers_results,
                        HISTORY_NODES = His_results,
                        CUR_LOCATION = cur_location,
                        PRE = pre_goal_point, 
                    )


    return User_Prompt1

def form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_point, target, cur_location, Frontier_list, available_targets) -> str:

    def convert_entry(entry):
        # 从字符串中提取坐标和像素计数
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'

    if pre_goal_point == []:
        pre_goal_point = 'No Movements'

    # 遍历字典，并使用转换函数
    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'
    
    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"
    
    User_Prompt = Single_Agent_Decision_Prompt_Frontier1.format(
        SCENE_INFORMATION = Scene_Information,
        SCENE_OBJECTS = semantic_segmentation,
        FRONTIERS_RESULTS = Frontiers_results,
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
        AVAILABLE_TARGETS = available_targets,
    )
    

    return User_Prompt

def form_prompt_for_Frontier_Reason(Frontier) -> str:
    User_Prompt = Frontier_Choice_Reason.format(
        FRONTIER = Frontier, 
    )

    return User_Prompt

def form_prompt_for_History_Reason() -> str:

    User_Prompt = History_Choice_Reason.format()
    return User_Prompt

# 恶魔代理人
def form_devils_advocate_prompt(
    agent_id: int, 
    agent_data: dict, 
    ) -> str:
    # 动态构建"其他代理提案"部分
    other_proposals_text = ""
    for proposal in agent_data['other_agents_proposals']:
        other_proposals_text += (
            f"Robot: {proposal['agent']} | Target: {proposal['target']}\n"
            f"Confidence: {proposal['confidence']:.4f}\n"
            f"Reason: *\"{proposal['reason']}\"*\n\n"
        )
    
    # 移除非必要空格和换行符
    my_reason = agent_data['your_previous_proposal']['reason'].replace('\n', ' ').strip()

    User_Prompt = Devil_Prompt.format(
        AGENT_ID = agent_id,
        TARGET = agent_data['your_previous_proposal']['target'],
        CONFIDENCE = round(agent_data['your_previous_proposal']['confidence'], 4),
        OTHER_AGENTS = len(agent_data['other_agents_proposals']),
        OTHER_PROPOSAL = other_proposals_text.strip(),
        MY_REASON = my_reason,
        AVAILABLE_TARGETS = agent_data['available_targets'],
        CRITICAL_FOCUS = agent_data['critical_focus'],
        # REMOVED_TARGET =  ', '.join(removed_target),
        
    )
    
    # 动态生成提示
    return User_Prompt
# 恶魔代理人-History
def form_devils_advocate_prompt_history(
    agent_id: int, 
    agent_data: dict, 
    ) -> str:
    # 动态构建"其他代理提案"部分
    other_proposals_text = ""
    for proposal in agent_data['other_agents_proposals']:
        other_proposals_text += (
            f"Robot: {proposal['agent']}\n"
            f"Reason: *\"{proposal['reason']}\"*\n\n"
        )
    
    # 移除非必要空格和换行符
    my_reason = agent_data['your_previous_proposal']['reason'].replace('\n', ' ').strip()

    User_Prompt = Devil_Prompt.format(
        AGENT_ID = agent_id,
        TYPE = agent_data['your_previous_proposal']['target'],
        OTHER_AGENTS = len(agent_data['other_agents_proposals']),
        OTHER_PROPOSAL = other_proposals_text.strip(),
        MY_REASON = my_reason,
        AVAILABLE_TARGETS = agent_data['available_targets'],
        CRITICAL_FOCUS = agent_data['critical_focus'],
        # REMOVED_TARGET =  ', '.join(removed_target),
        
    )
    
    # 动态生成提示
    return User_Prompt

# R2反思
def form_decision_update_prompt(
    agent_id: int, 
    agent_data: dict,
    rebuttal_reason: str, 
    new_target: str, 
    recommendation: str, 
    available_targets: str) -> str:

    my_reason = agent_data['your_previous_proposal']['reason'].replace('\n', ' ').strip()
    
    User_Prompt = R2_Decision_Update_Prompt.format(
        AGENT_ID = agent_id,
        MY_TARGET = agent_data['your_previous_proposal']['target'],
        MY_CONFIDENCE = round(agent_data['your_previous_proposal']['confidence'],4),
        MY_REASON = my_reason,
        REBUTTAL_REASON = rebuttal_reason,
        NEW_TARGET = new_target,
        RECOMMENDATION = recommendation,
        AVAILABLE_TARGETS = available_targets
    )
    

    return User_Prompt


def form_R2Prompt_acc(robot, discussion_rules, my_target, target, choice, converted0, converted1) -> str: # taget是新智能体的target，my_target是本智能体的target
    User_Prompt = R2Prompt_acc.format(
        ROBOT = robot,
        discussion_rules = discussion_rules,
        TARGET = target,
        MY_TARGET = my_target,
        converted0 = converted0,
        converted1 = converted1,
        CHOICE = choice,
    )
    return User_Prompt

def form_R2Prompt_es(robot, discussion_rules, goal, my_target, target, choice, es) -> str: # taget是新智能体的target，my_target是本智能体的target
    User_Prompt = R2Prompt_es.format(
        ROBOT = robot,
        discussion_rules = discussion_rules,
        GOAL = goal,
        TARGET = target,
        MY_TARGET = my_target,
        ES = es,
        CHOICE = choice,
    )
    return User_Prompt













# 废弃！！！
def form_prompt_for_DecisionVLM_History(pre_goal_point, target, cur_location, confidence, History_nodes) -> str:

    # def convert_entry(entry):
    #     # 从字符串中提取坐标和像素计数
    #     centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
    #     number_str = entry.split('number: ')[1][:-1]
    #     return f'[Coordinates: {centroid_str}]'
    
    if all(num == 0 for num in pre_goal_point):
        pre_goal_point = 'No Movements'
    # 遍历字典，并使用转换函数
    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a") + i)}: [Coordinates: {History_nodes[i]}; Confidence: {confidence[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'
    
    User_Prompt = Single_Agent_Decision_Prompt_History.format(
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
        HISTORY_NODES = His_results
    )
    

    return User_Prompt







def extract_scene_image_description_results(text):
    pattern = re.compile(r'Scene image description module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scene_object_detection_results(text):
    pattern = re.compile(r'Scene object detection module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scenario_exploration_analysis_results(text):
    pattern = re.compile(r'Scenario exploration analysis module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches

def form_prompt_for_DecisionVLM_MetaPreprocess() -> str:
    return Meta_Agent_Preprocess_Prompt

def form_prompt_for_Module_Decision(target, Preprocess_prompt, objs) -> str:
    Use_prompt = ''
    object_detection = ''
    for item in objs:
        name, confidence, coords = item
        coord_pairs = coords[0].split(',')
        coord1 = coord_pairs[0], coord_pairs[1]
        coord2 = coord_pairs[2], coord_pairs[3]
        detection = f"{name}: {confidence}, <({int(float(coord1[0]))}, {int(float(coord1[1]))}), ({int(float(coord2[0]))}, {int(float(coord2[1]))})>\n"
        object_detection += detection
    task_prompt_decription = '''
Provide a detailed description of the scene image on the right. Requirements include, but are not limited to, the main content of the image, the objects present, the environment (such as living room, bedroom, etc.), and possible potential indoor objects. Please note that the description should be detailed enough for someone who has not seen the image to have a clear visual perception of your explanation.

'''

    task_prompt_detection = '''
This is the object detection results for the right scene image in the format - [Name, Confidence Score, Coordinates of the Bounding Box (The Upper left and lower right corners) <(x1, y1), (x2, y2)>]:
{OBJECT_DETECTION}
Considering the existing object recognition results and the right scene image, list the names of all recognizable objects of the image. Be sure to rank these objects by predicted confidence or importance and briefly describe each object and its context.

'''

#     task_prompt_analysis = '''
# Please analyze the following elements based on the image: 1. the spatial layout and passable areas; 2. any potential obstacles; and 3. identify which areas may contain critical information or objects based on the current mission objectives {TARGET}. In conjunction with the above analysis, assess whether the areas in the image are worth exploring further and give your reasons for your recommendation.

# '''

    task_prompt_analysis = '''
Consider the spatial layout and passable areas, any potential obstacles based on the scene image on the right. Evaluate whether the areas in the image are worth exploring further in light of the current mission objectives {TARGET}, taking into account the above considerations, and give your reasons for your recommendation.    

'''
    des = ", ".join(extract_scene_image_description_results(Preprocess_prompt))
    det = ", ".join(extract_scene_object_detection_results(Preprocess_prompt))
    ana = ", ".join(extract_scenario_exploration_analysis_results(Preprocess_prompt))
    is_description = True if 'yes' in des.lower() else False
    is_detection = True if 'yes' in det.lower() else False
    is_analysis = True if 'yes' in ana.lower()  else False
    
    if is_description and not is_detection and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + end_prompt
    elif is_detection and not is_description and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: <Input your scene image object detection>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection)
    elif is_analysis and not is_description and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) rationale for scene exploration: <Input your rationale for scene exploration>
'''
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_analysis.format(TARGET=target)
    elif is_description and is_detection and not is_analysis:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) scene image object detection: <Input your scene image object detection>
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + end_prompt
    elif is_description and is_analysis and not is_detection:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) rationale for scene exploration: <Input your rationale for scene exploration>
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and not is_analysis and not is_description:    
        end_prompt = '''
Your output format is as follows:
(1) scene image object detection: <Input your scene image object detection>
(2) rationale for scene exploration: <Input your rationale for scene exploration>
'''   
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
            '(2) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    elif is_detection and is_analysis and is_description:
        end_prompt = '''
Your output format is as follows:
(1) scene image description: <Input your scene image description>
(2) scene image object detection: <Input your scene image object detection>
(3) rationale for scene exploration: <Input your rationale for scene exploration>
'''     
        Use_prompt = Module_Decision_Prompt + '(1) ' + task_prompt_decription + \
            '(2) ' + task_prompt_detection.format(OBJECT_DETECTION=object_detection) + \
                '(3) ' + task_prompt_analysis.format(TARGET=target) + end_prompt
    else:
        ### 跳过Module Prompt这一阶段
        Use_prompt = None
    
    return Use_prompt    




# def form_prompt_for_DecisionVLM_Meta() -> str:
#     return Meta_Agent_Decision_Prompt




def contains_yes_or_no(VLM_Pred: str) -> str:
    if "Yes" in VLM_Pred:
        return "Yes"
    elif "No" in VLM_Pred:
        return "No"
    else:
        return "Neither"

def Perception_weight_decision(VLM_Rel: list, VLM_Pred: str) -> str:
    b_decision = contains_yes_or_no(VLM_Pred)
    if b_decision == "Neither":
        return b_decision
    
    x, y = VLM_Rel
    
    if b_decision == "Yes":
        weighted_yes_prob = x 
        weighted_no_prob = y * (1 - x)
    else:  # b_decision == "No"
        weighted_yes_prob = x * (1 - y)
        weighted_no_prob = y 
    
    total_prob = weighted_yes_prob + weighted_no_prob
    if total_prob == 0:  # 避免划分零的问题
        return b_decision
    
    weighted_yes_prob /= total_prob
    weighted_no_prob /= total_prob
    
    return weighted_yes_prob, weighted_no_prob

def contains_decision(VLM_Pred: str) -> str:
    if "A" in VLM_Pred:
        return "A"
    elif "B" in VLM_Pred:
        return "B"
    elif "C" in VLM_Pred:
        return "C"
    elif "D" in VLM_Pred:
        return "D"
    else:
        return "Neither"

def Perception_weight_decision4(VLM_Rel: list, VLM_Pred: str) -> str:
    decision = contains_decision(VLM_Pred)
    if decision == "Neither":
        return decision
    
    # 假定 VLM_Rel 的長度為 4，分別對應四種決策的相對權重
    assert len(VLM_Rel) == 4, "VLM_Rel must contain weights for four decisions (A, B, C, D)."
    
    # 分配對應的權重
    weights = {
        "A": VLM_Rel[0],
        "B": VLM_Rel[1],
        "C": VLM_Rel[2],
        "D": VLM_Rel[3]
    }

    # 加权权重
    weights[decision] = weights[decision] 
    
    # 根據決策計算加權概率
    total_weight = sum(VLM_Rel)
    weighted_probs = {key: val/total_weight for key, val in weights.items()}
    
    # 確保總概率為 1
    if total_weight == 0:  # 避免除以零
        return "Invalid weights provided."
    
    return weighted_probs["A"],weighted_probs["B"],weighted_probs["C"],weighted_probs["D"]


def contains_decision26(VLM_Pred: str) -> str:
    for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
        if char in VLM_Pred:
            return char
    return "Neither"

def Perception_weight_decision26(VLM_Rel: list, VLM_Pred: str) -> dict:
    decision = contains_decision26(VLM_Pred)
    if decision == "Neither":
        return {"Neither": 1.0}
    
    assert len(VLM_Rel) == 26, "VLM_Rel must contain weights for 26 decisions (1-26)."
    
    weights = {chr(ord("a")+i): VLM_Rel[i] for i in range(26)}  # Assign weights to each number
    
    # 加权权重
    weights[decision] = weights[decision] * 26

    total_weight = sum(VLM_Rel)
    if total_weight == 0:
        return {"Invalid weights provided.": 1.0}

    weighted_probs = [val/total_weight for _, val in weights.items()]
    
    return weighted_probs
