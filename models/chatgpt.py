import copy
from utils.my_utils import load_json, dump_json, get_state_detail, get_state_three_segment
import requests
import json
import time
import re
import csv
import io
import pandas as pd
import numpy as np

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "YOUR_KEY_HERE"
}

four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
eight_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
direction_dict_ori = {"T": "through", "L": "turn-left", "R": "turn-right"}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

incoming_lane_2_outgoing_road = {
    "NT": "South",
    "NL": "East",
    "ST": "North",
    "SL": "West",
    "ET": "West",
    "EL": "South",
    "WT": "East",
    "WL": "North"
}

class ChatGPTTLCS_Wait_Time_Forecast(object):
    def __init__(self, GPT_version, intersection, inter_name, phase_num, log_dir, dataset):
        # init road length
        roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.roads = roads
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        for r in roads:
            self.length_dict[roads[r]["location"]] = int(roads[r]["length"])

        self.phases = four_phase_list if phase_num == 4 else eight_phase_list

        self.gpt_version = GPT_version
        self.last_action = "ETWT"
        self.system_prompt = load_json("./prompts/prompt_domain_knowledge.json")["system_prompt"]
        self.state_action_prompt_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_state_action_prompt_domain_knowledge.json"
        self.error_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_error_prompts_domain_knowledge.json"
        self.state_action_prompt = []
        self.errors = []

        self.temp_action_logger = ""

    def choose_action(self, env):
        self.temp_action_logger = ""
        state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)
        flow_num = 0
        for road in state:
            flow_num += state[road]["queue_len"] + sum(state[road]["cells"])

        if flow_num == 0:
            action_code = self.action2code("ETWT")
            self.state_action_prompt.append({"state": state, "prompt": [], "action": "ETWT"})
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            self.temp_action_logger = action_code

            return

        signal_text = ""

        # chain-of-thought
        retry_counter = 0
        while signal_text not in self.phases:
            try:
                if retry_counter > 10:
                    signal_text = "ETWT"
                    break
                state_txt, max_queue_len = self.state2table(state)
                prompt = self.getPrompt(state_txt, avg_speed)
                data = {
                    "model": self.gpt_version,
                    "messages": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.0
                }
                response = requests.post(url, headers=headers, data=json.dumps(data)).json()
                analysis = response['choices'][0]['message']['content']
                retry_counter += 1
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signal_text = re.findall(signal_answer_pattern, analysis)[-1]

            except Exception as e:
                self.errors.append({"error": str(e), "prompt": prompt})
                dump_json(self.errors, self.error_file)
                time.sleep(5)

        prompt.append({"role": "assistant", "content": analysis})
        action_code = self.action2code(signal_text)
        self.state_action_prompt.append({"state": state, "state_incoming": state_incoming, "prompt": prompt, "action": signal_text})
        dump_json(self.state_action_prompt, self.state_action_prompt_file)

        self.temp_action_logger = action_code
        self.last_action = signal_text

    '''
    ============ Class Utils ============
    '''

    def state2table(self, state):
        state_txt = "lane,early queued,average waiting time,segment 1,segment 2,segment 3,segment 4\n"
        max_queue_len = 0
        for lane in state:
            queue_len = int(state[lane]['queue_len'])
            avg_wait_time = int(state[lane]['avg_wait_time'])
            max_queue_len = queue_len if queue_len > max_queue_len else max_queue_len
            state_txt += f"{location_dict_detail[lane[0]]} {direction_dict[lane[1]]} lane,{queue_len},{avg_wait_time}s"

            for i, n in enumerate(state[lane]['cells']):
                n = int(n)
                state_txt += f",{n}"
            state_txt += "\n"

        return state_txt, max_queue_len

    def getPrompt(self, state_txt, avg_speed):
        # fill information
        signals_text = ""
        for i, p in enumerate(self.phases):
            signals_text += phase_explanation_dict_detail[p] + "\n"

        prompt = [
            {"role": "system",
             "content": self.system_prompt},
            {"role": "user",
             "content": "A traffic light regulates a four-section intersection with northern, southern, eastern, and "
                        "western sections, each containing two lanes: one for through traffic and one for left-turns. "
                        f"The eastern and western lanes are {int(self.length_dict['East'])} meters long, while the northern and southern lanes are "
                        f"{int(self.length_dict['North'])} meters in length. Each lane is further divided into four segments. Segment 1 spans from the "
                        "10m mark of the lane to segment 2. Segment 2 begins at the 1/10 mark of the lane and links segment "
                        "1 to segment 3. Segment 3 starts at the 1/3 mark of the lane and links segment 2 to segment 4. "
                        "Segment 4 begins at the 2/3 mark of the lane, spanning from the end of segment 3 to the lane's end.\n\n"
                        "The current lane statuses are:\n" + state_txt + "\n" +
                        "This CSV table shows lane statuses, with the first column representing lanes, the second column "
                        "displaying early queued vehicle counts, the third column showing the average time that early "
                        "queued vehicles have waited in previous phases, and columns 4-7 indicating approaching vehicle "
                        "counts in the four lane segments.\n\n"
                        "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                        f"vehicles are at an average speed of {int(avg_speed)}m/s. If they can arrive at the intersection during the next "
                        "phase, they may merge into the appropriate waiting queues (if they are NOT allowed to pass) or "
                        "pass the intersection (if they are allowed to pass).\n\n"
                        f"The traffic light has {len(self.phases)} signal phases. Each signal relieves vehicles' flow in the two specific "
                        "lanes. The lanes relieving vehicles' flow under each traffic light phase are listed below:\n" +
                        signals_text +
                        "\nThe next signal phase will persist for 30 seconds.\n\n"
                        "Please follow the following steps to provide your analysis (pay attention to accurate variable calculations in each step):\n"
                        "- Step 1: Calculate the ranges of the four lane segments in different lanes.\n"
                        "- Step 2: Identify the lane segments that vehicles travel on can potentially reach the intersection within the next phase.\n"
                        "- Step 3: Analyzing the CSV table, identify the traffic conditions (early queued vehicle count, average waiting time, and the approaching vehicle count in segments identified in Step 2) in each lane.\n"
                        "- Step 4: If no vehicle is permitted to pass the intersection within the next phase, analyze:\n"
                        "a) The total cumulative waiting times of ALL early queued vehicles that will accumulate by the END of the next phase in each lane.\n"
                        "b) The total waiting times of ALL vehicles from reachable segments within the next phase in each lane.\n"
                        "c) The total waiting time of ALL queuing vehicles analyzed above in each lane.\n"
                        "- Step 5: Considering the total waiting time, analyze the potential congestion level of the two allowed lanes of each signal if vehicles on these lanes cannot be relieved in the next phase.\n"
                        "- Step 6: Considering the potential congestion level of the two allowed lanes of each signal, identify the most effective traffic signal that will most significantly improve the traffic condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal.\n\n"
                        "Requirements:\n"
                        "- Let's think step by step.\n"
                        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
             }
        ]

        return prompt

    def action2code(self, action):
        code = self.phases[action]

        return code

class ChatGPTTLCS_Commonsense(object):
    def __init__(self, GPT_version, intersection, inter_name, phase_num, log_dir, dataset):
        # init road length
        roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.roads = roads
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        for r in roads:
            self.length_dict[roads[r]["location"]] = int(roads[r]["length"])

        self.phases = four_phase_list if phase_num == 4 else eight_phase_list

        self.gpt_version = GPT_version
        self.last_action = "ETWT"
        self.system_prompt = load_json("./prompts/prompt_commonsense.json")["system_prompt"]
        self.state_action_prompt_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_state_action_prompt_commonsense_no_calculation.json"
        self.error_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_error_prompts_commonsense_no_calculation.json"
        self.state_action_prompt = []
        self.errors = []

        self.temp_action_logger = ""

    def choose_action(self, env):
        self.temp_action_logger = ""
        state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)
        flow_num = 0
        for road in state:
            flow_num += state[road]["queue_len"] + sum(state[road]["cells"])

        if flow_num == 0:
            action_code = self.action2code("ETWT")
            self.state_action_prompt.append({"state": state, "prompt": [], "action": "ETWT"})
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            self.temp_action_logger = action_code

            return

        signal_text = ""

        # chain-of-thought
        retry_counter = 0
        while signal_text not in self.phases:
            if retry_counter > 10:
                signal_text = "ETWT"
                break
            try:
                state_txt = self.state2table(state)
                prompt = self.getPrompt(state_txt)
                data = {
                    "model": self.gpt_version,
                    "messages": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.0
                }
                response = requests.post(url, headers=headers, data=json.dumps(data)).json()
                analysis = response['choices'][0]['message']['content']
                retry_counter += 1
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signal_text = re.findall(signal_answer_pattern, analysis)[-1]

            except Exception as e:
                self.errors.append({"error": str(e), "prompt": prompt})
                dump_json(self.errors, self.error_file)
                time.sleep(3)

        prompt.append({"role": "assistant", "content": analysis})
        action_code = self.action2code(signal_text)
        self.state_action_prompt.append({"state": state, "state_incoming": state_incoming, "prompt": prompt, "action": signal_text})
        dump_json(self.state_action_prompt, self.state_action_prompt_file)

        self.temp_action_logger = action_code
        self.last_action = signal_text

    '''
    ============ Class Utils ============
    '''

    def state2table(self, state):
        state_txt = ""
        for p in self.phases:
            lane_1 = p[:2]
            lane_2 = p[2:]
            queue_len_1 = int(state[lane_1]['queue_len'])
            queue_len_2 = int(state[lane_2]['queue_len'])

            seg_1_lane_1 = state[lane_1]['cells'][0]
            seg_2_lane_1 = state[lane_1]['cells'][1]
            seg_3_lane_1 = state[lane_1]['cells'][2] + state[lane_1]['cells'][3]

            seg_1_lane_2 = state[lane_2]['cells'][0]
            seg_2_lane_2 = state[lane_2]['cells'][1]
            seg_3_lane_2 = state[lane_2]['cells'][2] + state[lane_2]['cells'][3]

            state_txt += (f"Signal: {p}\n"
                          f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                          f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                          f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                          f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                          f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

        return state_txt

    def getPrompt(self, state_txt):
        # fill information
        signals_text = ""
        for i, p in enumerate(self.phases):
            signals_text += phase_explanation_dict_detail[p] + "\n"

        prompt = [
            {"role": "system",
             "content": self.system_prompt},
            {"role": "user",
             "content": "A crossroad connects two roads: the north-south and east-west. The traffic light is located at "
                        "the intersection of the two roads. The north-south road is divided into two sections by the intersection: "
                        "the north and south. Similarly, the east-west road is divided into the east and west. Each section "
                        "has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. "
                        "Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. "
                        "In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
                        "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                        "vehicles will arrive at the intersection in the future.\n\n"
                        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                        "specific lanes. The state of the intersection is listed below. It describes:\n"
                        "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
                        "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                        + state_txt +
                        "Please answer:\n"
                        "Which is the most effective traffic signal that will most significantly improve the traffic "
                        "condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?\n\n"
                        "Note:\n"
                        "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant "
                        "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to "
                        "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
                        "Requirements:\n"
                        "- Let's think step by step.\n"
                        "- You can only choose one of the signals listed above.\n"
                        "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                        "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                        "- Your choice can only be given after finishing the analysis.\n"
                        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
             },
        ]

        return prompt

    def action2code(self, action):
        code = self.phases[action]

        return code

