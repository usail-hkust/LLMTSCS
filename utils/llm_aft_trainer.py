from utils.my_utils import load_json, dump_json, get_state_detail, state2text, getPrompt, action2code, code2action, eight_phase_list, four_phase_list, torch_gc
import os
import time
import numpy as np
import wandb
from utils.cityflow_env import CityFlowEnv
import utils.config as config
from utils.aft_rank_loss_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_int8_training
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from copy import deepcopy
import re
import json
import shutil
import copy
import random

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


class LLM_CGPR_Collector:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.data_buffer = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left"
        )
        self.tokenizer.pad_token_id = 0

        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
            "do_sample": False,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "num_beam_groups": 4,
            "diversity_penalty": 1.0,
            "num_beams": 4,
            "num_return_sequences": 4
        }

    def initialize_critic(self):
        round_num = 99
        traffic_file = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('.')[0]

        dic_adv_colight_agent_conf_extra = {
            "MODEL_NAME": "AdvancedColight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
        dic_critic_agent_conf = merge(dic_adv_colight_agent_conf_extra, config.DIC_BASE_AGENT_CONF)

        if dic_critic_agent_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
            dic_critic_agent_conf["EPSILON"] = 0
            dic_critic_agent_conf["MIN_EPSILON"] = 0

        critic_agents = []
        compare_dic_traffic_env_conf = deepcopy(self.dic_traffic_env_conf)
        compare_dic_traffic_env_conf["LIST_STATE_FEATURE"] = dic_critic_agent_conf["LIST_STATE_FEATURE"]
        for j in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_critic_agent_conf["MODEL_NAME"]
            agent = config.DIC_AGENTS[agent_name](
                dic_agent_conf=dic_critic_agent_conf,
                dic_traffic_env_conf=compare_dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=0,
                intersection_id=str(j)
            )
            critic_agents.append(agent)

        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            critic_agents[i].load_network(f'round_{round_num}_inter_0', file_path=f"./model_weights/{dic_critic_agent_conf['MODEL_NAME']}/{traffic_file}/")

        self.critic_agents = critic_agents
        self.dic_critic_agent_conf = dic_critic_agent_conf

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()
        self.initialize_critic()

    def collect(self):
        print("================ Start Training ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        state = self.env.reset()
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        # data buffer for training data collection
        self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest")['input_ids'].to('cuda')

            response_ids = self.llm_model.generate(input_ids=inputs, **self.generation_kwargs)
            response_ids = response_ids.reshape(-1, 4, response_ids.size(1))
            responses = []
            for i in range(response_ids.size(0)):
                responses.append(self.tokenizer.batch_decode(response_ids[i], skip_special_tokens=True))

            rewards = []
            all_decoded_responses = []
            all_sampled_rewards = []
            critic_actions = []
            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                action_response = responses[i][random.randint(0, 3)][len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, action_response)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": action_response})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                # critic agents
                one_state, _ = self.env.get_state(self.dic_critic_agent_conf["LIST_STATE_FEATURE"])
                critic_agent_action, q_value = self.critic_agents[i].choose_action_with_value(step_num, one_state)
                critic_actions.append(code2action(critic_agent_action[i]))
                rewards.append(q_value[i][action2code(signal_text)])

                # collect responses
                prompt_responses = []
                sampled_rewards = []
                for res_i in range(4):
                    sampled_response = res[res_i][len(prompts[i]):]
                    sampled_signals = re.findall(signal_answer_pattern, sampled_response)
                    sampled_signal_text = sampled_signals[-1] if len(sampled_signals) > 0 else "ETWT"
                    if len(sampled_signals) == 0 or sampled_signal_text not in four_phase_list:
                        sampled_rewards.append(0)
                    else:
                        sampled_rewards.append(float(q_value[i][action2code(sampled_signal_text)]))

                    prompt_responses.append(sampled_response)
                all_decoded_responses.append(prompt_responses)
                all_sampled_rewards.append(sampled_rewards)

            next_state, _, done, _ = self.env.step(action_list)

            for i, res in enumerate(responses):
                if vehicle_nums[i] > 0:
                    new_d = {"query": prompts[i],
                             "responses": all_decoded_responses[i],
                             "scores": all_sampled_rewards[i]}
                    com_score = new_d["scores"][0]
                    all_same = True
                    for s in new_d["scores"]:
                        if s != com_score:
                            all_same = False

                    if not all_same:
                        self.data_buffer.append(new_d)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            print("Rewards:", sum(rewards), "Fail Num:", fail_num)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            if not os.path.exists("./data/cgpr"):
                os.makedirs("./data/cgpr")
            dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")
            torch_gc()

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/collect_reward": total_reward,
            "env/collect_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_travel_time": total_travel_time}
        print("Collect:", results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)

        print("Collection time: ", time.time() - start_time)
        self.env.batch_log_2()

        if not os.path.exists("./data/cgpr"):
            os.makedirs("./data/cgpr")
        dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")

    def train_test(self):
        print("================ Start Data Collection ================")
        self.collect()

    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums


class LLM_CGPR_Trainer:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
            device_map=device_map,
        )
        gradient_accumulation_steps = self.dic_agent_conf["BATCH_SIZE"] // self.dic_agent_conf["MINI_BATCH_SIZE"]
        self.training_args = TrainingArguments(output_dir=f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
                                               num_train_epochs=self.dic_agent_conf["EPOCHS"],
                                               per_device_train_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               per_device_eval_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                               learning_rate=self.dic_agent_conf['LEARNING_RATE'],
                                               bf16=True,
                                               logging_steps=1,
                                               evaluation_strategy="steps",
                                               save_strategy="steps",
                                               eval_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_total_limit=3,
                                               load_best_model_at_end=True,
                                               model_max_length=2048)

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
            padding=True
        )
        self.tokenizer.pad_token_id = 0

        # init lora
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # self.llm_model = prepare_model_for_int8_training(self.llm_model)
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def train(self):
        print("================ Start Training ================")
        data = load_dataset("json", data_files=f"./data/cgpr/cgpr_{self.dic_agent_conf['CGPR_DATA_PATH']}")

        train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=2024)
        train_data = train_val["train"].shuffle(seed=2024)
        val_data = train_val["test"].shuffle(seed=2024)

        self.llm_model.train()
        data_module = make_supervised_data_module(self.tokenizer, train_data, val_data, mix=True if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else False)
        self.trainer = RankTrainer(model=self.llm_model, tokenizer=self.tokenizer, args=self.training_args, **data_module)

        # self.llm_model.config.use_cache = False
        # self.llm_model = torch.compile(self.llm_model)
        self.trainer.train()

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, truncation=True, max_length=2048, padding=True, return_tensors='pt').to('cuda')

            response_ids = self.llm_model.generate(input_ids=inputs["input_ids"], **self.test_generation_kwargs)
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

            next_state, rewards, done, _ = self.env.step(action_list)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/test_reward": total_reward,
            "env/test_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_travel_time": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        print("================ Start PPO Fine-Tuning ================")
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        rounds = self.dic_traffic_env_conf["NUM_ROUNDS"]
        last_10_results = {"env/test_reward_over": [],
                           "env/test_avg_queue_len_over": [],
                           "env/test_queuing_vehicle_num_over": [],
                           "env/test_avg_waiting_time_over": [],
                           "env/test_avg_travel_time_over": []}
        for r in range(rounds):
            # train
            self.train()

            # test
            results = self.test(logger, r)
            for ele in last_10_results:
                last_10_results[ele].append(results[ele[:-5]])

            main_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}"
            ckpt_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}"
            if not os.path.isdir(main_path):
                os.mkdir(main_path)
            if not os.path.isdir(ckpt_path):
                os.mkdir(ckpt_path)

            self.llm_model.save_pretrained(
                f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}")

        logger.log(last_10_results)
        wandb.finish()

        self.llm_model.save_pretrained(f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}")

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums


class LLM_Inference:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
            padding=True
        )
        self.tokenizer.pad_token_id = 0

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, truncation=True, max_length=2048, padding=True, return_tensors='pt').to('cuda')

            responses = []
            previous_flag = 0
            for i in range(len(current_states)):
                if (i + 1) % 16 == 0 or i + 1 >= len(current_states):
                    response_ids = self.llm_model.generate(input_ids=inputs["input_ids"][previous_flag:i+1], **self.test_generation_kwargs)
                    responses += self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    previous_flag = i + 1

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            critic_actions = []
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                state_action_log[i][-1]["response"] = res
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards
