import copy

from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
from .my_utils import get_state_detail, dump_json, load_json
import os
import time
import shutil
import numpy as np

four_phase_list = {0: 'ETWT', 1: 'NTST', 2: 'ELWL', 3: 'NLSL'}
phase2code = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}

def test_transfer(data_dir, dic_traffic_env_conf, dic_agent_conf):
    traffic_file_source = dic_traffic_env_conf['TRAFFIC_FILE_SOURCE'].split('.')[0]
    round_num = 99
    model_path = f'./model/{dic_agent_conf["MODEL_NAME"]}/' + dic_traffic_env_conf['TRAFFIC_FILE'] + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    work_path = "./records_transfer/" + dic_traffic_env_conf['TRAFFIC_FILE'] + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_path = "./logs_transfer/" + dic_traffic_env_conf['TRAFFIC_FILE'] + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(work_path):
        os.makedirs(work_path)
        shutil.copy(f"{data_dir}/{dic_traffic_env_conf['TRAFFIC_FILE']}", work_path)
        shutil.copy(f"{data_dir}/{dic_traffic_env_conf['ROADNET_FILE']}", work_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # init each agent
    dic_path = {"PATH_TO_DATA": data_dir, "PATH_TO_MODEL": model_path, "PATH_TO_WORK_DIRECTORY": work_path}
    if not os.path.exists("./records_case_study"):
        os.makedirs("./records_case_study")

    if dic_agent_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    inter_agents = []
    compare_dic_traffic_env_conf = deepcopy(dic_traffic_env_conf)
    compare_dic_traffic_env_conf["LIST_STATE_FEATURE"] = dic_agent_conf["LIST_STATE_FEATURE"]
    for j in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_agent_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=compare_dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(j)
        )
        inter_agents.append(agent)

    for i in range(len(inter_agents)):
        inter_agents[i].load_network(f'round_{round_num}_inter_0', file_path=f"./model_weights/{dic_agent_conf['MODEL_NAME']}/{traffic_file_source}/")
        if "Colight" in dic_agent_conf['MODEL_NAME']:
            new_q_network = inter_agents[i].build_network()
            inter_agents[i].q_network = inter_agents[i].build_network_from_copy_only_weight(new_q_network, inter_agents[i].q_network)

    env = CityFlowEnv(
        path_to_log=log_path,
        path_to_work_directory=work_path,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    done = False

    step_num = 0
    total_reward = 0.0
    queue_length_episode = []
    waiting_time_episode = []
    total_time = dic_traffic_env_conf["RUN_COUNTS"]
    state = env.reset()
    all_queues = []

    while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):

        for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
            # action agents
            action_list = inter_agents[i].choose_action(step_num, state)

        next_state, reward, done, _ = env.step(action_list)
        state = next_state
        step_num += 1

        # calculate logger results
        total_reward += sum(reward)
        queue_length_inter = []
        for inter in env.list_intersection:
            queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            for n in inter.dic_feature['lane_num_waiting_vehicle_in']:
                if n > 0:
                    all_queues.append(n)
        queue_length_episode.append(sum(queue_length_inter))

        # waiting time
        waiting_times = []
        for veh in env.waiting_vehicle_list:
            waiting_times.append(env.waiting_vehicle_list[veh]['time'])
        waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

    # wandb logger
    vehicle_travel_times = {}
    for inter in env.list_intersection:
        arrive_left_times = inter.dic_vehicle_arrive_leave_time
        for veh in arrive_left_times:
            if "shadow" in veh:
                continue
            enter_time = arrive_left_times[veh]["enter_time"]
            leave_time = arrive_left_times[veh]["leave_time"]
            if not np.isnan(enter_time):
                leave_time = leave_time if not np.isnan(leave_time) else dic_traffic_env_conf["RUN_COUNTS"]
                if veh not in vehicle_travel_times:
                    vehicle_travel_times[veh] = [leave_time - enter_time]
                else:
                    vehicle_travel_times[veh].append(leave_time - enter_time)

    total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

    over_results = {
        "test_reward_over": total_reward,
        "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
        "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
        "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
        "test_avg_travel_time_over": total_travel_time}

    return over_results
