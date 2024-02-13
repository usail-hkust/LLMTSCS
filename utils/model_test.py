from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
from .my_utils import get_state, get_state_detail, eight_phase_list
import json
import os
import numpy as np
from tqdm import tqdm

def test(model_dir, data_dir, cnt_round, run_cnt, _dic_traffic_env_conf, logger):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_DATA": data_dir, "PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if dic_traffic_env_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    try:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path
        )

        done = False

        total_time = dic_traffic_env_conf["RUN_COUNTS"]
        state = env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        state_action_log = [[] for _ in range(dic_traffic_env_conf['NUM_INTERSECTIONS'])]

        for step_num in tqdm(range(int(total_time / dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done:
                break

            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
                if dic_traffic_env_conf["MODEL_NAME"] in ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                                          "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"]:
                    one_state = state
                    action_list = agents[i].choose_action(step_num, one_state)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            # log statistic state & action
            for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
                # log
                intersection = env.intersection_dict[env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed, "action": eight_phase_list[action_list[i]]})

            next_state, reward, done, _ = env.step(action_list)

            state = next_state

            # calculate logger results
            total_reward += sum(reward)
            queue_length_inter = []
            for inter in env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
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

        results = {
            "test_reward": total_reward,
            "test_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time": total_travel_time}
        logger.log(results)

        env.batch_log_2()
        env.end_cityflow()

        over_results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}

        return over_results, state_action_log

    except Exception as e:
        print(e)
        print("============== error occurs in model_test ============")

        return None
