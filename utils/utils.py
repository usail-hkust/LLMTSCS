from .pipeline import Pipeline
from .oneline import OneLine
from . import config
import wandb
import copy
import numpy as np
import time
import os

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                       dic_traffic_env_conf=dic_traffic_env_conf,
                       dic_path=dic_path,
                       roadnet=roadnet,
                       trafficflow=trafficflow)
        round_results = ppl.run(round=i, multi_process=False)
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'], round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'find <dir> -type f ! -name "round_<round>_inter_*.h5" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_MODEL"]).replace("<round>", str(int(dic_traffic_env_conf["NUM_ROUNDS"] - 1)))
        cmd_delete_work = 'find <dir> -type f ! -name "state_action.json" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL']}-{roadnet}-{trafficflow}-{len(dic_traffic_env_conf['PHASE'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    print("pipeline_wrapper end")
    return

def oneline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        oneline = OneLine(dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                          dic_path=merge(config.DIC_PATH, dic_path),
                          roadnet=roadnet,
                          trafficflow=trafficflow
                          )
        round_results = oneline.train(round=i)
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'],
                              round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'rm -rf <dir>'.replace("<dir>", dic_path["PATH_TO_MODEL"])
        cmd_delete_work = 'find <dir> -type f ! -name "state_action.json" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL_NAME']}-{roadnet}-{trafficflow}-{len(dic_agent_conf['FIXED_TIME'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    return

