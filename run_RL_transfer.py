from utils.utils import merge
from utils import error
from utils import config
import os
import time
from utils.RL_transfer_test import test_transfer
import argparse
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo", type=str, default='AdvancedMPLight')
    parser.add_argument("-mod", type=str, default="AdvancedMPLight")
    parser.add_argument("-model", type=str, default="AdvancedMPLight")
    parser.add_argument("-proj_name", type=str, default="RL_Test")
    parser.add_argument("-eightphase", action="store_true", default=False)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("-workers", type=int, default=1)
    parser.add_argument("-dataset", type=str, default="jinan")
    parser.add_argument("-traffic_file", type=str, default="anon_3_4_jinan_synthetic_24000_60min.json")
    parser.add_argument("-traffic_file_source", type=str, default="anon_3_4_jinan_real.json")

    return parser.parse_args()


def main(in_args):
    traffic_file_list = []

    if in_args.dataset == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24h_6000.json",
                             "anon_3_4_jinan_synthetic_24000_60min.json"]
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json",
                             "anon_4_4_hangzhou_synthetic_24h_12000.json",
                             "anon_4_4_hangzhou_synthetic_24000_60min.json"]
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        template = "NewYork"

    if "24h" in in_args.traffic_file:
        count = 86400

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException('Flow file does not exist.')
    except error.flowFileException as e:
        print(e)

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,

        "MODEL_NAME": in_args.model,
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "TRAFFIC_FILE_SOURCE": in_args.traffic_file_source,
        "ROADNET_FILE": f"roadnet_{road_net}.json",

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],

        "DIC_REWARD_INFO": {
            "pressure": 0
        },
    }

    dic_base_agent_conf = config.DIC_BASE_AGENT_CONF

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_" +
                                      time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_" +
                                               time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net))
    }

    # case study agents
    if in_args.model == 'MPLight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "MPLight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_num",
            ],
        }
    elif in_args.model == 'AdvancedColight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "AdvancedColight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
    elif in_args.model == 'Colight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "Colight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
    elif in_args.model == 'EfficientColight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "EfficientColight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
    elif in_args.model == 'AttendLight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "Attend",
            "LIST_STATE_FEATURE": [
                "num_in_seg_attend",
            ],
        }
    elif in_args.model == 'PressLight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "EfficientPressLight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue",
            ],
        }
    elif in_args.model == 'AdvancedMPLight':
        dic_agent_conf_extra = {
            "MODEL_NAME": "AdvancedMPLight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
            ],
        }
    else:
        raise NotImplementedError
    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"] = dic_agent_conf_extra["LIST_STATE_FEATURE"]
    dic_agent_conf = merge(dic_agent_conf_extra, dic_base_agent_conf)
    dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

    logger = wandb.init(
        group=f"{dic_traffic_env_conf['MODEL_NAME']}-{dic_traffic_env_conf['TRAFFIC_FILE_SOURCE']}-to-{dic_traffic_env_conf['TRAFFIC_FILE']}",
        project=dic_traffic_env_conf['PROJECT_NAME'],
        name="round_0",
        config=dic_traffic_env_conf,
    )
    results = test_transfer(dic_path_extra["PATH_TO_DATA"], dic_traffic_env_conf, dic_agent_conf)
    logger.log(results)
    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
