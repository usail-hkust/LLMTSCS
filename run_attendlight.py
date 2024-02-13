from utils.utils import pipeline_wrapper, merge
from utils import config, error
import time
from multiprocessing import Process
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='AttendLight')
    parser.add_argument("--mod", type=str, default="Attend")
    parser.add_argument("--model", type=str, default="AttendLight")
    parser.add_argument("--proj_name", type=str, default="chatgpt-TSCS-Transfer")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--multi_process", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hangzhou", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--duration", type=int, default=30)
    return parser.parse_args()


def main(in_args=None):
    traffic_file_list = []

    if in_args.dataset == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json"]
        num_rounds = 100
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_32000_60min.json"]
        num_rounds = 100
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        num_rounds = 100
        template = "NewYork"

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException('Flow file does not exist.')
    except error.flowFileException as e:
        print(e)
        return
    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)
    process_list = []
    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": num_rounds,
        "NUM_GENERATORS": in_args.gen,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_intersections,
        "RUN_COUNTS": count,

        "MODEL_NAME": in_args.mod,
        "MODEL": in_args.model,
        "PROJECT_NAME": in_args.proj_name,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
        "TRAFFIC_SEPARATE": in_args.traffic_file,
        "LIST_STATE_FEATURE": [
            "num_in_seg_attend",
        ],

        "DIC_REWARD_INFO": {
            "pressure": -0.25,
        },
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                    'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_"
                                      + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_"
                                               + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
        "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
    }
    
    config.dic_traffic_env_conf['MIN_ACTION_TIME'] = in_args.duration
    config.dic_traffic_env_conf['MEASURE_TIME'] = in_args.duration
    deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    if in_args.multi_process:
        ppl = Process(target=pipeline_wrapper,
                      args=(deploy_dic_agent_conf,
                            deploy_dic_traffic_env_conf,
                            deploy_dic_path,
                            f'{template}-{road_net}',
                            in_args.traffic_file.split(".")[0]))
        process_list.append(ppl)
    else:
        pipeline_wrapper(dic_agent_conf=deploy_dic_agent_conf,
                         dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                         dic_path=deploy_dic_path,
                         roadnet=f'{template}-{road_net}',
                         trafficflow=in_args.traffic_file.split(".")[0])

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()

    main(args)

