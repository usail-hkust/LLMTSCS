import numpy as np
import torch
import random
import os
from ..agent import Agent
import traceback


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id=intersection_id)

        # ===== check num actions == num phases ============
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])
        # self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        # self.max_lane = dic_traffic_env_conf["MAX_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]

        self.len_feat = self.cal_input_len()
        self.num_feat = int(self.len_feat/12)
        self.min_q_weight = dic_traffic_env_conf["MIN_Q_W"]
        self.threshold = dic_traffic_env_conf["THRESHOLD"]
        
        if cnt_round == 0:

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.model = self.build_network()
            self.time_step = 0
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                self.time_step = 0
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def cal_input_len(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        for feat_name in used_feature:
            if "num_in_seg" in feat_name:
                N += 12*4
            elif "new_phase" in feat_name:
                N += 0
            else:
                N += 12
        return N

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.model = torch.load(os.path.join(file_path, "%s.pth" % file_name))
        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_TRANSFER_MODEL"]
        self.model = torch.load(os.path.join(file_path, "%s.pth" % file_name))
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        torch.save(self.model, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % file_name))

    def build_network(self):
        raise NotImplementedError

    @staticmethod
    def build_memory():
        return []
