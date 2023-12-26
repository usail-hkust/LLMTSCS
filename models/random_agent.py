"""
Random agent.
Return random action
"""

from .agent import Agent
import random
random.seed(42)


class RandomAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(RandomAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                0: 0
        }

    def choose_action(self, count, state):
        """choose the best action for current state """
        if state["cur_phase"][0] == -1:
            return self.action
        cur_phase = self.DIC_PHASE_MAP[state["cur_phase"][0]]

        # if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
        #     if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
        #         self.current_phase_time = 0
        #         self.action = (cur_phase + 1) % self.phase_length
        #         return (cur_phase + 1) % self.phase_length
        #     else:
        #         self.action = cur_phase
        #         self.current_phase_time += 1
        #         return cur_phase
        # else:
        #     if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
        #         self.current_phase_time = 0
        #         self.action = 1
        #         return 1
        #     else:
        #         self.current_phase_time += 1
        #         self.action = 0
        #         return 0
        return random.randint(0, 7)
