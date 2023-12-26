"""
Max-Pressure agent.
observation: [traffic_movement_pressure_queue].
Action: use greedy method select the phase with max value.
"""

from .agent import Agent

import numpy as np


class MaxPressureAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(MaxPressureAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])

        self.action = None
        if self.phase_length == 4:
            self.DIC_PHASE_MAP_4 = {  # for 4 phase
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        elif self.phase_length == 8:
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
        """
        As described by the definition, use traffic_movement_pressure
        to calcualte the pressure of each phase.
        """
        if state["cur_phase"][0] == -1:
            return self.action

        #  WT_ET
        tr_mo_pr = np.array(state["traffic_movement_pressure_queue"])
        phase_1 = tr_mo_pr[1] + tr_mo_pr[4]
        # NT_ST
        phase_2 = tr_mo_pr[7] + tr_mo_pr[10]
        # WL_EL
        phase_3 = tr_mo_pr[0] + tr_mo_pr[3]
        # NL_SL
        phase_4 = tr_mo_pr[6] + tr_mo_pr[9]

        if self.phase_length == 4:

            self.action = np.argmax([phase_1, phase_2, phase_3, phase_4])
        elif self.phase_length == 8:
            #  WL_WT
            phase_5 = tr_mo_pr[0] + tr_mo_pr[1]
            # EL_ET
            phase_6 = tr_mo_pr[3] + tr_mo_pr[4]
            # SL_ST
            phase_7 = tr_mo_pr[9] + tr_mo_pr[10]
            # NL_NT
            phase_8 = tr_mo_pr[6] + tr_mo_pr[7]
            self.action = np.argmax([phase_1, phase_2, phase_3, phase_4, phase_5, phase_6, phase_7, phase_8])

        if state["cur_phase"][0] == self.action:
            self.current_phase_time += 1
        else:
            self.current_phase_time = 0

        return self.action
