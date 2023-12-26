"""
MPLight agent, based on FRAP model structure.
Observations: [cur_phase, traffic_movement_pressure_num]
Reward: -Pressure
"""

from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Activation, Embedding, Conv2D, concatenate, add,\
    multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent, slice_tensor, relation
from tensorflow.keras import backend as K
import numpy as np
import random


class MPLightAgent(NetworkAgent):
    """
    optimize the build_network function
    assert the features are [ cur_phase, other feature]
    in this part, the determined feature name are removed
    """
    def build_network(self):
        dic_input_node = {"feat1": Input(shape=(8,), name="input_cur_phase"),
                          "feat2": Input(shape=(12,), name="input_feat2")}

        p = Activation('sigmoid')(Embedding(2, 4, input_length=8)(dic_input_node["feat1"]))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        dic_index = {
            "WL": 0,
            "WT": 1,
            "EL": 3,
            "ET": 4,
            "NL": 6,
            "NT": 7,
            "SL": 9,
            "ST": 10,
        }
        for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
            idx = dic_index[m]
            tmp_vec = d(
                Lambda(slice_tensor, arguments={"index": idx}, name="vec_%d" % i)(dic_input_node["feat2"]))
            tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)

        if self.num_actions == 8 or self.num_actions == 4:
            list_phase_pressure = []
            lane_embedding = Dense(16, activation="relu", name="lane_embedding")
            for phase in self.dic_traffic_env_conf["PHASE_LIST"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(add([lane_embedding(dic_lane[m1]),
                                                lane_embedding(dic_lane[m2])], name=phase))
            # [batch, 4, 3], initialed zeros
            constant_o = Lambda(relation, arguments={"phase_list": self.dic_traffic_env_conf["PHASE_LIST"]},
                                name="constant_o")(dic_input_node["feat2"])
        relation_embedding = Embedding(2, 4, name="relation_embedding")(constant_o)

        # rotate the phase pressure
        list_phase_pressure_recomb = []
        num_phase = len(list_phase_pressure)

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")
        if num_phase == 8:
            feature_map = Reshape((8, 7, 32))(list_phase_pressure_recomb)
        else:
            feature_map = Reshape((4, 3, 32))(list_phase_pressure_recomb)
        lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1),
                           activation="relu", name="lane_conv")(feature_map)
        relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                               name="relation_conv")(relation_embedding)
        combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")

        # [batch, 4, 3, D_DENSE]
        hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                              name="combine_conv")(combine_feature)
        # [batch, 4, 3, 1 ]
        before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="before_merge")(hidden_layer)
        if self.num_actions == 8:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((8, 7))(before_merge))
        else:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((4, 3))(before_merge))

        network = Model(inputs=[dic_input_node[feature_name] for feature_name in ["feat1", "feat2"]],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()

        return network

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == "cur_phase":
                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE'][s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, states):
        dic_state_feature_arrays = {}  # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(self.dic_traffic_env_conf['PHASE'][s[feature_name][0]])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       used_feature]

        q_values = self.q_network.predict(state_input)
        # e-greedy
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = np.random.randint(len(q_values[0]), size=len(q_values))
        else:
            action = np.argmax(q_values, axis=1)

        return action

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        # used_feature = ["phase_2", "phase_num_vehicle"]
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        _state = [[], []]
        _next_state = [[], []]
        _action = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _action.append(action)
            _reward.append(reward)
        # well prepared states
        _state2 = [np.array(ss) for ss in _state]
        _next_state2 = [np.array(ss) for ss in _next_state]

        cur_qvalues = self.q_network.predict(_state2)
        next_qvalues = self.q_network_bar.predict(_next_state2)
        # [batch, 4]
        target = np.copy(cur_qvalues)

        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = _state2
        self.Y = target