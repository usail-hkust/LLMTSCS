import numpy as np
from tensorflow.keras.layers import Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import random
import os
from .agent import Agent
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

        self.Xs, self.Y = None, None
        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]

        if cnt_round == 0:

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.q_network = self.build_network()
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))

                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] *
                                    self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

        # decay the epsilon
        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_TRANSFER_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name),
                                        custom_objects={"Selector": Selector})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def build_network(self):
        raise NotImplementedError

    @staticmethod
    def build_memory():
        return []

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network

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

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _, _ = sample_slice[i]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])
            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append(np.array([state[feature_name]]))
                _next_state.append(np.array([next_state[feature_name]]))

            target = self.q_network.predict(_state)

            next_state_qvalues = self.q_network_bar.predict(_next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y)

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if "cur_phase" in feature:
                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE'][s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, state):
        """choose the best action for current state """
        state_input = self.convert_state_to_input(state)
        q_values = self.q_network.predict(state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = random.randrange(len(q_values[0]))
        else:  # exploitation
            action = np.argmax(q_values[0])
        return action

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])


class Selector(Layer):

    def __init__(self, select, d_phase_encoding, d_action, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.d_phase_encoding = d_phase_encoding
        self.d_action = d_action
        self.select_neuron = K.constant(value=self.select, shape=(1, self.d_phase_encoding))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)

    def call(self, x):
        batch_size = K.shape(x)[0]
        constant = K.tile(self.select_neuron, (batch_size, 1))
        return K.min(K.cast(K.equal(x, constant), dtype="float32"), axis=-1, keepdims=True)

    def get_config(self):
        config = {"select": self.select, "d_phase_encoding": self.d_phase_encoding, "d_action": self.d_action}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return [batch_size, self.d_action]


def slice_tensor(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 3:
        return x[:, index, :]
    elif len(x_shape) == 2:
        return Reshape((1, ))(x[:, index])


def relation(x, phase_list):
    relations = []
    num_phase = len(phase_list)
    if num_phase == 8:
        for p1 in phase_list:
            zeros = [0, 0, 0, 0, 0, 0, 0]
            count = 0
            for p2 in phase_list:
                if p1 == p2:
                    continue
                m1 = p1.split("_")
                m2 = p2.split("_")
                if len(list(set(m1 + m2))) == 3:
                    zeros[count] = 1
                count += 1
            relations.append(zeros)
        relations = np.array(relations).reshape((1, 8, 7))
    else:
        relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))
    batch_size = K.shape(x)[0]
    constant = K.constant(relations)
    constant = K.tile(constant, (batch_size, 1, 1))
    return constant


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        # [batch,agent,dim]->[batch,1,agent,dim]
        # [batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))