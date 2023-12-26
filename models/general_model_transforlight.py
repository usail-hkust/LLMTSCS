import torch
from .transfo.network_agent import NetworkAgent
import numpy as np
import copy
from .transfo.decision_transformer import DecisionTransformer
import torch.nn.functional as F

NUM_DEVICE=0

class GeneralTransformerLight(NetworkAgent):

    def build_network(self):
        network = DecisionTransformer(state_dim=12*self.num_feat,
            act_dim=4,
            max_length=20,
            max_ep_len=1000,
            hidden_size=256,
            n_layer=10,
            n_head=4,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,)
        return network

    def choose_action(self, states):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        cur_states_len = len(states)
        batch_Xs1 = torch.tensor(state_input, dtype=torch.float32).squeeze(0).reshape(-1, 12*self.num_feat)
        actions = torch.zeros((cur_states_len, 4), dtype=torch.float32)
        rewards = torch.zeros((cur_states_len, 1), dtype=torch.float32)

        time_steps = torch.tensor([self.time_step]*cur_states_len).type(torch.long)

        _, q_values, _ = self.model(batch_Xs1, actions, rewards, time_steps)
        action = np.argmax(q_values.cpu().detach().numpy(), axis=1)
        self.time_step += 1
        return action

    def prepare_samples(self, memory):
        state, action, next_state, p_reward, ql_reward = memory
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        memory_size = len(action)
        _state = [[], None]
        _next_state = [[], None]
        for feat_name in used_feature:
            if feat_name == "new_phase":
                _state[1] = np.array(state[feat_name])
                _next_state[1] = np.array(next_state[feat_name])
            else:
                _state[0].append(np.array(state[feat_name]).reshape(memory_size, 12, -1))
                _next_state[0].append(np.array(next_state[feat_name]).reshape(memory_size, 12, -1))
                
        # ========= generate reaward information ===============
        if "pressure" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys():
            my_reward = p_reward
        else:
            my_reward = ql_reward
        
        return [np.concatenate(_state[0], axis=-1), _state[1]], action, [np.concatenate(_next_state[0], axis=-1), _next_state[1]], my_reward

    def train_network(self, memory):
        
        _state, _action, _, _reward = self.prepare_samples(memory)
        # ==== shuffle the samples ============
        random_index = np.random.permutation(len(_action))
        _state[0] = _state[0][random_index, :, :]
        _action = np.array(_action)[random_index]
        _reward = np.array(_reward)[random_index]
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        num_batch = int(np.floor((len(_action) / batch_size)))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        for epoch in range(epochs):
            for ba in range(int(num_batch)):
                batch_Xs1 = torch.tensor([_state[0][ba*batch_size:(ba+1)*batch_size, :, :]], dtype=torch.float32).\
                                squeeze(0).reshape(batch_size, 12*self.num_feat)
                
                batch_r = torch.tensor(_reward[ba*batch_size:(ba+1)*batch_size], dtype=torch.float32).view(-1,1)
                batch_a = F.one_hot(torch.tensor(_action[ba*batch_size:(ba+1)*batch_size]), 4).type(torch.float32)
                time_step = torch.arange(0, batch_size).type(torch.long)
                _, pre_actions, _ = self.model(batch_Xs1, batch_a, batch_r, time_step)
                loss = loss_fn(pre_actions, batch_a)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
                optimizer.step()
                print("===== Epoch {} | Batch {} / {} | Loss {}".format(epoch, ba, num_batch, loss))
