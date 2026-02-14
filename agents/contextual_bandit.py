
import numpy as np
import math
from collections import defaultdict


class HeuristicEvaluator:

    def evaluate(self, observation, action):
        if not isinstance(observation, np.ndarray) or len(observation) < 1:
            return -500.0

        obs_len = len(observation)
        mission_phase = int(observation[0]) if obs_len > 0 else 0


        if obs_len > 10:
            threat_level = np.sum(np.abs(observation[obs_len//2:]))
        else:
            threat_level = 0


        action_type = self._infer_action_type(action)


        h_value = -500.0

        if threat_level > 5:
            if action_type == 'remove':
                h_value = -50.0
            elif action_type == 'analyse':
                h_value = -100.0
            elif action_type == 'sleep':
                h_value = -900.0
        else:
            if action_type == 'monitor':
                h_value = -200.0
            elif action_type == 'deploy_decoy':
                h_value = -250.0
            elif action_type == 'remove':
                h_value = -700.0

        return h_value

    def _infer_action_type(self, action):
        action = int(action)
        if action == 0:
            return 'sleep'
        elif 1 <= action <= 16:
            return 'monitor'
        elif 17 <= action <= 32:
            return 'analyse'
        elif action in [33, 34]:
            return 'deploy_decoy'
        elif action in [35, 36]:
            return 'remove'
        elif action in [37, 38]:
            return 'restore'
        else:
            return 'unknown'


class TabularContextualUCB:

    def __init__(self, agent_id, num_actions=41, c_param=1.5, gamma=0.95):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.c_param = c_param
        self.gamma = gamma


        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.N = defaultdict(lambda: np.zeros(num_actions))


        self.total_steps = 0
        self.state_visits = defaultdict(int)


        self.min_c_param = 0.5
        self.max_c_param = 2.0


        self.heuristic = HeuristicEvaluator()

    def get_context_key(self, observation):
        if not isinstance(observation, np.ndarray):
            return 0

        obs_len = len(observation)
        if obs_len == 0:
            return 0


        mission_phase = int(observation[0]) if obs_len > 0 else 0


        features = []


        features.append(mission_phase)


        obs_array = np.array(observation, dtype=np.float32)
        obs_abs = np.abs(obs_array)


        mean_val = np.mean(obs_abs)
        mean_level = min(int(mean_val * 20), 9)
        features.append(mean_level)


        std_val = np.std(obs_array)
        std_level = min(int(std_val * 10), 9)
        features.append(std_level)


        non_zero_ratio = np.sum(obs_abs > 0.01) / max(obs_len, 1)
        non_zero_level = min(int(non_zero_ratio * 10), 9)
        features.append(non_zero_level)


        if obs_len > 10:

            first_half = obs_abs[:obs_len//2]
            first_sum = np.sum(first_half)
            first_level = min(int(first_sum / 10), 9)
            features.append(first_level)


            second_half = obs_abs[obs_len//2:]
            second_sum = np.sum(second_half)
            second_level = min(int(second_sum / 10), 9)
            features.append(second_level)
        else:
            features.extend([0, 0])


        if obs_len > 5:
            max_idx = np.argmax(obs_abs)
            min_idx = np.argmin(obs_abs)
            max_pos = min(int(max_idx / max(obs_len / 10, 1)), 9)
            min_pos = min(int(min_idx / max(obs_len / 10, 1)), 9)
            features.extend([max_pos, min_pos])
        else:
            features.extend([0, 0])


        context_key = tuple(features)

        return context_key

    def select_action(self, observation, valid_actions=None):
        context = self.get_context_key(observation)


        if valid_actions is None:
            valid_actions = list(range(self.num_actions))
        valid_actions = [int(a) for a in valid_actions]


        N_context = self.state_visits[context]


        ucb_values = np.full(self.num_actions, -np.inf)

        for action in valid_actions:
            Q_value = self.Q[context][action]
            N_action = self.N[context][action]

            if N_action == 0:


                heuristic_val = self.heuristic.evaluate(observation, action)

                base_bonus = 100.0
                decay_factor = 1.0 / (1.0 + self.total_steps / 2000.0 + N_context / 1000.0)
                exploration_bonus = base_bonus * decay_factor
                ucb_values[action] = heuristic_val + exploration_bonus
            else:


                avg_Q = Q_value / N_action if N_action > 0 else 0.0


                adaptive_c = self.c_param * (1.0 - min(0.7, N_context / 5000))
                adaptive_c = max(self.min_c_param, min(self.max_c_param, adaptive_c))


                if N_context > 0:

                    exploration = adaptive_c * math.sqrt(
                        math.log(N_context + 1) / (N_action + 1e-6)
                    )

                    exploration *= (1.0 - min(0.3, N_action / 100))
                else:
                    exploration = float('inf')

                ucb_values[action] = avg_Q + exploration


        best_action = int(np.argmax(ucb_values))

        return best_action

    def update(self, observation, action, reward, next_observation=None, done=False):
        context = self.get_context_key(observation)
        action = int(action)


        reward = np.clip(reward, -1000, 1000)


        if next_observation is None or done:
            old_Q = self.Q[context][action]
            N = self.N[context][action]


            if N < 5:
                alpha = 0.8
            elif N < 20:
                alpha = 0.5
            elif N < 100:
                alpha = 0.2
            else:

                alpha = 0.1 / (1.0 + N / 100)


            self.Q[context][action] = (1 - alpha) * old_Q + alpha * reward


        else:
            next_context = self.get_context_key(next_observation)


            next_Q_values = self.Q[next_context] / (self.N[next_context] + 1e-6)
            max_next_Q = np.max(next_Q_values[next_Q_values > -np.inf]) if np.any(next_Q_values > -np.inf) else 0.0


            td_target = reward + self.gamma * max_next_Q
            td_target = np.clip(td_target, -1000, 1000)


            old_Q = self.Q[context][action]
            N = self.N[context][action]

            if N < 5:
                alpha = 0.8
            elif N < 20:
                alpha = 0.5
            elif N < 100:
                alpha = 0.2
            else:
                alpha = 0.1 / (1.0 + N / 100)


            self.Q[context][action] = (1 - alpha) * old_Q + alpha * td_target


        self.N[context][action] += 1
        self.state_visits[context] += 1
        self.total_steps += 1

    def get_policy_info(self):
        num_states = len(self.Q)
        total_visits = sum(self.state_visits.values())

        return {
            'num_states_visited': num_states,
            'total_steps': self.total_steps,
            'average_visits_per_state': total_visits / max(num_states, 1)
        }
