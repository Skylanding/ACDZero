
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCTSNode:

    def __init__(self, state_key, prior: float = 1.0):
        self.state_key = state_key
        self.prior = prior


        self.value_sum = 0.0
        self.visit_count = 0


        self.children: Dict[int, 'MCTSNode'] = {}


        self.is_expanded = False


        self.parent: Optional['MCTSNode'] = None


        self.state: Optional[np.ndarray] = None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_child(self, action: int, prior: float) -> 'MCTSNode':
        child = MCTSNode(state_key=(self.state_key, action), prior=prior)
        child.parent = self
        self.children[action] = child
        return child

    def best_child(self, c_param: float = 1.5) -> Tuple[int, 'MCTSNode']:
        if not self.children:
            return None, None

        best_score = float('-inf')
        best_action = None
        best_node = None

        for action, child in self.children.items():

            exploitation = child.value()
            exploration = c_param * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_node = child

        return best_action, best_node

    def select_action(self, temperature: float = 1.0) -> int:
        if not self.children:
            return None

        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:

            best_idx = np.argmax(visit_counts)
            return actions[best_idx]
        else:

            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()
            action_idx = np.random.choice(len(actions), p=probs)
            return actions[action_idx]


class MuZeroMCTS:

    def __init__(
        self,
        num_simulations: int = 50,
        c_param: float = 1.5,
        discount: float = 0.95,
        use_value_network: bool = True
    ):
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.discount = discount
        self.use_value_network = use_value_network


        self.node_cache: Dict[Any, MCTSNode] = {}


        self.value_network: Optional[nn.Module] = None


        self.dynamics_model: Optional[nn.Module] = None


        self.use_heuristic_transition = True

    def set_value_network(self, network: nn.Module):
        self.value_network = network
        self.use_value_network = True

    def set_dynamics_model(self, model: nn.Module):
        self.dynamics_model = model

    def _hash_state(self, observation: np.ndarray) -> Any:
        obs = np.array(observation)
        if len(obs) == 0:
            return 0

        features = []


        features.extend([
            float(np.mean(obs)),
            float(np.std(obs)),
            float(np.sum(obs > 0)),
        ])


        if len(obs) > 50:

            features.append(float(obs[0]) if len(obs) > 0 else 0.0)


            num_agents = 5
            obs_per_agent = len(obs) // num_agents if num_agents > 0 else len(obs)

            for i in range(min(num_agents, 5)):
                start_idx = i * obs_per_agent
                end_idx = start_idx + obs_per_agent
                if end_idx <= len(obs):
                    agent_obs = obs[start_idx:end_idx]
                    threat_start = len(agent_obs) // 2
                    threat_features = agent_obs[threat_start:]
                    threat_level = float(np.sum(np.abs(threat_features)))
                    features.append(threat_level)


        features = [round(f, 4) for f in features]
        return tuple(features)

    def _predict_value(self, observation: np.ndarray) -> float:
        if self.value_network is not None:

            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                value = self.value_network(obs_tensor).item()
            return value
        else:

            obs = np.array(observation)
            if len(obs) == 0:
                return -500.0


            threat_level = np.sum(np.abs(obs[len(obs)//2:]))
            value = -threat_level * 10.0
            return value

    def _predict_prior(self, observation: np.ndarray, action: int, num_actions: int) -> float:

        return 1.0 / num_actions

    def _predict_next_state(self, observation: np.ndarray, action: int) -> np.ndarray:
        if self.dynamics_model is not None:

            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_tensor = torch.LongTensor([action]).unsqueeze(0)
            with torch.no_grad():
                next_state, reward = self.dynamics_model(obs_tensor, action_tensor)
            return next_state.squeeze().numpy()
        else:


            obs = np.array(observation).copy()


            rng = np.random.RandomState(int(action) % 1000)


            num_dims = len(obs)


            i = np.arange(num_dims)
            action_effect = np.sin((action + 1) * (i + 1) * 0.1) * 0.05


            noise = rng.randn(num_dims) * 0.01 * (action + 1)

            next_state = obs + action_effect + noise


            next_state = np.clip(next_state, obs.min() - 1.0, obs.max() + 1.0)

            return next_state

    def search(
        self,
        observation: np.ndarray,
        valid_actions: List[int],
        num_actions: int
    ) -> Dict[int, float]:
        state_key = self._hash_state(observation)


        if state_key in self.node_cache:
            root = self.node_cache[state_key]
        else:
            root = MCTSNode(state_key=state_key)

            root.state = observation.copy()
            self.node_cache[state_key] = root


        for _ in range(self.num_simulations):

            node = root
            path = [node]
            actions = []

            while node.is_expanded and node.children:
                action, child = node.best_child(self.c_param)
                if child is None:
                    break
                node = child
                path.append(node)
                actions.append(action)


            if not node.is_expanded:


                for action in valid_actions:
                    prior = self._predict_prior(observation, action, num_actions)
                    child = node.add_child(action, prior)


                    next_state = self._predict_next_state(observation, action)

                    child.state = next_state
                node.is_expanded = True


            if node.children:


                best_action, best_child = node.best_child(self.c_param)
                if best_child is not None and hasattr(best_child, 'state'):

                    value = self._predict_value(best_child.state)
                else:

                    value = self._predict_value(observation)
            else:

                value = self._predict_value(observation)


            for node in reversed(path):
                node.visit_count += 1
                node.value_sum += value
                value *= self.discount


        action_probs = {}
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for action in valid_actions:
                if action in root.children:
                    prob = root.children[action].visit_count / total_visits
                    action_probs[action] = prob
                else:
                    action_probs[action] = 0.0
        else:

            for action in valid_actions:
                action_probs[action] = 1.0 / len(valid_actions)

        return action_probs

    def get_top_k_actions(
        self,
        observation: np.ndarray,
        valid_actions: List[int],
        k: int = 10
    ) -> List[int]:
        action_probs = self.search(observation, valid_actions, len(valid_actions))


        sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)


        top_k = [action for action, prob in sorted_actions[:k]]

        return top_k

    def reset(self):
        self.node_cache.clear()


class MuZeroValueNetwork(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class MuZeroPolicyNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logits = self.net(observation)
        return F.softmax(logits, dim=-1)
