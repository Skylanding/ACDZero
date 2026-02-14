"""
Lightweight MuZero-style PUCT search used as a policy improver.

This is intentionally minimal: it assumes a model interface with
 - initial_inference(obs) -> (policy_logits, value, latent)
 - recurrent_inference(latent, action) -> (policy_logits, value, reward, next_latent)

The search tree stores priors and visit counts and outputs an improved
action distribution pi (visit counts with temperature).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F


@dataclass
class SearchNode:
    prior: float
    value_sum: float = 0.0
    visit_count: int = 0
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    reward: float = 0.0
    latent: Optional[torch.Tensor] = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def softmax_sample(counts: List[int], temperature: float) -> List[float]:
    visits = torch.tensor(counts, dtype=torch.float32)
    if temperature == 0:
        # deterministic: pick argmax
        probs = torch.zeros_like(visits)
        probs[visits.argmax()] = 1.0
        return probs.tolist()
    scaled = visits ** (1.0 / temperature)
    scaled = scaled / (scaled.sum() + 1e-8)
    return scaled.tolist()


class MuZeroPlanner:
    def __init__(
        self,
        model,
        num_simulations: int = 32,
        c_puct: float = 1.5,
        gamma: float = 0.99,
        temperature_train: float = 1.0,
        temperature_eval: float = 0.1,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.3,
        use_dynamic_c_puct: bool = True,
        c_base: float = 19652,
        c_init: float = 1.25,
    ):
        """
        model: provides initial_inference / recurrent_inference
        num_simulations: how many PUCT rollouts to run
        c_puct: exploration constant (fallback if dynamic disabled)
        gamma: discount for backups
        temperature_train/temperature_eval: visit-count temperature for train/eval
        dirichlet_*: root exploration noise hyperparams (train only)
        use_dynamic_c_puct: if True, use MuZero-style log schedule
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.temperature_train = temperature_train
        self.temperature_eval = temperature_eval
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.use_dynamic_c_puct = use_dynamic_c_puct
        self.c_base = c_base
        self.c_init = c_init
        self.is_training = True

    def set_training(self, flag: bool):
        self.is_training = flag

    def _node_c_puct(self, total_visits: int):
        if not self.use_dynamic_c_puct:
            return self.c_puct
        return math.log((1 + total_visits + self.c_base) / self.c_base) + self.c_init

    def run(self, obs) -> Tuple[int, List[float]]:
        root_policy, root_value, root_latent = self.model.initial_inference(obs)
        root_policy = F.softmax(root_policy, dim=-1).detach().cpu()

        # Root exploration noise (train only)
        if self.is_training and self.dirichlet_epsilon > 0:
            noise = torch.distributions.Dirichlet(
                torch.full_like(root_policy, self.dirichlet_alpha)
            ).sample()
            root_policy = (1 - self.dirichlet_epsilon) * root_policy + self.dirichlet_epsilon * noise

        root = SearchNode(prior=1.0, value_sum=root_value.item(), visit_count=1, latent=root_latent)
        for action, p in enumerate(root_policy.tolist()):
            root.children[action] = SearchNode(prior=p)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            action_taken = []

            # Selection
            while node.children:
                total_visits = sum(child.visit_count for child in node.children.values())
                c_puct = self._node_c_puct(total_visits)
                best_score = -1e9
                best_action = None
                for a, child in node.children.items():
                    q = child.value
                    u = c_puct * child.prior * math.sqrt(total_visits + 1e-8) / (1 + child.visit_count)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_action = a
                a = best_action
                action_taken.append(a)
                node = node.children[a]
                search_path.append(node)

                if node.latent is None:
                    break

            # Expansion with dynamics
            parent = search_path[-2] if len(search_path) >= 2 else root
            latent = parent.latent
            if latent is None:
                # Should not happen; root always has latent
                continue

            policy_logits, value, reward, next_latent = self.model.recurrent_inference(latent, action_taken[-1])
            policy = F.softmax(policy_logits, dim=-1).detach().cpu()

            node.latent = next_latent
            node.reward = reward.item()
            for action, p in enumerate(policy.tolist()):
                node.children[action] = SearchNode(prior=p)

            # Backup
            g = value.item()
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += g
                g = n.reward + self.gamma * g

        counts = [child.visit_count for child in root.children.values()]
        temp = self.temperature_train if self.is_training else self.temperature_eval
        pi = softmax_sample(counts, temp)
        # Action selection: sample from pi
        action_dist = torch.distributions.Categorical(torch.tensor(pi))
        action = action_dist.sample().item()
        return action, pi

