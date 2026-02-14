
import os
import sys
import time
import math
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from agents.muzero_mcts import MuZeroMCTS, MuZeroValueNetwork, MuZeroPolicyNetwork
from training.envs import create_hybrid_env
from training.observations import cast_observations_to_float32
from training.paths import create_timestamped_run_dirs, resolve_training_data_dir


class MCTSActionFilter:

    def __init__(self, c_param=1.5, num_simulations=50, use_value_network=False):

        self.mcts = MuZeroMCTS(
            num_simulations=num_simulations,
            c_param=c_param,
            discount=0.95,
            use_value_network=use_value_network
        )


        self.c_param = c_param


        self.action_costs = {
            'Sleep': 0,
            'Monitor': 1,
            'Analyse': 2,
            'DeployDecoy': 2,
            'Remove': 3,
            'Restore': 5,
            'BlockTraffic': 1,
            'AllowTraffic': 1,
        }


        self.action_id_to_type = {}
        self.action_labels = []
        self.action_space_size = 41


        self.value_network = None
        self.obs_dim = None


    def _init_action_space(self, env, obs_dim=None):
        try:

            first_agent = list(env.agents)[0] if hasattr(env, 'agents') else None
            if first_agent:

                if hasattr(env, 'action_space') and callable(env.action_space):
                    action_space = env.action_space(first_agent)
                elif hasattr(env, 'action_space'):
                    action_space = env.action_space
                else:
                    action_space = None

                if action_space is not None and hasattr(action_space, 'n'):
                    self.action_space_size = action_space.n


                if hasattr(env, 'action_labels'):
                    self.action_labels = env.action_labels(first_agent)
                elif hasattr(env, 'get_action_labels'):
                    self.action_labels = env.get_action_labels(first_agent)


                if self.action_labels:
                    for action_id, label in enumerate(self.action_labels):
                        action_type = self._parse_action_type_from_label(label)
                        self.action_id_to_type[action_id] = action_type


                if obs_dim is not None and self.mcts.use_value_network:
                    self.obs_dim = obs_dim
                    self.value_network = MuZeroValueNetwork(obs_dim=obs_dim)
                    self.mcts.set_value_network(self.value_network)
        except Exception as e:
            print(f"Warning: error while initializing action space: {e}")

            self.action_space_size = 41

    def _parse_action_type_from_label(self, label):
        label_str = str(label).lower()
        if 'sleep' in label_str:
            return 'Sleep'
        elif 'monitor' in label_str:
            return 'Monitor'
        elif 'analyse' in label_str or 'analyze' in label_str:
            return 'Analyse'
        elif 'decoy' in label_str:
            return 'DeployDecoy'
        elif 'remove' in label_str:
            return 'Remove'
        elif 'restore' in label_str:
            return 'Restore'
        elif 'block' in label_str:
            return 'BlockTraffic'
        elif 'allow' in label_str:
            return 'AllowTraffic'
        else:
            return 'Unknown'

    def _get_action_type(self, action):
        return self.action_id_to_type.get(int(action), 'Unknown')

    def _hash_state(self, observation):
        obs = np.array(observation)
        if len(obs) == 0:
            return hash(0)

        features = []


        features.extend([
            float(np.mean(obs)),
            float(np.std(obs)),
            float(np.min(obs)),
            float(np.max(obs)),
            float(np.sum(obs > 0)),
            float(np.sum(obs < 0)),
        ])


        if len(obs) > 50:

            features.append(float(obs[0]) if len(obs) > 0 else 0.0)


            num_agents = 5
            obs_per_agent = len(obs) // num_agents if num_agents > 0 else len(obs)

            zone_threats = []
            for i in range(num_agents):
                start_idx = i * obs_per_agent
                end_idx = start_idx + obs_per_agent
                if end_idx <= len(obs):
                    agent_obs = obs[start_idx:end_idx]

                    threat_start = len(agent_obs) // 2
                    threat_features = agent_obs[threat_start:]
                    threat_level = float(np.sum(np.abs(threat_features)))
                    zone_threats.append(threat_level)


            features.extend(zone_threats[:5])


            global_threat = float(np.sum(np.abs(obs[len(obs)//2:])))
            features.append(global_threat)


            if len(obs) > 100:
                firewall_features = obs[50:100]
                features.extend([
                    float(np.sum(firewall_features > 0)),
                    float(np.mean(np.abs(firewall_features))),
                ])


        features = [round(f, 2) for f in features]
        return hash(tuple(features))


    def _heuristic_value(self, observation, action):
        obs = np.array(observation)
        action_type = self._get_action_type(action)


        mission_phase = obs[0] if len(obs) > 0 else 0


        threat_start = len(obs) // 2 if len(obs) > 20 else 0
        threat_level = float(np.sum(np.abs(obs[threat_start:]))) if len(obs) > threat_start else 0.0

        reward = 0.0


        if threat_level > 5.0:
            if action_type in ['Remove', 'Restore']:
                reward += 50.0
            elif action_type == 'Sleep':
                reward -= 100.0


        if threat_level < 2.0:
            if action_type == 'Monitor':
                reward += 20.0
            elif action_type in ['Remove', 'Restore']:
                reward -= 30.0


        if mission_phase > 0.5:
            if action_type in ['Monitor', 'Analyse']:
                reward += 30.0


        cost = self.action_costs.get(action_type, 1)
        reward -= cost * 2.0

        return reward

    def get_top_k_actions(self, observation, all_actions, k=10):

        valid_actions = [int(a) for a in all_actions]
        top_k_actions = self.mcts.get_top_k_actions(
            observation=observation,
            valid_actions=valid_actions,
            k=k
        )

        return top_k_actions

    def update_from_experience(self, observation, action, reward, next_observation=None, gamma=0.95):


        reward = np.clip(reward, -1000, 1000)


        state_key = self.mcts._hash_state(observation)
        if state_key in self.mcts.node_cache:
            node = self.mcts.node_cache[state_key]
            action = int(action)

            if action in node.children:
                child = node.children[action]

                child.visit_count += 1
                child.value_sum += reward

    def get_exploration_bonus(self, observation, action):
        state_key = self.mcts._hash_state(observation)
        action = int(action)


        if state_key in self.mcts.node_cache:
            node = self.mcts.node_cache[state_key]
            if action in node.children:
                visit_count = node.children[action].visit_count
            else:
                visit_count = 0
        else:
            visit_count = 0


        bonus = 10.0 / (visit_count + 1)
        return bonus


class SimplePolicyNetwork(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_action_probs(self, obs):
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)


class HybridMCTSRLAgent:

    def __init__(self, agent_id, action_filter, obs_dim, action_dim, top_k=10, use_policy_net=True):
        self.agent_id = agent_id
        self.action_filter = action_filter
        self.top_k = top_k
        self.use_policy_net = use_policy_net


        if use_policy_net:
            self.policy_net = SimplePolicyNetwork(obs_dim, action_dim, hidden_dim=128)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5)
            self.value_net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=5e-5)


        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }


        self.mcts_visits = defaultdict(lambda: defaultdict(int))

    def _hash_state(self, observation):
        obs = np.array(observation)
        if len(obs) == 0:
            return hash(0)
        features = [
            float(np.mean(obs)),
            float(np.std(obs)),
            float(np.sum(obs > 0)),
        ]
        return hash(tuple(features))

    def get_candidate_actions(self, observation, all_actions):
        return self.action_filter.get_top_k_actions(observation, all_actions, k=self.top_k)

    def select_action(self, observation, all_actions):

        candidate_actions = self.get_candidate_actions(observation, all_actions)

        if not self.use_policy_net or len(candidate_actions) == 0:

            if len(candidate_actions) > 0:
                return np.random.choice(candidate_actions)
            else:
                return np.random.choice(all_actions)


        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        else:
            observation = observation.astype(np.float32)
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        with torch.no_grad():

            all_probs = self.policy_net.get_action_probs(obs_tensor).squeeze().numpy()


            candidate_probs = all_probs[candidate_actions]
            candidate_probs = candidate_probs / candidate_probs.sum()


            for i, action in enumerate(candidate_actions):
                bonus = self.get_exploration_bonus(observation, action)
                candidate_probs[i] += 0.1 * bonus


            candidate_probs = candidate_probs / candidate_probs.sum()


            action_idx = np.random.choice(len(candidate_actions), p=candidate_probs)
            action = candidate_actions[action_idx]


            log_prob = torch.log(torch.FloatTensor([all_probs[action]]))

        return int(action), log_prob

    def get_exploration_bonus(self, observation, action):
        return self.action_filter.get_exploration_bonus(observation, action)

    def update_mcts_stats(self, observation, action):
        state_key = self._hash_state(observation)
        self.mcts_visits[state_key][int(action)] += 1

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)

    def update_policy(self, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        if not self.use_policy_net or len(self.buffer['states']) == 0:
            return


        states = torch.FloatTensor(np.array(self.buffer['states']))
        actions = torch.LongTensor(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        old_log_probs = torch.cat(self.buffer['log_probs'])


        with torch.no_grad():
            values = self.value_net(states).squeeze()


        returns = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_reward = 0
            discounted_reward = rewards[i] + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns)


        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        for _ in range(2):

            logits = self.policy_net(states)
            new_log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze()


            ratio = torch.exp(torch.clamp(action_log_probs - old_log_probs, -10, 10))


            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()


            kl_div = (old_log_probs - action_log_probs).mean()
            kl_penalty = 0.01 * kl_div.abs()
            policy_loss = policy_loss + kl_penalty


            value_pred = self.value_net(states).squeeze()
            value_clipped = returns + torch.clamp(value_pred - returns, -10, 10)
            value_loss = torch.max(
                F.mse_loss(value_pred, returns),
                F.mse_loss(value_pred, value_clipped)
            )


            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * new_log_probs).sum(dim=-1).mean()
            entropy_bonus = 0.01 * entropy


            total_loss = policy_loss + 0.5 * value_loss - entropy_bonus


            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.optimizer.step()
            self.value_optimizer.step()


        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }


def train_hybrid_mcts_ppo(iterations=100, steps=100, output_dir="experiment/checkpoints", gpu_id=None, top_k=10, use_deterministic_red=True, use_centralized_obs=True, num_simulations=50):

    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU {gpu_id}")


    output_path, checkpoint_dir = create_timestamped_run_dirs(output_dir, "hybrid_mcts_ppo")
    unified_data_dir = resolve_training_data_dir(output_dir)
    unified_csv = unified_data_dir / "hybrid_mcts_ppo_training_data.csv"


    print("Creating CAGE environment...")
    env = create_hybrid_env(
        use_deterministic_red=use_deterministic_red,
        use_centralized_obs=use_centralized_obs,
    )


    print("Initializing MuZero MCTS action filter...")
    print(f"   MCTS simulations: {num_simulations}")
    action_filter = MCTSActionFilter(
        c_param=1.5,
        num_simulations=num_simulations,
        use_value_network=False
    )


    obs, info = env.reset()
    first_agent = list(env.agents)[0] if hasattr(env, 'agents') else None
    if first_agent and first_agent in obs:
        obs_dim = len(obs[first_agent])
        action_filter._init_action_space(env, obs_dim=obs_dim)
    else:
        action_filter._init_action_space(env)


    if 'obs' not in locals():
        obs, info = env.reset()


    agent_obs_dims = {}
    for agent_id in env.agents:
        actual_obs = obs[agent_id]
        if isinstance(actual_obs, np.ndarray):
            obs_dim = int(actual_obs.shape[0]) if len(actual_obs.shape) > 0 else int(actual_obs.size)
        elif hasattr(actual_obs, '__len__'):
            obs_dim = int(len(actual_obs))
        else:
            obs_dim = 1
        agent_obs_dims[agent_id] = obs_dim


    first_agent = list(env.agents)[0]

    if hasattr(env, 'action_space') and callable(env.action_space):
        action_space = env.action_space(first_agent)
    elif hasattr(env, 'action_space'):
        action_space = env.action_space
    else:
        action_space = None

    if action_space is not None and hasattr(action_space, 'n'):
        action_dim = int(action_space.n)
    else:
        action_dim = int(action_filter.action_space_size)



    agents = {}
    for agent_id in env.agents:
        agent_obs_dim = agent_obs_dims[agent_id]
        agents[agent_id] = HybridMCTSRLAgent(
            agent_id,
            action_filter,
            obs_dim=agent_obs_dim,
            action_dim=action_dim,
            top_k=top_k,
            use_policy_net=True
        )
        


    training_data = {
        'iterations': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'mcts_filter_stats': [],
    }


    smoothed_reward = None
    ema_alpha = 0.1

    best_reward = float('-inf')
    start_time = time.time()

    print(f"\nStart Hybrid MCTS-PPO training")
    print(f"   Iterations: {iterations}")
    print(f"   Steps per episode: {steps}")
    print(f"   MCTS top-k actions: {top_k} (from {action_filter.action_space_size} actions)")
    print(f"   Output directory: {output_path}\n")


    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    for iteration in range(iterations):
        episode_start = time.time()


        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0


        obs = cast_observations_to_float32(obs, env.agents)


        filter_stats = {
            'total_actions': 0,
            'filtered_actions': 0,
        }

        for step in range(steps):
            actions = {}


            log_probs_dict = {}
            for agent_id in env.agents:
                agent_obs = obs[agent_id]
                agent = agents[agent_id]


                all_actions = list(range(action_filter.action_space_size))
                candidate_actions = agent.get_candidate_actions(agent_obs, all_actions)

                filter_stats['total_actions'] += len(all_actions)
                filter_stats['filtered_actions'] += len(candidate_actions)


                result = agent.select_action(agent_obs, all_actions)
                if isinstance(result, tuple):
                    action, log_prob = result
                    log_probs_dict[agent_id] = log_prob
                else:
                    action = result
                    log_probs_dict[agent_id] = torch.tensor(0.0)

                actions[agent_id] = action


            obs_next, rewards, terminations, truncations, info = env.step(actions)


            step_reward = sum(rewards.values())
            episode_reward += step_reward
            episode_length += 1


            num_agents = len(env.agents) if len(env.agents) > 0 else len(actions)
            for agent_id, action in actions.items():
                agent_obs = obs[agent_id]
                agent_obs_next = obs_next[agent_id]
                agent_reward = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                log_prob = log_probs_dict.get(agent_id, torch.tensor(0.0))


                if num_agents > 0 and step_reward != 0:
                    shared_reward = step_reward / num_agents

                    total_reward = agent_reward * 0.9 + shared_reward * 0.1
                else:
                    total_reward = agent_reward


                total_reward = max(-500.0, min(500.0, total_reward))


                agents[agent_id].store_transition(
                    agent_obs, action, total_reward, agent_obs_next, done, log_prob
                )


                action_filter.update_from_experience(
                    agent_obs, action, total_reward,
                    next_observation=agent_obs_next, gamma=0.95
                )


                agents[agent_id].update_mcts_stats(agent_obs, action)

            obs = obs_next


            if any(terminations.values()) or any(truncations.values()):
                break


        update_frequency = 5
        if (iteration + 1) % update_frequency == 0:
            for agent_id, agent in agents.items():
                if len(agent.buffer['states']) > 0:
                    agent.update_policy()


        if smoothed_reward is None:
            smoothed_reward = episode_reward
        else:
            smoothed_reward = ema_alpha * episode_reward + (1 - ema_alpha) * smoothed_reward


        training_data['iterations'].append(iteration)
        training_data['episode_rewards'].append(episode_reward)
        training_data['episode_lengths'].append(episode_length)
        training_data['mcts_filter_stats'].append(filter_stats)


        if episode_reward > best_reward:
            best_reward = episode_reward


        elapsed = time.time() - episode_start
        filter_ratio = filter_stats['filtered_actions'] / filter_stats['total_actions'] if filter_stats['total_actions'] > 0 else 0

        print(f"[{iteration+1:4d}/{iterations}] HybridMCTS-PPO | "
              f"Reward: {episode_reward:8.2f} | "
              f"Smoothed: {smoothed_reward:8.2f} | "
              f"Best: {best_reward:8.2f} | "
              f"Length: {episode_length:3d} | "
              f"Filter: {filter_ratio:.1%} | "
              f"Time: {elapsed:.1f}s")


        sys.stdout.flush()
        sys.stderr.flush()


        if True:

            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            data_file = checkpoint_dir / "training_data.json"
            with open(data_file, 'w') as f:
                json.dump(training_data, f, indent=2)


            csv_file = checkpoint_dir / "training_data.csv"
            try:
                import pandas as pd
                df = pd.DataFrame({
                    'iteration': training_data['iterations'],
                    'episode_reward': training_data['episode_rewards'],
                    'episode_length': training_data['episode_lengths'],
                })
                df.to_csv(csv_file, index=False)


                df.to_csv(unified_csv, index=False)
            except ImportError:

                with open(csv_file, 'w') as f:
                    f.write('iteration,episode_reward,episode_length\n')
                    for i, r, l in zip(
                        training_data['iterations'],
                        training_data['episode_rewards'],
                        training_data['episode_lengths']
                    ):
                        f.write(f'{i},{r},{l}\n')


                with open(unified_csv, 'w') as f:
                    f.write('iteration,episode_reward,episode_length\n')
                    for i, r, l in zip(
                        training_data['iterations'],
                        training_data['episode_rewards'],
                        training_data['episode_lengths']
                    ):
                        f.write(f'{i},{r},{l}\n')

    total_time = time.time() - start_time

    print(f"\nTraining completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Mean reward: {np.mean(training_data['episode_rewards']):.2f}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Training data saved to: {checkpoint_dir}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid MCTS-PPO")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k candidate actions selected by MCTS")
    parser.add_argument("--deterministic-red", action="store_true", default=True,
                       help="Use deterministic Red agent (recommended for MCTS)")
    parser.add_argument("--centralized-obs", action="store_true", default=True,
                       help="Use global observation wrapper (recommended for full observability)")
    parser.add_argument("--num-simulations", type=int, default=50,
                       help="MCTS simulation count (MuZero parameter, default 50)")

    args = parser.parse_args()

    train_hybrid_mcts_ppo(
        iterations=args.iterations,
        steps=args.steps,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
        top_k=args.top_k,
        use_deterministic_red=args.deterministic_red,
        use_centralized_obs=args.centralized_obs,
        num_simulations=args.num_simulations
    )
