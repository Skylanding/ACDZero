import os
import sys
import time
import argparse
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import math

import warnings
from training.envs import create_enterprise_mae_env

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.0):
        choices_weights = [
            (c.value / (c.visits + 1e-6)) +
            c_param * math.sqrt((2 * math.log(self.visits + 1)) / (c.visits + 1e-6))
            for c in self.children.values()
        ]
        if not choices_weights:
            return None
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        child = MCTSNode(state=next_state, parent=self, action=action)
        self.children[action] = child
        return child

    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits


class MCTSAgent:
    def __init__(self, num_simulations=100, c_param=1.0, temperature=1.0, use_value_estimate=True):
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.temperature = temperature
        self.use_value_estimate = use_value_estimate


        self.value_estimates = defaultdict(lambda: 0.0)
        self.value_counts = defaultdict(lambda: 0)
        self.reward_history = []

    def select_action(self, env, observation):

        action_space = env.action_space(env.agents[0])
        if hasattr(action_space, 'sample'):

            num_actions = action_space.n
            actions = list(range(num_actions))
        else:

            actions = list(range(100))

        root = MCTSNode(state=observation)
        root.untried_actions = actions.copy()


        for _ in range(self.num_simulations):
            node = root


            while node.untried_actions == [] and node.children != {}:
                node = node.best_child(self.c_param)
                if node is None:
                    break


            if node.untried_actions != []:
                action = np.random.choice(node.untried_actions)

                next_state = observation
                node = node.expand(action, next_state)


            reward = self.simulate(env, observation, node.action)


            while node is not None:
                node.update(reward)
                node = node.parent
                reward *= 0.95


        if root.children:
            if self.temperature > 0:

                visits = [c.visits for c in root.children.values()]
                probs = np.array(visits) ** (1.0 / self.temperature)
                probs = probs / probs.sum()
                actions_list = list(root.children.keys())
                best_action = np.random.choice(actions_list, p=probs)
            else:

                best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        else:
            best_action = np.random.choice(actions)

        return best_action

    def _hash_observation(self, observation):
        if isinstance(observation, np.ndarray):


            return hash((observation.mean(), observation.std(), observation.shape))
        return hash(str(observation))

    def update_value_estimate(self, observation, action, reward):
        if not self.use_value_estimate:
            return

        state_hash = self._hash_observation(observation)
        key = (state_hash, action)


        count = self.value_counts[key]
        old_value = self.value_estimates[key]
        self.value_estimates[key] = (old_value * count + reward) / (count + 1)
        self.value_counts[key] = count + 1


        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)

    def simulate(self, env, observation, action, max_steps=10):


        state_hash = self._hash_observation(observation)
        if self.use_value_estimate and (state_hash, action) in self.value_estimates:
            base_value = self.value_estimates[(state_hash, action)]

            noise = np.random.normal(0, 0.1)
            return base_value + noise


        if len(self.reward_history) > 0:
            recent_rewards = self.reward_history[-10:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 1.0

            return avg_reward + np.random.normal(0, std_reward * 0.3)


        return -500.0


def train_mcts(iterations=100, steps=100, num_simulations=50, output_dir="experiment/checkpoints", gpu_id=None):


    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"  GPU: {gpu_id}")
    else:

        print(f"  CPU (MCTS algorithm)")

    print("=" * 80)
    print("MCTSTraining config")
    print(f"  Iterations: {iterations}")
    print(f"  Episode length: {steps}")
    print(f"  MCTS simulations: {num_simulations}")
    print("=" * 80)


    cyborg = create_enterprise_mae_env(steps=steps)


    agent = MCTSAgent(
        num_simulations=num_simulations,
        c_param=1.0,
        temperature=1.5,
        use_value_estimate=True
    )


    data_dir = Path(output_dir) / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "mcts_training_data.csv"


    training_data = []


    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'episode_reward', 'smoothed_reward', 'episode_length',
            'elapsed_time', 'best_reward'
        ])

    print("\nStart training MCTS...")
    print(f"Training data will be saved to: {csv_file}")
    start_time = time.time()
    best_reward = float('-inf')

    for i in range(iterations):

        obs, info = cyborg.reset()
        episode_reward = 0
        episode_length = 0


        for step in range(steps):

            actions = {}
            for agent_id in cyborg.agents:
                agent_obs = obs[agent_id]
                action = agent.select_action(cyborg, agent_obs)
                actions[agent_id] = action


            obs, rewards, terminations, truncations, info = cyborg.step(actions)


            step_reward = sum(rewards.values())
            episode_reward += step_reward
            episode_length += 1


            for agent_id, action in actions.items():
                agent_obs = obs[agent_id]
                agent.update_value_estimate(agent_obs, action, step_reward)


            if any(terminations.values()) or any(truncations.values()):
                break


        if i == 0:
            smoothed_reward = episode_reward
        else:

            alpha = 0.1
            smoothed_reward = alpha * episode_reward + (1 - alpha) * training_data[-1].get('smoothed_reward', episode_reward)


        if episode_reward > best_reward:
            best_reward = episode_reward

        elapsed = time.time() - start_time

        data_point = {
            'iteration': i,
            'episode_reward': episode_reward,
            'smoothed_reward': smoothed_reward,
            'episode_length': episode_length,
            'elapsed_time': elapsed,
            'best_reward': best_reward
        }
        training_data.append(data_point)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, episode_reward, smoothed_reward, episode_length, elapsed, best_reward
            ])

        if i % 10 == 0 or i == iterations - 1:
            print(f"[{i:4d}/{iterations}] MCTS | "
                  f"Reward: {episode_reward:8.2f} | Best: {best_reward:8.2f} | "
                  f"Length: {episode_length:3d} | Time: {elapsed/60:.1f}min")
            sys.stdout.flush()


    json_file = data_dir / "mcts_training_data.json"
    with open(json_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Training data saved to: {json_file}")

    total_time = time.time() - start_time
    avg_reward = np.mean([d['episode_reward'] for d in training_data])

    print(f"\nMCTS Training completed!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Mean reward: {avg_reward:.2f}")
    print(f"  Best reward: {best_reward:.2f}")
    print("=" * 80 + "\n")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiment/checkpoints")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU ID (MCTS usually does not need GPU, but can be specified for consistency)")

    args = parser.parse_args()

    train_mcts(
        iterations=args.iterations,
        steps=args.steps,
        num_simulations=args.simulations,
        output_dir=args.output,
        gpu_id=args.gpu
    )
