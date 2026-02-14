import inspect
import time
import os

from statistics import mean, stdev
from typing import Any
from rich import print

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BaseWrapper, BlueFlatWrapper, BlueFixedActionWrapper, EnterpriseMAE

import numpy as np

from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import register_env

import warnings
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


sys.stdout.flush()

warnings.filterwarnings("ignore", category=DeprecationWarning)


def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,

        steps=100,
    )
    cyborg = CybORG(scenario_generator=sg)
    cyborg = EnterpriseMAE(cyborg)
    return cyborg


NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}


def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]


register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
env = env_creator_CC4({})


algo_config = (
    DQNConfig().framework("torch")

    .debugging(logger_config={"logdir":"logs/DQN_Complicated_SleepRed", "type":"ray.tune.logger.TBXLogger"})
    .environment(env="CC4")

    .resources(
        num_gpus=1,
        num_cpus_per_worker=1,
        num_learner_workers=0,
    )
    .multi_agent(
        policies={
            ray_agent: PolicySpec(
                policy_class=None,
                observation_space=env.observation_space(cyborg_agent),
                action_space=env.action_space(cyborg_agent),
                config={"gamma": 0.85},
            )
            for cyborg_agent, ray_agent in POLICY_MAP.items()
        },
        policy_mapping_fn=policy_mapper,
    )
)

check_env(env)
print("=" * 80)
print("Initializing algorithm...")
algo = algo_config.build()
print("Algorithm initialization completed!")
print("=" * 80)


TRAIN_ITERATIONS = 1000
PRINT_INTERVAL = 10

print("\n" + "=" * 80)
print(f"Start training")
print(f"  Total iterations: {TRAIN_ITERATIONS}")
print(f"  Print interval: every {PRINT_INTERVAL} iterations")
print(f"  Episode length: 100 steps")
print(f"  GPU: 3")
print("=" * 80 + "\n")

start_time = time.time()
best_reward = float('-inf')

for i in range(TRAIN_ITERATIONS):
    train_info = algo.train()


    if i % PRINT_INTERVAL == 0 or i == TRAIN_ITERATIONS - 1:
        elapsed = time.time() - start_time
        reward_mean = train_info.get("episode_reward_mean", 0)
        reward_max = train_info.get("episode_reward_max", 0)
        reward_min = train_info.get("episode_reward_min", 0)
        episode_len_mean = train_info.get("episode_len_mean", 0)


        if reward_mean > best_reward:
            best_reward = reward_mean


        iterations_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
        eta_seconds = (TRAIN_ITERATIONS - i - 1) / iterations_per_sec if iterations_per_sec > 0 else 0


        print(f"[{i:4d}/{TRAIN_ITERATIONS}] "
              f"Reward: mean={reward_mean:8.2f} | max={reward_max:8.2f} | min={reward_min:8.2f} | "
              f"best={best_reward:8.2f}")
        print(f"         Episode length: {episode_len_mean:.1f} | "
              f"Elapsed: {elapsed/60:6.1f}min | "
              f"Speed: {iterations_per_sec:.2f} iter/s | "
              f"ETA: {eta_seconds/60:6.1f}min")
        print("-" * 80)
        sys.stdout.flush()

total_time = time.time() - start_time
print("\n" + "=" * 80)
print("Training completed!")
print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"  Average speed: {TRAIN_ITERATIONS/total_time:.2f} iterations/sec")
print(f"  Best mean reward: {best_reward:.2f}")
print("=" * 80 + "\n")


checkpoint_path = "checkpoint_cc4"
print(f"Saving model to: {checkpoint_path}")
algo.save(checkpoint_path)
print("Model saved successfully!\n")


print("Start evaluation...")
output = algo.evaluate()
print("\nEvaluation results:")
print(f"  Mean episode length: {output['evaluation']['episode_len_mean']:.1f}")
print(f"  Mean reward: {output['evaluation']['episode_reward_mean']:.2f}")
print(f"  Reward std: {output['evaluation']['episode_reward_std']:.2f}")
print("=" * 80)
