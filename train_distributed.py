import os
import sys
import time
import argparse
import json
import csv
import numpy as np
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
import warnings
from training.envs import create_cc4_env

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_cc4_env_from_config(env_config: dict):
    return create_cc4_env(env_config)


env_creator_CC4 = create_cc4_env_from_config


def train_algorithm(algorithm="PPO", gpu_ids="2,3", iterations=1000, steps=100,
                   gamma=0.95, lr=3e-4, output_dir="experiment/checkpoints"):


    if isinstance(gpu_ids, int):
        gpu_ids = str(gpu_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    gpu_list = [int(x.strip()) for x in str(gpu_ids).split(",")]
    num_gpus = len(gpu_list)


    use_single_gpu = (num_gpus == 1)

    print("=" * 80)
    print(f"Training config: {algorithm}")
    print(f"  GPU: {gpu_ids} ({num_gpus}GPUs, {'single-GPU mode' if use_single_gpu else 'multi-GPU mode'})")
    print(f"  Iterations: {iterations}")
    print(f"  Episode length: {steps}")
    print(f"  Discount factor: {gamma}")
    print(f"  Learning rate: {lr}")
    print("=" * 80)


    register_env(name="CC4", env_creator=lambda config: create_cc4_env_from_config(config))
    env = create_cc4_env_from_config({"steps": steps})


    NUM_AGENTS = 5
    POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

    def policy_mapper(agent_id, episode, worker, **kwargs):
        return POLICY_MAP[agent_id]


    if algorithm == "PPO":
        algo_config = (
            PPOConfig()
            .framework("torch")
            .environment(env="CC4")
            .resources(
                num_gpus=1 if use_single_gpu else num_gpus,
                num_cpus_per_worker=2,
                num_learner_workers=0,
            )
            .multi_agent(
                policies={
                    ray_agent: PolicySpec(
                        observation_space=env.observation_space(cyborg_agent),
                        action_space=env.action_space(cyborg_agent),
                        config={
                            "gamma": gamma,

                            "model": {
                                "fcnet_hiddens": [256, 256],
                                "fcnet_activation": "tanh",
                                "vf_share_layers": False,
                            }
                        },
                    )
                    for cyborg_agent, ray_agent in POLICY_MAP.items()
                },
                policy_mapping_fn=policy_mapper,
            )
            .training(
                lr=min(lr, 1e-4),
                train_batch_size=2000,
                sgd_minibatch_size=64,
                num_sgd_iter=5,

                grad_clip=0.5,

                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,


                use_gae=True,
                lambda_=0.95,
            )
        )
    else:
        algo_config = (
            DQNConfig()
            .framework("torch")
            .environment(env="CC4")
            .resources(
                num_gpus=1 if use_single_gpu else num_gpus,
                num_cpus_per_worker=2,
            )
            .multi_agent(
                policies={
                    ray_agent: PolicySpec(
                        observation_space=env.observation_space(cyborg_agent),
                        action_space=env.action_space(cyborg_agent),
                        config={
                            "gamma": gamma,

                            "model": {
                                "fcnet_hiddens": [256, 256],
                                "fcnet_activation": "tanh",
                            }
                        },
                    )
                    for cyborg_agent, ray_agent in POLICY_MAP.items()
                },
                policy_mapping_fn=policy_mapper,
            )
            .training(
                lr=min(lr, 1e-4),
                train_batch_size=32,

                grad_clip=10.0,

                target_network_update_freq=500,
                tau=1.0,
            )
            .exploration(
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 10000,
                }
            )
        )


    algo = algo_config.build()


    data_dir = Path(output_dir) / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / f"{algorithm.lower()}_training_data.csv"


    training_data = []


    print(f"\nStart training {algorithm}...")
    print(f"Training data will be saved to: {csv_file}")
    start_time = time.time()
    best_reward = float('-inf')


    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'episode_reward_mean', 'episode_reward_max',
            'episode_reward_min', 'episode_len_mean', 'policy_loss',
            'vf_loss', 'entropy', 'kl', 'elapsed_time', 'best_reward'
        ])

    for i in range(iterations):
        try:
            train_info = algo.train()
        except (ValueError, RuntimeError) as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                print(f"\nWarning: detected NaN values (iteration {i})")
                print("Trying to recover training...")

                if i > 10:
                    print("Training partially completed, saving current state...")
                    checkpoint_path = f"{output_dir}/{algorithm.lower()}_distributed_error"
                    algo.save(checkpoint_path)
                    print(f"Checkpoint saved to: {checkpoint_path}")
                raise RuntimeError(f"Training failed: {e}. Suggestions: 1)reduce learning rate to 5e-5 2)reduce batch size 3)use a single GPU")
            else:
                raise

        elapsed = time.time() - start_time


        reward_mean = train_info.get("episode_reward_mean", 0)
        if isinstance(reward_mean, float) and (np.isnan(reward_mean) or np.isinf(reward_mean)):
            print(f"\nWarning: detected NaN/Inf reward values (iteration {i})")
            print("Skip logging for this iteration")
            continue


        reward_mean = train_info.get("episode_reward_mean", 0)
        reward_max = train_info.get("episode_reward_max", 0)
        reward_min = train_info.get("episode_reward_min", 0)
        episode_len = train_info.get("episode_len_mean", 0)


        info = train_info.get("info", {})
        policy_loss = info.get("learner", {}).get("default_policy", {}).get("policy_loss", 0)
        vf_loss = info.get("learner", {}).get("default_policy", {}).get("vf_loss", 0)
        entropy = info.get("learner", {}).get("default_policy", {}).get("entropy", 0)
        kl = info.get("learner", {}).get("default_policy", {}).get("kl", 0)

        if reward_mean > best_reward:
            best_reward = reward_mean


        data_point = {
            'iteration': i,
            'episode_reward_mean': reward_mean,
            'episode_reward_max': reward_max,
            'episode_reward_min': reward_min,
            'episode_len_mean': episode_len,
            'policy_loss': policy_loss,
            'vf_loss': vf_loss,
            'entropy': entropy,
            'kl': kl,
            'elapsed_time': elapsed,
            'best_reward': best_reward
        }
        training_data.append(data_point)


        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, reward_mean, reward_max, reward_min, episode_len,
                policy_loss, vf_loss, entropy, kl, elapsed, best_reward
            ])

        if i % 50 == 0 or i == iterations - 1:
            print(f"[{i:4d}/{iterations}] {algorithm} | "
                  f"Reward: {reward_mean:8.2f} | Best: {best_reward:8.2f} | "
                  f"Time: {elapsed/60:.1f}min")
            sys.stdout.flush()


    json_file = data_dir / f"{algorithm.lower()}_training_data.json"
    with open(json_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Training data saved to: {json_file}")


    checkpoint_path = f"{output_dir}/{algorithm.lower()}_distributed"
    algo.save(checkpoint_path)
    print(f"\n{algorithm} Model saved to: {checkpoint_path}")


    print("\nStart evaluation...")
    output = algo.evaluate()
    final_reward = output['evaluation']['episode_reward_mean']
    final_std = output['evaluation']['episode_reward_std']

    total_time = time.time() - start_time
    print(f"\n{algorithm} Training completed!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Final mean reward: {final_reward:.2f} ± {final_std:.2f}")
    print(f"  Model saved to: {checkpoint_path}")
    print("=" * 80 + "\n")

    return final_reward, checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN"])
    parser.add_argument("--gpus", type=str, default="3",
                        help="GPU ID(s), use '3' for single GPU or '2,3' for multi-GPU")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", type=str, default="checkpoints")

    args = parser.parse_args()

    train_algorithm(
        algorithm=args.algorithm,
        gpu_ids=args.gpus,
        iterations=args.iterations,
        steps=args.steps,
        gamma=args.gamma,
        lr=args.lr,
        output_dir=args.output
    )
