
import os
import sys
import time
import argparse
import json
import csv
import numpy as np

import torch

from agents.contextual_bandit import TabularContextualUCB
from training.envs import create_enterprise_mae_env
from training.observations import cast_observations_to_float32
from training.paths import create_timestamped_run_dirs, resolve_training_data_dir


def train_contextual_bandit(
    iterations=100,
    steps=100,
    output_dir="experiment/checkpoints",
    c_param=2.0,
    gamma=0.95,
    gpu_id=None
):


    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(f"Contextual Bandit training - Tabular UCB")
    print(f"  Iterations: {iterations}")
    print(f"  Episode length: {steps}")
    print(f"  UCB parameter c: {c_param}")
    print(f"  Discount factor: {gamma}")
    print(f"  Device: {device}")
    print("=" * 80)


    print("\nCreating CAGE environment...")
    env = create_enterprise_mae_env(steps=steps)


    print("Initializing Contextual Bandit agents...")
    agents = {}
    for agent_id in env.agents:

        action_space = env.action_space(agent_id)
        num_actions = action_space.n if hasattr(action_space, 'n') else 41

        agents[agent_id] = TabularContextualUCB(
            agent_id=agent_id,
            num_actions=num_actions,
            c_param=c_param,
            gamma=gamma
        )


    output_path, checkpoint_dir = create_timestamped_run_dirs(output_dir, "contextual_bandit")
    unified_data_dir = resolve_training_data_dir(output_dir)


    training_data = {
        'iterations': [],
        'episode_rewards': [],
        'episode_lengths': [],
    }


    smoothed_reward = None
    ema_alpha = 0.1

    best_reward = float('-inf')
    start_time = time.time()

    print(f"\nStart Contextual Bandit training")
    print(f"   Iterations: {iterations}")
    print(f"   Steps per episode: {steps}")
    print(f"   Output directory: {output_path}\n")

    for iteration in range(iterations):
        episode_start = time.time()


        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0


        obs = cast_observations_to_float32(obs, env.agents)

        for step in range(steps):
            actions = {}


            for agent_id in env.agents:
                agent_obs = obs[agent_id]
                agent = agents[agent_id]


                valid_actions = None
                if hasattr(env, 'action_mask'):
                    try:
                        mask = env.action_mask(agent_id)
                        if mask is not None:
                            valid_actions = np.where(mask)[0].tolist()
                    except:
                        pass


                try:
                    action = agent.select_action(agent_obs, valid_actions)
                    actions[agent_id] = action
                except Exception as e:
                    print(f"Warning: agent {agent_id} failed to select action: {e}")
                    actions[agent_id] = 0


            try:
                obs_next, rewards, terminations, truncations, info = env.step(actions)
            except Exception as e:
                print(f"Warning: environment step failed: {e}")
                break


            step_reward = sum(rewards.values())
            episode_reward += step_reward
            episode_length += 1


            num_agents = len(env.agents) if len(env.agents) > 0 else len(actions)
            for agent_id, action in actions.items():
                agent_obs = obs[agent_id]
                agent_obs_next = obs_next[agent_id]
                agent_reward = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(agent_id, False)


                if num_agents > 0 and step_reward != 0:
                    shared_reward = step_reward / num_agents

                    total_reward = agent_reward * 0.95 + shared_reward * 0.05
                else:
                    total_reward = agent_reward


                total_reward = max(-800.0, min(800.0, total_reward))


                agents[agent_id].update(
                    observation=agent_obs,
                    action=action,
                    reward=total_reward,
                    next_observation=agent_obs_next,
                    done=done
                )

            obs = obs_next


            if any(terminations.values()) or any(truncations.values()):
                break


        if smoothed_reward is None:
            smoothed_reward = episode_reward
        else:

            ema_alpha_smooth = 0.2
            smoothed_reward = ema_alpha_smooth * episode_reward + (1 - ema_alpha_smooth) * smoothed_reward


        training_data['iterations'].append(iteration)
        training_data['episode_rewards'].append(episode_reward)
        training_data['episode_lengths'].append(episode_length)


        if episode_reward > best_reward:
            best_reward = episode_reward


        elapsed = time.time() - episode_start
        print(f"[{iteration+1:4d}/{iterations}] ContextualBandit | "
              f"Reward: {episode_reward:8.2f} | "
              f"Smoothed: {smoothed_reward:8.2f} | "
              f"Best: {best_reward:8.2f} | "
              f"Length: {episode_length:3d} | "
              f"Time: {elapsed:.1f}s")


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


            unified_csv = unified_data_dir / "contextual_bandit_training_data.csv"
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


            unified_csv = unified_data_dir / "contextual_bandit_training_data.csv"
            with open(unified_csv, 'w') as f:
                f.write('iteration,episode_reward,episode_length\n')
                for i, r, l in zip(
                    training_data['iterations'],
                    training_data['episode_rewards'],
                    training_data['episode_lengths']
                ):
                    f.write(f'{i},{r},{l}\n')


    total_time = time.time() - start_time


    csv_file = checkpoint_dir / "training_data.csv"
    try:
        import pandas as pd
        df = pd.DataFrame({
            'iteration': training_data['iterations'],
            'episode_reward': training_data['episode_rewards'],
            'episode_length': training_data['episode_lengths'],
        })
        df.to_csv(csv_file, index=False)


        unified_csv = unified_data_dir / "contextual_bandit_training_data.csv"
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

        unified_csv = unified_data_dir / "contextual_bandit_training_data.csv"
        with open(unified_csv, 'w') as f:
            f.write('iteration,episode_reward,episode_length\n')
            for i, r, l in zip(
                training_data['iterations'],
                training_data['episode_rewards'],
                training_data['episode_lengths']
            ):
                f.write(f'{i},{r},{l}\n')

    print(f"\nTraining completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Mean reward: {np.mean(training_data['episode_rewards']):.2f}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Training data saved to: {checkpoint_dir}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contextual Bandit agent")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="experiment/checkpoints", help="Output directory")
    parser.add_argument("--c-param", type=float, default=1.0, help="UCB exploration coefficient")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")

    args = parser.parse_args()

    train_contextual_bandit(
        iterations=args.iterations,
        steps=args.steps,
        output_dir=args.output_dir,
        c_param=args.c_param,
        gamma=args.gamma,
        gpu_id=args.gpu
    )
