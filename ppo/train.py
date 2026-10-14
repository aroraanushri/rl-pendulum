"""Train PPO on CartPole-v1. Run: python -m ppo.train"""
import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import ActorCritic, PPOConfig, compute_gae, ppo_update, save_checkpoint


def train(config: PPOConfig, seed: int, checkpoint: Path, device: torch.device):
    if min(config.total_timesteps, config.rollout_steps, config.minibatch_size, config.update_epochs) < 1:
        raise ValueError("Training counts must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    try:
        observation, _ = env.reset(seed=seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        model = ActorCritic(obs_dim, action_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        completed_rewards = deque(maxlen=20)
        episode_reward = 0.0
        steps = 0
        while steps < config.total_timesteps:
            observations, actions, log_probs, rewards = [], [], [], []
            values, terminations, truncations, timeout_values = [], [], [], []
            for _ in range(min(config.rollout_steps, config.total_timesteps - steps)):
                action, log_prob, value = model.act(observation, device)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                timeout_value = 0.0
                if truncated and not terminated:
                    with torch.no_grad():
                        final_obs = torch.as_tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
                        timeout_value = float(model.get_value(final_obs).item())
                observations.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                terminations.append(terminated)
                truncations.append(truncated)
                timeout_values.append(timeout_value)
                steps += 1
                episode_reward += reward
                if terminated or truncated:
                    completed_rewards.append(episode_reward)
                    episode_reward = 0.0
                    observation, _ = env.reset()
                else:
                    observation = next_observation

            with torch.no_grad():
                last_obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                last_value = float(model.get_value(last_obs).item())
            advantages, returns = compute_gae(
                np.asarray(rewards, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
                np.asarray(terminations, dtype=bool),
                np.asarray(truncations, dtype=bool),
                np.asarray(timeout_values, dtype=np.float32),
                last_value, config.gamma, config.gae_lambda,
            )
            stats = ppo_update(
                model, optimizer, np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.int64), np.asarray(log_probs, dtype=np.float32),
                advantages, returns, config, device,
            )
            mean_reward = float(np.mean(completed_rewards)) if completed_rewards else float("nan")
            print(f"steps={steps} | mean last-20 episode reward={mean_reward:.1f} | "
                  f"policy_loss={stats['policy_loss']:.3f} | value_loss={stats['value_loss']:.3f}")
        save_checkpoint(checkpoint, model, obs_dim, action_dim, config, steps)
        print(f"Checkpoint saved: {checkpoint}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="PPO training on CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("ppo/cartpole_ppo.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    train(PPOConfig(total_timesteps=args.total_timesteps, rollout_steps=args.rollout_steps),
          args.seed, args.checkpoint, torch.device(device_name))


if __name__ == "__main__":
    main()
