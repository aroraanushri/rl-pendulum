"""Evaluate PPO on CartPole-v1. Run: python -m ppo.evaluate"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import load_checkpoint


def evaluate(checkpoint: Path, episodes: int = 10, render: bool = False, seed: int = 123):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = torch.device("cpu")
    model = load_checkpoint(checkpoint, device)
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    rewards = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done, reward_sum = False, 0.0
            while not done:
                action, _, _ = model.act(observation, device, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += reward
                done = terminated or truncated
            rewards.append(reward_sum)
            print(f"Episode {episode + 1}: {reward_sum:.0f}")
    finally:
        env.close()
    print(f"Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate CartPole PPO checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("ppo/cartpole_ppo.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render, args.seed)


if __name__ == "__main__":
    main()
