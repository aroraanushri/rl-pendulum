"""Discrete-action PPO: actor-critic, GAE, and clipped policy updates.

The core math has no Gymnasium dependency so it can be tested in isolation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 1024
    total_timesteps: int = 100_000


class ActorCritic(nn.Module):
    """Separate actor (action logits) and critic (state value) for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        distribution = Categorical(logits=self.actor(obs))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(obs).squeeze(-1)

    def get_value(self, obs: torch.Tensor):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: np.ndarray, device: torch.device, deterministic: bool = False):
        obs = torch.as_tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            action = self.actor(obs).argmax(dim=-1)
        else:
            action = Categorical(logits=self.actor(obs)).sample()
        _, log_prob, _, value = self.get_action_and_value(obs, action)
        return int(action.item()), float(log_prob.item()), float(value.item())


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    timeout_values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE handles true termination, time limits, and incomplete rollout ends.

    Terminated transitions do not bootstrap. Truncated ones bootstrap using
    the final observation's value, never the reset observation's value.
    Either episode boundary stops advantage recursion into the next episode.
    """
    n = len(rewards)
    if not all(len(x) == n for x in (values, terminated, truncated, timeout_values)):
        raise ValueError("All rollout arrays must have the same length")
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        done = bool(terminated[t] or truncated[t])
        if terminated[t]:
            next_value = 0.0
        elif truncated[t]:
            next_value = float(timeout_values[t])
        else:
            next_value = float(values[t + 1]) if t + 1 < n else float(last_value)
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        last_gae = delta + gamma * gae_lambda * (not done) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values.astype(np.float32)


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Perform PPO gradient updates only on the current on-policy rollout."""
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    acts = torch.as_tensor(actions, dtype=torch.long, device=device)
    log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    target_values = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if len(obs) == 0:
        raise ValueError("Cannot update from an empty rollout")
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    stats: dict[str, float] = {}
    for _ in range(config.update_epochs):
        for indices in torch.randperm(len(obs), device=device).split(config.minibatch_size):
            _, new_log_prob, entropy, predicted_values = model.get_action_and_value(obs[indices], acts[indices])
            ratio = (new_log_prob - log_probs[indices]).exp()
            policy_loss = -torch.min(
                ratio * adv[indices],
                ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef) * adv[indices],
            ).mean()
            value_loss = 0.5 * ((predicted_values - target_values[indices]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "approx_kl": float(((ratio - 1) - (new_log_prob - log_probs[indices])).mean().item()),
                }
    return stats


def save_checkpoint(path: Path, model: ActorCritic, obs_dim: int, action_dim: int, config: PPOConfig, steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_id": "CartPole-v1",
        "config": asdict(config),
        "steps": steps,
    }, path)


def load_checkpoint(path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["env_id"] != "CartPole-v1":
        raise ValueError("Checkpoint was not trained for CartPole-v1")
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["action_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
