# rl-pendulum

Two reinforcement learning approaches to **Gymnasium CartPole-v1**: the original tabular Q-Learning and a new neural-network PPO implementation. Despite the repository name, this is currently CartPole, **not Pendulum-v1**.

## Q-Learning (original, unchanged)

```bash
pip install "gymnasium[classic-control]" numpy
cd q-learn
python train_carpole.py
python test_carpole.py
```

The Q-learning scripts use a relative path for `q_table.npy`, so run them from inside `q-learn/`.

## PPO (new)

Requirements: Python 3.10+, PyTorch, NumPy, Gymnasium. From the **repository root**:

```bash
pip install -r requirements-ppo.txt
python -m ppo.train --total-timesteps 100000 --seed 42
python -m ppo.evaluate --episodes 10
# Optional: see the agent balance the pole with a desktop window
python -m ppo.evaluate --episodes 1 --render
```

The checkpoint is `ppo/cartpole_ppo.pt` and is excluded from version control. Smoke test (does not prove convergence):

```bash
python -m ppo.train --total-timesteps 2048 --rollout-steps 512
python -m ppo.evaluate --episodes 2
```

PPO uses: a categorical **actor** policy, a **critic** predicting state values, fresh on-policy rollouts, **GAE**, a **clipped policy objective**, critic value loss, an entropy bonus and gradient clipping. A terminated episode does not bootstrap; a time-limit truncated episode bootstraps from the last observation. See `ppo/agent.py` for the hyperparameters.

## Comparison

| | Q-Learning | PPO |
|---|---|---|
| Learns | Tabular Q-values | Neural actor policy and critic |
| Observation | Discretized 4D state | Raw 4D state |
| Action selection | Epsilon-greedy | Stochastic categorical policy |
| Update | Bellman target | Clipped policy gradient + value loss |
| Data | Can be reused off-policy | Fresh on-policy trajectories |
| Trained result | `q-learn/q_table.npy` | `ppo/cartpole_ppo.pt` |

Evaluate across several seeds and episodes before concluding either method is better. The default PPO training budget is a starting point, **not a guaranteed target score**.

## Tests and CI

```bash
python -m unittest discover -s tests -v
```

Six tests cover sampled actions, rollout bootstrapping, terminal boundaries, truncation bootstrapping, PPO gradient updates, and checkpoint reload. The GitHub Actions workflow runs tests plus a short end-to-end CartPole smoke test.

## Files

- `q-learn/`: existing Q-learning baseline and saved Q-table
- `ppo/agent.py`: PPO math and neural networks
- `ppo/train.py`: PPO training CLI
- `ppo/evaluate.py`: trained-policy evaluation CLI
- `tests/test_ppo.py`: unit tests
- `requirements-ppo.txt`: PPO dependencies

## Original CartPole demo

https://github.com/user-attachments/assets/38894248-697b-4ce5-826a-223beb7a003d
