# rl-pendulum

Q-learning example for the `CartPole-v1` environment using Gymnasium. The project trains a small tabular agent, stores the learned Q-table in `q-learn/q_table.npy`, and then reuses that table for a simple test run.

## Project Structure

- `q-learn/train_carpole.py` - trains the agent and saves `q_table.npy`
- `q-learn/test_carpole.py` - loads the saved Q-table and runs one evaluation episode
- `q-learn/q_table.npy` - generated training output used by the test script

## Requirements

- Python 3.10 or newer
- `gymnasium[classic-control]`
- `numpy`

Install the dependencies with:

```bash
pip install "gymnasium[classic-control]" numpy
```

## Training

Run the training script from inside the `q-learn` folder:

```bash
cd q-learn
python train_carpole.py
```

The script runs for 5000 episodes by default, learns a Q-table with discretized state bins, and writes the result to `q_table.npy` in the same folder.

## Testing

After training, run the test script to watch the agent use the saved table:

```bash
cd q-learn
python test_carpole.py
```

The test script opens the environment in render mode, loads `q_table.npy`, and plays one episode using the learned policy.

## Notes

- If `q_table.npy` is missing, the training script creates a fresh table and the test script exits with `no qtable found`.
- You can tune learning behavior by editing the constants inside `train_carpole.py`, such as the number of episodes, epsilon decay, learning rate, and discount factor.
