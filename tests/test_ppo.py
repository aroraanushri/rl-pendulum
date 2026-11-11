"""PPO math and checkpoint tests, runnable without Gymnasium."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ppo.agent import ActorCritic, PPOConfig, compute_gae, load_checkpoint, ppo_update, save_checkpoint


class TestPPO(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cpu")

    def test_action_shapes(self):
        model = ActorCritic(4, 2)
        actions, logp, entropy, values = model.get_action_and_value(torch.zeros(8, 4))
        for tensor in (actions, logp, entropy, values):
            self.assertEqual(tuple(tensor.shape), (8,))
        self.assertTrue(torch.all((actions >= 0) & (actions < 2)))
        self.assertIn(model.act(np.zeros(4, dtype=np.float32), self.device, True)[0], (0, 1))

    def test_gae_rollout_boundary(self):
        advantage, returns = compute_gae(
            np.array([1., 1.]), np.zeros(2), np.zeros(2, bool),
            np.zeros(2, bool), np.zeros(2), 2., 1., 1.,
        )
        np.testing.assert_allclose(advantage, [4., 3.])
        np.testing.assert_allclose(returns, [4., 3.])

    def test_true_termination_does_not_leak(self):
        advantage, _ = compute_gae(
            np.array([1., 2., 3.]), np.zeros(3), np.array([False, True, False]),
            np.zeros(3, bool), np.zeros(3), 4., 1., 1.,
        )
        np.testing.assert_allclose(advantage, [3., 2., 7.])

    def test_truncation_bootstraps_final_observation(self):
        advantage, _ = compute_gae(
            np.array([1., 2.]), np.array([0., 100.]), np.zeros(2, bool),
            np.array([True, False]), np.array([5., 0.]), 3., 1., 1.,
        )
        np.testing.assert_allclose(advantage, [6., -95.])

    def test_update_changes_weights_and_is_finite(self):
        model = ActorCritic(4, 2)
        observations = np.random.default_rng(4).normal(size=(32, 4)).astype(np.float32)
        with torch.no_grad():
            actions, logp, _, _ = model.get_action_and_value(torch.tensor(observations))
        before = [p.detach().clone() for p in model.parameters()]
        stats = ppo_update(
            model, torch.optim.Adam(model.parameters(), lr=1e-3),
            observations, actions.numpy(), logp.numpy(),
            np.linspace(-1, 1, 32, dtype=np.float32), np.ones(32, np.float32),
            PPOConfig(update_epochs=2, minibatch_size=8), self.device,
        )
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(np.isfinite(list(stats.values())).all())

    def test_checkpoint_roundtrip(self):
        model = ActorCritic(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, 4, 2, PPOConfig(), 128)
            loaded = load_checkpoint(path, self.device)
            obs = torch.randn(4, 4)
            torch.testing.assert_close(model.actor(obs), loaded.actor(obs))
            torch.testing.assert_close(model.critic(obs), loaded.critic(obs))


if __name__ == "__main__":
    unittest.main()
