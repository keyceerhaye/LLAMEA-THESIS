import numpy as np

from llamea.mada.ds_ts import DiscountedThompsonSampler


def test_sampler_prefers_higher_reward_arm():
    sampler = DiscountedThompsonSampler(
        ["alpha", "beta"],
        discount=1.0,
        prior_variance=1.0,
        reward_variance=0.25,
        tau_max=2.0,
    )

    for _ in range(10):
        sampler.update("mutation", "alpha", reward=1.0)
    for _ in range(10):
        sampler.update("mutation", "beta", reward=-1.0)

    arm, _, _ = sampler.select_arm("mutation")
    snapshot = sampler.get_state_snapshot("mutation")

    assert arm == "alpha"
    assert snapshot["alpha"]["mean"] > snapshot["beta"]["mean"]


def test_discount_reduces_effective_counts():
    sampler = DiscountedThompsonSampler(
        ["alpha", "beta"], discount=0.5, prior_variance=1.0, reward_variance=1.0
    )

    sampler.update("mutation", "alpha", reward=1.0)
    sampler.update("mutation", "alpha", reward=1.0)
    snapshot = sampler.get_state_snapshot("mutation")

    assert snapshot["alpha"]["count"] < 2.0


def test_reward_override_applies_to_updates():
    sampler = DiscountedThompsonSampler(
        ["alpha", "beta"], discount=1.0, prior_variance=1.0, reward_variance=0.1
    )
    sampler.update("mutation", "alpha", reward=5.0, reward_override=0.0)
    snapshot = sampler.get_state_snapshot("mutation")

    assert np.isclose(snapshot["alpha"]["mean"], 0.0, atol=1e-3)




