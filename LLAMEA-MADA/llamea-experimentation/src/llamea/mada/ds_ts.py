"""Discounted Thompson Sampling with Gaussian priors."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ArmState:
    """Holds the sufficient statistics for a single arm."""

    discounted_count: float = 0.0
    discounted_sum: float = 0.0
    discounted_sum_sq: float = 0.0
    posterior_mean: float = 0.0
    posterior_var: float = 1.0
    last_theta: float = 0.0


@dataclass
class BlockState:
    """Tracks the arms associated with a specific block identifier."""

    arms: Dict[str, ArmState] = field(default_factory=dict)
    counter: int = 0


class DiscountedThompsonSampler:
    """Implements DS-TS with Gaussian priors as described in the MADA plan."""

    def __init__(
        self,
        arm_names: List[str],
        discount: float = 0.97,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        reward_variance: float = 1.0,
        tau_max: float = 5.0,
        epsilon: float = 1e-6,
    ):
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        self.arm_names = arm_names
        self.discount = discount
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.reward_variance = reward_variance
        self.tau_max = tau_max
        self.epsilon = epsilon
        self._blocks: Dict[str, BlockState] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def select_arm(self, block_id: str) -> Tuple[str, float, str]:
        """Sample an arm according to the discounted posterior for ``block_id``."""

        block_state = self._get_block_state(block_id)
        best_arm = self.arm_names[0]
        best_theta = float("-inf")
        snapshot_id = f"{block_id}:{block_state.counter}"

        for arm_name, arm_state in block_state.arms.items():
            self._update_posterior(arm_state)
            std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
            std_dev = min(std_dev, self.tau_max)
            theta = random.gauss(arm_state.posterior_mean, std_dev)
            arm_state.last_theta = theta
            if theta > best_theta:
                best_theta = theta
                best_arm = arm_name

        return best_arm, best_theta, snapshot_id

    def update(
        self,
        block_id: str,
        arm_name: str,
        reward: float,
        reward_override: float | None = None,
    ) -> None:
        """Update the posterior for ``arm_name`` using the provided reward."""

        block_state = self._get_block_state(block_id)
        self._apply_discount(block_state)
        arm_state = block_state.arms[arm_name]
        value = reward if reward_override is None else reward_override
        arm_state.discounted_count += 1.0
        arm_state.discounted_sum += value
        arm_state.discounted_sum_sq += value**2
        self._update_posterior(arm_state)
        block_state.counter += 1

    def get_state_snapshot(self, block_id: str) -> Dict[str, Dict[str, float]]:
        """Return a serialisable view of the current posterior statistics."""

        block_state = self._get_block_state(block_id)
        snapshot: Dict[str, Dict[str, float]] = {}
        for arm_name, arm_state in block_state.arms.items():
            snapshot[arm_name] = {
                "count": arm_state.discounted_count,
                "mean": arm_state.posterior_mean,
                "var": arm_state.posterior_var,
                "theta": arm_state.last_theta,
            }
        return snapshot

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_block_state(self, block_id: str) -> BlockState:
        if block_id not in self._blocks:
            arms = {name: ArmState() for name in self.arm_names}
            self._blocks[block_id] = BlockState(arms=arms)
        return self._blocks[block_id]

    def _apply_discount(self, block_state: BlockState) -> None:
        if self.discount == 1.0:
            return
        for arm_state in block_state.arms.values():
            arm_state.discounted_count *= self.discount
            arm_state.discounted_sum *= self.discount
            arm_state.discounted_sum_sq *= self.discount

    def _update_posterior(self, arm_state: ArmState) -> None:
        count = max(self.epsilon, arm_state.discounted_count)
        sample_mean = arm_state.discounted_sum / count
        if arm_state.discounted_count <= self.epsilon:
            observed_var = self.reward_variance
        else:
            mean_sq = sample_mean**2
            moment = arm_state.discounted_sum_sq / count
            observed_var = max(
                self.epsilon, moment - mean_sq if moment > mean_sq else self.reward_variance
            )

        inv_prior = 1.0 / self.prior_variance
        inv_likelihood = count / max(self.epsilon, observed_var)
        posterior_var = 1.0 / max(self.epsilon, inv_prior + inv_likelihood)
        posterior_var = min(self.tau_max**2, posterior_var)
        posterior_mean = posterior_var * (
            self.prior_mean * inv_prior + sample_mean * inv_likelihood
        )

        arm_state.posterior_mean = posterior_mean
        arm_state.posterior_var = posterior_var





