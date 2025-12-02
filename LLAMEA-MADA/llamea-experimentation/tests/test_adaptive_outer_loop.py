"""
Smoke tests for the adaptive outer loop (MADA 2.0 controller).

These tests run entirely offline using Dummy_LLM and a trivial evaluation
function so we can validate the control flow without network calls or heavy
dependencies.
"""

import random
import sys
from pathlib import Path

# Ensure the src package is importable when tests are executed from repo root.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamea import Dummy_LLM, LLaMEA
from llamea.utils import DiscountedThompsonSampling


def _trivial_eval(solution, logger=None):
    """Cheap evaluation function: fitness = (code length mod 17) + noise."""
    score = len(solution.code or "") % 17 + random.random()
    solution.set_scores(score, f"score={score:.3f}")
    return solution


def test_discounted_thompson_sampling_basic():
    """DiscountedThompsonSampling should select and update without crashing."""
    dts = DiscountedThompsonSampling(n_arms=2, gamma=0.9, tau_max=3.0)
    for _ in range(5):
        arm = dts.select_arm()
        assert arm in (0, 1)
        dts.update(arm, reward=random.random())
    snap = dts.snapshot()
    assert len(snap) == 2
    for entry in snap:
        assert "count" in entry and "mean" in entry


def test_adaptive_outer_loop_runs():
    """LLaMEA with adaptive_outer_loop=True should run to completion."""
    random.seed(42)
    llm = Dummy_LLM()
    optimizer = LLaMEA(
        f=_trivial_eval,
        llm=llm,
        n_parents=2,
        n_offspring=1,
        budget=6,
        log=False,
        adaptive_outer_loop=True,
        outer_loop_gamma=0.9,
        outer_loop_tau=3.0,
    )
    best = optimizer.run()
    # We only care that it finishes and returns a Solution
    assert best is not None
    assert hasattr(best, "fitness")
    print(f"Adaptive outer loop dry run fitness: {best.fitness}")


def test_adaptive_outer_loop_metadata_logged():
    """Each offspring should have outer_loop_event metadata."""
    random.seed(1)
    llm = Dummy_LLM()
    optimizer = LLaMEA(
        f=_trivial_eval,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        budget=4,
        log=False,
        adaptive_outer_loop=True,
    )
    optimizer.run()
    # Expect at least one offspring with outer_loop_event in its metadata
    events = [
        sol.get_metadata("outer_loop_event")
        for sol in optimizer.run_history
        if sol.get_metadata("outer_loop_event") is not None
    ]
    assert len(events) > 0, "No outer_loop_event metadata found in run_history"
    for ev in events:
        assert "selected_arm" in ev
        assert "reward" in ev


if __name__ == "__main__":
    test_discounted_thompson_sampling_basic()
    test_adaptive_outer_loop_runs()
    test_adaptive_outer_loop_metadata_logged()
    print("All adaptive outer loop tests passed.")


