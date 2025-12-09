"""
Unit tests for MADA 4.0 implementation.

Tests cover:
- BlockParser: Context bundle extraction, union-merge, placeholder detection
- DS-TS: Discount decay, tau_max clamping, arm selection
- MADAOperator: Recombination, innovation, context merging, lineage tracking
"""

import hashlib
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llamea.mada.parser import (
    BlockParser,
    ContextBundle,
    ParsedAlgorithm,
    detect_unused_helpers,
    union_merge_contexts,
)
from llamea.mada.ds_ts import DiscountedThompsonSampler, ArmState
from llamea.mada.operator import MADAOperator, OffspringProposal
from llamea.solution import Solution


# =============================================================================
# Sample Algorithm Code for Testing
# =============================================================================

SAMPLE_OPTIMIZER_CODE = '''
import numpy as np

class SampleOptimizer:
    """A sample optimizer for testing."""
    
    def __init__(self, budget=10000):
        self.budget = budget
        self.population = []
        self.archive = []
        self.best_fitness = np.inf
    
    def __call__(self, func):
        self.initialize_population(func)
        for _ in range(self.budget // 10):
            parents = self.parent_selection(self.population)
            offspring = self.recombination(parents)
            offspring = self.mutation(offspring)
            self.population = self.survivor_selection(self.population, offspring)
        return self.best_fitness
    
    def initialize_population(self, func):
        """Initialize random population."""
        self.population = [np.random.uniform(-5, 5, 5) for _ in range(10)]
    
    def update_archive(self, solution, fitness):
        """Update the archive with good solutions."""
        self.archive.append((solution, fitness))
    
    def parent_selection(self, population):
        """Select parents using tournament selection."""
        return population[:5]
    
    def recombination(self, parents):
        """Blend crossover recombination."""
        offspring = []
        for i in range(0, len(parents)-1, 2):
            child = 0.5 * parents[i] + 0.5 * parents[i+1]
            offspring.append(child)
        return offspring
    
    def mutation(self, candidate):
        """Gaussian mutation."""
        return candidate + np.random.normal(0, 0.1, len(candidate))
    
    def survivor_selection(self, population, offspring):
        """Truncation selection."""
        combined = population + offspring
        return combined[:len(population)]
'''

SAMPLE_OPTIMIZER_2_CODE = '''
import numpy as np

class SampleOptimizer2:
    """Another optimizer with different helpers."""
    
    def __init__(self, budget=10000):
        self.budget = budget
        self.sigma = 0.5
        self.history = []
    
    def calculate_velocity(self, solution):
        """Calculate velocity for PSO-like behavior."""
        return np.random.uniform(-1, 1, len(solution))
    
    def parent_selection(self, population):
        """Roulette wheel selection."""
        return population[:3]
    
    def recombination(self, parents):
        """Arithmetic crossover."""
        return [np.mean(parents, axis=0)]
    
    def mutation(self, candidate):
        """Adaptive mutation using sigma."""
        velocity = self.calculate_velocity(candidate)
        return candidate + self.sigma * velocity
    
    def survivor_selection(self, population, offspring):
        """Elitist selection."""
        return sorted(population + offspring, key=lambda x: np.sum(x**2))[:len(population)]
'''


# =============================================================================
# BlockParser Tests
# =============================================================================

class TestBlockParser:
    """Tests for BlockParser and Context Bundle extraction."""
    
    def test_extract_basic_blocks(self):
        """Parser should extract all 4 functional blocks."""
        parser = BlockParser()
        parsed = parser.extract(SAMPLE_OPTIMIZER_CODE)
        
        assert parsed.class_name == "SampleOptimizer"
        assert "parent_selection" in parsed.blocks
        assert "recombination" in parsed.blocks
        assert "mutation" in parsed.blocks
        assert "survivor_selection" in parsed.blocks
    
    def test_extract_context_bundle(self):
        """Parser should extract context bundle with helpers and init."""
        parser = BlockParser()
        parsed = parser.extract(SAMPLE_OPTIMIZER_CODE)
        context = parsed.get_context_bundle()
        
        assert isinstance(context, ContextBundle)
        assert "numpy" in str(context.imports) or "np" in str(context.imports)
        assert "update_archive" in context.other_methods
        assert "initialize_population" in context.other_methods
        assert "budget" in context.init_assignments
    
    def test_context_bundle_helper_names(self):
        """Context bundle should report all helper method names."""
        parser = BlockParser()
        parsed = parser.extract(SAMPLE_OPTIMIZER_CODE)
        context = parsed.get_context_bundle()
        
        helper_names = context.get_all_helper_names()
        assert "update_archive" in helper_names
        assert "initialize_population" in helper_names
    
    def test_context_bundle_to_prompt_string(self):
        """Context bundle should format correctly for LLM prompts."""
        parser = BlockParser()
        parsed = parser.extract(SAMPLE_OPTIMIZER_CODE)
        context = parsed.get_context_bundle()
        
        prompt_str = context.to_prompt_string()
        assert "# Imports:" in prompt_str or "helper methods" in prompt_str.lower()
    
    def test_hashes_are_unique(self):
        """Different blocks should have different hashes."""
        parser = BlockParser()
        parsed = parser.extract(SAMPLE_OPTIMIZER_CODE)
        
        hashes = list(parsed.hashes.values())
        assert len(hashes) == len(set(hashes)), "All block hashes should be unique"
    
    def test_placeholder_detection(self):
        """Parser should detect placeholder blocks when methods are missing."""
        minimal_code = '''
class MinimalOptimizer:
    def __init__(self):
        pass
    def __call__(self, func):
        return 0
'''
        parser = BlockParser()
        parsed = parser.extract(minimal_code)
        
        # Should have placeholder blocks
        assert parsed.coverage["placeholder_ratio"] > 0


class TestContextUnionMerge:
    """Tests for union-merging Context Bundles from multiple parents."""
    
    def test_union_merge_single_context(self):
        """Single context should pass through unchanged."""
        ctx = ContextBundle(
            imports=["import numpy as np"],
            init_assignments={"budget": "self.budget = 10000"},
            other_methods={"helper": "def helper(self): pass"},
        )
        
        merged, conflicts = union_merge_contexts([ctx])
        
        assert merged.imports == ctx.imports
        assert merged.init_assignments == ctx.init_assignments
        assert conflicts == {}
    
    def test_union_merge_dedup_imports(self):
        """Duplicate imports should be deduplicated."""
        ctx1 = ContextBundle(imports=["import numpy as np"])
        ctx2 = ContextBundle(imports=["import numpy as np", "import random"])
        
        merged, _ = union_merge_contexts([ctx1, ctx2])
        
        # Should have unique imports
        assert len(merged.imports) == 2
    
    def test_union_merge_combines_helpers(self):
        """Helpers from both parents should be combined."""
        ctx1 = ContextBundle(other_methods={"helper_a": "def helper_a(self): pass"})
        ctx2 = ContextBundle(other_methods={"helper_b": "def helper_b(self): pass"})
        
        merged, _ = union_merge_contexts([ctx1, ctx2])
        
        assert "helper_a" in merged.other_methods
        assert "helper_b" in merged.other_methods
    
    def test_union_merge_init_assignments(self):
        """Init assignments should be unioned, preferring first."""
        ctx1 = ContextBundle(init_assignments={"budget": "self.budget = 100"})
        ctx2 = ContextBundle(init_assignments={"budget": "self.budget = 200", "sigma": "self.sigma = 0.5"})
        
        merged, _ = union_merge_contexts([ctx1, ctx2])
        
        assert "budget" in merged.init_assignments
        assert "sigma" in merged.init_assignments
        # First context's budget should be preferred
        assert "100" in merged.init_assignments["budget"]
    
    def test_union_merge_conflict_detection(self):
        """Conflicting helpers with same name but different code should be detected."""
        ctx1 = ContextBundle(other_methods={"helper": "def helper(self): return 1"})
        ctx2 = ContextBundle(other_methods={"helper": "def helper(self): return 2"})
        
        # prefer_first=True should keep first version
        merged, conflicts = union_merge_contexts([ctx1, ctx2], prefer_first=True)
        
        # No conflict recorded when prefer_first=True (first version kept)
        assert len(merged.other_methods) == 1
        assert "return 1" in merged.other_methods["helper"]


class TestDetectUnusedHelpers:
    """Tests for detecting unused helper methods."""
    
    def test_detects_unused(self):
        """Should detect helpers that are never called."""
        code = '''
class Test:
    def used_helper(self):
        pass
    
    def unused_helper(self):
        pass
    
    def main(self):
        self.used_helper()
'''
        unused = detect_unused_helpers(code, {"used_helper", "unused_helper"})
        assert "unused_helper" in unused
        assert "used_helper" not in unused
    
    def test_all_used(self):
        """Should return empty set when all helpers are used."""
        code = '''
class Test:
    def helper_a(self):
        pass
    
    def helper_b(self):
        self.helper_a()
    
    def main(self):
        self.helper_a()
        self.helper_b()
'''
        unused = detect_unused_helpers(code, {"helper_a", "helper_b"})
        assert len(unused) == 0


# =============================================================================
# DS-TS Bandit Tests
# =============================================================================

class TestDiscountedThompsonSampler:
    """Tests for DS-TS bandit implementation."""
    
    def test_initialization(self):
        """Sampler should initialize with correct arms."""
        arms = ["alpha", "beta", "innovation"]
        sampler = DiscountedThompsonSampler(arms, discount=0.97, tau_max=3.0)
        
        # Should be able to select from any arm
        arm, theta, snapshot = sampler.select_arm("test_block")
        assert arm in arms
    
    def test_discount_decay(self):
        """Discounted counts should decay over time."""
        sampler = DiscountedThompsonSampler(
            ["alpha", "beta"], 
            discount=0.5,  # High decay for test
            tau_max=5.0,
        )
        
        # Update alpha arm
        sampler.update("block", "alpha", reward=1.0)
        state_before = sampler.get_state_snapshot("block")
        count_before = state_before["alpha"]["count"]
        
        # Another update should decay previous count
        sampler.update("block", "beta", reward=0.5)
        state_after = sampler.get_state_snapshot("block")
        
        # Alpha's count should have decayed
        assert state_after["alpha"]["count"] < count_before
    
    def test_tau_max_clamping(self):
        """Posterior variance should be clamped by tau_max."""
        sampler = DiscountedThompsonSampler(
            ["alpha"], 
            discount=0.97,
            tau_max=2.0,
        )
        
        # With no updates, variance should be at prior
        state = sampler.get_state_snapshot("block")
        # Variance should not exceed tau_max^2 = 4.0
        assert state["alpha"]["var"] <= 4.0
    
    def test_reward_updates_posterior(self):
        """Positive rewards should increase posterior mean."""
        sampler = DiscountedThompsonSampler(
            ["alpha"], 
            discount=0.99,
            tau_max=5.0,
        )
        
        # Get initial state
        state_before = sampler.get_state_snapshot("block")
        mean_before = state_before["alpha"]["mean"]
        
        # Update with positive reward
        sampler.update("block", "alpha", reward=1.0)
        sampler.update("block", "alpha", reward=1.0)
        sampler.update("block", "alpha", reward=1.0)
        
        state_after = sampler.get_state_snapshot("block")
        mean_after = state_after["alpha"]["mean"]
        
        # Mean should have increased
        assert mean_after > mean_before
    
    def test_arm_selection_probabilistic(self):
        """Arm selection should be probabilistic based on posterior."""
        sampler = DiscountedThompsonSampler(
            ["alpha", "beta", "innovation"],
            discount=0.97,
            tau_max=3.0,
        )
        
        # Train alpha to be much better
        for _ in range(10):
            sampler.update("block", "alpha", reward=1.0)
        for _ in range(10):
            sampler.update("block", "beta", reward=-0.5)
        
        # Sample many times
        selections = [sampler.select_arm("block")[0] for _ in range(100)]
        
        # Alpha should be selected more often
        alpha_count = selections.count("alpha")
        assert alpha_count > 30, "Alpha should be selected frequently after high rewards"


# =============================================================================
# MADAOperator Tests
# =============================================================================

class TestMADAOperator:
    """Tests for MADAOperator offspring generation."""
    
    @pytest.fixture
    def mock_algorithm_manager(self):
        """Create a mock AlgorithmManager."""
        manager = MagicMock()
        manager.extract_algorithm_code.return_value = "def mutation(self, x): return x"
        manager.refine_algorithm.return_value = "# Name: Test\n```python\nclass Test:\n    pass\n```"
        manager.extract_algorithm_name.return_value = "Test"
        manager.generate_block_snippet.return_value = "```python\ndef mutation(self, x): return x * 2\n```"
        return manager
    
    @pytest.fixture
    def sample_solutions(self):
        """Create sample parent solutions."""
        alpha = Solution(
            code=SAMPLE_OPTIMIZER_CODE,
            name="SampleOptimizer",
            generation=0,
        )
        alpha.fitness = 0.8
        alpha.aucs = [0.8] * 10
        
        beta = Solution(
            code=SAMPLE_OPTIMIZER_2_CODE,
            name="SampleOptimizer2",
            generation=0,
        )
        beta.fitness = 0.6
        beta.aucs = [0.6] * 10
        
        return alpha, beta
    
    def test_operator_initialization(self, mock_algorithm_manager):
        """Operator should initialize with correct bandits."""
        operator = MADAOperator(
            mock_algorithm_manager,
            discount=0.95,
            tau_max=4.0,
            reward_variance=0.3,
        )
        
        assert operator.discount == 0.95
        assert operator.tau_max == 4.0
        assert len(operator.bandits) == 4  # 4 functional blocks
    
    def test_strategy_weights_normalization(self, mock_algorithm_manager):
        """Strategy weights should be normalized."""
        operator = MADAOperator(
            mock_algorithm_manager,
            strategy_weights={"innovation": 2.0, "recombination": 2.0, "legacy": 1.0},
        )
        
        weights = operator.strategy_weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, "Weights should sum to 1.0"
    
    def test_ensure_blocks_caches(self, mock_algorithm_manager, sample_solutions):
        """ensure_blocks should cache parsed results."""
        operator = MADAOperator(mock_algorithm_manager)
        alpha, _ = sample_solutions
        
        parsed1 = operator.ensure_blocks(alpha)
        parsed2 = operator.ensure_blocks(alpha)
        
        assert parsed1 is parsed2, "Should return cached result"
    
    def test_context_bundle_extraction(self, mock_algorithm_manager, sample_solutions):
        """Should extract context bundles from solutions."""
        operator = MADAOperator(mock_algorithm_manager)
        alpha, _ = sample_solutions
        
        context = operator._get_context_bundle(alpha)
        
        assert isinstance(context, ContextBundle)
        assert len(context.other_methods) > 0
    
    def test_update_bandits(self, mock_algorithm_manager):
        """update_bandits should propagate rewards to correct arms."""
        operator = MADAOperator(mock_algorithm_manager)
        
        lineage = {
            "strategy": "innovation",
            "decisions": [
                {"block": "mutation", "arm": "alpha", "bandit": True},
                {"block": "recombination", "arm": "innovation", "bandit": True},
            ],
        }
        
        # Should not raise
        operator.update_bandits(lineage, reward=0.5)
        
        # Check that bandits were updated
        mutation_state = operator.bandits["mutation"].get_state_snapshot("mutation")
        assert mutation_state["alpha"]["count"] > 0
    
    def test_lineage_tracking(self, mock_algorithm_manager, sample_solutions):
        """Offspring should include complete lineage information."""
        operator = MADAOperator(
            mock_algorithm_manager,
            strategy_weights={"innovation": 0.0, "recombination": 1.0, "legacy": 0.0},
        )
        alpha, beta = sample_solutions
        
        proposal = operator.generate_offspring(
            parents=[alpha, beta],
            focal_parent=alpha,
            population_summary="Test population",
        )
        
        assert proposal.lineage is not None
        assert "strategy" in proposal.lineage
        assert "decisions" in proposal.lineage
        assert "parents" in proposal.lineage


# =============================================================================
# Integration Tests
# =============================================================================

class TestMADA40Integration:
    """Integration tests for complete MADA 4.0 workflow."""
    
    def test_full_recombination_workflow(self):
        """Test complete recombination with context merging."""
        # Create mock manager
        manager = MagicMock()
        manager.extract_algorithm_code.return_value = "def test(): pass"
        
        # Create operator
        operator = MADAOperator(
            manager,
            strategy_weights={"innovation": 0.0, "recombination": 1.0, "legacy": 0.0},
        )
        
        # Create solutions
        alpha = Solution(code=SAMPLE_OPTIMIZER_CODE, name="Alpha", generation=0)
        alpha.fitness = 0.8
        alpha.aucs = [0.8]
        
        beta = Solution(code=SAMPLE_OPTIMIZER_2_CODE, name="Beta", generation=0)
        beta.fitness = 0.6
        beta.aucs = [0.6]
        
        # Generate offspring
        proposal = operator.generate_offspring(
            parents=[alpha, beta],
            focal_parent=alpha,
            population_summary="Test",
        )
        
        # Verify offspring
        assert proposal.strategy == "recombination"
        assert len(proposal.lineage["decisions"]) == 4  # One per block
        assert proposal.code is not None
        assert len(proposal.code) > 0
    
    def test_bandit_adaptation_over_time(self):
        """Test that bandits adapt based on rewards over time."""
        manager = MagicMock()
        operator = MADAOperator(manager, discount=0.9)
        
        # Simulate evolution with rewards
        for i in range(20):
            # Simulate innovation being rewarded
            lineage = {
                "decisions": [
                    {"block": "mutation", "arm": "innovation", "bandit": True},
                ]
            }
            operator.update_bandits(lineage, reward=1.0 if i % 2 == 0 else -0.5)
        
        # Get bandit state
        state = operator.bandits["mutation"].get_state_snapshot("mutation")
        
        # Innovation arm should have been updated
        assert state["innovation"]["count"] > 0


class TestSnippetValidation:
    """Tests for the new snippet attribute validation functionality (MADA 4.0 fix)."""
    
    def test_validate_snippet_with_valid_attributes(self):
        """Test that snippets using only allowed attributes pass validation."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def mutation(self, candidate):
    mutated = candidate + self.F * (self.archive[0] - candidate)
    mutated = np.clip(mutated, self.lb, self.ub)
    return mutated
'''
        allowed_attrs = {"F", "archive", "lb", "ub", "dim", "pop_size"}
        allowed_methods = {"update_archive", "__call__"}
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert is_valid
        assert len(invalid_refs) == 0
    
    def test_validate_snippet_with_forbidden_domain(self):
        """Test that snippets using 'domain' instead of 'lb/ub' are rejected."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def mutation(self, candidate):
    mutated = np.clip(candidate, self.domain[0], self.domain[1])
    return mutated
'''
        allowed_attrs = {"F", "lb", "ub", "dim"}
        allowed_methods = set()
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert not is_valid
        assert "domain" in invalid_refs
        assert "domain" in suggestions
        assert "lb" in suggestions["domain"] or "ub" in suggestions["domain"]
    
    def test_validate_snippet_with_forbidden_bounds(self):
        """Test that snippets using 'bounds' instead of 'lb/ub' are rejected."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def recombination(self, parents):
    child = parents[0] + self.F * (parents[1] - parents[2])
    child = np.clip(child, self.bounds[:, 0], self.bounds[:, 1])
    return child
'''
        allowed_attrs = {"F", "CR", "lb", "ub", "archive"}
        allowed_methods = set()
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert not is_valid
        assert "bounds" in invalid_refs
    
    def test_validate_snippet_with_forbidden_population(self):
        """Test that snippets using 'population' instead of 'pop' are rejected."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def mutation(self, candidate):
    idx = np.random.choice(len(self.population), 3, replace=False)
    return self.population[idx[0]]
'''
        allowed_attrs = {"pop", "archive", "F"}
        allowed_methods = set()
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert not is_valid
        assert "population" in invalid_refs
        assert "population" in suggestions
        assert "pop" in suggestions["population"]
    
    def test_get_correction_hint(self):
        """Test that correction hints are generated properly."""
        from llamea.mada.parser import get_attribute_correction_hint
        
        invalid_refs = {"domain", "population", "unknown_attr"}
        suggestions = {
            "domain": ["lb", "ub"],
            "population": ["pop"],
        }
        
        hint = get_attribute_correction_hint(invalid_refs, suggestions)
        
        assert "self.domain" in hint
        assert "self.lb" in hint or "self.ub" in hint
        assert "self.population" in hint
        assert "self.pop" in hint
        assert "unknown_attr" in hint
    
    def test_validate_snippet_with_syntax_error(self):
        """Test that snippets with syntax errors are rejected gracefully."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def mutation(self, candidate)
    return candidate  # Missing colon
'''
        allowed_attrs = {"F"}
        allowed_methods = set()
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert not is_valid
        assert "__syntax_error__" in invalid_refs
    
    def test_validate_snippet_allows_method_calls(self):
        """Test that calling allowed helper methods passes validation."""
        from llamea.mada.parser import validate_snippet_attributes
        
        snippet = '''
def survivor_selection(self, population, offspring):
    for ind in offspring:
        self.update_archive(ind, 0.5)
    return population
'''
        allowed_attrs = {"archive", "archive_fitness"}
        allowed_methods = {"update_archive", "__call__"}
        
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            snippet, allowed_attrs, allowed_methods
        )
        
        assert is_valid
        assert len(invalid_refs) == 0


class TestContextBundleEnhancements:
    """Tests for enhanced ContextBundle prompt generation."""
    
    def test_context_bundle_includes_bounds_hint(self):
        """Test that context prompt includes hint about using lb/ub."""
        context = ContextBundle(
            imports=["import numpy as np"],
            init_assignments={"lb": "self.lb = -5.0", "ub": "self.ub = 5.0", "dim": "self.dim = 5"}
        )
        
        prompt_str = context.to_prompt_string()
        
        assert "self.lb" in prompt_str
        assert "self.ub" in prompt_str
        assert "NOT self.domain" in prompt_str or "lb and self.ub for bounds" in prompt_str
    
    def test_context_bundle_includes_pop_hint(self):
        """Test that context prompt includes hint about using pop."""
        context = ContextBundle(
            imports=["import numpy as np"],
            init_assignments={"pop": "self.pop = []", "fitness": "self.fitness = []"}
        )
        
        prompt_str = context.to_prompt_string()
        
        assert "self.pop" in prompt_str
        assert "NOT self.population" in prompt_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

