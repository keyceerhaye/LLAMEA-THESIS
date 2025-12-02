"""Main orchestration logic for the MADA operator."""

from __future__ import annotations

import ast
import hashlib
import random
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..solution import Solution
from .ds_ts import DiscountedThompsonSampler
from .parser import BlockParser, ParsedAlgorithm


@dataclass
class OffspringProposal:
    """Lightweight container describing a generated offspring."""

    code: str
    class_name: str
    description: str
    lineage: Dict[str, object]
    strategy: str
    parent_ids: List[str]
    diagnostics: Dict[str, object] = field(default_factory=dict)


class MADAOperator:
    """Generates offspring using DS-TS guided recomposition of code blocks."""

    def __init__(
        self,
        algorithm_manager,
        parser: Optional[BlockParser] = None,
        discount: float = 0.97,
        tau_max: float = 3.0,
        rng_seed: Optional[int] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
    ):
        self.algorithm_manager = algorithm_manager
        self.parser = parser or BlockParser()
        self.random = random.Random(rng_seed)
        arms = ["alpha", "beta", "innovation"]
        self.bandits = {
            block: DiscountedThompsonSampler(
                arms,
                discount=discount,
                tau_max=tau_max,
                reward_variance=0.25,
            )
            for block in self.parser.target_blocks
        }
        self.block_cache: Dict[str, ParsedAlgorithm] = {}
        self.strategy_weights = self._normalize_weights(strategy_weights)
        self.child_counter = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_offspring(
        self,
        parents: Sequence[Solution],
        focal_parent: Optional[Solution] = None,
        population_summary: str = "",
    ) -> OffspringProposal:
        """Return code + lineage for a new offspring according to the plan."""

        if not parents:
            raise ValueError("MADAOperator requires at least one parent.")

        ordered = sorted(parents, key=lambda sol: sol.fitness, reverse=True)
        alpha = ordered[0]
        beta = ordered[1] if len(ordered) > 1 else ordered[0]
        focal = focal_parent or self.random.choice(ordered)
        strategy = self._sample_strategy()

        try:
            if strategy == "legacy":
                return self._legacy_offspring(focal, population_summary)
            if strategy == "recombination":
                return self._recombination_offspring(alpha, beta)
            return self._innovative_offspring(alpha, beta, population_summary)
        except Exception:
            # Fallback to legacy mutation if anything goes wrong.
            return self._legacy_offspring(focal, population_summary)

    def update_bandits(self, lineage: Optional[Dict[str, object]], reward: float) -> None:
        """Propagate the observed reward back into the DS-TS bandits."""

        if not lineage:
            return
        for decision in lineage.get("decisions", []):
            if not decision.get("bandit", False):
                continue
            block = decision["block"]
            arm = decision["arm"]
            sampler = self.bandits.get(block)
            if sampler is None:
                continue
            sampler.update(
                block_id=block,
                arm_name=arm,
                reward=reward,
                reward_override=decision.get("reward_override"),
            )

    def ensure_blocks(self, solution: Solution) -> ParsedAlgorithm:
        """Parse and cache block metadata for ``solution``."""

        if solution.id in self.block_cache:
            return self.block_cache[solution.id]
        parsed = self.parser.extract(solution.code or "")
        self.block_cache[solution.id] = parsed
        if hasattr(solution, "set_mada_blocks"):
            solution.set_mada_blocks(parsed.to_metadata())
        else:
            solution.metadata["mada_blocks"] = parsed.to_metadata()
        return parsed

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _legacy_offspring(
        self, parent: Solution, population_summary: str
    ) -> OffspringProposal:
        self.algorithm_manager.tried_algorithms = f"Current population:\n{population_summary}"
        self.algorithm_manager.last_algorithm = (
            f"# Name: {parent.description or parent.name}\n# Code:\n```python\n{parent.code}\n```"
        )
        message = self.algorithm_manager.refine_algorithm(
            parent.fitness,
            float(np.std(parent.aucs)) if getattr(parent, "aucs", None) else 0.0,
            parent.description or parent.name,
            getattr(parent, "detailed_aucs", [0, 0, 0, 0, 0]),
        )
        code = self.algorithm_manager.extract_algorithm_code(message)
        class_name = self._extract_class_name(code, default=f"{parent.name}Legacy")
        lineage = {
            "strategy": "legacy",
            "decisions": [],
            "parents": [parent.id],
        }
        description = self.algorithm_manager.extract_algorithm_name(message) or class_name
        return OffspringProposal(
            code=code,
            class_name=class_name,
            description=description,
            lineage=lineage,
            strategy="legacy",
            parent_ids=[parent.id],
        )

    def _recombination_offspring(
        self, alpha: Solution, beta: Solution
    ) -> OffspringProposal:
        parsed_alpha = self.ensure_blocks(alpha)
        parsed_beta = self.ensure_blocks(beta)
        overrides: Dict[str, str] = {}
        decisions: List[Dict[str, object]] = []
        parent_ids = set()

        for block in self.parser.target_blocks:
            donor_label, parsed, donor = self.random.choice(
                [("alpha", parsed_alpha, alpha), ("beta", parsed_beta, beta)]
            )
            overrides[block] = parsed.blocks[block]
            decisions.append(
                {
                    "block": block,
                    "arm": f"recomb_{donor_label}",
                    "parent_id": donor.id,
                    "source_hash": parsed.hashes[block],
                    "bandit": False,
                }
            )
            parent_ids.add(donor.id)

        child_name = self._child_name(parsed_alpha.class_name, suffix="Recomb")
        code = self.parser.assemble(parsed_alpha, overrides=overrides, class_name=child_name)
        lineage = {
            "strategy": "recombination",
            "decisions": decisions,
            "parents": list(parent_ids),
        }
        description = f"Recombined variant of {alpha.name}/{beta.name}"
        return OffspringProposal(
            code=code,
            class_name=child_name,
            description=description,
            lineage=lineage,
            strategy="recombination",
            parent_ids=list(parent_ids),
        )

    def _innovative_offspring(
        self, alpha: Solution, beta: Solution, population_summary: str
    ) -> OffspringProposal:
        parsed_alpha = self.ensure_blocks(alpha)
        parsed_beta = self.ensure_blocks(beta)
        overrides: Dict[str, str] = {}
        decisions: List[Dict[str, object]] = []
        parent_ids = {alpha.id, beta.id}

        for block in self.parser.target_blocks:
            sampler = self.bandits[block]
            arm, theta, snapshot = sampler.select_arm(block)
            snippet: Optional[str] = None
            reward_override = None
            parent_id = None
            source_hash = ""

            if arm == "alpha":
                snippet = parsed_alpha.blocks[block]
                parent_id = alpha.id
                source_hash = parsed_alpha.hashes[block]
            elif arm == "beta":
                snippet = parsed_beta.blocks[block]
                parent_id = beta.id
                source_hash = parsed_beta.hashes[block]
            else:
                snippet, meta = self._request_innovation_block(
                    block, parsed_alpha, population_summary
                )
                if snippet is None:
                    snippet = parsed_alpha.blocks[block]
                    reward_override = 0.0
                    parent_id = alpha.id
                    source_hash = parsed_alpha.hashes[block]
                else:
                    parent_id = meta.get("parent_id")
                    source_hash = meta.get("hash", "")

            overrides[block] = snippet
            decisions.append(
                {
                    "block": block,
                    "arm": arm,
                    "theta": theta,
                    "snapshot": snapshot,
                    "parent_id": parent_id,
                    "source_hash": source_hash,
                    "bandit": True,
                    "reward_override": reward_override,
                }
            )

        child_name = self._child_name(parsed_alpha.class_name, suffix="MADA")
        code = self.parser.assemble(parsed_alpha, overrides=overrides, class_name=child_name)
        lineage = {
            "strategy": "innovation",
            "decisions": decisions,
            "parents": list(parent_ids),
            "population_context": population_summary,
        }
        description = f"MADA innovation guided by bandits (α={alpha.name}, β={beta.name})"
        return OffspringProposal(
            code=code,
            class_name=child_name,
            description=description,
            lineage=lineage,
            strategy="innovation",
            parent_ids=list(parent_ids),
        )

    # ------------------------------------------------------------------ #
    # Innovation helpers
    # ------------------------------------------------------------------ #
    def _request_innovation_block(
        self,
        block: str,
        template: ParsedAlgorithm,
        population_summary: str,
    ) -> tuple[Optional[str], Dict[str, object]]:
        """Ask the AlgorithmManager for a fresh block implementation."""

        baseline = template.blocks[block]
        signature = baseline.splitlines()[0] if baseline else f"def {block}(self, *args, **kwargs):"
        prompt = textwrap.dedent(
            f"""
            The current population summary is:\n{population_summary}\n
            Improve the `{block}` method of the optimizer class `{template.class_name}`.
            Use the exact signature `{signature}` and return only the method definition
            inside a Python code block. Keep helper references consistent with the class.

            Existing implementation for reference:
            ```python
            {baseline}
            ```
            """
        ).strip()

        try:
            snippet_message = self.algorithm_manager.generate_block_snippet(
                block_name=block,
                prompt=prompt,
            )
            snippet_code = self.algorithm_manager.extract_algorithm_code(snippet_message)
        except Exception:
            return None, {}

        cleaned = textwrap.dedent(snippet_code).strip()
        if not self._valid_block(cleaned, block):
            return None, {}

        block_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return cleaned, {"hash": block_hash, "parent_id": "innovation"}

    def _valid_block(self, snippet: str, block: str) -> bool:
        if not snippet.startswith("def "):
            return False
        if not snippet.split("(")[0].endswith(block):
            return False
        try:
            ast.parse(textwrap.dedent(snippet))
            return True
        except SyntaxError:
            return False

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _sample_strategy(self) -> str:
        roll = self.random.random()
        cumulative = 0.0
        for strategy, weight in self.strategy_weights.items():
            cumulative += weight
            if roll <= cumulative:
                return strategy
        return "innovation"

    def _child_name(self, base: str, suffix: str) -> str:
        self.child_counter += 1
        return f"{base}{suffix}{self.child_counter}"

    def _extract_class_name(self, code: str, default: str) -> str:
        for line in code.splitlines():
            if line.startswith("class "):
                return line.split()[1].split("(")[0].strip().strip(":")
        return default

    def _normalize_weights(
        self, weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        default = {
            "innovation": 0.4,
            "recombination": 0.4,
            "legacy": 0.2,
        }
        if not weights:
            return default
        sanitized = {
            key: max(0.0, weights.get(key, default[key])) for key in default
        }
        total = sum(sanitized.values())
        if total <= 0:
            return default
        return {key: value / total for key, value in sanitized.items()}

