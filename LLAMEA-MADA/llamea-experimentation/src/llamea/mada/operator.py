"""Main orchestration logic for the MADA operator."""

from __future__ import annotations

import ast
import hashlib
import random
import textwrap
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..solution import Solution
from .ds_ts import DiscountedThompsonSampler
from .parser import (
    BlockParser,
    ContextBundle,
    ParsedAlgorithm,
    detect_unused_helpers,
    union_merge_contexts,
    validate_snippet_attributes,
    get_attribute_correction_hint,
    expand_attributes_with_aliases,
)

PROMPT_GUARDRAILS = [
    "- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.",
    "- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.",
    "- Do not introduce orphaned dependencies; use helpers from the Context Block.",
    "- Preserve class structure: only modify the requested block.",
]


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
    """
    Generates offspring using DS-TS guided recomposition of code blocks.
    
    MADA 4.0 Features:
    - Hierarchical evolutionary cycle with three strategies (innovation/recombination/legacy)
    - DS-TS bandits per block with configurable discount and tau_max
    - 4+1 Architecture: Union-merge of Context Blocks from donors
    - Optional semantic linter for conflict resolution
    """

    def __init__(
        self,
        algorithm_manager,
        parser: Optional[BlockParser] = None,
        discount: float = 0.97,
        tau_max: float = 3.0,
        reward_variance: float = 0.25,
        rng_seed: Optional[int] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        guardrail_window: int = 10,
        enable_semantic_linter: bool = False,
        placeholder_threshold: float = 0.5,
    ):
        self.algorithm_manager = algorithm_manager
        self.parser = parser or BlockParser()
        self.random = random.Random(rng_seed)
        self.discount = discount
        self.tau_max = tau_max
        self.reward_variance = reward_variance
        
        arms = ["alpha", "beta", "innovation"]
        self.bandits = {
            block: DiscountedThompsonSampler(
                arms,
                discount=discount,
                tau_max=tau_max,
                reward_variance=reward_variance,
            )
            for block in self.parser.target_blocks
        }
        self.block_cache: Dict[str, ParsedAlgorithm] = {}
        self.strategy_weights = self._normalize_weights(strategy_weights)
        self.child_counter = 0
        self.placeholder_threshold = placeholder_threshold
        self.guardrail_window = guardrail_window
        self._violation_history: Deque[str] = deque(maxlen=guardrail_window)
        self.enable_semantic_linter = enable_semantic_linter
        self._context_cache: Dict[str, ContextBundle] = {}
        self.innovation_cache: Dict[str, ParsedAlgorithm] = {}

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
        beta = self._pick_diverse_parent(ordered, alpha)
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

    def update_bandits(
        self,
        lineage: Optional[Dict[str, object]],
        reward: float,
        reward_detail: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Propagate the observed reward back into the DS-TS bandits.

        reward_detail can carry:
        - per_block_rewards: dict[block_id] -> float (e.g., raw delta per block)
        - constraint_violation: bool flag to penalize all bandit arms
        - constraint_penalty: optional override value when constraint_violation is True
        """

        if not lineage:
            return

        per_block_rewards = {}
        constraint_penalty = None
        if reward_detail:
            per_block_rewards = reward_detail.get("per_block_rewards", {}) or {}
            if reward_detail.get("constraint_violation"):
                constraint_penalty = reward_detail.get("constraint_penalty", -abs(reward))

        for decision in lineage.get("decisions", []):
            if not decision.get("bandit", False):
                continue
            block = decision["block"]
            arm = decision["arm"]
            sampler = self.bandits.get(block)
            if sampler is None:
                continue

            override = decision.get("reward_override")
            if override is None and block in per_block_rewards:
                override = per_block_rewards[block]
            if override is None and constraint_penalty is not None:
                override = constraint_penalty

            sampler.update(
                block_id=block,
                arm_name=arm,
                reward=reward,
                reward_override=override,
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
        self._log_placeholder_ratio(solution, parsed)
        return parsed

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _legacy_offspring(
        self, parent: Solution, population_summary: str
    ) -> OffspringProposal:
        self._push_guardrail_feedback()
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
        """
        MADA 4.0 Pure-Crossover Recombination (Section 5.4).
        
        Performs strict, stable recombination using a restricted 2-Arm Bandit
        (Alpha/Beta only) without LLM-driven innovation.
        
        Uses the 4+1 Architecture: Context Block is union-merged from both parents.
        """
        parsed_alpha = self.ensure_blocks(alpha)
        parsed_beta = self.ensure_blocks(beta)
        overrides: Dict[str, str] = {}
        decisions: List[Dict[str, object]] = []
        parent_ids = set()
        helper_bundle = self._empty_helper_bundle()

        donor_parsed_map = {alpha.id: parsed_alpha, beta.id: parsed_beta}

        # Use the two supplied parents as the donor pool. Previously this
        # referenced a non-existent ``ordered`` variable, triggering a
        # NameError and aborting recombination.
        donor_pool = [alpha, beta] if alpha.id != beta.id else [alpha]

        for block in self.parser.target_blocks:
            donor = self.random.choice(donor_pool)
            parsed = donor_parsed_map.get(donor.id) or self.ensure_blocks(donor)
            donor_label = "alpha" if donor.id == alpha.id else ("beta" if donor.id == beta.id else "pool")
            snippet = parsed.blocks[block]
            decision = {
                "block": block,
                "arm": f"recomb_{donor_label}",
                "parent_id": donor.id,
                "source_hash": parsed.hashes[block],
                "bandit": False,
            }
            decisions.append(decision)
            parent_ids.add(donor.id)

            success, updated_bundle, reason = self._verify_dependencies(
                block,
                parsed_alpha,
                snippet,
                helper_bundle,
                donor_parsed=parsed,
            )
            if success:
                helper_bundle = updated_bundle
                overrides[block] = snippet
            else:
                overrides[block] = parsed_alpha.blocks[block]
                decision["dependency_issue"] = reason
                self._record_violation(reason)
                decision["parent_id"] = alpha.id
                decision["source_hash"] = parsed_alpha.hashes[block]
                parent_ids.add(alpha.id)

        # MADA 4.0: Union-merge Context Blocks from both parents
        merged_context, conflict_map = self._merge_contexts_for_offspring(
            [alpha, beta], decisions
        )
        if conflict_map:
            # Record conflicts for guardrail feedback
            for original, renamed in conflict_map.items():
                self._record_violation(f"helper_conflict:{original}")
        
        # Convert merged context to helper bundle format
        context_bundle = self._convert_context_to_helper_bundle(merged_context, conflict_map)
        
        # Merge with existing helper_bundle (dependency-resolved helpers take priority)
        for name, code in context_bundle["methods"].items():
            if name not in helper_bundle["methods"]:
                helper_bundle["methods"][name] = code
        for attr, stmt in context_bundle["init_assignments"].items():
            if attr not in helper_bundle["init_assignments"]:
                helper_bundle["init_assignments"][attr] = stmt

        changed_blocks = sorted(
            {
                decision["block"]
                for decision in decisions
                if decision.get("parent_id") != alpha.id
            }
        )
        child_name = self._child_name(
            parsed_alpha.class_name, strategy="Recomb", changed_blocks=changed_blocks
        )
        code = self.parser.assemble(
            parsed_alpha,
            overrides=overrides,
            class_name=child_name,
            helper_overrides=helper_bundle,
        )
        
        # MADA 4.0: Apply semantic linter if enabled
        code = self._apply_semantic_linter(code)
        
        lineage = {
            "strategy": "recombination",
            "decisions": decisions,
            "parents": list(parent_ids),
            "context_conflicts": conflict_map,
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
        """
        MADA 5.0 Holistic Innovation path.
        
        Uses DS-TS bandits to select block sources:
        - Alpha arm: Extract block from highest-fitness parent
        - Beta arm: Extract block from lowest-fitness parent (diversity)
        - Innovation arm: Whole-class semantic refinement via LLM (context-aware)
        
        Uses the 4+1 Architecture for context merging.
        """
        parsed_alpha = self.ensure_blocks(alpha)
        parsed_beta = self.ensure_blocks(beta)
        overrides: Dict[str, str] = {}
        decisions: List[Dict[str, object]] = []
        parent_ids = {alpha.id, beta.id}
        helper_bundle = self._empty_helper_bundle()

        for block in self.parser.target_blocks:
            sampler = self.bandits[block]
            arm, theta, snapshot = sampler.select_arm(block)
            snippet: Optional[str] = None
            reward_override = None
            parent_id = None
            source_hash = ""
            verify_source: Optional[ParsedAlgorithm] = None

            if arm == "alpha":
                snippet = parsed_alpha.blocks[block]
                parent_id = alpha.id
                source_hash = parsed_alpha.hashes[block]
                verify_source = parsed_alpha
            elif arm == "beta":
                snippet = parsed_beta.blocks[block]
                parent_id = beta.id
                source_hash = parsed_beta.hashes[block]
                verify_source = parsed_beta
            else:
                # MADA 5.0: Holistic semantic refinement using full parent code
                parsed_innov, meta = self._request_semantic_refinement(
                    block,
                    alpha,
                    beta,
                    population_summary,
                )
                if parsed_innov is not None:
                    self.innovation_cache[block] = parsed_innov
                    snippet = parsed_innov.blocks.get(block)
                    verify_source = parsed_innov
                    parent_id = meta.get("parent_id")
                    source_hash = meta.get("hash", "")
                if snippet is None:
                    snippet = parsed_alpha.blocks[block]
                    reward_override = 0.0
                    parent_id = alpha.id
                    source_hash = parsed_alpha.hashes[block]
                    verify_source = parsed_alpha

            decision = {
                "block": block,
                "arm": arm,
                "theta": theta,
                "snapshot": snapshot,
                "parent_id": parent_id,
                "source_hash": source_hash,
                "bandit": True,
                "reward_override": reward_override,
            }

            if snippet is not None:
                success, updated_bundle, reason = self._verify_dependencies(
                    block,
                    parsed_alpha,
                    snippet,
                    helper_bundle,
                    donor_parsed=verify_source,
                )
                if success:
                    helper_bundle = updated_bundle
                    overrides[block] = snippet
                else:
                    decision["dependency_issue"] = reason
                    self._record_violation(reason)
                    snippet = parsed_alpha.blocks[block]
                    overrides[block] = snippet
                    decision["parent_id"] = alpha.id
                    decision["source_hash"] = parsed_alpha.hashes[block]
                    if reward_override is None and arm == "innovation":
                        decision["reward_override"] = 0.0
            else:
                overrides[block] = parsed_alpha.blocks[block]

            decisions.append(decision)

        # MADA 4.0: Union-merge Context Blocks from both parents
        merged_context, conflict_map = self._merge_contexts_for_offspring(
            [alpha, beta], decisions
        )
        if conflict_map:
            for original, renamed in conflict_map.items():
                self._record_violation(f"helper_conflict:{original}")
        
        # Convert merged context to helper bundle format
        context_bundle = self._convert_context_to_helper_bundle(merged_context, conflict_map)
        
        # Merge with existing helper_bundle
        for name, code in context_bundle["methods"].items():
            if name not in helper_bundle["methods"]:
                helper_bundle["methods"][name] = code
        for attr, stmt in context_bundle["init_assignments"].items():
            if attr not in helper_bundle["init_assignments"]:
                helper_bundle["init_assignments"][attr] = stmt

        changed_blocks = sorted(
            decision["block"]
            for decision in decisions
            if decision.get("arm") == "innovation"
        )
        child_name = self._child_name(
            parsed_alpha.class_name, strategy="MADA", changed_blocks=changed_blocks
        )
        code = self.parser.assemble(
            parsed_alpha,
            overrides=overrides,
            class_name=child_name,
            helper_overrides=helper_bundle,
        )
        
        # MADA 4.0: Apply semantic linter if enabled
        code = self._apply_semantic_linter(code)
        
        lineage = {
            "strategy": "innovation",
            "decisions": decisions,
            "parents": list(parent_ids),
            "population_context": population_summary,
            "context_conflicts": conflict_map,
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
        context_bundle: Optional[ContextBundle] = None,
        parent_b_snippet: Optional[str] = None,
    ) -> tuple[Optional[str], Dict[str, object]]:
        """
        Ask the AlgorithmManager for a fresh block implementation.
        
        MADA 4.0 Section 5.2: Innovation arm generates novel block via LLM.
        Uses the Context Block to inform the LLM about available helpers/state.
        Pre-validates generated code to reject blocks with undefined attributes.
        """
        self._push_guardrail_feedback()
        baseline = template.blocks[block]
        signature = baseline.splitlines()[0] if baseline else f"def {block}(self, *args, **kwargs):"
        
        # Collect allowed attributes and methods for validation
        allowed_attrs = set(template.init_attributes)
        allowed_methods = set(template.other_methods.keys()) | set(self.parser.target_blocks)
        
        # Add attributes from context bundle if available
        if context_bundle is not None:
            allowed_attrs |= context_bundle.get_all_attribute_names()
            allowed_methods |= context_bundle.get_all_helper_names()
            context_info = context_bundle.to_prompt_string()
        else:
            context_info = self._format_helper_context(template, block)
        
        # Add common optimizer attributes that are always valid
        allowed_attrs |= {"budget", "dim", "f_opt", "x_opt", "lb", "ub", "evals", "pop", "fitness"}
        # Expand aliases (e.g., lb <-> lower_bounds, pop <-> population)
        allowed_attrs = expand_attributes_with_aliases(allowed_attrs)
        
        guardrail_text = self._format_guardrail_block()
        guardrail_section = f"\n{guardrail_text}\n" if guardrail_text else ""
        
        # Check if algorithm_manager has the enhanced prompt builder
        prompt_builder = getattr(self.algorithm_manager, "build_block_innovation_prompt", None)
        if callable(prompt_builder):
            prompt = prompt_builder(
                block_name=block,
                class_name=template.class_name,
                signature=signature,
                baseline=baseline,
                context=context_info,
                population_summary=population_summary,
                allowed_attributes=list(allowed_attrs),
                parent_a_snippet=baseline if block == "recombination" else None,
                parent_b_snippet=parent_b_snippet if block == "recombination" else None,
            )
        else:
            # Fallback to legacy prompt format
            prompt = textwrap.dedent(
                f"""
                The current population summary is:\n{population_summary}\n
                Improve the `{block}` method of the optimizer class `{template.class_name}`.
                Use the exact signature `{signature}` and return only the method definition
                inside a Python code block. Keep helper references consistent with the class.

                Context Block (available helpers and state):
                {context_info}

                {guardrail_section}

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

        # MADA 4.0: Pre-validate that snippet only uses allowed attributes
        is_valid, invalid_refs, suggestions = validate_snippet_attributes(
            cleaned, allowed_attrs, allowed_methods
        )
        
        if not is_valid:
            # Record the specific invalid references for guardrail feedback
            hint = get_attribute_correction_hint(invalid_refs, suggestions)
            for ref in invalid_refs:
                self._record_violation(f"invalid_attr:{ref}")
            
            # Log the rejection for debugging
            logger = getattr(self.algorithm_manager, "logger", None)
            if logger and hasattr(logger, "log_conversation"):
                try:
                    logger.log_conversation(
                        f"\n[MADA Block Rejected: {block}] Invalid attribute references:\n{hint}\n"
                    )
                except Exception:
                    pass
            
            return None, {"rejected": True, "invalid_refs": list(invalid_refs)}

        block_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return cleaned, {"hash": block_hash, "parent_id": "innovation"}

    def _request_semantic_refinement(
        self,
        block: str,
        alpha: Solution,
        beta: Solution,
        population_summary: str,
    ) -> tuple[Optional[ParsedAlgorithm], Dict[str, object]]:
        """
        MADA 5.0: Whole-class semantic refinement for a target block.
        Sends the entire parent code to the LLM so dependencies are preserved.
        """
        builder = getattr(self.algorithm_manager, "build_semantic_refinement_prompt", None)
        requester = getattr(self.algorithm_manager, "generate_semantic_refinement", None)
        if not callable(builder) or not callable(requester):
            return None, {}

        guardrails = self._format_guardrail_block()
        delta_summary = self._delta_summary(alpha, beta)
        prompt = builder(
            block_name=block,
            parent_a_code=alpha.code or "",
            parent_b_code=beta.code or "",
            population_summary=population_summary,
            instruction=f"Perform a semantic refinement of the {block} logic; preserve other blocks.",
            guardrails=guardrails,
            delta_summary=delta_summary,
        )

        try:
            message = requester(block_name=block, prompt=prompt)
            class_code = self.algorithm_manager.extract_algorithm_code(message)
        except Exception:
            return None, {}

        parsed = self.parser.extract(class_code or "")
        snippet = parsed.blocks.get(block)
        if not snippet:
            return None, {}

        return parsed, {
            "hash": parsed.hashes.get(block, ""),
            "parent_id": "innovation",
            "source_code": class_code,
        }

    def _valid_block(self, snippet: str, block: str) -> bool:
        if not snippet.startswith("def "):
            self._record_violation(f"invalid_block:{block}")
            return False
        if not snippet.split("(")[0].endswith(block):
            self._record_violation(f"invalid_block:{block}")
            return False
        try:
            ast.parse(textwrap.dedent(snippet))
            return True
        except SyntaxError:
            self._record_violation(f"invalid_block:{block}")
            return False

    # ------------------------------------------------------------------ #
    # Dependency + helper utilities
    # ------------------------------------------------------------------ #
    def _format_helper_context(self, template: ParsedAlgorithm, block: str) -> str:
        bundle = template.helper_groups.get(block) or {}
        helpers = sorted(list(bundle.get("helpers", [])))
        attributes = sorted(list(bundle.get("attributes", [])))
        lines = []
        if helpers:
            lines.append(f"Available helper methods: {', '.join(helpers)}")
        else:
            lines.append("Available helper methods: none detected; reuse shared helpers only.")
        if attributes:
            lines.append(f"Shared attributes referenced: {', '.join(attributes)}")
        else:
            lines.append("Shared attributes referenced: none beyond default state.")
        return "\n".join(lines)

    def _empty_helper_bundle(self) -> Dict[str, Dict[str, str]]:
        return {"methods": {}, "init_assignments": {}}

    def _clone_helper_bundle(
        self, bundle: Optional[Dict[str, Dict[str, str]]]
    ) -> Dict[str, Dict[str, str]]:
        bundle = bundle or {}
        return {
            "methods": dict(bundle.get("methods", {})),
            "init_assignments": dict(bundle.get("init_assignments", {})),
        }

    # ------------------------------------------------------------------ #
    # MADA 4.0: Context Bundle Union-Merge (Section 5.6)
    # ------------------------------------------------------------------ #
    def _get_context_bundle(self, solution: Solution) -> ContextBundle:
        """Get or cache the context bundle for a solution."""
        if solution.id in self._context_cache:
            return self._context_cache[solution.id]
        
        parsed = self.ensure_blocks(solution)
        context = parsed.get_context_bundle()
        self._context_cache[solution.id] = context
        return context

    def _merge_contexts_for_offspring(
        self,
        donors: List[Solution],
        decisions: List[Dict[str, object]],
    ) -> Tuple[ContextBundle, Dict[str, str]]:
        """
        MADA 4.0 Section 5.6: Union-merge Context Blocks from all donors.
        
        The offspring inherits the union of all helper functions and state
        variables from all parents used in the recombination.
        """
        # Collect unique donors based on decisions
        donor_ids = set()
        for decision in decisions:
            parent_id = decision.get("parent_id")
            if parent_id:
                donor_ids.add(parent_id)
        
        # Get context bundles for each unique donor
        contexts = []
        for donor in donors:
            if donor.id in donor_ids:
                contexts.append(self._get_context_bundle(donor))
        
        if not contexts:
            return ContextBundle(), {}
        
        return union_merge_contexts(contexts, prefer_first=True)

    def _delta_summary(self, alpha: Solution, beta: Solution) -> str:
        """Summarize high-level differences between two parents for prompting."""
        try:
            pa = self.ensure_blocks(alpha)
            pb = self.ensure_blocks(beta)
            differing_blocks = []
            for block in self.parser.target_blocks:
                if pa.hashes.get(block) != pb.hashes.get(block):
                    differing_blocks.append(block)
            if not differing_blocks:
                return "Parents are very similar across target blocks."
            return "Blocks that differ: " + ", ".join(differing_blocks)
        except Exception:
            return "(delta summary unavailable)"

    def _pick_diverse_parent(self, ordered: List[Solution], alpha: Solution) -> Solution:
        """Choose a second parent with different code when possible."""
        for candidate in ordered[1:]:
            if (candidate.code or "") != (alpha.code or ""):
                return candidate
        return ordered[1] if len(ordered) > 1 else ordered[0]

    def _apply_semantic_linter(self, code: str) -> str:
        """
        MADA 4.0 Section 5.6: Apply semantic linter to resolve conflicts.
        
        Runs a lightweight linter pass to:
        1. Rename conflicting helper methods
        2. Update call sites to use renamed methods
        3. Remove unused helpers
        """
        if not self.enable_semantic_linter:
            return code
        
        # Check if algorithm_manager has the linter method
        linter = getattr(self.algorithm_manager, "run_semantic_linter", None)
        if not callable(linter):
            return code
        
        try:
            return linter(code, timeout_fallback=code)
        except Exception:
            return code

    def _convert_context_to_helper_bundle(
        self,
        merged_context: ContextBundle,
        conflict_map: Dict[str, str],
    ) -> Dict[str, Dict[str, str]]:
        """Convert merged context bundle to the helper_bundle format for assembly."""
        bundle = self._empty_helper_bundle()
        
        # Add all other_methods (class helpers) to the bundle
        for name, code in merged_context.other_methods.items():
            if name != "__init__":
                bundle["methods"][name] = code
        
        # Add init_assignments
        bundle["init_assignments"].update(merged_context.init_assignments)
        
        return bundle

    def _verify_dependencies(
        self,
        block: str,
        template: ParsedAlgorithm,
        snippet: str,
        helper_bundle: Dict[str, Dict[str, str]],
        donor_parsed: Optional[ParsedAlgorithm] = None,
    ) -> tuple[bool, Dict[str, Dict[str, str]], Optional[str]]:
        working_bundle = self._clone_helper_bundle(helper_bundle)
        success, working_bundle, reason = self._ensure_snippet_support(
            snippet,
            template,
            donor_parsed,
            working_bundle,
            visited=set(),
        )
        if not success:
            return False, helper_bundle, reason
        return True, working_bundle, None

    def _ensure_snippet_support(
        self,
        snippet: str,
        template: ParsedAlgorithm,
        donor_parsed: Optional[ParsedAlgorithm],
        helper_bundle: Dict[str, Dict[str, str]],
        visited: Set[str],
    ) -> tuple[bool, Dict[str, Dict[str, str]], Optional[str]]:
        deps = self._dependencies_from_snippet(snippet)
        available_helpers = set(template.other_methods.keys()) | set(
            helper_bundle["methods"].keys()
        )

        for helper in deps["helpers"]:
            if helper in visited or helper in available_helpers or helper in self.parser.target_blocks:
                continue
            if donor_parsed is None:
                return False, helper_bundle, f"missing_helper:{helper}"
            helper_snippet = self._locate_helper_snippet(donor_parsed, helper)
            if helper_snippet is None:
                return False, helper_bundle, f"missing_helper:{helper}"
            helper_bundle["methods"][helper] = helper_snippet
            visited.add(helper)
            success, helper_bundle, reason = self._ensure_snippet_support(
                helper_snippet,
                template,
                donor_parsed,
                helper_bundle,
                visited,
            )
            if not success:
                return False, helper_bundle, reason
            available_helpers.add(helper)

        missing_attrs = [
            attr
            for attr in deps["attributes"]
            if attr not in template.init_attributes
            and attr not in helper_bundle["init_assignments"]
        ]
        if missing_attrs:
            assignments = self._resolve_attribute_assignments(missing_attrs, donor_parsed)
            if assignments is None:
                return False, helper_bundle, f"missing_attributes:{','.join(missing_attrs)}"
            helper_bundle["init_assignments"].update(assignments)

        return True, helper_bundle, None

    def _locate_helper_snippet(
        self, parsed: Optional[ParsedAlgorithm], helper_name: str
    ) -> Optional[str]:
        if parsed is None:
            return None
        if helper_name in parsed.other_methods:
            return parsed.other_methods[helper_name]
        if helper_name in parsed.blocks:
            return parsed.blocks[helper_name]
        return None

    def _resolve_attribute_assignments(
        self, attributes: List[str], donor_parsed: Optional[ParsedAlgorithm]
    ) -> Optional[Dict[str, str]]:
        if donor_parsed is None:
            return None
        assignments: Dict[str, str] = {}
        for attr in attributes:
            stmt = donor_parsed.init_assignments.get(attr)
            if stmt is None:
                return None
            assignments[attr] = stmt
        return assignments

    def _dependencies_from_snippet(self, snippet: str) -> Dict[str, Set[str]]:
        try:
            module = ast.parse(textwrap.dedent(snippet))
        except SyntaxError:
            return {"helpers": set(), "attributes": set()}
        func_def = next(
            (node for node in module.body if isinstance(node, ast.FunctionDef)),
            None,
        )
        if func_def is None:
            return {"helpers": set(), "attributes": set()}
        visitor = _SnippetDependencyVisitor()
        visitor.visit(func_def)
        return {"helpers": visitor.helpers, "attributes": visitor.attributes}

    def _log_placeholder_ratio(self, solution: Solution, parsed: ParsedAlgorithm) -> None:
        coverage = getattr(parsed, "coverage", {}) or {}
        ratio = coverage.get("placeholder_ratio")
        if ratio is None or ratio < self.placeholder_threshold:
            return
        logger = getattr(self.algorithm_manager, "logger", None)
        if logger and hasattr(logger, "log_conversation"):
            name = solution.name or parsed.class_name
            message = (
                f"[MADA] Placeholder ratio {ratio:.2f} detected for {name}. "
                "Verify parser heuristics or provide richer prompts."
            )
            try:
                logger.log_conversation("system", message)
            except Exception:
                pass
        self._record_violation("high_placeholder_ratio")

    def _record_violation(self, reason: Optional[str]) -> None:
        if not reason:
            return
        bucket = reason.split(":", 1)[0]
        self._violation_history.append(bucket)

    def _guardrail_feedback(self) -> str:
        if not self._violation_history:
            return ""
        counts = Counter(self._violation_history)
        top = counts.most_common(3)
        fragments = []
        for key, count in top:
            label = self._friendly_violation_label(key)
            fragments.append(f"{label}: {count}")
        return "; ".join(fragments)

    def _friendly_violation_label(self, key: str) -> str:
        mapping = {
            "missing_helper": "missing helper support",
            "missing_attributes": "missing __init__ attributes",
            "high_placeholder_ratio": "high placeholder ratio",
            "invalid_block": "invalid block syntax",
            "invalid_attr": "undefined attribute used",
        }
        return mapping.get(key, key.replace("_", " "))

    def _guardrail_lines(self) -> List[str]:
        lines = list(PROMPT_GUARDRAILS)
        summary = self._guardrail_feedback()
        if summary:
            lines.append(f"- Recent issues observed: {summary}")
        return lines

    def _format_guardrail_block(self) -> str:
        lines = self._guardrail_lines()
        if not lines:
            return ""
        return "Please respect the following guardrails:\n" + "\n".join(lines)

    def _push_guardrail_feedback(self) -> None:
        updater = getattr(self.algorithm_manager, "update_guardrail_feedback", None)
        if not callable(updater):
            return
        summary = self._guardrail_feedback()
        try:
            updater(summary)
        except Exception:
            pass

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

    def _child_name(
        self, base: str, strategy: str, changed_blocks: Optional[List[str]] = None
    ) -> str:
        self.child_counter += 1
        tag = self._summarize_blocks_for_name(changed_blocks)
        base_stub = base[:18].replace(" ", "")
        return f"{base_stub}_{strategy}_{tag}_{self.child_counter:04d}"

    def _summarize_blocks_for_name(
        self, blocks: Optional[Sequence[str]]
    ) -> str:
        if not blocks:
            return "stable"
        trimmed = [block.replace("_", "") for block in blocks[:3]]
        tag = "_".join(trimmed)
        if len(blocks) > 3:
            tag += "_more"
        return tag or "stable"

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


class _SnippetDependencyVisitor(ast.NodeVisitor):
    """Minimal visitor to inspect helper calls/attribute usage."""

    def __init__(self):
        self.helpers: Set[str] = set()
        self.attributes: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self":
                self.helpers.add(func.attr)
                self.visit(func.value)
            else:
                self.visit(func)
        else:
            self.visit(func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            self.attributes.add(node.attr)
        self.generic_visit(node)
