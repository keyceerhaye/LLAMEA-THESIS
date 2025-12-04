"""Main orchestration logic for the MADA operator."""

from __future__ import annotations

import ast
import hashlib
import math
import random
import textwrap
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set

import numpy as np

from ..solution import Solution
from .ds_ts import DiscountedThompsonSampler
from .parser import BlockParser, ParsedAlgorithm, PLACEHOLDER_SIGNATURES

PROMPT_GUARDRAILS = [
    "- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.",
    "- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.",
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


class PlaceholderAbort(Exception):
    """Raised when a parent is too placeholder-heavy for recombination."""

    def __init__(self, parent: Solution, ratio: float, block: Optional[str] = None):
        self.parent = parent
        self.ratio = ratio
        self.block = block
        super().__init__(
            f"Placeholder pressure ({ratio:.2f}) for {parent.name}"
            + (f" block {block}" if block else "")
        )


class InnovationAbort(Exception):
    """Raised when innovation fell back to alpha for most blocks."""

    def __init__(self, parent: Solution, reason: str):
        self.parent = parent
        self.reason = reason
        super().__init__(reason)


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
        guardrail_window: int = 10,
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
        self.default_strategy_weights = dict(self.strategy_weights)
        self.boosted_strategy_weights = self._normalize_weights(
            {"innovation": 0.5, "recombination": 0.2, "legacy": 0.3}
        )
        self._boost_generations_remaining = 0
        self.strategy_failure_counts = Counter()
        self.strategy_cooldowns: Dict[str, int] = {
            key: 0 for key in self.strategy_weights
        }
        self.failure_threshold = 2
        self.cooldown_length = 5
        self.child_counter = 0
        self.placeholder_threshold = 0.5
        self.placeholder_block_threshold = 0.3
        self.guardrail_window = guardrail_window
        self._violation_history: Deque[str] = deque(maxlen=guardrail_window)
        (
            self.expected_signature_headers,
            self.expected_signature_args,
        ) = self._build_expected_signatures()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_offspring(
        self,
        parents: Sequence[Solution],
        focal_parent: Optional[Solution] = None,
        population_summary: str = "",
        force_strategy: Optional[str] = None,
    ) -> OffspringProposal:
        """Return code + lineage for a new offspring according to the plan."""

        if not parents:
            raise ValueError("MADAOperator requires at least one parent.")

        ordered = sorted(parents, key=lambda sol: sol.fitness, reverse=True)
        alpha = ordered[0]
        beta = ordered[1] if len(ordered) > 1 else ordered[0]
        focal = focal_parent or self.random.choice(ordered)
        if force_strategy and force_strategy not in self.strategy_weights:
            raise ValueError(f"force_strategy must be one of {list(self.strategy_weights)}")
        self._tick_strategy_cooldowns()
        strategy = force_strategy or self._sample_strategy()

        try:
            if strategy == "legacy":
                return self._legacy_offspring(focal, population_summary)
            if strategy == "recombination":
                return self._recombination_offspring(alpha, beta)
            return self._innovative_offspring(alpha, beta, population_summary)
        except PlaceholderAbort as exc:
            self._log_strategy_event(
                f"[MADA] Placeholder pressure detected ({exc.ratio:.2f}); "
                f"requesting legacy rewrite for {exc.parent.name}."
            )
            return self._legacy_offspring(exc.parent, population_summary)
        except InnovationAbort as exc:
            self._log_strategy_event(
                f"[MADA] Innovation aborted: {exc.reason}. Falling back to legacy."
            )
            return self._legacy_offspring(exc.parent, population_summary)
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

    def boost_exploration(self, duration: int = 3) -> None:
        """Temporarily bias sampling toward exploration-heavy weights."""

        duration = max(1, int(duration))
        self.strategy_weights = dict(self.boosted_strategy_weights)
        self._boost_generations_remaining = duration
        self._log_strategy_event(
            f"[MADA] Exploration boost enabled for {duration} generations."
        )

    def reset_strategy_weights(self, message: Optional[str] = None) -> None:
        """Restore strategy weights to their defaults."""

        if (
            self._boost_generations_remaining == 0
            and self.strategy_weights == self.default_strategy_weights
        ):
            return
        self.strategy_weights = dict(self.default_strategy_weights)
        self._boost_generations_remaining = 0
        note = message or "[MADA] Strategy weights reset to defaults."
        self._log_strategy_event(note)

    def step_generation(self) -> None:
        """Decay any temporary exploration boost once per generation."""

        if self._boost_generations_remaining > 0:
            self._boost_generations_remaining -= 1
            if self._boost_generations_remaining == 0:
                self.reset_strategy_weights(
                    "[MADA] Exploration boost window elapsed; weights restored."
                )

    def record_strategy_outcome(
        self, strategy: str, success: bool, error: Optional[str] = None
    ) -> None:
        """Track repeated failures per strategy so we can temporarily disable them."""

        if strategy not in self.strategy_weights:
            return
        if success:
            self.strategy_failure_counts[strategy] = 0
            return

        self.strategy_failure_counts[strategy] += 1
        self._log_strategy_event(
            f"[MADA] Strategy {strategy} failure count="
            f"{self.strategy_failure_counts[strategy]} (error: {error or 'n/a'})"
        )
        if self.strategy_failure_counts[strategy] >= self.failure_threshold:
            self.strategy_cooldowns[strategy] = self.cooldown_length
            self.strategy_failure_counts[strategy] = 0
            self._record_violation(f"{strategy}_disabled")
            self._log_strategy_event(
                f"[MADA] Disabled {strategy} strategy for "
                f"{self.cooldown_length} offspring due to repeated failures."
            )
            self._push_guardrail_feedback()

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
        parsed_alpha = self.ensure_blocks(alpha)
        parsed_beta = self.ensure_blocks(beta)
        ratio_alpha = self._block_placeholder_ratio(parsed_alpha)
        if ratio_alpha >= self.placeholder_block_threshold:
            raise PlaceholderAbort(alpha, ratio_alpha)
        ratio_beta = self._block_placeholder_ratio(parsed_beta)
        if ratio_beta >= self.placeholder_block_threshold:
            raise PlaceholderAbort(beta, ratio_beta)
        overrides: Dict[str, str] = {}
        decisions: List[Dict[str, object]] = []
        parent_ids = set()
        helper_bundle = self._empty_helper_bundle()

        for block in self.parser.target_blocks:
            donor_label, parsed, donor = self.random.choice(
                [("alpha", parsed_alpha, alpha), ("beta", parsed_beta, beta)]
            )
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
        helper_bundle = self._empty_helper_bundle()
        fallback_blocks = 0
        total_blocks = len(self.parser.target_blocks) or 1

        for block in self.parser.target_blocks:
            sampler = self.bandits[block]
            arm, theta, snapshot = sampler.select_arm(block)
            snippet: Optional[str] = None
            reward_override = None
            parent_id = None
            source_hash = ""
            fallback_to_alpha = False

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
                    fallback_to_alpha = True
                else:
                    parent_id = meta.get("parent_id")
                    source_hash = meta.get("hash", "")

            verify_source = None
            if arm == "alpha":
                verify_source = parsed_alpha
            elif arm == "beta":
                verify_source = parsed_beta
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
                    fallback_to_alpha = True
                    if reward_override is None and arm == "innovation":
                        decision["reward_override"] = 0.0
            else:
                overrides[block] = parsed_alpha.blocks[block]
                fallback_to_alpha = True

            decisions.append(decision)
            if fallback_to_alpha:
                fallback_blocks += 1

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
        lineage = {
            "strategy": "innovation",
            "decisions": decisions,
            "parents": list(parent_ids),
            "population_context": population_summary,
        }
        if fallback_blocks >= math.ceil(total_blocks / 2):
            raise InnovationAbort(
                alpha,
                f"{fallback_blocks}/{total_blocks} blocks reverted to alpha",
            )
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

        self._push_guardrail_feedback()
        baseline = template.blocks[block]
        signature = baseline.splitlines()[0] if baseline else f"def {block}(self, *args, **kwargs):"
        helper_context = self._format_helper_context(template, block)
        guardrail_text = self._format_guardrail_block()
        guardrail_section = f"\n{guardrail_text}\n" if guardrail_text else ""
        prompt = textwrap.dedent(
            f"""
            The current population summary is:\n{population_summary}\n
            Improve the `{block}` method of the optimizer class `{template.class_name}`.
            Use the exact signature `{signature}` and return only the method definition
            inside a Python code block. Keep helper references consistent with the class.

            Helper/context for `{block}`:
            {helper_context}

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
        cleaned, _ = self._repair_signature(block, cleaned)
        if not self._valid_block(cleaned, block):
            return None, {}
        if self._is_placeholder_snippet(cleaned):
            self._record_violation(f"placeholder_snippet:{block}")
            return None, {}

        block_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return cleaned, {"hash": block_hash, "parent_id": "innovation"}

    def _valid_block(self, snippet: str, block: str) -> bool:
        if not snippet.startswith("def "):
            self._record_violation(f"invalid_block:{block}")
            return False
        if not snippet.split("(")[0].endswith(block):
            self._record_violation(f"invalid_block:{block}")
            return False
        try:
            module = ast.parse(textwrap.dedent(snippet))
        except SyntaxError:
            self._record_violation(f"invalid_block:{block}")
            return False
        func_node = next((node for node in module.body if isinstance(node, ast.FunctionDef)), None)
        if func_node is None or func_node.name != block:
            self._record_violation(f"invalid_block:{block}")
            return False
        return self._signature_matches(func_node, block)

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

    def _log_strategy_event(self, message: str) -> None:
        logger = getattr(self.algorithm_manager, "logger", None)
        if not (logger and hasattr(logger, "log_conversation")):
            return
        try:
            logger.log_conversation("system", message)
        except TypeError:
            try:
                logger.log_conversation(message)
            except Exception:
                pass
        except Exception:
            pass

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
            "innovation_disabled": "innovation cooldown",
            "recombination_disabled": "recombination cooldown",
            "legacy_disabled": "legacy cooldown",
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
    def _build_expected_signatures(self) -> tuple[Dict[str, str], Dict[str, List[str]]]:
        headers: Dict[str, str] = {}
        args_map: Dict[str, List[str]] = {}
        for block, template in PLACEHOLDER_SIGNATURES.items():
            dedented = textwrap.dedent(template).strip()
            lines = dedented.splitlines()
            if not lines:
                continue
            headers[block] = lines[0].strip()
            try:
                node = ast.parse(dedented).body[0]
                if isinstance(node, ast.FunctionDef):
                    args_map[block] = [arg.arg for arg in node.args.args]
                else:
                    args_map[block] = []
            except SyntaxError:
                args_map[block] = []
        return headers, args_map

    def _repair_signature(self, block: str, snippet: str) -> tuple[str, bool]:
        expected = self.expected_signature_headers.get(block)
        if not expected:
            return snippet, False
        dedented = textwrap.dedent(snippet).splitlines()
        if not dedented:
            return snippet, False
        current = dedented[0].strip()
        normalized_expected = expected.strip()
        if current == normalized_expected:
            return snippet, False
        dedented[0] = normalized_expected
        repaired = "\n".join(dedented)
        self._log_strategy_event(
            f"[MADA] Auto-repaired signature for `{block}` to `{normalized_expected}`."
        )
        return repaired, True

    def _is_placeholder_snippet(self, snippet: Optional[str]) -> bool:
        if not snippet:
            return True
        return 'Placeholder generated by MADA' in snippet

    def _block_placeholder_ratio(self, parsed: ParsedAlgorithm) -> float:
        total = len(self.parser.target_blocks) or 1
        placeholder_blocks = 0
        for block in self.parser.target_blocks:
            snippet = parsed.blocks.get(block, "")
            if self._is_placeholder_snippet(snippet):
                placeholder_blocks += 1
        return placeholder_blocks / total

    def _signature_matches(self, func_node: ast.FunctionDef, block: str) -> bool:
        expected = self.expected_signature_args.get(block)
        if not expected:
            return True
        actual = [arg.arg for arg in func_node.args.args]
        if actual == expected:
            return True
        self._record_violation(f"invalid_signature:{block}")
        return False

    def _tick_strategy_cooldowns(self) -> None:
        for strategy, remaining in list(self.strategy_cooldowns.items()):
            if remaining > 0:
                self.strategy_cooldowns[strategy] = remaining - 1

    def _active_strategy_weights(self) -> Dict[str, float]:
        active = {
            strategy: weight
            for strategy, weight in self.strategy_weights.items()
            if self.strategy_cooldowns.get(strategy, 0) == 0
        }
        if not active:
            return self.strategy_weights
        total = sum(active.values())
        if total <= 0:
            return self.strategy_weights
        return {strategy: weight / total for strategy, weight in active.items()}

    def _sample_strategy(self) -> str:
        weights = self._active_strategy_weights()
        roll = self.random.random()
        cumulative = 0.0
        for strategy, weight in weights.items():
            cumulative += weight
            if roll <= cumulative:
                return strategy
        return next(iter(weights.keys()), "innovation")

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
