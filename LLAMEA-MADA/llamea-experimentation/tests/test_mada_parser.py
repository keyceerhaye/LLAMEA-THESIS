import os
import sys
import textwrap
import unittest


TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llamea.mada.operator import MADAOperator  # noqa: E402
from llamea.mada.parser import BlockParser  # noqa: E402


class _DummyManager:
    """Minimal stand-in for the AlgorithmManager used in tests."""

    logger = None


class BlockParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = BlockParser()

    def test_semantic_classification_and_metadata(self):
        code = textwrap.dedent(
            """
            import numpy as np

            class FancyOptimizer:
                def __init__(self, budget=10):
                    self.budget = budget
                    self.archive = []

                def select_parents(self, population):
                    ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
                    return ranked[:2]

                def crossover_recombine(self, parents):
                    child = parents[0]
                    self.sync_archive(child)
                    self.archive.append(child)
                    return child

                def mutate_candidate(self, candidate):
                    return candidate + np.random.randn()

                def survivor_selection_step(self, population, offspring):
                    self.archive.append(offspring[0])
                    return population

                def sync_archive(self, candidate):
                    self.archive.append(candidate)
                    return candidate
            """
        )

        parsed = self.parser.extract(code)

        self.assertTrue(parsed.blocks["parent_selection"].startswith("def select_parents"))
        self.assertTrue(parsed.blocks["recombination"].startswith("def crossover_recombine"))
        self.assertEqual(parsed.block_sources["mutation"], "mutate_candidate")
        self.assertEqual(parsed.block_sources["survivor_selection"], "survivor_selection_step")

        recomb_bundle = parsed.helper_groups["recombination"]
        self.assertIn("sync_archive", recomb_bundle["helpers"])
        self.assertIn("archive", recomb_bundle["attributes"])

        self.assertEqual(parsed.method_roles["sync_archive"], "helper")
        self.assertEqual(parsed.coverage["placeholder_blocks"], 0)

    def test_placeholder_reason_and_helper_groups_default(self):
        code = textwrap.dedent(
            """
            class MinimalOpt:
                def parent_selection(self, population):
                    return population

                def recombination(self, parents):
                    return parents[0]

                def mutation(self, candidate):
                    return candidate
            """
        )

        parsed = self.parser.extract(code)

        self.assertEqual(parsed.coverage["placeholder_blocks"], 1)
        self.assertEqual(parsed.block_sources["survivor_selection"], "__placeholder__")
        self.assertEqual(
            parsed.helper_groups["survivor_selection"]["helpers"],
            set(),
        )
        self.assertEqual(
            parsed.placeholder_reasons["survivor_selection"], "no_candidate"
        )


class DependencyVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = BlockParser()
        self.operator = MADAOperator(algorithm_manager=_DummyManager(), parser=self.parser)

    def test_verify_dependencies_clones_helpers_and_init_assignments(self):
        template_code = textwrap.dedent(
            """
            class TemplateOpt:
                def __init__(self, budget=10):
                    self.budget = budget

                def parent_selection(self, population):
                    return population

                def recombination(self, parents):
                    return parents[0]

                def mutation(self, candidate):
                    return candidate

                def survivor_selection(self, population, offspring):
                    return population
            """
        )
        donor_code = textwrap.dedent(
            """
            class DonorOpt:
                def __init__(self):
                    self.archive = []

                def parent_selection(self, population):
                    return population

                def recombination(self, parents):
                    return parents[0]

                def mutation(self, candidate):
                    return candidate

                def survivor_selection(self, population, offspring):
                    return population

                def sync_archive(self, candidate):
                    self.archive.append(candidate)
                    return candidate
            """
        )

        snippet = textwrap.dedent(
            """
            def recombination(self, parents):
                child = parents[0]
                self.sync_archive(child)
                if len(self.archive) > 5:
                    self.archive.pop(0)
                return child
            """
        )

        template = self.parser.extract(template_code)
        donor = self.parser.extract(donor_code)
        helper_bundle = self.operator._empty_helper_bundle()

        success, updated_bundle, reason = self.operator._verify_dependencies(
            "recombination",
            template,
            snippet,
            helper_bundle,
            donor_parsed=donor,
        )

        self.assertTrue(success, msg=f"dependency verification failed: {reason}")
        self.assertIn("sync_archive", updated_bundle["methods"])
        self.assertIn("archive", updated_bundle["init_assignments"])

    def test_verify_dependencies_fails_without_helpers(self):
        template_code = textwrap.dedent(
            """
            class BaseOpt:
                def parent_selection(self, population):
                    return population

                def recombination(self, parents):
                    return parents[0]

                def mutation(self, candidate):
                    return candidate

                def survivor_selection(self, population, offspring):
                    return population
            """
        )

        missing_helper_snippet = textwrap.dedent(
            """
            def mutation(self, candidate):
                self.unknown_helper()
                return candidate
            """
        )

        template = self.parser.extract(template_code)
        helper_bundle = self.operator._empty_helper_bundle()

        success, _, reason = self.operator._verify_dependencies(
            "mutation",
            template,
            missing_helper_snippet,
            helper_bundle,
            donor_parsed=None,
        )

        self.assertFalse(success)
        self.assertTrue(reason.startswith("missing_helper"))


if __name__ == "__main__":
    unittest.main()

