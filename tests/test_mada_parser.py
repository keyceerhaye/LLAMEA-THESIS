from llamea.mada.parser import BlockParser


SIMPLE_ALGO = """
import numpy as np

class DemoOptim:
    def __init__(self, budget=10):
        self.budget = budget

    def __call__(self, func):
        return func(np.zeros(1))

    def parent_selection(self, population):
        return population

    def recombination(self, parents):
        return parents[0]

    def mutation(self, candidate):
        return candidate

    def survivor_selection(self, population, offspring):
        return population
"""


def test_parser_extracts_blocks_and_hashes():
    parser = BlockParser()
    parsed = parser.extract(SIMPLE_ALGO)

    assert parsed.class_name == "DemoOptim"
    for block in parser.target_blocks:
        assert block in parsed.blocks
        assert block in parsed.hashes
        assert len(parsed.blocks[block]) > 0


def test_parser_assemble_applies_overrides():
    parser = BlockParser()
    parsed = parser.extract(SIMPLE_ALGO)
    override = "def mutation(self, candidate):\n    return candidate * 2"

    rebuilt = parser.assemble(parsed, overrides={"mutation": override}, class_name="NewOpt")

    assert "class NewOpt" in rebuilt
    assert "return candidate * 2" in rebuilt




