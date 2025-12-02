"""
Centralized prompt snippets used by the adaptive outer loop controller.

Having a dedicated module keeps the high-level behavioural instructions
discoverable and makes it easier to reuse the same wording across
different strategies without sprinkling long strings through the code.
"""

MUTATION_EXPLORE = (
    "Generate a new optimization algorithm that is deliberately different "
    "from the ones explored so far. Prioritise novel structures or search "
    "patterns over incremental tweaks."
)

MUTATION_SIMPLIFY = (
    "Refine and simplify the selected algorithm to make it more efficient "
    "and robust. Remove redundant logic, tighten parameter handling, and "
    "clarify the control flow without changing the core objective."
)

CROSSOVER_PURE = (
    "Combine the provided functional blocks exactly as given. Reuse the "
    "parent implementations verbatim and focus purely on assembling a hybrid "
    "algorithm. Do not invent new helper logic—only stitch together the "
    "supplied parent components."
)


__all__ = [
    "MUTATION_EXPLORE",
    "MUTATION_SIMPLIFY",
    "CROSSOVER_PURE",
]


