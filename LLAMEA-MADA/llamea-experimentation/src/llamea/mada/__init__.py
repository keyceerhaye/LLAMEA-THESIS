"""MADA integration helpers for LLaMEA."""

from .ds_ts import DiscountedThompsonSampler
from .operator import MADAOperator, OffspringProposal
from .parser import (
    BlockParser,
    ContextBundle,
    ParsedAlgorithm,
    detect_unused_helpers,
    expand_attributes_with_aliases,
    get_attribute_correction_hint,
    union_merge_contexts,
    validate_snippet_attributes,
    FORBIDDEN_ATTRIBUTE_ALIASES,
)

__all__ = [
    "BlockParser",
    "ContextBundle",
    "DiscountedThompsonSampler",
    "expand_attributes_with_aliases",
    "FORBIDDEN_ATTRIBUTE_ALIASES",
    "MADAOperator",
    "OffspringProposal",
    "ParsedAlgorithm",
    "detect_unused_helpers",
    "get_attribute_correction_hint",
    "union_merge_contexts",
    "validate_snippet_attributes",
]




