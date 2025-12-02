"""AST utilities for splitting and assembling optimizer blocks."""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PLACEHOLDER_SIGNATURES = {
    "__init__": (
        "def __init__(self, budget=10000):\n"
        "        self.budget = budget\n"
        "        self.dim = None\n"
        "        self.history = []"
    ),
    "__call__": (
        "def __call__(self, func):\n"
        "        raise NotImplementedError('Optimizer must implement __call__')"
    ),
    "parent_selection": "def parent_selection(self, population):\n        return population",
    "recombination": "def recombination(self, parents):\n        return parents[0]",
    "mutation": "def mutation(self, candidate):\n        return candidate",
    "survivor_selection": (
        "def survivor_selection(self, population, offspring):\n        return population"
    ),
}


@dataclass
class ParsedAlgorithm:
    """Container for all structural parts needed to rebuild an algorithm."""

    class_name: str
    imports: List[str] = field(default_factory=list)
    helpers: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    other_methods: List[str] = field(default_factory=list)
    blocks: Dict[str, str] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, object]:
        """Return a serialisable representation for Solution.metadata."""

        return {
            "class_name": self.class_name,
            "hashes": self.hashes,
            "blocks": {name: {"code": code, "hash": self.hashes[name]} for name, code in self.blocks.items()},
        }


class BlockParser:
    """Splits optimizer code into reusable AST-backed blocks."""

    def __init__(self, target_blocks: Optional[List[str]] = None):
        self.target_blocks = target_blocks or list(PLACEHOLDER_SIGNATURES.keys())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, source: str) -> ParsedAlgorithm:
        """Extract the block map and skeleton from ``source``."""

        try:
            module = ast.parse(source)
        except SyntaxError:
            return self._synthetic_algorithm()

        imports: List[str] = []
        helpers: List[str] = []
        class_defs: List[ast.ClassDef] = []

        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(self._unparse(node))
            elif isinstance(node, ast.FunctionDef):
                helpers.append(self._dedent(self._unparse(node)))
            elif isinstance(node, ast.ClassDef):
                class_defs.append(node)

        class_def = self._select_primary_class(class_defs)
        if class_def is None:
            return self._synthetic_algorithm()

        blocks: Dict[str, str] = {}
        other_methods: List[str] = []
        for element in class_def.body:
            if isinstance(element, ast.FunctionDef):
                snippet = self._dedent(self._unparse(element))
                if element.name in self.target_blocks:
                    blocks[element.name] = snippet
                else:
                    other_methods.append(snippet)

        docstring = ast.get_docstring(class_def)
        parsed = ParsedAlgorithm(
            class_name=class_def.name,
            imports=imports,
            helpers=helpers,
            docstring=docstring,
            other_methods=other_methods,
            blocks={},
            hashes={},
        )

        for block in self.target_blocks:
            snippet = blocks.get(block, self._placeholder(block))
            parsed.blocks[block] = snippet
            parsed.hashes[block] = self._hash(snippet)

        return parsed

    def assemble(
        self,
        parsed: ParsedAlgorithm,
        overrides: Optional[Dict[str, str]] = None,
        class_name: Optional[str] = None,
    ) -> str:
        """Rebuild a Python module from ``parsed`` plus optional block overrides."""

        overrides = overrides or {}
        class_lines: List[str] = []

        if parsed.docstring:
            doc = textwrap.indent(f'"""{parsed.docstring}"""', "    ")
            class_lines.append(doc)

        for method in parsed.other_methods:
            class_lines.append(self._indent(method))

        for block in self.target_blocks:
            snippet = overrides.get(block, parsed.blocks.get(block) or self._placeholder(block))
            class_lines.append(self._indent(snippet))

        body = "\n\n".join(line.rstrip() for line in class_lines) or "    pass"
        cls_name = class_name or parsed.class_name
        class_code = f"class {cls_name}:\n{body}"

        sections = [
            "\n".join(parsed.imports).strip(),
            "\n\n".join(parsed.helpers).strip(),
            class_code,
        ]
        return "\n\n".join(section for section in sections if section.strip()) + "\n"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _select_primary_class(
        self, class_defs: List[ast.ClassDef]
    ) -> Optional[ast.ClassDef]:
        if not class_defs:
            return None
        for candidate in class_defs:
            if any(isinstance(node, ast.FunctionDef) and node.name == "__call__" for node in candidate.body):
                return candidate
        return class_defs[0]

    def _hash(self, snippet: str) -> str:
        return hashlib.sha256(snippet.encode("utf-8")).hexdigest()

    def _placeholder(self, block: str) -> str:
        placeholder = PLACEHOLDER_SIGNATURES.get(
            block,
            f"def {block}(self, *args, **kwargs):\n        return None",
        )
        doc = '"""Placeholder generated by MADA."""'
        if "\n" in placeholder:
            lines = placeholder.splitlines()
            header = lines[0]
            body = "\n".join(lines[1:])
            return f"{header}\n        {doc}\n{body if body else '        return None'}"
        return f"{placeholder}\n        {doc}\n        return None"

    def _indent(self, snippet: str) -> str:
        return textwrap.indent(textwrap.dedent(snippet).strip() + "\n", "    ")

    def _dedent(self, snippet: str) -> str:
        return textwrap.dedent(snippet).strip()

    def _unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except AttributeError:
            # Python < 3.9 fallback (should not trigger, but kept for safety)
            return ""

    def _synthetic_algorithm(self) -> ParsedAlgorithm:
        blocks = {name: self._placeholder(name) for name in self.target_blocks}
        hashes = {name: self._hash(code) for name, code in blocks.items()}
        helpers = ["def evaluate_candidate(self, candidate):\n    return candidate"]
        other_methods = [
            "def __init__(self, budget=10000):\n    self.budget = budget\n    self.history = []",
            "def __call__(self, func):\n    return getattr(self, 'budget', 0)",
        ]
        return ParsedAlgorithm(
            class_name="MADAOptimizer",
            imports=["import numpy as np"],
            helpers=helpers,
            other_methods=other_methods,
            blocks=blocks,
            hashes=hashes,
        )



