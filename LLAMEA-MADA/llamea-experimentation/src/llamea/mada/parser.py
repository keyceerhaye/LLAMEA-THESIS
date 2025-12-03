"""AST utilities for splitting and assembling optimizer blocks."""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

PLACEHOLDER_SIGNATURES = {
    "parent_selection": "def parent_selection(self, population):\n        return population",
    "recombination": "def recombination(self, parents):\n        return parents[0]",
    "mutation": "def mutation(self, candidate):\n        return candidate",
    "survivor_selection": (
        "def survivor_selection(self, population, offspring):\n        return population"
    ),
}

ROLE_HINTS: Dict[str, Dict[str, Tuple[Tuple[str, float], ...]]] = {
    "parent_selection": {
        "names": (
            ("parent", 0.8),
            ("select", 0.6),
            ("choose", 0.5),
            ("mate", 0.4),
        ),
        "args": (("population", 0.6), ("parents", 0.6), ("fitness", 0.2)),
        "keywords": (("fitness", 0.3), ("selection", 0.4), ("tournament", 0.3)),
    },
    "recombination": {
        "names": (
            ("recomb", 0.9),
            ("crossover", 0.9),
            ("mate", 0.5),
            ("combine", 0.5),
        ),
        "args": (("parents", 0.6), ("offspring", 0.4), ("child", 0.4)),
        "keywords": (("blend", 0.3), ("averag", 0.3), ("intermediate", 0.3)),
    },
    "mutation": {
        "names": (("mutat", 1.0), ("perturb", 0.4), ("explore", 0.3)),
        "args": (("candidate", 0.6), ("solution", 0.4), ("point", 0.2)),
        "keywords": (("noise", 0.4), ("gaussian", 0.4), ("perturb", 0.4)),
    },
    "survivor_selection": {
        "names": (
            ("surviv", 0.9),
            ("replace", 0.5),
            ("elit", 0.4),
            ("environment", 0.3),
        ),
        "args": (("population", 0.5), ("offspring", 0.5), ("archive", 0.4)),
        "keywords": (
            ("truncate", 0.4),
            ("rank", 0.4),
            ("fitness", 0.3),
            ("selection", 0.3),
        ),
    },
}


@dataclass
class ParsedAlgorithm:
    """Container for all structural parts needed to rebuild an algorithm."""

    class_name: str
    imports: List[str] = field(default_factory=list)
    helpers: Dict[str, str] = field(default_factory=dict)
    docstring: Optional[str] = None
    other_methods: Dict[str, str] = field(default_factory=dict)
    blocks: Dict[str, str] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)
    method_roles: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    helper_groups: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    block_sources: Dict[str, str] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    placeholder_reasons: Dict[str, str] = field(default_factory=dict)
    init_attributes: Set[str] = field(default_factory=set)
    init_assignments: Dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, object]:
        """Return a serialisable representation for Solution.metadata."""

        def _serialise(dep_map: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Dict[str, List[str]]]:
            serialised: Dict[str, Dict[str, List[str]]] = {}
            for name, entry in dep_map.items():
                serialised[name] = {key: sorted(value) for key, value in entry.items()}
            return serialised

        return {
            "class_name": self.class_name,
            "hashes": self.hashes,
            "blocks": {
                name: {"code": code, "hash": self.hashes[name]}
                for name, code in self.blocks.items()
            },
            "method_roles": self.method_roles,
            "block_sources": self.block_sources,
            "dependencies": _serialise(self.dependencies),
            "helper_groups": _serialise(self.helper_groups),
            "coverage": self.coverage,
            "placeholder_reasons": self.placeholder_reasons,
        }


class BlockParser:
    """Splits optimizer code into reusable AST-backed blocks."""

    def __init__(
        self,
        target_blocks: Optional[List[str]] = None,
        llm_labeler: Optional[Callable[[str], Optional[str]]] = None,
        enable_llm_labeler: bool = False,
        role_threshold: float = 1.0,
    ):
        self.target_blocks = target_blocks or list(PLACEHOLDER_SIGNATURES.keys())
        self.llm_labeler = llm_labeler
        self.enable_llm_labeler = enable_llm_labeler
        self.role_threshold = role_threshold

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
        imported_symbols: Set[str] = set()
        helpers: Dict[str, str] = {}
        class_defs: List[ast.ClassDef] = []

        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(self._unparse(node))
                imported_symbols.update(self._collect_import_names(node))
            elif isinstance(node, ast.FunctionDef):
                helpers[node.name] = self._dedent(self._unparse(node))
            elif isinstance(node, ast.ClassDef):
                class_defs.append(node)

        class_def = self._select_primary_class(class_defs)
        if class_def is None:
            return self._synthetic_algorithm()

        method_sources: Dict[str, str] = {}
        method_nodes: Dict[str, ast.FunctionDef] = {}
        method_roles: Dict[str, str] = {}
        method_dependencies: Dict[str, Dict[str, Set[str]]] = {}
        candidate_map: Dict[str, Dict[str, object]] = {}
        ambiguous_roles: Set[str] = set()

        for element in class_def.body:
            if not isinstance(element, ast.FunctionDef):
                continue
            snippet = self._dedent(self._unparse(element))
            method_sources[element.name] = snippet
            method_nodes[element.name] = element
            dependencies = self._analyze_dependencies(element, imported_symbols)
            method_dependencies[element.name] = dependencies

            if element.name in self.target_blocks:
                candidate_map[element.name] = {
                    "score": 3.0,
                    "method": element.name,
                    "origin": "direct",
                }
                method_roles[element.name] = element.name
                continue

            try:
                role, score, origin = self._classify_method(element, snippet)
            except Exception:
                role, score, origin = None, 0.0, "error"

            if role:
                if score >= self.role_threshold:
                    method_roles[element.name] = role
                else:
                    method_roles[element.name] = "helper"
                existing = candidate_map.get(role)
                if existing is None or score > existing["score"]:
                    candidate_map[role] = {
                        "score": score,
                        "method": element.name,
                        "origin": origin,
                    }
                if score < self.role_threshold and origin != "direct":
                    ambiguous_roles.add(role)
            else:
                method_roles[element.name] = "helper"

        docstring = ast.get_docstring(class_def)
        parsed = ParsedAlgorithm(
            class_name=class_def.name,
            imports=imports,
            helpers=helpers,
            docstring=docstring,
            other_methods={},
            blocks={},
            hashes={},
            method_roles=method_roles,
            dependencies=method_dependencies,
        )

        assigned_methods: Set[str] = set()
        placeholder_count = 0

        for block in self.target_blocks:
            candidate = candidate_map.get(block)
            if candidate and (
                candidate["origin"] == "direct"
                or candidate["score"] >= self.role_threshold
            ):
                method_name = candidate["method"]
                snippet = method_sources.get(method_name, self._placeholder(block))
                parsed.blocks[block] = snippet
                parsed.hashes[block] = self._hash(snippet)
                parsed.block_sources[block] = method_name
                assigned_methods.add(method_name)
                dependencies = method_dependencies.get(method_name, {})
                parsed.helper_groups[block] = {
                    "helpers": set(dependencies.get("helpers", set())),
                    "attributes": set(dependencies.get("attributes", set())),
                    "imports": set(dependencies.get("imports", set())),
                }
            else:
                placeholder = self._placeholder(block)
                parsed.blocks[block] = placeholder
                parsed.hashes[block] = self._hash(placeholder)
                placeholder_count += 1
                parsed.helper_groups[block] = {
                    "helpers": set(),
                    "attributes": set(),
                    "imports": set(),
                }
                parsed.block_sources[block] = "__placeholder__"
                if block in ambiguous_roles:
                    parsed.placeholder_reasons[block] = "ambiguous_classification"
                else:
                    parsed.placeholder_reasons[block] = "no_candidate"

        for element in class_def.body:
            if not isinstance(element, ast.FunctionDef):
                continue
            if element.name in assigned_methods:
                continue
            snippet = method_sources[element.name]
            parsed.other_methods[element.name] = snippet

            if element.name == "__init__":
                assignments = self._collect_init_assignments(element)
                parsed.init_assignments.update(assignments)
                parsed.init_attributes.update(assignments.keys())

        total_blocks = len(self.target_blocks) or 1
        parsed.coverage = {
            "total_blocks": len(self.target_blocks),
            "real_blocks": len(self.target_blocks) - placeholder_count,
            "placeholder_blocks": placeholder_count,
            "placeholder_ratio": placeholder_count / total_blocks,
            "helper_methods": len(parsed.other_methods),
            "module_helpers": len(parsed.helpers),
        }

        return parsed

    def assemble(
        self,
        parsed: ParsedAlgorithm,
        overrides: Optional[Dict[str, str]] = None,
        class_name: Optional[str] = None,
        helper_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        """Rebuild a Python module from ``parsed`` plus optional block overrides."""

        overrides = overrides or {}
        helper_overrides = helper_overrides or {}
        merged_methods: Dict[str, str] = dict(parsed.other_methods)
        class_lines: List[str] = []

        extra_methods = helper_overrides.get("methods", {})
        for name, snippet in extra_methods.items():
            if name not in merged_methods and name not in self.target_blocks:
                merged_methods[name] = self._dedent(snippet)

        init_assignments = helper_overrides.get("init_assignments", {})
        if init_assignments:
            merged_methods = self._ensure_init_assignments(merged_methods, init_assignments)

        if parsed.docstring:
            doc = textwrap.indent(f'"""{parsed.docstring}"""', "    ")
            class_lines.append(doc)

        for snippet in merged_methods.values():
            class_lines.append(self._indent(snippet))

        for block in self.target_blocks:
            snippet = overrides.get(block, parsed.blocks.get(block) or self._placeholder(block))
            class_lines.append(self._indent(snippet))

        body = "\n\n".join(line.rstrip() for line in class_lines) or "    pass"
        cls_name = class_name or parsed.class_name
        class_code = f"class {cls_name}:\n{body}"

        sections = [
            "\n".join(parsed.imports).strip(),
            "\n\n".join(parsed.helpers.values()).strip(),
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

    def _collect_import_names(self, node: ast.AST) -> Set[str]:
        names: Set[str] = set()
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        return names

    def _analyze_dependencies(
        self,
        node: ast.FunctionDef,
        imported_symbols: Set[str],
    ) -> Dict[str, Set[str]]:
        visitor = _DependencyVisitor(imported_symbols)
        visitor.visit(node)
        return {
            "helpers": visitor.helper_calls,
            "attributes": visitor.attributes,
            "imports": visitor.imports,
        }

    def _collect_init_assignments(self, node: ast.FunctionDef) -> Dict[str, str]:
        assignments: Dict[str, str] = {}
        for stmt in ast.walk(node):
            target = None
            if isinstance(stmt, ast.Assign):
                for possible in stmt.targets:
                    attr_name = self._attr_name(possible)
                    if attr_name:
                        target = attr_name
                        break
            elif isinstance(stmt, ast.AnnAssign):
                target = self._attr_name(stmt.target)
            if target:
                try:
                    assignments[target] = self._dedent(self._unparse(stmt))
                except Exception:
                    assignments[target] = f"self.{target} = None"
        return assignments

    def _attr_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr
        return None

    def _classify_method(
        self,
        node: ast.FunctionDef,
        snippet: str,
    ) -> Tuple[Optional[str], float, str]:
        name = node.name.lower()
        if name in self.target_blocks:
            return name, 3.0, "direct"

        arg_names = [arg.arg.lower() for arg in node.args.args[1:]]
        body = snippet.lower()
        best_role: Optional[str] = None
        best_score = 0.0

        for role, hints in ROLE_HINTS.items():
            score = 0.0
            for token, weight in hints["names"]:
                if token in name:
                    score += weight
            for token, weight in hints["args"]:
                if any(token in arg for arg in arg_names):
                    score += weight
            for token, weight in hints["keywords"]:
                if token in body:
                    score += weight
            if score > best_score:
                best_role = role
                best_score = score

        if best_role and best_score >= self.role_threshold:
            return best_role, best_score, "heuristic"

        if self.enable_llm_labeler and self.llm_labeler:
            label = self._llm_labeler(snippet)
            if label in self.target_blocks:
                return label, max(best_score, self.role_threshold), "llm"

        return best_role, best_score, "heuristic"

    def _llm_labeler(self, snippet: str) -> Optional[str]:
        if not self.llm_labeler:
            return None
        try:
            return self.llm_labeler(snippet)
        except Exception:
            return None

    def _ensure_init_assignments(
        self,
        methods: Dict[str, str],
        new_assignments: Dict[str, str],
    ) -> Dict[str, str]:
        if "__init__" not in methods:
            lines = ["def __init__(self, *args, **kwargs):", "    super().__init__(*args, **kwargs)"]
            for stmt in new_assignments.values():
                lines.append(f"    {stmt}")
            methods["__init__"] = "\n".join(lines)
            return methods

        snippet = methods["__init__"]
        try:
            parsed = ast.parse(textwrap.dedent(snippet))
            func_def = next(
                (node for node in parsed.body if isinstance(node, ast.FunctionDef)),
                None,
            )
            if func_def is None:
                raise ValueError
            existing_attrs: Set[str] = set()
            for stmt in ast.walk(func_def):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        attr_name = self._attr_name(target)
                        if attr_name:
                            existing_attrs.add(attr_name)
                elif isinstance(stmt, ast.AnnAssign):
                    attr_name = self._attr_name(stmt.target)
                    if attr_name:
                        existing_attrs.add(attr_name)
            for attr, stmt in new_assignments.items():
                if attr in existing_attrs:
                    continue
                func_def.body.append(ast.parse(stmt).body[0])
            methods["__init__"] = self._dedent(ast.unparse(func_def))
            return methods
        except Exception:
            buffer = [snippet.rstrip(), ""]
            for stmt in new_assignments.values():
                buffer.append(f"        {stmt.strip()}")
            methods["__init__"] = "\n".join(buffer)
            return methods

    def _synthetic_algorithm(self) -> ParsedAlgorithm:
        blocks = {name: self._placeholder(name) for name in self.target_blocks}
        hashes = {name: self._hash(code) for name, code in blocks.items()}
        helpers = {"evaluate_candidate": "def evaluate_candidate(self, candidate):\n    return candidate"}
        other_methods = {
            "__init__": "def __init__(self, budget=10000):\n    self.budget = budget\n    self.history = []",
            "__call__": "def __call__(self, func):\n    return getattr(self, 'budget', 0)",
        }
        return ParsedAlgorithm(
            class_name="MADAOptimizer",
            imports=["import numpy as np"],
            helpers=helpers,
            other_methods=other_methods,
            blocks=blocks,
            hashes=hashes,
            helper_groups={block: {"helpers": set(), "attributes": set(), "imports": set()} for block in blocks},
            coverage={
                "total_blocks": len(self.target_blocks),
                "real_blocks": 0,
                "placeholder_blocks": len(self.target_blocks),
                "placeholder_ratio": 1.0,
                "helper_methods": len(other_methods),
                "module_helpers": len(helpers),
            },
            placeholder_reasons={block: "synthetic" for block in blocks},
        )


class _DependencyVisitor(ast.NodeVisitor):
    """Collects helper/attribute usage for a method."""

    def __init__(self, imported_symbols: Set[str]):
        self.imported_symbols = imported_symbols
        self.helper_calls: Set[str] = set()
        self.attributes: Set[str] = set()
        self.imports: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self":
                self.helper_calls.add(func.attr)
                # Skip visiting the attribute itself so it is not counted as a state access.
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

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.imported_symbols:
            self.imports.add(node.id)

