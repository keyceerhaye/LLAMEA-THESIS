"""Utility helpers for validating optimizer snippets with AST analysis."""

from __future__ import annotations

import ast
import builtins
from typing import Iterable, List, Optional, Set

_DEFAULT_ALLOWED_NAMES: Set[str] = set(dir(builtins))
_COMMON_LIBS = {"np", "numpy", "math", "random", "statistics", "itertools"}


def default_allowed_names(extra: Optional[Iterable[str]] = None) -> Set[str]:
    """Return the baseline set of identifiers allowed without local bindings."""

    allowed = set(_DEFAULT_ALLOWED_NAMES)
    allowed.update(_COMMON_LIBS)
    if extra:
        allowed.update(extra)
    return allowed


class _UndefinedNameVisitor(ast.NodeVisitor):
    """Track Name loads that were never defined within the current scopes."""

    def __init__(self, allowed: Set[str]):
        self.allowed = allowed
        self.invalid: Set[str] = set()
        self._scopes: List[Set[str]] = [set()]

    # ------------------------------------------------------------------ #
    # Scope helpers
    # ------------------------------------------------------------------ #
    def _push_scope(self, initial=None):
        self._scopes.append(set(initial or ()))

    def _pop_scope(self):
        self._scopes.pop()

    def _define(self, name: Optional[str]):
        if not name:
            return
        self._scopes[-1].add(name)

    def _is_defined(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self._scopes))

    def _define_from_target(self, target: ast.AST):
        if isinstance(target, ast.Name):
            self._define(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_from_target(elt)
        elif isinstance(target, ast.Starred):
            self._define_from_target(target.value)
        # Attributes (e.g., self.x) are intentional and should not register

    # ------------------------------------------------------------------ #
    # Node visitors
    # ------------------------------------------------------------------ #
    def visit_FunctionDef(self, node: ast.FunctionDef):
        params = {arg.arg for arg in node.args.args}
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        params.update(arg.arg for arg in node.args.kwonlyargs)
        self._push_scope(params)
        self.generic_visit(node)
        self._pop_scope()

    def visit_Lambda(self, node: ast.Lambda):
        params = {arg.arg for arg in node.args.args}
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        self._push_scope(params)
        self.generic_visit(node)
        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef):
        self._define(node.name)
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for target in node.targets:
            self._define_from_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value:
            self.visit(node.value)
        self._define_from_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.value)
        # AugAssign target is already defined; no need to re-register

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self._define_from_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_comprehension(self, node: ast.comprehension):
        self.visit(node.iter)
        self._define_from_target(node.target)
        for if_clause in node.ifs:
            self.visit(if_clause)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._define_from_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._define(name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            self._define(name)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            if not self._is_defined(node.id) and node.id not in self.allowed:
                self.invalid.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._define(node.id)
        self.generic_visit(node)


def find_undefined_names(
    source: str,
    allowed_names: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Return the set of identifier names referenced in ``source`` without a binding.

    Args:
        source: Python snippet (function/class/module) to inspect.
        allowed_names: Optional pre-approved globals (defaults include builtins, np, math).
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Syntax errors are handled elsewhere; validation should not duplicate that work.
        return set()

    visitor = _UndefinedNameVisitor(allowed_names or default_allowed_names())
    visitor.visit(tree)
    return visitor.invalid


def detect_population_contract_issues(source: str) -> List[str]:
    """
    Inspect optimizer code and report tuple contract violations in block methods.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    issues: List[str] = []
    visitor = _PopulationContractVisitor(issues)
    visitor.visit(tree)
    return issues


class _PopulationContractVisitor(ast.NodeVisitor):
    """Locate block methods and collect tuple contract issues."""

    _TARGET_BLOCKS = {
        "parent_selection",
        "recombination",
        "mutation",
        "survivor_selection",
    }

    def __init__(self, issues: List[str]):
        self.issues = issues

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name not in self._TARGET_BLOCKS:
            return
        analyzer = _BlockContractAnalyzer(node.name)
        analyzer.visit(node)
        self.issues.extend(analyzer.issues)


class _BlockContractAnalyzer(ast.NodeVisitor):
    """Check a specific block method for tuple handling issues."""

    _COLLECTION_NAMES = {"population", "offspring", "parents"}
    _TUPLE_TOKENS = {"candidate", "fitness"}
    _CALL_GUARDS = {
        "func",
        "self.func",
        "self.mutation",
        "self.recombination",
        "mutation",
        "recombination",
    }

    def __init__(self, block_name: str):
        self.block_name = block_name
        self.issues: List[str] = []
        self._forbidden_names = {"archive", "arch_cand", "arch_fit"}

    def visit_For(self, node: ast.For):
        iter_name = _reference_name(node.iter)
        if iter_name in self._COLLECTION_NAMES:
            if not isinstance(node.target, (ast.Tuple, ast.List)):
                self.issues.append(f"missing_destructuring:{self.block_name}:{iter_name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_label = _call_name(node.func)
        if func_label in self._CALL_GUARDS and any(
            _is_candidate_tuple(arg, self._TUPLE_TOKENS) for arg in node.args
        ):
            self.issues.append(f"tuple_call:{self.block_name}:{func_label}")
        # flag population.sort() and similar on tuple collections
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "sort"
            and _reference_name(node.func.value) in self._COLLECTION_NAMES
        ):
            self.issues.append(f"forbidden_sort:{self.block_name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self._forbidden_names:
            self.issues.append(f"forbidden_archive:{self.block_name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in {
            "archive",
            "arch_cand",
            "arch_fit",
        }:
            self.issues.append(f"forbidden_archive:{self.block_name}")
        self.generic_visit(node)


def _reference_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        value = _call_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_candidate_tuple(node: ast.AST, tokens: Set[str]) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return False
    if len(node.elts) != 2:
        return False
    names = []
    for elt in node.elts:
        if isinstance(elt, ast.Name):
            names.append(elt.id)
    return set(names) >= tokens