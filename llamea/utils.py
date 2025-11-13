import ast
import re
from difflib import SequenceMatcher
from typing import List
import numpy as np


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    """Raise a timeout exception"""
    raise TimeoutError


def apply_unified_diff(text: str, diff: str) -> str:
    """
    Apply a unified diff to the given text using pure Python implementation.

    This function parses and applies unified diff patches without relying on
    external system commands, making it cross-platform compatible.

    Args:
        text: The original text to patch.
        diff: The unified diff (as produced by `git diff`, `difflib.unified_diff`, etc.).

    Returns:
        The patched text as a string.

    Raises:
        ValueError: If the diff format is invalid or cannot be applied.
    """
    
    # Ensure text ends with newline for consistent processing
    if not text.endswith("\n"):
        text += "\n"

    # Normalize diff format
    d = diff.lstrip()
    if not d.startswith("--- "):
        diff = f"--- a\n+++ a\n{diff}"

    # Ensure diff ends with newline
    if not diff.endswith("\n"):
        diff += "\n"

    lines = text.splitlines(keepends=True)
    diff_lines = diff.splitlines()
    
    # Parse the diff header
    i = 0
    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
        i += 1
    
    if i >= len(diff_lines):
        # No hunks found, return original text
        return text
    
    result_lines = lines[:]
    
    # Process each hunk
    while i < len(diff_lines):
        if not diff_lines[i].startswith("@@"):
            i += 1
            continue
            
        # Parse hunk header: @@ -start,count +start,count @@
        hunk_header = diff_lines[i]
        i += 1
        
        # Extract line numbers from hunk header
        import re
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", hunk_header)
        if not match:
            raise ValueError(f"Invalid hunk header: {hunk_header}")
        
        old_start = int(match.group(1)) - 1  # Convert to 0-based indexing
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3)) - 1  # Convert to 0-based indexing
        new_count = int(match.group(4)) if match.group(4) else 1
        
        # Collect hunk lines
        hunk_lines = []
        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            if diff_lines[i].startswith((" ", "+", "-")):
                hunk_lines.append(diff_lines[i])
            i += 1
        
        # Apply the hunk
        result_lines = _apply_hunk(result_lines, hunk_lines, old_start, old_count)
    
    return "".join(result_lines)


def _apply_hunk(lines: List[str], hunk_lines: List[str], old_start: int, old_count: int) -> List[str]:
    """Apply a single hunk to the lines."""
    result = lines[:old_start]
    
    hunk_pos = 0
    old_pos = old_start
    
    while hunk_pos < len(hunk_lines):
        hunk_line = hunk_lines[hunk_pos]
        
        if hunk_line.startswith(" "):
            # Context line - should match
            expected = hunk_line[1:]
            if not expected.endswith("\n"):
                expected += "\n"
            
            if old_pos < len(lines) and lines[old_pos].rstrip() == expected.rstrip():
                result.append(lines[old_pos])
                old_pos += 1
            else:
                result.append(expected)
                old_pos += 1
            
        elif hunk_line.startswith("-"):
            # Deletion - skip the original line
            old_pos += 1
            
        elif hunk_line.startswith("+"):
            # Addition - add the new line
            new_line = hunk_line[1:]
            if not new_line.endswith("\n"):
                new_line += "\n"
            result.append(new_line)
        
        hunk_pos += 1
    
    # Add remaining lines after the hunk
    result.extend(lines[old_pos:])
    
    return result


def discrete_power_law_distribution(n, beta):
    """
    Power law distribution function from:
    # Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen. 2017.
    # Fast genetic algorithms.
    # In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
    # Association for Computing Machinery, New York, NY, USA, 777–784.
    # https://doi.org/10.1145/3071178.3071301
    """

    def discrete_power_law(n, alpha, beta):
        half_n = int(n / 2)
        C_beta_half_n = 0
        for i in range(1, half_n + 1):
            C_beta_half_n += i ** (-beta)
        probability_alpha = C_beta_half_n ** (-1) * alpha ** (-beta)
        return probability_alpha

    half_n = int(n / 2)
    elements = [alpha for alpha in range(1, half_n + 1)]
    probabilities = [discrete_power_law(n, alpha, beta) for alpha in elements]
    if elements == []:
        return 0.05
    else:
        sample = np.random.choice(elements, p=probabilities)
        return sample / n


def code_distance(a, b):
    """Return a rough distance between two solutions based on their ASTs.

    The function accepts either :class:`Solution` objects or raw code strings
    and computes ``1 - similarity`` of their abstract syntax trees using
    :class:`difflib.SequenceMatcher` on the dumped AST representations.
    ``1.0`` is returned on parsing errors or when the inputs cannot be
    processed.

    Args:
        a: The first solution or Python source code.
        b: The second solution or Python source code.

    Returns:
        float: A value in ``[0, 1]`` indicating dissimilarity of the code.
    """

    code_a = getattr(a, "code", a)
    code_b = getattr(b, "code", b)
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
        return 1 - SequenceMatcher(None, ast.dump(tree_a), ast.dump(tree_b)).ratio()
    except Exception:
        return 1.0
