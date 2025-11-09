"""
Semantic equivalence clustering for self-consistency evaluation.

This module implements clustering of candidate solutions by semantic equivalence,
which is used to compute self-consistency rewards. Candidates are grouped based on:
1. Tool evaluation results (e.g., which tests they pass)
2. Normalized output equivalence (for math/QA tasks)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class Cluster:
    """Represents a cluster of semantically equivalent candidates.

    Attributes:
        members: List of candidate indices in this cluster
        size: Number of candidates in the cluster
        avg_tool_score: Average tool evaluation score across cluster members
        is_tool_correct: Whether cluster members are considered tool-correct
        representative_idx: Index of the representative candidate (usually first)
        tool_signature: Tuple representing the tool evaluation signature (e.g., which tests passed)
    """
    members: List[int]
    size: int
    avg_tool_score: float
    is_tool_correct: bool
    representative_idx: int
    tool_signature: Optional[Tuple] = None

    def fraction_of_total(self, total: int) -> float:
        """Compute fraction of total candidates in this cluster."""
        return self.size / total if total > 0 else 0.0


def normalize_text(text: str) -> str:
    """Normalize text for equivalence comparison.

    Performs the following normalization:
    - Lowercase
    - Strip leading/trailing whitespace
    - Remove extra internal whitespace
    - Remove common punctuation (for math answers)

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text string
    """
    # Lowercase and strip
    text = text.lower().strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # For math answers, strip common punctuation
    text = text.replace(',', '').replace('$', '').replace('%', '')

    return text


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer from text.

    Handles common formats:
    - Plain numbers: "42", "3.14"
    - Boxed answers: "\\boxed{42}"
    - Answer statements: "The answer is 42"
    - Fractions: "1/2" -> 0.5

    Args:
        text: Text potentially containing a numeric answer

    Returns:
        Extracted numeric value or None if not found
    """
    # Try boxed answer first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        text = boxed_match.group(1)

    # Try "answer is X" pattern
    answer_match = re.search(r'answer is:?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if answer_match:
        try:
            return float(answer_match.group(1))
        except ValueError:
            pass

    # Try fraction pattern
    fraction_match = re.search(r'([+-]?\d+)/(\d+)', text)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                return numerator / denominator
        except ValueError:
            pass

    # Try plain number (most permissive)
    number_match = re.search(r'([+-]?\d+\.?\d*)', text)
    if number_match:
        try:
            return float(number_match.group(1))
        except ValueError:
            pass

    return None


def create_tool_signature(tool_result: Dict[str, Any]) -> Tuple:
    """Create a hashable signature from tool evaluation result.

    This signature is used to group candidates with identical tool behavior.

    Args:
        tool_result: Tool evaluation result dictionary

    Returns:
        Tuple representing the tool signature
    """
    score = tool_result.get('score', 0.0)
    passed = tool_result.get('passed', False)

    # For code tests, include which tests passed
    if 'test_results' in tool_result:
        test_results = tuple(tool_result['test_results'])
        return (passed, score, test_results)

    # For simple pass/fail
    return (passed, round(score, 4))


def cluster_by_tool_behavior(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
) -> List[Cluster]:
    """Cluster candidates primarily by tool evaluation behavior.

    Candidates with identical tool signatures are grouped together.
    This is the primary clustering strategy for code tasks.

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation results (one per candidate)

    Returns:
        List of Cluster objects
    """
    signature_to_indices: Dict[Tuple, List[int]] = {}

    for idx, tool_result in enumerate(tool_results):
        signature = create_tool_signature(tool_result)
        if signature not in signature_to_indices:
            signature_to_indices[signature] = []
        signature_to_indices[signature].append(idx)

    clusters = []
    for signature, indices in signature_to_indices.items():
        # Compute average tool score for this cluster
        scores = [tool_results[idx].get('score', 0.0) for idx in indices]
        avg_score = np.mean(scores)

        # Determine if cluster is tool-correct
        # Consider correct if avg score > 0.8 or all explicitly passed
        is_correct = avg_score > 0.8 or all(
            tool_results[idx].get('passed', False) for idx in indices
        )

        cluster = Cluster(
            members=indices,
            size=len(indices),
            avg_tool_score=avg_score,
            is_tool_correct=is_correct,
            representative_idx=indices[0],
            tool_signature=signature,
        )
        clusters.append(cluster)

    return clusters


def cluster_by_normalized_text(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
) -> List[Cluster]:
    """Cluster candidates by normalized text equivalence.

    Used for tasks where the answer format is standardized (math, QA).
    Candidates with identical normalized text are grouped together.

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation results (one per candidate)

    Returns:
        List of Cluster objects
    """
    normalized_to_indices: Dict[str, List[int]] = {}

    for idx, candidate in enumerate(candidates):
        normalized = normalize_text(candidate)
        if normalized not in normalized_to_indices:
            normalized_to_indices[normalized] = []
        normalized_to_indices[normalized].append(idx)

    clusters = []
    for normalized_text, indices in normalized_to_indices.items():
        # Compute average tool score for this cluster
        scores = [tool_results[idx].get('score', 0.0) for idx in indices]
        avg_score = np.mean(scores)

        # Determine if cluster is tool-correct
        is_correct = avg_score > 0.8 or all(
            tool_results[idx].get('passed', False) for idx in indices
        )

        cluster = Cluster(
            members=indices,
            size=len(indices),
            avg_tool_score=avg_score,
            is_tool_correct=is_correct,
            representative_idx=indices[0],
        )
        clusters.append(cluster)

    return clusters


def cluster_by_numeric_answer(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
    tolerance: float = 1e-6,
) -> List[Cluster]:
    """Cluster candidates by extracted numeric answer.

    Used for math tasks. Extracts numeric answers and groups
    candidates with the same (or very close) numeric values.

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation results
        tolerance: Tolerance for numeric equality

    Returns:
        List of Cluster objects
    """
    # Extract numeric answers
    numeric_answers = [extract_numeric_answer(cand) for cand in candidates]

    # Group by numeric value
    answer_to_indices: Dict[Optional[float], List[int]] = {}

    for idx, answer in enumerate(numeric_answers):
        # Find existing cluster with close answer
        found = False
        if answer is not None:
            for existing_answer in answer_to_indices.keys():
                if existing_answer is not None and abs(answer - existing_answer) < tolerance:
                    answer_to_indices[existing_answer].append(idx)
                    found = True
                    break

        if not found:
            if answer not in answer_to_indices:
                answer_to_indices[answer] = []
            answer_to_indices[answer].append(idx)

    clusters = []
    for answer, indices in answer_to_indices.items():
        # Compute average tool score for this cluster
        scores = [tool_results[idx].get('score', 0.0) for idx in indices]
        avg_score = np.mean(scores)

        # Determine if cluster is tool-correct
        is_correct = avg_score > 0.8 or all(
            tool_results[idx].get('passed', False) for idx in indices
        )

        cluster = Cluster(
            members=indices,
            size=len(indices),
            avg_tool_score=avg_score,
            is_tool_correct=is_correct,
            representative_idx=indices[0],
        )
        clusters.append(cluster)

    return clusters


def cluster_candidates(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
    mode: str = 'auto',
) -> List[Cluster]:
    """Cluster candidate solutions by semantic equivalence.

    This is the main entry point for clustering. The clustering strategy
    is determined by the mode parameter:

    - 'tool': Cluster by tool evaluation signature (for code)
    - 'text': Cluster by normalized text (for QA)
    - 'numeric': Cluster by extracted numeric answer (for math)
    - 'auto': Automatically detect based on tool results

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation result dictionaries
        mode: Clustering mode

    Returns:
        List of Cluster objects, sorted by size (descending)

    Raises:
        ValueError: If candidates and tool_results have different lengths
    """
    if len(candidates) != len(tool_results):
        raise ValueError(
            f"Candidates ({len(candidates)}) and tool_results ({len(tool_results)}) "
            f"must have the same length"
        )

    if len(candidates) == 0:
        return []

    # Auto-detect mode based on tool results
    if mode == 'auto':
        # If tool results have test_results, use tool-based clustering
        if any('test_results' in tr for tr in tool_results):
            mode = 'tool'
        # If candidates look like math (contain numbers/boxed), use numeric
        elif any(extract_numeric_answer(c) is not None for c in candidates):
            mode = 'numeric'
        # Otherwise, use text-based
        else:
            mode = 'text'

    # Apply appropriate clustering strategy
    if mode == 'tool':
        clusters = cluster_by_tool_behavior(candidates, tool_results)
    elif mode == 'numeric':
        clusters = cluster_by_numeric_answer(candidates, tool_results)
    elif mode == 'text':
        clusters = cluster_by_normalized_text(candidates, tool_results)
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")

    # Sort clusters by size (descending)
    clusters.sort(key=lambda c: c.size, reverse=True)

    return clusters
