# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Wilton Moore

"""
Core fuzzy entity matching functionality.

High-performance fuzzy matching with normalization, caching, and multiple algorithms.
Optimized for entity deduplication with proper memory management.

Copyright (c) 2024 Will Moore
Licensed under the MIT License - see LICENSE file for details.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

try:
    import rapidfuzz
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    import difflib

try:
    from cachetools import LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False


# Type definitions for better API clarity
Algorithm = Literal["auto", "ratio", "partial", "token", "levenshtein"]


class FuzzyMatcherError(Exception):
    """Base exception for fuzzy matching operations."""

    def __init__(self, message: str, code: str = "unknown"):
        super().__init__(message)
        self.code = code


@dataclass(slots=True, frozen=True)
class MatchResult:
    """Result from a fuzzy matching operation."""

    text: str
    score: float  # Always 0-1 scale
    algorithm: str
    original_index: Optional[int] = None


@dataclass(slots=True, frozen=True)
class CacheStats:
    """Statistics about cache performance."""

    hit_rate: float
    size: int
    max_size: int
    hits: int
    misses: int


# Capped caches with LRU eviction
if CACHETOOLS_AVAILABLE:
    _match_cache = LRUCache(maxsize=50_000)
    _preprocess_cache = LRUCache(maxsize=100_000)
else:
    # Fallback to simple dicts with manual size management
    _match_cache = {}
    _preprocess_cache = {}

_cache_hits = 0
_cache_misses = 0


def find_matches(
    query: str,
    candidates: List[str],
    threshold: float = 80.0,
    include_scores: bool = False,
    limit: int = 10,
    algorithm: Algorithm = "auto",
    normalize: bool = True,
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Find fuzzy matches for a query string against candidate strings.

    Uses RapidFuzz (fast FuzzyWuzzy-compatible) with intelligent algorithm selection.
    Default strategy: token-based matching for dirty inputs, ratio for clean names.

    Args:
        query: String to find matches for
        candidates: List of candidate strings to match against
        threshold: Minimum similarity score (0-100, default 80.0)
        include_scores: If True, return (match, score) tuples
        limit: Maximum number of matches to return
        algorithm: Algorithm to use ("auto", "ratio", "partial", "token", "levenshtein")
        normalize: Apply Unicode normalization and text cleaning (default True)

    Returns:
        List of matching strings or (string, score) tuples if include_scores=True
        Scores are always 0-100 scale (RapidFuzz standard)

    Raises:
        FuzzyMatcherError: If query is empty or invalid parameters

    Examples:
        >>> find_matches("Sony Music", ["Sony Music Entertainment", "Warner Music"])
        ["Sony Music Entertainment"]

        >>> find_matches("Sony Music", candidates, include_scores=True, threshold=90)
        [("Sony Music Entertainment", 95.2)]
    """
    global _cache_hits, _cache_misses

    if not query or not query.strip():
        raise FuzzyMatcherError("Query string cannot be empty", "empty_query")

    if not candidates:
        return []

    # Validate threshold (0-100 scale)
    if not 0.0 <= threshold <= 100.0:
        raise FuzzyMatcherError("Threshold must be between 0.0 and 100.0", "invalid_threshold")

    # Create cache key including normalization flag
    cache_key = _create_cache_key(query, candidates, threshold, algorithm, normalize)

    # Check cache first
    if cache_key in _match_cache:
        _cache_hits += 1
        cached_results = _match_cache[cache_key]
        return _format_results(cached_results[:limit], include_scores)

    _cache_misses += 1

    # Normalize query and candidates if requested
    if normalize:
        processed_query = _normalize_text(query)
        processed_candidates = [_normalize_text(c) for c in candidates]
    else:
        processed_query = query
        processed_candidates = candidates

    # Find matches using selected algorithm
    matches = _find_matches_with_algorithm(
        processed_query, processed_candidates, candidates, threshold, algorithm
    )

    # Sort by score descending, with deterministic tie-breaking
    matches.sort(key=lambda x: (-x.score, x.original_index or 0))

    # Cache results (LRUCache handles size automatically)
    _match_cache[cache_key] = matches

    # Return formatted results
    return _format_results(matches[:limit], include_scores)


def clear_cache() -> None:
    """Clear all caches - same pattern as production helpers."""
    global _match_cache, _preprocess_cache, _cache_hits, _cache_misses
    _match_cache.clear()
    _preprocess_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def get_cache_stats() -> CacheStats:
    """Get cache performance statistics."""
    total_requests = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total_requests if total_requests > 0 else 0.0

    # Calculate max size based on cache type
    if CACHETOOLS_AVAILABLE:
        max_size = getattr(_match_cache, 'maxsize', 50_000) + getattr(_preprocess_cache, 'maxsize', 100_000)
    else:
        max_size = 150_000  # Fallback total

    return CacheStats(
        hit_rate=hit_rate,
        size=len(_match_cache) + len(_preprocess_cache),
        max_size=max_size,
        hits=_cache_hits,
        misses=_cache_misses,
    )


# Private helpers - extracted from production ETL pipeline


def _normalize_text(text: str) -> str:
    """
    Normalize text for better matching using Unicode NFKC and casefold.

    Applies:
    - Unicode NFKC normalization (canonical decomposition + composition)
    - Case folding (more aggressive than lowercase)
    - Punctuation removal
    """
    if not text:
        return text

    # Check cache first
    if text in _preprocess_cache:
        return _preprocess_cache[text]

    # Unicode normalization + case folding
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()

    # Remove punctuation using Unicode categories
    cleaned = "".join(
        ch for ch in normalized
        if unicodedata.category(ch)[0] != "P"
    )

    # Normalize whitespace
    result = re.sub(r"\s+", " ", cleaned).strip()

    # Cache result
    _preprocess_cache[text] = result
    return result


def _find_matches_with_algorithm(
    query: str,
    processed_candidates: List[str],
    original_candidates: List[str],
    threshold: float,
    algorithm: Algorithm
) -> List[MatchResult]:
    """Find matches using the specified algorithm."""
    if RAPIDFUZZ_AVAILABLE:
        return _find_matches_rapidfuzz(query, processed_candidates, original_candidates, threshold, algorithm)
    else:
        return _find_matches_fallback(query, processed_candidates, original_candidates, threshold)


def _find_matches_rapidfuzz(
    query: str,
    processed_candidates: List[str],
    original_candidates: List[str],
    threshold: float,
    algorithm: Algorithm
) -> List[MatchResult]:
    """Find matches using RapidFuzz library with 0-100 scoring."""
    matches = []

    if algorithm == "auto":
        # Token-based strategy for dirty inputs (default)
        # Use WRatio (weighted ratio) which combines multiple algorithms
        for orig_idx, (original, processed) in enumerate(zip(original_candidates, processed_candidates)):
            # Use token_set_ratio for entity names (handles reordering + subsets)
            score = fuzz.token_set_ratio(query, processed)

            if score >= threshold:
                matches.append(
                    MatchResult(
                        text=original,
                        score=score,
                        algorithm="auto",
                        original_index=orig_idx,
                    )
                )

    else:
        # Use specific algorithm
        if algorithm == "ratio":
            alg_func = fuzz.ratio
        elif algorithm == "partial":
            alg_func = fuzz.partial_ratio
        elif algorithm == "token":
            alg_func = fuzz.token_set_ratio
        elif algorithm == "levenshtein":
            # Use ratio as Levenshtein approximation
            alg_func = fuzz.ratio
        else:
            raise FuzzyMatcherError(f"Unknown algorithm: {algorithm}", "invalid_algorithm")

        for orig_idx, (original, processed) in enumerate(zip(original_candidates, processed_candidates)):
            score = alg_func(query, processed)

            if score >= threshold:
                matches.append(
                    MatchResult(
                        text=original,
                        score=score,
                        algorithm=algorithm,
                        original_index=orig_idx,
                    )
                )

    return matches


def _find_matches_fallback(
    query: str,
    processed_candidates: List[str],
    original_candidates: List[str],
    threshold: float
) -> List[MatchResult]:
    """Fallback matching using difflib when RapidFuzz is not available."""
    matches = []

    # Convert threshold from 0-100 to 0-1 for difflib
    difflib_threshold = threshold / 100.0

    for orig_idx, (original, processed) in enumerate(zip(original_candidates, processed_candidates)):
        # Use difflib's SequenceMatcher
        matcher = difflib.SequenceMatcher(None, query, processed)
        score = matcher.ratio() * 100.0  # Convert back to 0-100 scale

        if score >= threshold:
            matches.append(
                MatchResult(
                    text=original,
                    score=score,
                    algorithm="difflib",
                    original_index=orig_idx,
                )
            )

    return matches


def _create_cache_key(
    query: str, candidates: List[str], threshold: float, algorithm: Algorithm, normalize: bool
) -> str:
    """Create a cache key for the matching operation."""
    # Create a hash of the candidates list for efficiency
    candidates_str = "|".join(sorted(candidates))
    candidates_hash = hashlib.md5(candidates_str.encode("utf-8")).hexdigest()[:8]

    return f"{query}:{candidates_hash}:{threshold}:{algorithm}:{normalize}"


def _format_results(
    matches: List[MatchResult], include_scores: bool
) -> Union[List[str], List[Tuple[str, float]]]:
    """Format match results based on include_scores parameter."""
    if include_scores:
        return [(match.text, match.score) for match in matches]
    else:
        return [match.text for match in matches]


# Cache management for non-LRUCache fallback
def _manage_cache_size():
    """Manage cache size when not using LRUCache."""
    if not CACHETOOLS_AVAILABLE:
        max_size = 50_000
        if len(_match_cache) > max_size:
            # Remove oldest 25% of entries (simple FIFO)
            items_to_remove = len(_match_cache) - int(max_size * 0.75)
            keys_to_remove = list(_match_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                _match_cache.pop(key, None)

        if len(_preprocess_cache) > 100_000:
            items_to_remove = len(_preprocess_cache) - 75_000
            keys_to_remove = list(_preprocess_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                _preprocess_cache.pop(key, None)
