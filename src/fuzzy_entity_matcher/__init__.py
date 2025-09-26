# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Wilton Moore

"""
Fuzzy Entity Matcher - High-performance entity deduplication with intelligent caching.

Simple API for finding similar entities with configurable algorithms and thresholds.
Optimized for production-scale data processing with memory-efficient caching.
"""

from importlib.metadata import PackageNotFoundError, version

from .core import CacheStats, FuzzyMatcherError, MatchResult, clear_cache, find_matches, get_cache_stats

try:
    __version__ = version("fuzzy-entity-matcher")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "find_matches",
    "clear_cache",
    "get_cache_stats",
    "MatchResult",
    "CacheStats",
    "FuzzyMatcherError",
]
