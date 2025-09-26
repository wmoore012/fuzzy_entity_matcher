# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Benchmark tests for fuzzy entity matcher using pytest-benchmark.

These benchmarks measure performance and memory usage of matching operations
across different scenarios and data sizes.
"""

import gc
import time
import tracemalloc
from typing import List

import pytest

from fuzzy_entity_matcher.core import find_matches, clear_cache


@pytest.fixture
def music_entities_1k() -> List[str]:
    """Generate 1000 realistic music entity names for benchmarking."""
    base_names = [
        "Sony Music Entertainment",
        "Universal Music Group",
        "Warner Music Group",
        "Atlantic Records",
        "Capitol Records",
        "Columbia Records",
        "RCA Records",
        "Def Jam Recordings",
        "Interscope Records",
        "Republic Records",
        "Epic Records",
        "Elektra Records",
        "Arista Records",
        "Geffen Records",
        "Island Records",
        "Virgin Records",
        "EMI Records",
        "Parlophone Records",
        "Blue Note Records",
        "Nonesuch Records"
    ]

    # Generate variations and combinations
    entities = []
    for i in range(1000):
        base = base_names[i % len(base_names)]

        # Add variations
        if i % 5 == 0:
            entities.append(f"{base} LLC")
        elif i % 5 == 1:
            entities.append(f"{base} Inc.")
        elif i % 5 == 2:
            entities.append(f"{base} Records")
        elif i % 5 == 3:
            entities.append(base.replace(" ", "-"))
        else:
            entities.append(f"{base} {i}")

    return entities


@pytest.fixture
def dirty_entities_1k() -> List[str]:
    """Generate 1000 dirty/messy entity names for benchmarking."""
    clean_names = [
        "Sony Music Entertainment",
        "Universal Music Group",
        "Warner Music Group",
        "Atlantic Records",
        "Capitol Records"
    ]

    dirty_entities = []
    for i in range(1000):
        base = clean_names[i % len(clean_names)]

        # Add various types of "dirt"
        if i % 10 == 0:
            dirty_entities.append(f"  {base.upper()}  ")  # Case + whitespace
        elif i % 10 == 1:
            dirty_entities.append(base.replace(" ", "_"))  # Underscores
        elif i % 10 == 2:
            dirty_entities.append(f"{base} (Official)")  # Parentheses
        elif i % 10 == 3:
            dirty_entities.append(f"The {base}")  # Extra words
        elif i % 10 == 4:
            dirty_entities.append(base.replace("Music", "Música"))  # Unicode
        elif i % 10 == 5:
            dirty_entities.append(f"{base}™")  # Special chars
        elif i % 10 == 6:
            dirty_entities.append(" ".join(reversed(base.split())))  # Reordered
        elif i % 10 == 7:
            dirty_entities.append(base.replace("Records", "Rec."))  # Abbreviations
        elif i % 10 == 8:
            dirty_entities.append(f"{base} - {i}")  # Hyphens + numbers
        else:
            dirty_entities.append(base)  # Some clean ones

    return dirty_entities


def warmup_matcher(entities: List[str]):
    """Warm up the matcher to avoid first-call penalty."""
    find_matches("Sony Music", entities[:10], threshold=80.0)


class TestFuzzyMatcherBenchmarks:
    """Benchmark tests for fuzzy matching operations."""

    def test_clean_entities_1k_auto_algorithm(self, benchmark, music_entities_1k):
        """Benchmark auto algorithm with 1K clean music entities."""
        clear_cache()  # Start fresh
        warmup_matcher(music_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "Sony Music Entertainment",
                music_entities_1k,
                threshold=80.0,
                algorithm="auto"
            )
        finally:
            gc.enable()

        assert len(result) > 0
        assert "Sony Music Entertainment" in result

    def test_dirty_entities_1k_token_algorithm(self, benchmark, dirty_entities_1k):
        """Benchmark token algorithm with 1K dirty entities."""
        clear_cache()
        warmup_matcher(dirty_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "Sony Music",
                dirty_entities_1k,
                threshold=75.0,
                algorithm="token"
            )
        finally:
            gc.enable()

        assert len(result) > 0

    def test_ratio_algorithm_performance(self, benchmark, music_entities_1k):
        """Benchmark ratio algorithm (fastest but least flexible)."""
        clear_cache()
        warmup_matcher(music_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "Universal Music Group",
                music_entities_1k,
                threshold=85.0,
                algorithm="ratio"
            )
        finally:
            gc.enable()

        assert len(result) > 0

    def test_partial_algorithm_performance(self, benchmark, music_entities_1k):
        """Benchmark partial ratio algorithm (good for substring matches)."""
        clear_cache()
        warmup_matcher(music_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "Warner",  # Partial match
                music_entities_1k,
                threshold=70.0,
                algorithm="partial"
            )
        finally:
            gc.enable()

        assert len(result) > 0

    def test_with_scores_performance(self, benchmark, music_entities_1k):
        """Benchmark performance when returning scores."""
        clear_cache()
        warmup_matcher(music_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "Atlantic Records",
                music_entities_1k,
                threshold=80.0,
                include_scores=True,
                limit=5
            )
        finally:
            gc.enable()

        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(score, (int, float)) for _, score in result)

    def test_normalization_impact(self, benchmark, dirty_entities_1k):
        """Benchmark impact of text normalization."""
        clear_cache()
        warmup_matcher(dirty_entities_1k)

        gc.disable()
        try:
            result = benchmark(
                find_matches,
                "SONY MUSIC™",
                dirty_entities_1k,
                threshold=75.0,
                normalize=True  # Test with normalization
            )
        finally:
            gc.enable()

        assert len(result) > 0

    def test_cache_performance(self, benchmark, music_entities_1k):
        """Benchmark cache hit performance."""
        # Pre-populate cache
        find_matches("Sony Music", music_entities_1k, threshold=80.0)

        gc.disable()
        try:
            # This should hit cache
            result = benchmark(
                find_matches,
                "Sony Music",
                music_entities_1k,
                threshold=80.0
            )
        finally:
            gc.enable()

        assert len(result) > 0


class TestMemoryBenchmarks:
    """Memory usage benchmarks using tracemalloc."""

    def test_memory_usage_1k_entities(self, music_entities_1k):
        """Measure memory usage for 1K entity matching."""
        clear_cache()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Perform multiple matches to stress test
        for query in ["Sony Music", "Universal", "Warner", "Atlantic", "Capitol"]:
            find_matches(query, music_entities_1k, threshold=80.0)

        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory difference
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_memory_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

        tracemalloc.stop()

        # Memory usage should be reasonable (< 50MB for 1K entities)
        assert total_memory_mb < 50
        print(f"Memory usage: {total_memory_mb:.2f} MB for 1K entities")

    def test_cache_memory_efficiency(self, music_entities_1k):
        """Test that cache doesn't grow unbounded."""
        clear_cache()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Perform many different queries to test cache limits
        for i in range(100):
            query = f"Test Query {i}"
            find_matches(query, music_entities_1k[:100], threshold=80.0)

        snapshot2 = tracemalloc.take_snapshot()

        # Perform more queries
        for i in range(100, 200):
            query = f"Test Query {i}"
            find_matches(query, music_entities_1k[:100], threshold=80.0)

        snapshot3 = tracemalloc.take_snapshot()

        # Memory growth should be bounded
        growth1 = sum(stat.size_diff for stat in snapshot2.compare_to(snapshot1, 'lineno'))
        growth2 = sum(stat.size_diff for stat in snapshot3.compare_to(snapshot2, 'lineno'))

        tracemalloc.stop()

        # Second batch should use less memory due to cache eviction
        assert growth2 < growth1 * 1.5  # Allow some variance
        print(f"Memory growth: batch1={growth1/1024/1024:.2f}MB, batch2={growth2/1024/1024:.2f}MB")


def test_performance_regression_guard(music_entities_1k):
    """Guard against performance regressions with timing assertions."""
    clear_cache()
    warmup_matcher(music_entities_1k)

    # Time the operation
    start_time = time.perf_counter_ns()
    result = find_matches("Sony Music Entertainment", music_entities_1k, threshold=80.0)
    end_time = time.perf_counter_ns()

    duration_ms = (end_time - start_time) / 1_000_000
    ops_per_second = len(music_entities_1k) / (duration_ms / 1000)

    # Should process at least 10,000 comparisons/second (adjust based on requirements)
    assert ops_per_second > 10_000, f"Performance regression: {ops_per_second:.0f} ops/sec"
    assert len(result) > 0

    print(f"Performance: {ops_per_second:.0f} comparisons/sec, {duration_ms:.1f}ms total")
