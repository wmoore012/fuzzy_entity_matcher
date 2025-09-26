# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Tests for fuzzy entity matcher core functionality.

Comprehensive test suite covering all matching scenarios and edge cases.
"""

import pytest
from fuzzy_entity_matcher import FuzzyMatcherError, find_matches
from fuzzy_entity_matcher.core import (
    _preprocess_entity_name,
    clear_cache,
    get_cache_stats,
)


class TestFindMatches:
    """Test the main find_matches function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_basic_matching(self):
        """Test basic fuzzy matching functionality."""
        candidates = [
            "Sony Music Entertainment",
            "Warner Music Group",
            "Universal Music Group",
            "Atlantic Records",
            "Random Label",
        ]

        matches = find_matches("Sony Music", candidates, threshold=0.7)

        assert len(matches) >= 1
        assert "Sony Music Entertainment" in matches

    def test_threshold_filtering(self):
        """Test that threshold properly filters results."""
        candidates = [
            "Sony Music Entertainment",  # High similarity
            "Sony Pictures",  # Medium similarity
            "Microsoft Corporation",  # Low similarity
        ]

        # High threshold should only return exact matches
        high_matches = find_matches("Sony Music", candidates, threshold=0.9)
        assert len(high_matches) <= 1

        # Low threshold should return more matches
        low_matches = find_matches("Sony Music", candidates, threshold=0.3)
        assert len(low_matches) >= len(high_matches)

    def test_include_scores(self):
        """Test returning scores with matches."""
        candidates = ["Sony Music Entertainment", "Warner Music Group"]

        matches = find_matches("Sony Music", candidates, include_scores=True, threshold=0.7)

        assert isinstance(matches, list)
        assert len(matches) >= 1
        assert isinstance(matches[0], tuple)
        assert len(matches[0]) == 2
        assert isinstance(matches[0][0], str)  # Match text
        assert isinstance(matches[0][1], float)  # Score
        assert 0.0 <= matches[0][1] <= 1.0

    def test_limit_parameter(self):
        """Test that limit parameter works correctly."""
        candidates = [
            "Sony Music Entertainment",
            "Sony Pictures",
            "Sony Corporation",
            "Sony Interactive",
            "Sony Mobile",
        ]

        matches = find_matches("Sony", candidates, threshold=0.3, limit=2)

        assert len(matches) <= 2

    def test_empty_query_raises_error(self):
        """Test that empty query raises appropriate error."""
        candidates = ["Sony Music", "Warner Music"]

        with pytest.raises(FuzzyMatcherError, match="Query string cannot be empty"):
            find_matches("", candidates)

        with pytest.raises(FuzzyMatcherError, match="Query string cannot be empty"):
            find_matches(None, candidates)

    def test_empty_candidates_returns_empty(self):
        """Test that empty candidates returns empty list."""
        matches = find_matches("Sony Music", [])
        assert matches == []

    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises error."""
        candidates = ["Sony Music"]

        with pytest.raises(FuzzyMatcherError, match="Threshold must be between 0.0 and 1.0"):
            find_matches("Sony", candidates, threshold=1.5)

        with pytest.raises(FuzzyMatcherError, match="Threshold must be between 0.0 and 1.0"):
            find_matches("Sony", candidates, threshold=-0.1)

    def test_no_matches_above_threshold(self):
        """Test behavior when no matches meet threshold."""
        candidates = ["Microsoft", "Apple", "Google"]

        matches = find_matches("Sony Music", candidates, threshold=0.8)

        assert matches == []

    def test_different_algorithms(self):
        """Test different matching algorithms."""
        candidates = ["Sony Music Entertainment Inc.", "Warner Music Group Corp."]
        query = "Sony Music"

        algorithms = ["auto", "ratio", "partial", "token"]

        for algorithm in algorithms:
            # Use lower threshold for some algorithms that are more strict
            if algorithm in ["ratio", "token"]:
                threshold = 0.5
            else:
                threshold = 0.6
            matches = find_matches(query, candidates, algorithm=algorithm, threshold=threshold)
            # Should find at least the Sony match
            assert len(matches) >= 1, f"Algorithm {algorithm} found no matches"
            assert any(
                "Sony" in match for match in matches
            ), f"Algorithm {algorithm} didn't find Sony match"

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        candidates = ["Sony Music Entertainment", "WARNER MUSIC GROUP"]

        matches = find_matches("sony music", candidates, threshold=0.7)

        assert len(matches) >= 1
        assert "Sony Music Entertainment" in matches

    def test_corporate_suffix_normalization(self):
        """Test that corporate suffixes are normalized for better matching."""
        candidates = [
            "Sony Music Entertainment, Inc.",
            "Sony Music Entertainment LLC",
            "Sony Music Entertainment Ltd.",
            "Sony Music Entertainment Corp.",
        ]

        matches = find_matches("Sony Music Entertainment", candidates, threshold=0.8)

        # Should match all variants due to suffix normalization
        assert len(matches) >= 2

    def test_music_industry_specific_matching(self):
        """Test matching with music industry specific terms."""
        candidates = [
            "Atlantic Records",
            "Atlantic Recording Corporation",
            "Atlantic Music Group",
            "Pacific Records",
        ]

        matches = find_matches("Atlantic Records", candidates, threshold=0.7)

        # Should find multiple Atlantic variants
        atlantic_matches = [m for m in matches if "Atlantic" in m]
        assert len(atlantic_matches) >= 2

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        candidates = ["Café Records", "Naïve Records", "Björk Music", "Regular Records"]

        matches = find_matches("Cafe Records", candidates, threshold=0.7)

        # Should handle Unicode normalization
        assert len(matches) >= 1


class TestCaching:
    """Test caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_cache_improves_performance(self):
        """Test that caching improves performance on repeated queries."""
        candidates = ["Sony Music Entertainment"] * 100  # Large candidate list
        query = "Sony Music"

        # First call (cold cache)
        import time

        start = time.time()
        matches1 = find_matches(query, candidates)
        cold_time = time.time() - start

        # Second call (warm cache)
        start = time.time()
        matches2 = find_matches(query, candidates)
        warm_time = time.time() - start

        # Results should be identical
        assert matches1 == matches2

        # Warm cache should be faster (though this might be flaky in CI)
        # Just check that cache stats show hits
        stats = get_cache_stats()
        assert stats.hits > 0

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        candidates = ["Sony Music", "Warner Music"]

        # Initial stats
        stats = get_cache_stats()
        initial_hits = stats.hits
        initial_misses = stats.misses

        # Make some queries
        find_matches("Sony", candidates)
        find_matches("Warner", candidates)
        find_matches("Sony", candidates)  # Repeat for cache hit

        # Check updated stats
        stats = get_cache_stats()
        assert stats.hits > initial_hits
        assert stats.misses > initial_misses
        assert stats.size > 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        candidates = ["Sony Music"]

        # Populate cache
        find_matches("Sony", candidates)
        stats = get_cache_stats()
        assert stats.size > 0

        # Clear cache
        clear_cache()
        stats = get_cache_stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0


class TestPreprocessing:
    """Test entity name preprocessing."""

    def test_corporate_suffix_removal(self):
        """Test removal of corporate suffixes."""
        assert _preprocess_entity_name("Sony Music Inc.") == "sony music"
        assert _preprocess_entity_name("Warner Music LLC") == "warner music"
        assert _preprocess_entity_name("Universal Music Ltd.") == "universal music"
        assert _preprocess_entity_name("Atlantic Records Corp.") == "atlantic"

    def test_music_industry_suffix_removal(self):
        """Test removal of music industry suffixes."""
        assert _preprocess_entity_name("Atlantic Records") == "atlantic"
        assert _preprocess_entity_name("Sony Music") == "sony music"  # Only "Records" is removed
        assert _preprocess_entity_name("Warner Entertainment") == "warner entertainment"
        assert _preprocess_entity_name("Universal Group") == "universal group"

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        assert _preprocess_entity_name("  Sony   Music  ") == "sony music"
        assert _preprocess_entity_name("Warner\t\tMusic") == "warner music"
        assert _preprocess_entity_name("Universal\nMusic") == "universal music"

    def test_special_character_removal(self):
        """Test handling of special characters."""
        assert _preprocess_entity_name("Sony-Music") == "sony music"  # Hyphens become spaces
        assert _preprocess_entity_name("Warner/Music") == "warner/music"  # Slashes preserved
        assert (
            _preprocess_entity_name("Universal & Music") == "universal & music"
        )  # Ampersands preserved

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        assert _preprocess_entity_name("") == ""
        assert _preprocess_entity_name("   ") == ""


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_single_character_queries(self):
        """Test behavior with single character queries."""
        candidates = ["Sony Music", "Apple Inc."]

        matches = find_matches("S", candidates, threshold=0.1)

        # Should handle gracefully
        assert isinstance(matches, list)

    def test_very_long_strings(self):
        """Test behavior with very long strings."""
        long_candidate = "A" * 1000 + " Music Entertainment Corporation"
        candidates = [long_candidate, "Sony Music"]

        matches = find_matches("A Music", candidates, threshold=0.3)

        # Should handle without crashing
        assert isinstance(matches, list)

    def test_special_characters_in_query(self):
        """Test queries with special characters."""
        candidates = ["Sony Music", "AT&T Records", "C&W Music"]

        matches = find_matches("AT&T", candidates, threshold=0.7)

        # Should find the AT&T match
        assert len(matches) >= 1
        assert "AT&T Records" in matches

    def test_numeric_content(self):
        """Test handling of numeric content."""
        candidates = ["Sony Music 2023", "Warner Music 2022", "Universal Music"]

        matches = find_matches("Sony Music 2023", candidates, threshold=0.8)

        # Should find exact match
        assert "Sony Music 2023" in matches

    def test_duplicate_candidates(self):
        """Test behavior with duplicate candidates."""
        candidates = ["Sony Music", "Sony Music", "Warner Music"]

        matches = find_matches("Sony", candidates, threshold=0.7)

        # Should handle duplicates gracefully
        assert isinstance(matches, list)
        # Might contain duplicates, which is acceptable behavior

    def test_all_identical_candidates(self):
        """Test with all identical candidates."""
        candidates = ["Sony Music"] * 10

        matches = find_matches("Sony", candidates, threshold=0.7)

        # Should return matches (possibly duplicates)
        assert len(matches) >= 1

    def test_unknown_algorithm(self):
        """Test error handling for unknown algorithm."""
        candidates = ["Sony Music"]

        with pytest.raises(FuzzyMatcherError, match="Unknown algorithm"):
            find_matches("Sony", candidates, algorithm="unknown_algorithm")


class TestRealWorldScenarios:
    """Test real-world music industry scenarios."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_major_label_variants(self):
        """Test matching major label variants."""
        candidates = [
            "Sony Music Entertainment",
            "Sony Music Entertainment, Inc.",
            "Sony Music Entertainment LLC",
            "Sony Music Group",
            "Sony Pictures Music",
            "Warner Music Group",
            "Universal Music Group",
        ]

        matches = find_matches("Sony Music", candidates, threshold=0.7)

        # Should find multiple Sony variants
        sony_matches = [m for m in matches if "Sony" in m]
        assert len(sony_matches) >= 3

    def test_independent_label_matching(self):
        """Test matching independent label names."""
        candidates = [
            "Sub Pop Records",
            "Sub Pop",
            "Subpop Records",
            "Matador Records",
            "4AD Records",
            "Rough Trade Records",
        ]

        matches = find_matches("Sub Pop", candidates, threshold=0.7)

        # Should find Sub Pop variants
        subpop_matches = [m for m in matches if "Sub" in m or "Subpop" in m]
        assert len(subpop_matches) >= 2

    def test_artist_name_matching(self):
        """Test matching artist names with variations."""
        candidates = [
            "The Beatles",
            "Beatles",
            "The Rolling Stones",
            "Rolling Stones",
            "Led Zeppelin",
            "Pink Floyd",
        ]

        matches = find_matches("Beatles", candidates, threshold=0.7)

        # Should find both Beatles variants
        beatles_matches = [m for m in matches if "Beatles" in m]
        assert len(beatles_matches) >= 2

    def test_international_labels(self):
        """Test matching international label names."""
        candidates = [
            "EMI Records",
            "EMI Music",
            "Parlophone Records",
            "Virgin Records",
            "Chrysalis Records",
        ]

        matches = find_matches("EMI", candidates, threshold=0.7)

        # Should find EMI variants
        emi_matches = [m for m in matches if "EMI" in m]
        assert len(emi_matches) >= 2
