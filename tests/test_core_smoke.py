# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Smoke tests that prove fuzzy-entity-matcher works on real-world music data.
"""

import pytest
from fuzzy_entity_matcher import FuzzyMatcherError, find_matches


class TestRealWorldMusicData:
    """Test with actual music industry data patterns."""

    def test_find_matches_basic_labels(self):
        """Test basic label matching with real label names."""
        candidates = [
            "Sony Music Entertainment",
            "Atlantic Records",
            "Universal Music Group",
            "Warner Music Group",
            "EMI Records",
        ]

        # Test exact match
        results = find_matches("Sony Music Entertainment", candidates, include_scores=True)
        assert results[0][0] == "Sony Music Entertainment"
        assert results[0][1] == 1.0

        # Test partial match
        results = find_matches("Sony Music", candidates, include_scores=True, threshold=0.7)
        assert results[0][0] == "Sony Music Entertainment"
        assert 0.70 <= results[0][1] <= 1.0

    def test_find_matches_with_features(self):
        """Test matching with 'feat.' and other music-specific patterns."""
        candidates = [
            "Blinding Lights",
            "Blinding Lights (feat. Daft Punk)",
            "Blinding Lights - Radio Edit",
            "Blinding Lights (Remix)",
        ]

        results = find_matches("Blinding Lights feat Daft Punk", candidates, include_scores=True)
        # Should match the feat. version best
        assert "feat." in results[0][0] or "Daft Punk" in results[0][0]
        assert results[0][1] >= 0.75

    def test_find_matches_with_punctuation(self):
        """Test matching handles punctuation correctly."""
        candidates = [
            "Guns N' Roses",
            "Guns N Roses",
            "Guns and Roses",
            "AC/DC",
            "AC-DC",
        ]

        # Test apostrophe handling
        results = find_matches("Guns N Roses", candidates, include_scores=True)
        top_match = results[0][0]
        assert "Guns N" in top_match
        assert results[0][1] >= 0.85

        # Test slash/dash handling
        results = find_matches("AC DC", candidates, include_scores=True)
        top_match = results[0][0]
        assert "AC" in top_match and "DC" in top_match
        assert results[0][1] >= 0.80

    def test_find_matches_corporate_suffixes(self):
        """Test matching with corporate suffixes (Inc, LLC, etc)."""
        candidates = [
            "Sony Music Entertainment Inc.",
            "Sony Music Entertainment LLC",
            "Sony Music Entertainment",
            "Warner Bros. Records Inc.",
        ]

        results = find_matches("Sony Music Entertainment", candidates, include_scores=True)
        # Should match the base name best, but all Sony variants should score high
        sony_matches = [r for r in results if "Sony" in r[0]]
        assert len(sony_matches) >= 3
        assert all(score >= 0.85 for _, score in sony_matches)

    @pytest.mark.parametrize("query", ["", None])
    def test_find_matches_rejects_empty(self, query):
        """Test that empty queries are rejected properly."""
        candidates = ["A", "B", "C"]
        with pytest.raises((TypeError, ValueError, FuzzyMatcherError)):
            find_matches(query, candidates)

    def test_find_matches_empty_candidates(self):
        """Test behavior with empty candidate list."""
        results = find_matches("test query", [])
        assert results == []

    def test_find_matches_threshold_filtering(self):
        """Test that threshold parameter works correctly."""
        candidates = ["The Beatles", "The Rolling Stones", "Led Zeppelin", "Pink Floyd"]

        # High threshold should return fewer matches
        high_threshold_results = find_matches("Beatles", candidates, threshold=0.8)
        low_threshold_results = find_matches("Beatles", candidates, threshold=0.3)

        assert len(high_threshold_results) <= len(low_threshold_results)

        # The Beatles should be in results for "Beatles" query at reasonable threshold
        medium_threshold_results = find_matches("Beatles", candidates, threshold=0.7)
        medium_threshold_names = [
            r[0] if isinstance(r, tuple) else r for r in medium_threshold_results
        ]
        assert "The Beatles" in medium_threshold_names

    def test_find_matches_performance_baseline(self):
        """Test that matching performance meets baseline expectations."""
        import time

        # Create larger candidate list
        candidates = [f"Label {i} Records" for i in range(1000)]
        candidates.extend(
            ["Sony Music Entertainment", "Universal Music Group", "Warner Music Group"]
        )

        # Time the matching
        start_time = time.perf_counter()
        results = find_matches("Sony Music", candidates, limit=10, threshold=0.7)
        end_time = time.perf_counter()

        duration = end_time - start_time

        # Should complete in reasonable time (< 100ms for 1000 candidates)
        assert duration < 0.1, f"Matching took {duration:.3f}s, expected < 0.1s"

        # Should find the right match
        assert len(results) > 0
        top_match = results[0][0] if isinstance(results[0], tuple) else results[0]
        assert "Sony" in top_match

    def test_find_matches_unicode_handling(self):
        """Test matching with Unicode characters (international artists)."""
        candidates = ["Björk", "Sigur Rós", "Mötley Crüe", "Queensrÿche"]

        # Test that Unicode matching works
        results = find_matches("Bjork", candidates, include_scores=True)
        assert results[0][0] == "Björk"
        assert results[0][1] >= 0.65

        results = find_matches("Motley Crue", candidates, include_scores=True)
        assert results[0][0] == "Mötley Crüe"
        assert results[0][1] >= 0.80
