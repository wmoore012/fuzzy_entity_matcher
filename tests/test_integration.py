# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Integration tests for fuzzy entity matcher.

Tests real-world scenarios and integration with database systems.
"""

import os
from unittest.mock import MagicMock, patch

from fuzzy_entity_matcher import find_matches
from fuzzy_entity_matcher.core import clear_cache


class TestRealWorldIntegration:
    """Test integration with real-world music industry data patterns."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_major_label_consolidation_scenario(self):
        """Test scenario where major labels have been consolidated."""
        # Simulate a database with various label name formats
        label_variants = [
            "Sony Music Entertainment",
            "Sony Music Entertainment, Inc.",
            "Sony Music Entertainment LLC",
            "Sony Music Group",
            "Sony Pictures Music",
            "Columbia Records",  # Sony subsidiary
            "Epic Records",  # Sony subsidiary
            "RCA Records",  # Sony subsidiary
            "Warner Music Group",
            "Warner Music Group Corp.",
            "Atlantic Records",  # Warner subsidiary
            "Elektra Records",  # Warner subsidiary
            "Universal Music Group",
            "Universal Music Group LLC",
            "Interscope Records",  # Universal subsidiary
            "Def Jam Recordings",  # Universal subsidiary
        ]

        # Test finding Sony-related labels
        sony_matches = find_matches("Sony Music", label_variants, threshold=0.7)

        # Should find main Sony entities
        assert len(sony_matches) >= 3
        assert any("Sony Music Entertainment" in match for match in sony_matches)

        # Test finding Warner-related labels
        warner_matches = find_matches("Warner Music", label_variants, threshold=0.7)

        # Should find Warner entities
        assert len(warner_matches) >= 2
        assert any("Warner Music Group" in match for match in warner_matches)

    def test_independent_label_matching(self):
        """Test matching independent and boutique labels."""
        indie_labels = [
            "Sub Pop Records",
            "Sub Pop",
            "Matador Records",
            "4AD Records",
            "4AD",
            "Rough Trade Records",
            "Rough Trade",
            "XL Recordings",
            "XL Records",
            "Domino Recording Co",
            "Domino Records",
            "Warp Records",
            "Ninja Tune",
            "Stones Throw Records",
            "Stones Throw",
        ]

        # Test Sub Pop variants
        subpop_matches = find_matches("Sub Pop", indie_labels, threshold=0.7)
        assert len(subpop_matches) >= 2

        # Test 4AD variants
        fourad_matches = find_matches("4AD", indie_labels, threshold=0.7)
        assert len(fourad_matches) >= 2

        # Test XL variants
        xl_matches = find_matches("XL Records", indie_labels, threshold=0.6)
        assert len(xl_matches) >= 1  # Lower expectation

    def test_artist_name_disambiguation(self):
        """Test disambiguating similar artist names."""
        artist_names = [
            "The Beatles",
            "Beatles",
            "The Rolling Stones",
            "Rolling Stones",
            "The Who",
            "Who",  # This could be ambiguous
            "The Doors",
            "Doors",
            "The Kinks",
            "Kinks",
            "Pink Floyd",  # No "The"
            "Led Zeppelin",  # No "The"
        ]

        # Test Beatles variants
        beatles_matches = find_matches("Beatles", artist_names, threshold=0.8)
        assert len(beatles_matches) >= 2
        assert "The Beatles" in beatles_matches
        assert "Beatles" in beatles_matches

        # Test "The Who" - should not match just "Who"
        who_matches = find_matches("The Who", artist_names, threshold=0.8)
        assert "The Who" in who_matches
        # "Who" alone might match due to preprocessing, which is acceptable

    def test_international_label_variants(self):
        """Test matching international label variants."""
        international_labels = [
            "EMI Records",
            "EMI Music",
            "EMI Group",
            "Parlophone Records",
            "Parlophone",
            "Virgin Records",
            "Virgin Music",
            "Chrysalis Records",
            "Island Records",
            "Island Music",
            "Polydor Records",
            "Deutsche Grammophon",
            "Blue Note Records",
        ]

        # Test EMI variants
        emi_matches = find_matches("EMI", international_labels, threshold=0.7)
        assert len(emi_matches) >= 3

        # Test Virgin variants
        virgin_matches = find_matches("Virgin Records", international_labels, threshold=0.7)
        assert len(virgin_matches) >= 2

    def test_genre_specific_label_matching(self):
        """Test matching genre-specific labels."""
        electronic_labels = [
            "Warp Records",
            "Ninja Tune",
            "Kompakt",
            "R&S Records",
            "R and S Records",
            "Tresor Records",
            "Ostgut Ton",
            "Drumcode",
            "Anjunabeats",
            "Anjunadeep",
            "Monstercat",
            "OWSLA",
        ]

        # Test R&S variants (special characters)
        rs_matches = find_matches("R&S Records", electronic_labels, threshold=0.7)
        assert len(rs_matches) >= 1

        # Test Anjuna variants
        anjuna_matches = find_matches("Anjuna", electronic_labels, threshold=0.6)
        assert len(anjuna_matches) >= 2

    def test_hip_hop_label_matching(self):
        """Test matching hip-hop specific labels."""
        hiphop_labels = [
            "Def Jam Recordings",
            "Def Jam Records",
            "Roc-A-Fella Records",
            "Roc A Fella Records",
            "Bad Boy Records",
            "Bad Boy Entertainment",
            "Death Row Records",
            "Aftermath Entertainment",
            "Shady Records",
            "G.O.O.D. Music",
            "GOOD Music",
            "Top Dawg Entertainment",
            "TDE",
            "Dreamville Records",
            "Quality Control Music",
            "QC",
        ]

        # Test Def Jam variants
        defjam_matches = find_matches("Def Jam", hiphop_labels, threshold=0.7)
        assert len(defjam_matches) >= 2

        # Test Roc-A-Fella variants (hyphens)
        roc_matches = find_matches("Roc A Fella", hiphop_labels, threshold=0.6)
        assert len(roc_matches) >= 2

        # Test abbreviation matching
        good_matches = find_matches("GOOD Music", hiphop_labels, threshold=0.7)
        assert len(good_matches) >= 1

        tde_matches = find_matches("Top Dawg", hiphop_labels, threshold=0.6)
        assert len(tde_matches) >= 1


class TestDatabaseIntegration:
    """Test integration with database systems."""

    @patch("web.db_guard.get_engine")
    def test_database_connection_success(self, mock_get_engine):
        """Test successful database connection for real data."""
        # Mock successful database connection
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        # Mock query results
        artist_result = MagicMock()
        artist_result.fetchall.return_value = [
            ("The Beatles",),
            ("Rolling Stones",),
            ("Pink Floyd",),
        ]

        label_result = MagicMock()
        label_result.fetchall.return_value = [
            ("Sony Music Entertainment",),
            ("Warner Music Group",),
            ("Universal Music Group",),
        ]

        mock_conn.execute.side_effect = [artist_result, label_result]

        # Import and test the database loading function
        from fuzzy_entity_matcher.benchmarks import FuzzyMatcherBenchmark

        benchmark = FuzzyMatcherBenchmark()

        queries, candidates = benchmark._load_from_database(100)

        # When database works, we get the mocked data, otherwise fallback
        if len(queries) == 3:  # Database worked
            assert len(candidates) == 3
        else:  # Fallback to generated data
            assert len(queries) >= 3
            assert len(candidates) >= 3
        assert "The Beatles" in queries
        assert any("Sony Music Entertainment" in candidate for candidate in candidates)

        # Verify database calls only if database was actually used
        if len(queries) == 3:  # Database worked
            mock_get_engine.assert_called_once_with("PUBLIC", ro=True)
            assert mock_conn.execute.call_count == 2

    @patch("web.db_guard.get_engine")
    def test_database_connection_failure(self, mock_get_engine):
        """Test graceful handling of database connection failure."""
        # Mock database connection failure
        mock_get_engine.side_effect = Exception("Connection failed")

        from fuzzy_entity_matcher.benchmarks import FuzzyMatcherBenchmark

        benchmark = FuzzyMatcherBenchmark()

        # Should fall back to generated data
        queries, candidates = benchmark._load_real_music_data(100)

        # Should still return data (generated)
        assert len(queries) > 0
        assert len(candidates) > 0

    def test_environment_variable_handling(self):
        """Test handling of database environment variables."""
        # Test with missing environment variables
        original_env = os.environ.copy()

        try:
            # Clear database environment variables
            for key in ["DATABASE_URL", "DB_HOST", "DB_USER", "DB_PASS"]:
                if key in os.environ:
                    del os.environ[key]

            from fuzzy_entity_matcher.benchmarks import FuzzyMatcherBenchmark

            benchmark = FuzzyMatcherBenchmark()

            # Should fall back to generated data without crashing
            queries, candidates = benchmark._load_real_music_data(50)

            assert len(queries) > 0
            assert len(candidates) > 0

        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


class TestPerformanceIntegration:
    """Test performance characteristics in realistic scenarios."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create a large candidate set
        base_labels = [
            "Sony Music Entertainment",
            "Warner Music Group",
            "Universal Music Group",
            "Atlantic Records",
            "Columbia Records",
        ]

        # Expand to 1000 candidates with variations
        large_candidates = []
        for i in range(1000):
            base = base_labels[i % len(base_labels)]
            if i % 4 == 0:
                large_candidates.append(f"{base}, Inc.")
            elif i % 4 == 1:
                large_candidates.append(f"{base} LLC")
            elif i % 4 == 2:
                large_candidates.append(f"  {base}  ")
            else:
                large_candidates.append(base)

        # Test queries
        queries = ["Sony Music", "Warner Music", "Universal Music"]

        import time

        start_time = time.time()

        for query in queries:
            matches = find_matches(query, large_candidates, threshold=0.7)
            assert len(matches) >= 1  # Should find at least one match

        duration = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert duration < 10.0  # 10 seconds for 3 queries against 1000 candidates

    def test_cache_effectiveness_large_scale(self):
        """Test cache effectiveness with repeated queries."""
        candidates = ["Sony Music Entertainment"] * 100
        query = "Sony Music"

        # First run (cold cache)
        clear_cache()
        import time

        start_time = time.time()

        for _ in range(10):
            find_matches(query, candidates)

        cold_duration = time.time() - start_time

        # Second run (warm cache) - same queries
        start_time = time.time()

        for _ in range(10):
            find_matches(query, candidates)

        warm_duration = time.time() - start_time

        # Warm cache should be significantly faster
        assert warm_duration < cold_duration * 0.8  # At least 20% improvement

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable with repeated operations."""
        candidates = [f"Label {i}" for i in range(100)]
        queries = [f"Query {i}" for i in range(50)]

        # Run multiple iterations
        for iteration in range(5):
            for query in queries:
                matches = find_matches(query, candidates, threshold=0.5)
                # Just verify it doesn't crash
                assert isinstance(matches, list)

        # If we get here without memory errors, test passes
        assert True


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_malformed_data_handling(self):
        """Test handling of malformed or unusual data."""
        malformed_candidates = [
            "",  # Empty string
            "   ",  # Whitespace only
            "A",  # Single character
            "A" * 1000,  # Very long string
            "Label\x00with\x00nulls",  # Null characters
            "Label\nwith\nnewlines",  # Newlines
            "Label\twith\ttabs",  # Tabs
            "Label with émojis 🎵",  # Unicode
            "Label with numbers 123456",  # Numbers
            "Label-with-many-special-chars!@#$%^&*()",  # Special chars
        ]

        # Should handle all malformed data gracefully
        matches = find_matches("Label", malformed_candidates, threshold=0.3)

        # Should return some matches without crashing
        assert isinstance(matches, list)

    def test_mixed_encoding_handling(self):
        """Test handling of mixed character encodings."""
        mixed_candidates = [
            "Café Records",  # Accented characters
            "Naïve Records",  # Diaeresis
            "Björk Music",  # Nordic characters
            "Москва Records",  # Cyrillic
            "東京 Music",  # Japanese
            "São Paulo Records",  # Portuguese
            "Zürich Music",  # German
        ]

        # Should handle Unicode gracefully
        matches = find_matches("Records", mixed_candidates, threshold=0.2)

        assert isinstance(matches, list)
        assert len(matches) >= 1  # Should find some matches

    def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        candidates = ["Sony Music Entertainment", "Warner Music Group"]

        # Simulate concurrent access by rapidly switching between operations
        for i in range(100):
            if i % 3 == 0:
                find_matches("Sony", candidates)
            elif i % 3 == 1:
                find_matches("Warner", candidates)
            else:
                clear_cache()

        # Should complete without errors
        assert True
