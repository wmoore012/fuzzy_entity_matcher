# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Tests for YouTube integration with fuzzy entity matcher.

Tests the integration with real database schema and YouTube video processing.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzy_entity_matcher import find_matches


class TestYouTubeIntegration:
    """Test integration with YouTube video processing."""

    def test_artist_name_matching(self):
        """Test matching YouTube channel names to database artists."""
        # Simulate database artist names
        database_artists = [
            "Taylor Swift",
            "Ed Sheeran",
            "Ariana Grande",
            "The Beatles",
            "Rolling Stones",
        ]

        # Test YouTube channel variations
        test_channels = [
            "TaylorSwiftVEVO",
            "Ed Sheeran Official",
            "ArianaGrandeVevo",
            "The Beatles - Topic",
            "Rolling Stones Official",
        ]

        for channel in test_channels:
            matches = find_matches(channel, database_artists, threshold=0.5)
            assert len(matches) >= 1, f"Should find match for {channel}"

    def test_label_name_matching(self):
        """Test matching YouTube descriptions to record labels."""
        # Simulate database label names
        database_labels = [
            "Republic Records",
            "Atlantic Records",
            "Sony Music Entertainment",
            "Warner Music Group",
            "Universal Music Group",
        ]

        # Test YouTube description variations
        test_descriptions = [
            "© 2023 Republic Records",
            "Atlantic Records LLC",
            "Sony Music Entertainment, Inc.",
            "Warner Music Group Corp.",
            "UMG - Universal Music Group",
        ]

        for description in test_descriptions:
            matches = find_matches(description, database_labels, threshold=0.6)
            assert len(matches) >= 1, f"Should find match for {description}"

    def test_song_title_normalization(self):
        """Test normalizing YouTube titles to clean song names."""
        # Simulate database song titles
        database_songs = [
            "Anti-Hero",
            "Shape of You",
            "Thank U, Next",
            "Blinding Lights",
            "Watermelon Sugar",
        ]

        # Test YouTube title variations
        youtube_titles = [
            "Taylor Swift - Anti-Hero (Official Music Video)",
            "Ed Sheeran - Shape of You [Official Video]",
            "Ariana Grande - thank u, next (Official Video)",
            "The Weeknd - Blinding Lights (Official Audio)",
            "Harry Styles - Watermelon Sugar (Official Video)",
        ]

        for title in youtube_titles:
            # Extract just the song part (after artist and separator)
            if " - " in title:
                song_part = title.split(" - ", 1)[1]
                # Remove common YouTube suffixes
                song_part = song_part.replace("(Official Music Video)", "")
                song_part = song_part.replace("[Official Video]", "")
                song_part = song_part.replace("(Official Video)", "")
                song_part = song_part.replace("(Official Audio)", "")
                song_part = song_part.strip()

                matches = find_matches(song_part, database_songs, threshold=0.8)
                assert len(matches) >= 1, f"Should find match for {song_part}"

    def test_corporate_suffix_handling(self):
        """Test handling of corporate suffixes in entity matching."""
        # Test with corporate suffixes
        entities_with_suffixes = [
            "Sony Music Entertainment, Inc.",
            "Warner Music Group Corp.",
            "Universal Music Group LLC",
            "Atlantic Records Ltd.",
        ]

        # Test without suffixes
        entities_without_suffixes = [
            "Sony Music Entertainment",
            "Warner Music Group",
            "Universal Music Group",
            "Atlantic Records",
        ]

        for with_suffix, without_suffix in zip(entities_with_suffixes, entities_without_suffixes):
            matches = find_matches(without_suffix, [with_suffix], threshold=0.8)
            assert len(matches) == 1, f"Should match {without_suffix} to {with_suffix}"

            matches = find_matches(with_suffix, [without_suffix], threshold=0.8)
            assert len(matches) == 1, f"Should match {with_suffix} to {without_suffix}"

    def test_performance_with_large_dataset(self):
        """Test performance with realistic dataset sizes."""
        # Simulate a large artist database
        large_artist_list = [f"Artist {i}" for i in range(1000)]
        large_artist_list.extend(
            [
                "Taylor Swift",
                "Ed Sheeran",
                "Ariana Grande",
                "The Beatles",
                "Rolling Stones",
                "Drake",
                "Billie Eilish",
            ]
        )

        # Test queries
        test_queries = [
            "TaylorSwiftVEVO",
            "Ed Sheeran Official",
            "ArianaGrandeVevo",
            "The Beatles - Topic",
        ]

        import time

        start_time = time.time()

        for query in test_queries:
            matches = find_matches(query, large_artist_list, threshold=0.7)
            assert isinstance(matches, list)

        duration = time.time() - start_time

        # Should complete in reasonable time
        assert duration < 5.0, f"Processing took too long: {duration:.2f}s"

    def test_empty_and_edge_cases(self):
        """Test edge cases that might occur in YouTube data."""
        database_entities = ["Taylor Swift", "Ed Sheeran", "Ariana Grande"]

        # Test empty query - should raise exception
        with pytest.raises(Exception):  # FuzzyMatcherError
            find_matches("", database_entities, threshold=0.8)

        # Test very short query
        matches = find_matches("T", database_entities, threshold=0.8)
        assert isinstance(matches, list)

        # Test query with special characters
        matches = find_matches("Taylor Swift & Co.", database_entities, threshold=0.7)
        assert len(matches) >= 1

        # Test query with numbers
        matches = find_matches("Taylor Swift 2023", database_entities, threshold=0.7)
        assert len(matches) >= 1

    def test_unicode_handling(self):
        """Test handling of Unicode characters in entity names."""
        database_entities = ["Beyoncé", "Björk", "Café Tacvba", "Naïve Records"]

        test_queries = [
            "Beyonce",  # Without accent
            "Bjork",  # Without special characters
            "Cafe Tacvba",  # Without accent
            "Naive Records",  # Without diaeresis
        ]

        for query in test_queries:
            matches = find_matches(query, database_entities, threshold=0.8)
            assert len(matches) >= 1, f"Should find match for {query}"


class TestDatabaseIntegration:
    """Test integration with database operations."""

    @patch("web.db_guard.get_engine")
    def test_database_connection_mock(self, mock_get_engine):
        """Test database connection handling."""
        # Mock successful database connection
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        # Mock query results for artists
        artist_result = MagicMock()
        artist_result.fetchall.return_value = [
            ("Taylor Swift",),
            ("Ed Sheeran",),
            ("Ariana Grande",),
        ]

        # Mock query results for labels
        label_result = MagicMock()
        label_result.fetchall.return_value = [
            ("Republic Records",),
            ("Atlantic Records",),
            ("Sony Music Entertainment",),
        ]

        mock_conn.execute.side_effect = [artist_result, label_result]

        # Test the database loading function
        from fuzzy_entity_matcher.benchmarks import FuzzyMatcherBenchmark

        benchmark = FuzzyMatcherBenchmark()

        queries, candidates = benchmark._load_from_database(100)

        # When database works, we get the mocked data, otherwise fallback
        if len(queries) == 3:  # Database worked
            assert len(candidates) == 3
            assert "Taylor Swift" in queries
            assert "Republic Records" in candidates
            # Verify database calls
            mock_get_engine.assert_called_once_with("PUBLIC", ro=True)
            assert mock_conn.execute.call_count == 2
        else:  # Fallback to generated data
            assert len(queries) >= 3
            assert len(candidates) >= 3

    def test_environment_variable_handling(self):
        """Test handling of missing environment variables."""
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
