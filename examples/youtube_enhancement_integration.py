#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
YouTube Enhancement Integration for Fuzzy Entity Matcher

Enhances existing YouTube video data by adding:
1. normalized_title - Clean song names (NOT added to songs table)
2. primary_artist_id - Foreign key to existing artists (NEVER creates new artists)
3. song_id - Optional foreign key to songs table (one-to-many relationship)

Key principles:
- NEVER add new entries to artists or songs tables (fact tables)
- ONLY match to existing primary artists from artist_roles
- Keep original YouTube titles intact
- Create one-to-many relationship: one song -> many YouTube videos
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Add the src directory to the path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzy_entity_matcher import find_matches


@dataclass
class YouTubeVideoEnhancement:
    """Enhancement data for a YouTube video."""

    video_id: str
    original_title: str
    normalized_title: Optional[str] = None
    primary_artist_id: Optional[int] = None
    related_isrc: Optional[str] = None
    confidence_score: Optional[float] = None
    enhancement_notes: Optional[str] = None


class YouTubeEnhancer:
    """
    Enhances YouTube video data using fuzzy matching against existing database entities.

    NEVER creates new artists or songs - only links to existing ones.
    """

    def __init__(self, engine):
        self.engine = engine
        self.primary_artists = self._load_primary_artists()
        self.songs_catalog = self._load_songs_catalog()

    def _load_primary_artists(self) -> Dict[int, str]:
        """Load primary artists from database (those with primary role)."""
        from sqlalchemy import text

        with self.engine.connect() as conn:
            # Only get artists who have primary artist roles (assuming role_id 1 is primary)
            result = conn.execute(
                text(
                    """
                SELECT DISTINCT a.artist_id, a.artist_name
                FROM artists a
                JOIN song_artist_roles sar ON a.artist_id = sar.artist_id
                WHERE sar.role_id = 1
                ORDER BY a.artist_name
            """
                )
            )

            return {row.artist_id: row.artist_name for row in result.fetchall()}

    def _load_songs_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load songs catalog for potential linking."""
        from sqlalchemy import text

        with self.engine.connect() as conn:
            # First check what role_id represents primary artist
            result = conn.execute(
                text(
                    """
                SELECT sar.role_id, COUNT(*) as count
                FROM song_artist_roles sar
                GROUP BY sar.role_id
                ORDER BY count DESC
            """
                )
            )

            role_usage = result.fetchall()
            # Assume role_id 1 is primary artist (most common)
            primary_role_id = role_usage[0].role_id if role_usage else 1

            result = conn.execute(
                text(
                    """
                SELECT
                    s.isrc,
                    s.song_title,
                    GROUP_CONCAT(DISTINCT a.artist_name ORDER BY a.artist_name) as primary_artists
                FROM songs s
                LEFT JOIN song_artist_roles sar ON s.isrc = sar.isrc AND sar.role_id = :role_id
                LEFT JOIN artists a ON sar.artist_id = a.artist_id
                GROUP BY s.isrc, s.song_title
                HAVING primary_artists IS NOT NULL
                ORDER BY s.song_title
            """
                ),
                {"role_id": primary_role_id},
            )

            catalog = {}
            for row in result.fetchall():
                catalog[row.isrc] = {
                    "isrc": row.isrc,
                    "song_title": row.song_title,
                    "primary_artists": row.primary_artists or "",
                }

            return catalog

    def enhance_youtube_video(self, video_data: Dict[str, Any]) -> YouTubeVideoEnhancement:
        """
        Enhance a single YouTube video with normalized data.

        Args:
            video_data: Dict with keys: video_id, title, channel_name, description

        Returns:
            YouTubeVideoEnhancement with matched data
        """
        video_id = video_data["video_id"]
        original_title = video_data["title"]
        channel_name = video_data.get("channel_name", "")
        description = video_data.get("description", "")

        enhancement = YouTubeVideoEnhancement(video_id=video_id, original_title=original_title)

        # Step 1: Extract and normalize title using existing YouTube parser
        normalized_title = self._normalize_youtube_title(original_title)
        enhancement.normalized_title = normalized_title

        # Step 2: Match to primary artist (NEVER create new artists)
        primary_artist_id, artist_confidence = self._match_primary_artist(
            original_title, channel_name, description
        )
        enhancement.primary_artist_id = primary_artist_id
        enhancement.confidence_score = artist_confidence

        # Step 3: Try to link to existing song (optional)
        related_isrc, song_confidence = self._match_to_song(normalized_title, primary_artist_id)
        enhancement.related_isrc = related_isrc

        # Step 4: Add enhancement notes
        notes = []
        if primary_artist_id:
            artist_name = self.primary_artists[primary_artist_id]
            notes.append(f"Matched to artist: {artist_name}")
        if related_isrc:
            song_info = self.songs_catalog.get(related_isrc)
            if song_info:
                notes.append(f"Linked to song: {song_info['song_title']} ({related_isrc})")
        if not primary_artist_id:
            notes.append("No primary artist match found")

        enhancement.enhancement_notes = "; ".join(notes)

        return enhancement

    def _normalize_youtube_title(self, title: str) -> str:
        """
        Normalize YouTube title to clean song name.

        Uses existing YouTube parser logic but focuses on extracting clean song title.
        """
        # Import your existing YouTube parser
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "web"))
            from youtube_version_parser import parse_youtube_title

            # Use your elite parser
            parsed = parse_youtube_title(title, "")
            clean_title = parsed.get("title", title)

            # Additional cleanup for song matching
            clean_title = self._clean_for_song_matching(clean_title)

            return clean_title

        except ImportError:
            # Fallback to basic cleaning if parser not available
            return self._basic_title_cleanup(title)

    def _clean_for_song_matching(self, title: str) -> str:
        """Clean title specifically for song matching."""
        import re

        # Remove common YouTube noise
        title = re.sub(r"\(Official.*?\)", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\[Official.*?\]", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\(.*?Video\)", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\(.*?Audio\)", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\(Lyric.*?\)", "", title, flags=re.IGNORECASE)

        # Remove version indicators but keep them in normalized_title
        # (we want to track "Song (Slowed)" but match to "Song")
        version_indicators = [
            r"\(Slowed.*?\)",
            r"\(Sped Up.*?\)",
            r"\(Remix.*?\)",
            r"\(Acoustic.*?\)",
            r"\(Live.*?\)",
            r"\(Extended.*?\)",
        ]

        for pattern in version_indicators:
            title = re.sub(pattern, "", title, flags=re.IGNORECASE)

        # Clean whitespace
        title = re.sub(r"\s+", " ", title).strip()

        return title

    def _basic_title_cleanup(self, title: str) -> str:
        """Basic title cleanup if parser not available."""
        import re

        # Remove artist part if separated by " - "
        if " - " in title:
            title = title.split(" - ", 1)[1]

        # Remove common suffixes
        title = re.sub(r"\(Official.*?\)", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\[Official.*?\]", "", title, flags=re.IGNORECASE)

        return title.strip()

    def _match_primary_artist(
        self, title: str, channel_name: str, description: str
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Match to existing primary artist (NEVER create new artists).

        Returns:
            Tuple of (artist_id, confidence_score) or (None, None)
        """
        if not self.primary_artists:
            return None, None

        # Get list of primary artist names for matching
        artist_names = list(self.primary_artists.values())

        # Try different matching strategies
        candidates_to_try = []

        # Strategy 1: Channel name (highest priority)
        if channel_name:
            # Clean channel name (remove VEVO, Official, etc.)
            clean_channel = (
                channel_name.replace("VEVO", "")
                .replace("Official", "")
                .replace(" - Topic", "")
                .strip()
            )
            candidates_to_try.append((clean_channel, 0.9))  # High confidence for channel match

        # Strategy 2: Extract artist from title using your parser
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "web"))
            from youtube_version_parser import parse_youtube_title

            parsed = parse_youtube_title(title, channel_name)
            primary_artists = parsed.get("primary", [])

            for artist in primary_artists:
                candidates_to_try.append((artist, 0.8))  # Good confidence for parsed artist

        except ImportError:
            # Fallback: extract from title manually
            if " - " in title:
                potential_artist = title.split(" - ")[0].strip()
                candidates_to_try.append((potential_artist, 0.7))  # Medium confidence

        # Try fuzzy matching for each candidate
        best_match = None
        best_score = 0.0
        best_artist_id = None

        for candidate, base_confidence in candidates_to_try:
            if not candidate:
                continue

            # Use fuzzy matching with high threshold (we only want good matches)
            matches = find_matches(
                candidate,
                artist_names,
                threshold=0.85,  # High threshold - only confident matches
                include_scores=True,
                limit=1,
            )

            if matches:
                matched_name, similarity_score = matches[0]
                # Combined confidence: base confidence * similarity score
                combined_score = base_confidence * similarity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = matched_name
                    # Find artist_id for this name
                    for artist_id, name in self.primary_artists.items():
                        if name == matched_name:
                            best_artist_id = artist_id
                            break

        return best_artist_id, best_score if best_artist_id else None

    def _match_to_song(
        self, normalized_title: str, primary_artist_id: Optional[int]
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Try to match normalized title to existing song.

        Only matches if we have a primary artist match for additional confidence.
        """
        if not normalized_title or not primary_artist_id:
            return None, None

        # Get artist name for filtering
        artist_name = self.primary_artists.get(primary_artist_id)
        if not artist_name:
            return None, None

        # Filter songs to those involving this artist
        relevant_songs = []
        for song_info in self.songs_catalog.values():
            if artist_name.lower() in song_info["primary_artists"].lower():
                relevant_songs.append(song_info)

        if not relevant_songs:
            return None, None

        # Try fuzzy matching against song titles
        song_titles = [song["song_title"] for song in relevant_songs]

        matches = find_matches(
            normalized_title,
            song_titles,
            threshold=0.90,  # Very high threshold for song matching
            include_scores=True,
            limit=1,
        )

        if matches:
            matched_title, score = matches[0]
            # Find the ISRC for this title
            for song_info in relevant_songs:
                if song_info["song_title"] == matched_title:
                    return song_info["isrc"], score

        return None, None

    def enhance_batch(self, video_batch: List[Dict[str, Any]]) -> List[YouTubeVideoEnhancement]:
        """Enhance a batch of YouTube videos."""
        enhancements = []

        for video_data in video_batch:
            try:
                enhancement = self.enhance_youtube_video(video_data)
                enhancements.append(enhancement)
            except Exception as e:
                # Log error but continue processing
                print(f"Error enhancing video {video_data.get('video_id', 'unknown')}: {e}")
                # Create minimal enhancement
                enhancement = YouTubeVideoEnhancement(
                    video_id=video_data.get("video_id", ""),
                    original_title=video_data.get("title", ""),
                    enhancement_notes=f"Enhancement failed: {str(e)}",
                )
                enhancements.append(enhancement)

        return enhancements

    def update_youtube_videos_table(
        self, enhancements: List[YouTubeVideoEnhancement]
    ) -> Dict[str, int]:
        """
        Update youtube_videos table with enhancement data.

        Adds columns: normalized_title, primary_artist_id, song_id
        """
        from sqlalchemy import text

        stats = {"updated": 0, "errors": 0}

        with self.engine.begin() as conn:
            for enhancement in enhancements:
                try:
                    # Update the youtube_videos table
                    conn.execute(
                        text(
                            """
                        UPDATE youtube_videos
                        SET
                            normalized_title = :normalized_title,
                            primary_artist_id = :primary_artist_id,
                            related_isrc = :related_isrc,
                            enhancement_confidence = :confidence_score,
                            enhancement_notes = :enhancement_notes,
                            enhanced_at = NOW()
                        WHERE video_id = :video_id
                    """
                        ),
                        {
                            "video_id": enhancement.video_id,
                            "normalized_title": enhancement.normalized_title,
                            "primary_artist_id": enhancement.primary_artist_id,
                            "related_isrc": enhancement.related_isrc,
                            "confidence_score": enhancement.confidence_score,
                            "enhancement_notes": enhancement.enhancement_notes,
                        },
                    )

                    stats["updated"] += 1

                except Exception as e:
                    print(f"Error updating video {enhancement.video_id}: {e}")
                    stats["errors"] += 1

        return stats


def demonstrate_youtube_enhancement():
    """Demonstrate YouTube video enhancement with real database integration."""
    print("🎬 YouTube Video Enhancement Demo")
    print("=" * 50)

    # This would use your real database connection
    try:
        # Import your database connection
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "web"))
        from db_guard import get_engine

        engine = get_engine("PUBLIC", ro=False)  # Need write access for updates

        enhancer = YouTubeEnhancer(engine)

        print(f"📊 Loaded {len(enhancer.primary_artists)} primary artists")
        print(f"🎵 Loaded {len(enhancer.songs_catalog)} songs from catalog")

        # Get sample YouTube videos from database
        with engine.connect() as conn:
            from sqlalchemy import text

            result = conn.execute(
                text(
                    """
                SELECT video_id, title, channel_name, description
                FROM youtube_videos
                WHERE normalized_title IS NULL  -- Only unprocessed videos
                LIMIT 10
            """
                )
            )

            sample_videos = [
                {
                    "video_id": row.video_id,
                    "title": row.title,
                    "channel_name": row.channel_name or "",
                    "description": row.description or "",
                }
                for row in result.fetchall()
            ]

        if not sample_videos:
            print("⚠️  No unprocessed YouTube videos found in database")
            return

        print(f"\n🔄 Processing {len(sample_videos)} YouTube videos...")

        # Enhance the videos
        enhancements = enhancer.enhance_batch(sample_videos)

        # Show results
        print("\n📋 Enhancement Results:")
        print("-" * 80)

        for enhancement in enhancements:
            print(f"\n📹 Video: {enhancement.video_id}")
            print(f"   Original: {enhancement.original_title}")
            print(f"   Normalized: {enhancement.normalized_title}")

            if enhancement.primary_artist_id:
                artist_name = enhancer.primary_artists[enhancement.primary_artist_id]
                print(f"   🎤 Primary Artist: {artist_name} (ID: {enhancement.primary_artist_id})")
            else:
                print("   🎤 Primary Artist: Not matched")

            if enhancement.related_isrc:
                print(f"   🎵 Linked Song ISRC: {enhancement.related_isrc}")
            else:
                print("   🎵 Linked Song: Not matched")

            if enhancement.confidence_score:
                print(f"   📊 Confidence: {enhancement.confidence_score:.2%}")

            if enhancement.enhancement_notes:
                print(f"   📝 Notes: {enhancement.enhancement_notes}")

        # Update database
        print("\n💾 Updating database...")
        stats = enhancer.update_youtube_videos_table(enhancements)

        print(f"✅ Updated {stats['updated']} videos")
        if stats["errors"] > 0:
            print(f"❌ {stats['errors']} errors occurred")

        # Show summary statistics
        print("\n📈 Summary Statistics:")
        matched_artists = sum(1 for e in enhancements if e.primary_artist_id)
        linked_songs = sum(1 for e in enhancements if e.related_isrc)

        print(
            f"   Artist Match Rate: {matched_artists}/{len(enhancements)} ({matched_artists/len(enhancements):.1%})"
        )
        print(
            f"   Song Link Rate: {linked_songs}/{len(enhancements)} ({linked_songs/len(enhancements):.1%})"
        )

    except ImportError as e:
        print(f"⚠️  Database connection not available: {e}")
        print("💡 This demo requires access to your database schema")
    except Exception as e:
        print(f"❌ Error: {e}")


def create_database_schema_additions():
    """
    Generate SQL to add enhancement columns to youtube_videos table.
    """
    print("🗄️  Database Schema Additions")
    print("=" * 40)

    sql_additions = """
-- Add enhancement columns to youtube_videos table
ALTER TABLE youtube_videos
ADD COLUMN normalized_title VARCHAR(500) NULL COMMENT 'Clean song title for matching',
ADD COLUMN primary_artist_id INT NULL COMMENT 'Foreign key to artists table (primary artist only)',
ADD COLUMN song_id INT NULL COMMENT 'Foreign key to songs table (one-to-many relationship)',
ADD COLUMN enhancement_confidence DECIMAL(4,3) NULL COMMENT 'Confidence score for matches (0.000-1.000)',
ADD COLUMN enhancement_notes TEXT NULL COMMENT 'Notes about the enhancement process',
ADD COLUMN enhanced_at TIMESTAMP NULL COMMENT 'When enhancement was performed';

-- Add foreign key constraints
ALTER TABLE youtube_videos
ADD CONSTRAINT fk_youtube_videos_primary_artist
    FOREIGN KEY (primary_artist_id) REFERENCES artists(artist_id)
    ON DELETE SET NULL ON UPDATE CASCADE,
ADD CONSTRAINT fk_youtube_videos_song
    FOREIGN KEY (song_id) REFERENCES songs(song_id)
    ON DELETE SET NULL ON UPDATE CASCADE;

-- Add indexes for performance
CREATE INDEX idx_youtube_videos_normalized_title ON youtube_videos(normalized_title);
CREATE INDEX idx_youtube_videos_primary_artist_id ON youtube_videos(primary_artist_id);
CREATE INDEX idx_youtube_videos_song_id ON youtube_videos(song_id);
CREATE INDEX idx_youtube_videos_enhanced_at ON youtube_videos(enhanced_at);

-- Create view for enhanced YouTube videos with artist and song info
CREATE VIEW youtube_videos_enhanced AS
SELECT
    yv.video_id,
    yv.title as original_title,
    yv.normalized_title,
    yv.channel_name,
    yv.description,
    yv.view_count,
    yv.like_count,
    yv.comment_count,
    yv.published_at,
    yv.duration_seconds,

    -- Artist information
    a.artist_name as primary_artist_name,
    yv.primary_artist_id,

    -- Song information
    s.song_title,
    s.isrc,
    yv.song_id,

    -- Enhancement metadata
    yv.enhancement_confidence,
    yv.enhancement_notes,
    yv.enhanced_at,

    -- Calculated fields
    CASE
        WHEN yv.song_id IS NOT NULL THEN 'Linked to Song'
        WHEN yv.primary_artist_id IS NOT NULL THEN 'Artist Matched'
        ELSE 'Unmatched'
    END as enhancement_status

FROM youtube_videos yv
LEFT JOIN artists a ON yv.primary_artist_id = a.artist_id
LEFT JOIN songs s ON yv.song_id = s.song_id;
"""

    print("SQL to add enhancement columns:")
    print("-" * 40)
    print(sql_additions)

    return sql_additions


if __name__ == "__main__":
    print("🎵 YouTube Enhancement Integration")
    print("=" * 50)
    print()
    print("This integration enhances YouTube video data by:")
    print("✅ Adding normalized_title (clean song names)")
    print("✅ Adding primary_artist_id (links to existing artists only)")
    print("✅ Adding song_id (optional link to songs table)")
    print("✅ NEVER pollutes artists or songs tables")
    print("✅ Creates one-to-many relationship: song -> YouTube videos")
    print()

    # Show database schema additions
    create_database_schema_additions()
    print()

    # Run demonstration
    demonstrate_youtube_enhancement()

    print("\n✅ YouTube Enhancement Integration Complete!")
    print("\n💡 Usage Tips:")
    print("   • Run this after your existing YouTube ETL pipeline")
    print("   • Only processes videos where normalized_title IS NULL")
    print("   • High thresholds ensure only confident matches")
    print("   • One song can have many YouTube videos (official, slowed, etc.)")
    print("   • Query both directions: song->videos or video->song")
    print("   • Artists and songs tables remain clean fact tables")
