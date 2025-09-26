#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Test database connection for YouTube integration.

Run this to verify that the fuzzy matcher can connect to your database
and access real YouTube titles for testing.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    print("🔧 Environment variables loaded from .env")

    # Check for database credentials
    db_url = os.getenv("DATABASE_URL")
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    if db_url:
        print(f"✅ Found DATABASE_URL: {db_url[:20]}...")
    elif db_host and db_user and db_name:
        print(f"✅ Found DB credentials: {db_user}@{db_host}/{db_name}")
    else:
        print("❌ No database credentials found in .env")
        print("Expected: DATABASE_URL or DB_HOST, DB_USER, DB_PASSWORD, DB_NAME")
        sys.exit(1)

    # Try to import and connect
    from sqlalchemy import text

    # DEPRECATED: # DEPRECATED: from web.db_guard import
# Use SQLAlchemy directly get_engine
# Use SQLAlchemy create_engine directly

    print("📦 Imported database modules successfully")

    # Test connection
    engine = get_engine(schema="PUBLIC")
    with engine.begin() as conn:
        # Test basic query
        result = conn.execute(text("SELECT COUNT(*) as count FROM youtube_videos"))
        youtube_count = result.fetchone()[0]

        result = conn.execute(text("SELECT COUNT(*) as count FROM artists"))
        artist_count = result.fetchone()[0]

        print(f"🎬 Found {youtube_count:,} YouTube videos in database")
        print(f"🎤 Found {artist_count:,} artists in database")

        # Get a sample of real YouTube titles
        result = conn.execute(
            text(
                """
            SELECT title, channel_title
            FROM youtube_videos
            WHERE title IS NOT NULL
            AND title != ''
            AND channel_title IS NOT NULL
            ORDER BY fetched_at DESC
            LIMIT 5
        """
            )
        )

        print("\n📺 Sample YouTube titles from your database:")
        for i, row in enumerate(result.fetchall(), 1):
            title, channel = row
            print(f"   {i}. {title}")
            print(f"      Channel: {channel}")

        print("\n✅ Database connection successful!")
        print("You can now run the YouTube integration tests.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Database connection error: {e}")
    print("Check your .env file and database credentials")
    sys.exit(1)
