#!/usr/bin/env python3
"""
Database Migration Script for Auto Eval Job Scheduler
====================================================

PURPOSE:
This script adds performance indexes to an existing SQLite database for the Auto Eval
job scheduler system. It's designed to optimize query performance on large databases
(100+ MB) that were created before the index optimizations were added to the schema.

WHAT IT DOES:
- Adds an index on the 'submitted_at' column for faster job ordering/sorting
- Adds an index on the 'submitter_id' column for faster user-specific job filtering
- Runs ANALYZE to update SQLite query planner statistics
- Provides feedback on migration progress and database statistics

WHEN TO USE:
- You have an existing database.db file that's slow to load
- You've updated job_scheduler.py with the new indexed schema
- Your job tables are taking a long time to load in the web interface

EXAMPLE USAGE:
    python scripts/db_utils/migrate_database.py

EXPECTED OUTPUT:
    Migrating database database.db...
    Adding index on submitted_at column...
    ✓ Index on submitted_at created
    Adding index on submitter_id column...
    ✓ Index on submitter_id created
    Analyzing database for query optimization...
    ✓ Database migration completed successfully!
    📊 Total jobs in database: 1234
    📊 Database file size: 105.67 MB

SAFETY:
- Safe to run multiple times (checks if indexes already exist)
- Does not modify existing data, only adds indexes
- Creates a backup is recommended but not required (indexes can be dropped if needed)
"""

import os
import sqlite3
import sys


def migrate_database(db_path=None):
    """
    Add indexes to the database for better performance.

    Args:
        db_path (str, optional): Path to the database file.
                               Defaults to "database.db" in current directory.
    """
    if db_path is None:
        # Default to database.db in the current working directory
        db_file = "database.db"
        # If running from scripts/db_utils, look for database in project root
        if os.path.basename(os.getcwd()) == "db_utils":
            db_file = "../../database.db"
    else:
        db_file = db_path

    # Resolve relative paths
    db_file = os.path.abspath(db_file)

    if not os.path.exists(db_file):
        print(f"❌ Database file not found: {db_file}")
        print(
            f"💡 Make sure you're running this script from the project root directory,"
        )
        print(f"   or provide the correct path to your database.db file.")
        print(f"")
        print(f"Examples:")
        print(f"   python scripts/db_utils/migrate_database.py")
        print(f"   python scripts/db_utils/migrate_database.py /path/to/database.db")
        sys.exit(1)

    print(f"🔧 Migrating database: {db_file}")
    print(f"📁 Database location: {os.path.dirname(db_file)}")

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if indexes already exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_job_submitted_at';"
        )
        submitted_at_index_exists = cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_job_submitter_id';"
        )
        submitter_id_index_exists = cursor.fetchone() is not None

        # Add submitted_at index if it doesn't exist
        if not submitted_at_index_exists:
            print("⚡ Adding index on submitted_at column...")
            cursor.execute("CREATE INDEX ix_job_submitted_at ON job (submitted_at);")
            print("✓ Index on submitted_at created")
        else:
            print("✓ Index on submitted_at already exists")

        # Add submitter_id index if it doesn't exist
        if not submitter_id_index_exists:
            print("⚡ Adding index on submitter_id column...")
            cursor.execute("CREATE INDEX ix_job_submitter_id ON job (submitter_id);")
            print("✓ Index on submitter_id created")
        else:
            print("✓ Index on submitter_id already exists")

        # Commit changes
        conn.commit()

        # Analyze the database to update statistics for the query planner
        print("🔍 Analyzing database for query optimization...")
        cursor.execute("ANALYZE;")
        conn.commit()

        print("🎉 Database migration completed successfully!")
        print("")

        # Show some database statistics
        cursor.execute("SELECT COUNT(*) FROM job;")
        total_jobs = cursor.fetchone()[0]
        print(f"📊 Total jobs in database: {total_jobs:,}")

        # Get database file size
        file_size = os.path.getsize(db_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"📊 Database file size: {file_size_mb:.2f} MB")

        if total_jobs > 1000:
            print("")
            print("💡 With these indexes, your job tables should load much faster!")
            print("   Restart your FastAPI server to see the performance improvements.")

    except sqlite3.Error as e:
        print(f"❌ Error during migration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()


def main():
    """Main entry point for command line usage."""
    if len(sys.argv) > 2:
        print("Usage: python migrate_database.py [database_path]")
        print("       python migrate_database.py")
        print("       python migrate_database.py /path/to/database.db")
        sys.exit(1)

    db_path = sys.argv[1] if len(sys.argv) == 2 else None
    migrate_database(db_path)


if __name__ == "__main__":
    main()
