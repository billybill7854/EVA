"""
Database migration script
Handles database schema migrations
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from config.settings import get_settings

settings = get_settings()


def migrate_database():
    """Run database migrations"""
    try:
        # Create database engine
        engine = create_engine(settings.database_url, echo=True)
        
        with engine.connect() as conn:
            # Example migration: Add new columns
            # This is a placeholder for future migrations
            
            # Check if column exists
            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'is_stranger'"))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE users ADD COLUMN is_stranger BOOLEAN DEFAULT TRUE"))
                conn.commit()
                print("Added is_stranger column to users table")
            
            print("Database migration completed successfully!")
        
        return engine
    except Exception as e:
        print(f"Error running database migration: {str(e)}")
        raise


if __name__ == "__main__":
    migrate_database()
