"""
Database initialization script
Creates all tables and initializes the database
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from config.settings import get_settings
from models.database import Base

settings = get_settings()


def init_database():
    """Initialize database with all tables"""
    try:
        # Create database engine
        engine = create_engine(settings.database_url, echo=True)
        
        # Create all tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        print("Database initialized successfully!")
        print(f"Database URL: {settings.database_url}")
        
        return engine
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise


if __name__ == "__main__":
    init_database()
