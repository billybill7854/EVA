"""
Configuration settings for EVA Telegram Bot
Loads from .env file and environment variables
"""
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Main application settings"""
    
    # App Info
    app_name: str = "EVA Telegram Bot"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Telegram
    telegram_bot_token: str = "test_token"
    telegram_webhook_url: str = "http://localhost:8000"

    # Admin API (FastAPI /admin/* endpoints)
    # Set ADMIN_API_KEY to a long random string in production.
    admin_api_key: str = ""
    
    # NVIDIA APIs
    nvidia_api_key_main: str = "test_key_main"
    nvidia_api_key_tts: str = "test_key_tts"
    nvidia_api_key_stt: str = "test_key_stt"
    
    # NVIDIA Models
    nvidia_orchestrator_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_agent_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_chat_model_1: str = "google/diffusiongemma-26b-a4b-it"
    nvidia_chat_model_2: str = "moonshotai/kimi-k2.6"
    nvidia_tts_model: str = "magpie-tts-multilingual"
    nvidia_stt_model: str = "whisper-large-v3"
    nvidia_image_gen_model: str = "FLUX.1-dev"
    
    # PostgreSQL
    database_url: str = "sqlite:///eva.db"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "eva_db"
    database_user: str = "eva_user"
    database_password: str = "eva_password"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Pinecone
    pinecone_api_key: str = "test_key"
    pinecone_environment: str = "gcp-starter"
    pinecone_index_name: str = "eva-memory"
    
    # Google APIs
    google_client_id: str = "test_client_id"
    google_client_secret: str = "test_client_secret"
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    gmail_service_account_email: Optional[str] = None
    
    # Stripe
    stripe_api_key: str = "sk_test_dummy"
    stripe_webhook_secret: Optional[str] = None
    
    # Serper API (Google Search)
    serper_api_key: str = "test_serper_key"
    
    # Wise
    wise_api_token: Optional[str] = None
    wise_profile_id: Optional[str] = None
    
    # Primary User — Eva's owner
    # telegram_id: get yours by messaging @userinfobot on Telegram
    # Defaults are placeholders; set real values via env.
    primary_user_phone: str = ""
    primary_user_telegram_id: int = 0
    primary_user_username: str = ""          # your @username (without @)
    primary_user_name: str = ""              # your first name as it appears on Telegram
    
    # Admin API Key for secure admin endpoints
    admin_api_key: Optional[str] = None

    # Trusted contacts — people Eva treats with elevated access (not full owner access)
    # Comma-separated telegram IDs e.g. "111111111,222222222"
    trusted_contact_ids: str = ""

    # Eva's real Telegram user account (Pyrogram MTProto)
    # Get api_id and api_hash from https://my.telegram.org
    eva_phone: str = ""
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_session_name: str = "eva_session"
    telegram_session_dir: str = "."
    # Set to True once session file exists (first run requires OTP)
    telegram_user_mode: bool = False

    # Tool execution robustness
    tool_timeout_seconds: int = 20
    tool_max_calls_per_request: int = 3
    orchestrator_json_repair_attempts: int = 1
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
