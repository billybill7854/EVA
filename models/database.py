"""
Data models for EVA Telegram Bot
Using SQLAlchemy ORM for database persistence
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel

Base = declarative_base()


# ====================== SQLAlchemy Models ======================

class UserDB(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(String, unique=True, index=True)
    phone_number = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    is_primary_user = Column(Boolean, default=False)
    is_stranger = Column(Boolean, default=True)
    
    # Personality preference
    preferred_personality = Column(String, default="general")
    personality_config = Column(JSON, default={})
    
    # Settings
    language = Column(String, default="en")
    timezone = Column(String, default="UTC")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conversations = relationship("ConversationDB", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("MemoryDB", back_populates="user", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCallDB", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username or self.telegram_id}>"


class ConversationDB(Base):
    """Conversation history database model"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Message content
    message_type = Column(String, default="text")  # text, voice, image, etc
    user_message = Column(Text)
    eva_response = Column(Text, nullable=True)
    
    # Metadata
    detected_personality = Column(String, default="general")
    detected_intent = Column(String, nullable=True)
    tools_used = Column(JSON, default=[])
    
    # Context
    session_id = Column(String, nullable=True)
    context_data = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("UserDB", back_populates="conversations")
    
    def __repr__(self):
        return f"<Conversation {self.id}>"


class MemoryDB(Base):
    """Long-term memory database model (semantic + user preferences)"""
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Memory content
    memory_type = Column(String)  # preference, fact, habit, goal, relationship, etc
    content = Column(Text)
    embedding = Column(JSON, nullable=True)  # Vector embedding for semantic search
    
    # Importance & retention
    importance_score = Column(Integer, default=1)  # 1-10 scale
    last_referenced = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    
    # Metadata
    source = Column(String, nullable=True)  # From which conversation/tool
    tags = Column(JSON, default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete for privacy
    is_forgotten = Column(Boolean, default=False)
    forgotten_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("UserDB", back_populates="memories")
    
    def __repr__(self):
        return f"<Memory {self.id}: {self.memory_type}>"


class ToolCallDB(Base):
    """Tool execution log database model"""
    __tablename__ = "tool_calls"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Tool info
    tool_name = Column(String, index=True)
    action = Column(String)  # e.g., "send_email", "schedule_meeting"
    
    # Execution details
    status = Column(String)  # pending, executing, success, failed
    input_data = Column(JSON)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("UserDB", back_populates="tool_calls")
    
    def __repr__(self):
        return f"<ToolCall {self.id}: {self.tool_name}/{self.action}>"


# ====================== Pydantic Models ======================

class UserBase(BaseModel):
    """User base schema"""
    telegram_id: str
    phone_number: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema"""
    pass


class UserUpdate(BaseModel):
    """User update schema"""
    preferred_personality: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    personality_config: Optional[Dict[str, Any]] = None


class UserResponse(UserBase):
    """User response schema"""
    id: int
    is_primary_user: bool
    preferred_personality: str
    created_at: datetime
    last_active: datetime
    
    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    """Message creation schema"""
    message_type: str = "text"
    user_message: str
    session_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """Conversation response schema"""
    id: int
    message_type: str
    user_message: str
    eva_response: Optional[str] = None
    detected_personality: str
    detected_intent: Optional[str] = None
    tools_used: List[str] = []
    created_at: datetime
    
    class Config:
        from_attributes = True


class MemoryCreate(BaseModel):
    """Memory creation schema"""
    memory_type: str
    content: str
    importance_score: int = 1
    tags: List[str] = []
    source: Optional[str] = None


class MemoryResponse(BaseModel):
    """Memory response schema"""
    id: int
    memory_type: str
    content: str
    importance_score: int
    last_referenced: Optional[datetime] = None
    access_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ToolCallCreate(BaseModel):
    """Tool call creation schema"""
    tool_name: str
    action: str
    input_data: Dict[str, Any]


class ToolCallResponse(BaseModel):
    """Tool call response schema"""
    id: int
    tool_name: str
    action: str
    status: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
