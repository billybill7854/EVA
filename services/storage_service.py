import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import select, update, delete, and_, desc
    from models.database import Base, UserDB, ConversationDB, MemoryDB, ToolCallDB
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logger.warning("SQLAlchemy not available — falling back to in-memory storage")

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    logger.warning("Pinecone not available — semantic memory features disabled")

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis not available — short-term memory features disabled")

try:
    from config.settings import get_settings
    settings = get_settings()
except Exception as e:
    logger.warning(f"Settings not available: {e}")
    settings = None


class StorageService:

    def __init__(self):
        self._engine = None
        self._session_factory = None
        self.redis_client = None
        self.pinecone_index = None

        # In-memory fallback (used when SQLAlchemy not available)
        self._mem_users: Dict[str, Dict] = {}
        self._mem_conversations: Dict[int, Dict] = {}
        self._mem_memories: Dict[int, Dict] = {}
        self._mem_tool_calls: Dict[int, Dict] = {}
        self._counters = {"user": 0, "conv": 0, "mem": 0, "tool": 0}

    async def initialize(self):
        if not HAS_SQLALCHEMY:
            logger.warning("Running with in-memory storage — data will not persist")
            return

        db_url = settings.database_url if settings else "sqlite+aiosqlite:///eva.db"

        # Normalise URL for async drivers
        if db_url.startswith("sqlite:///"):
            db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self._engine = create_async_engine(db_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info(f"Storage initialised: {db_url}")

        # Redis (optional)
        if HAS_REDIS and settings and settings.redis_url:
            try:
                self.redis_client = await aioredis.from_url(settings.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Redis connected")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None

        # Pinecone (optional)
        if HAS_PINECONE and settings and settings.pinecone_api_key:
            try:
                pc = Pinecone(api_key=settings.pinecone_api_key)
                self.pinecone_index = pc.Index(settings.pinecone_index_name)
                logger.info(f"Pinecone connected to index: {settings.pinecone_index_name}")
            except Exception as e:
                logger.warning(f"Pinecone not available: {e}")
                self.pinecone_index = None

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------

    def _session(self) -> AsyncSession:
        if not self._session_factory:
            raise RuntimeError("DB not initialised")
        return self._session_factory()

    @property
    def _use_db(self) -> bool:
        return HAS_SQLALCHEMY and self._session_factory is not None

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def get_or_create_user(
        self,
        telegram_id: str,
        first_name: Optional[str] = None,
        username: Optional[str] = None,
        phone_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._use_db:
            return await self._db_get_or_create_user(telegram_id, first_name, username, phone_number)
        return self._mem_get_or_create_user(telegram_id, first_name, username, phone_number)

    async def _db_get_or_create_user(self, telegram_id, first_name, username, phone_number):
        async with self._session() as s:
            result = await s.execute(select(UserDB).where(UserDB.telegram_id == telegram_id))
            user = result.scalar_one_or_none()

            if user:
                user.last_active = datetime.utcnow()
                if first_name: user.first_name = first_name
                if username: user.username = username
                await s.commit()
                return self._user_to_dict(user)

            is_primary = settings and telegram_id == str(settings.primary_user_telegram_id)
            user = UserDB(
                telegram_id=telegram_id,
                first_name=first_name,
                username=username,
                phone_number=phone_number,
                is_primary_user=bool(is_primary),
                is_stranger=not bool(is_primary),
            )
            s.add(user)
            await s.commit()
            await s.refresh(user)
            logger.info(f"Created user: {telegram_id}")
            return self._user_to_dict(user)

    def _mem_get_or_create_user(self, telegram_id, first_name, username, phone_number):
        if telegram_id in self._mem_users:
            self._mem_users[telegram_id]["last_active"] = datetime.utcnow().isoformat()
            return self._mem_users[telegram_id]
        self._counters["user"] += 1
        is_primary = settings and telegram_id == str(settings.primary_user_telegram_id)
        user = {
            "id": self._counters["user"], "telegram_id": telegram_id,
            "first_name": first_name, "username": username, "phone_number": phone_number,
            "is_primary_user": bool(is_primary), "is_stranger": not bool(is_primary),
            "preferred_personality": "general", "personality_config": {},
            "language": "en", "timezone": "UTC",
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
        }
        self._mem_users[telegram_id] = user
        return user

    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        return await self.get_user_by_id(user_id)

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(select(UserDB).where(UserDB.id == user_id))
                user = result.scalar_one_or_none()
                return self._user_to_dict(user) if user else None
        for u in self._mem_users.values():
            if u["id"] == user_id:
                return u
        return None

    async def get_primary_users(self) -> List[Dict[str, Any]]:
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(select(UserDB).where(UserDB.is_primary_user == True))
                return [self._user_to_dict(u) for u in result.scalars().all()]
        return [u for u in self._mem_users.values() if u.get("is_primary_user")]

    async def update_user_personality(self, user_id: int, personality: str, config: Optional[Dict] = None) -> bool:
        if self._use_db:
            async with self._session() as s:
                await s.execute(
                    update(UserDB).where(UserDB.id == user_id).values(
                        preferred_personality=personality,
                        **({"personality_config": config} if config else {})
                    )
                )
                await s.commit()
                return True
        user = await self.get_user_by_id(user_id)
        if user:
            user["preferred_personality"] = personality
            if config: user["personality_config"] = config
            return True
        return False

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def create_conversation(
        self, user_id: int, message_type: str, user_message: str,
        personality: str, intent: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._use_db:
            async with self._session() as s:
                conv = ConversationDB(
                    user_id=user_id, message_type=message_type, user_message=user_message,
                    detected_personality=personality, detected_intent=intent, session_id=session_id,
                )
                s.add(conv)
                await s.commit()
                await s.refresh(conv)
                return self._conv_to_dict(conv)
        self._counters["conv"] += 1
        conv = {
            "id": self._counters["conv"], "user_id": user_id,
            "message_type": message_type, "user_message": user_message,
            "eva_response": None, "detected_personality": personality,
            "detected_intent": intent, "tools_used": [], "session_id": session_id,
            "context_data": {}, "created_at": datetime.utcnow().isoformat(), "completed_at": None,
        }
        self._mem_conversations[self._counters["conv"]] = conv
        return conv

    async def update_conversation_response(
        self, conversation_id: int, eva_response: str, tools_used: Optional[List[str]] = None,
    ) -> bool:
        if self._use_db:
            async with self._session() as s:
                await s.execute(
                    update(ConversationDB).where(ConversationDB.id == conversation_id).values(
                        eva_response=eva_response, tools_used=tools_used or [],
                        completed_at=datetime.utcnow(),
                    )
                )
                await s.commit()
                return True
        if conversation_id in self._mem_conversations:
            self._mem_conversations[conversation_id].update({
                "eva_response": eva_response, "tools_used": tools_used or [],
                "completed_at": datetime.utcnow().isoformat(),
            })
            return True
        return False

    async def get_conversation_history(
        self, user_id: int, limit: int = 50, days: int = 30,
    ) -> List[Dict[str, Any]]:
        since = datetime.utcnow() - timedelta(days=days)
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(
                    select(ConversationDB)
                    .where(and_(ConversationDB.user_id == user_id, ConversationDB.created_at >= since))
                    .order_by(desc(ConversationDB.created_at))
                    .limit(limit)
                )
                return [self._conv_to_dict(c) for c in result.scalars().all()]
        convs = [c for c in self._mem_conversations.values() if c["user_id"] == user_id]
        return sorted(convs, key=lambda x: x["created_at"], reverse=True)[:limit]

    async def get_last_interaction(self, user_id: int) -> Optional[datetime]:
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(
                    select(ConversationDB.created_at)
                    .where(ConversationDB.user_id == user_id)
                    .order_by(desc(ConversationDB.created_at))
                    .limit(1)
                )
                row = result.scalar_one_or_none()
                return row
        convs = [c for c in self._mem_conversations.values() if c["user_id"] == user_id]
        if not convs: return None
        latest = max(convs, key=lambda x: x["created_at"])
        ts = latest["created_at"]
        return datetime.fromisoformat(ts) if isinstance(ts, str) else ts

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def create_memory(
        self, user_id: int, memory_type: str, content: str,
        importance_score: int = 1, tags: Optional[List[str]] = None, source: Optional[str] = None,
    ) -> Optional[int]:
        if self._use_db:
            async with self._session() as s:
                mem = MemoryDB(
                    user_id=user_id, memory_type=memory_type, content=content,
                    importance_score=importance_score, tags=tags or [], source=source,
                )
                s.add(mem)
                await s.commit()
                await s.refresh(mem)
                return mem.id
        self._counters["mem"] += 1
        self._mem_memories[self._counters["mem"]] = {
            "id": self._counters["mem"], "user_id": user_id, "memory_type": memory_type,
            "content": content, "importance_score": importance_score,
            "tags": tags or [], "source": source, "access_count": 0,
            "last_referenced": None, "is_forgotten": False, "forgotten_at": None,
            "created_at": datetime.utcnow().isoformat(), "updated_at": datetime.utcnow().isoformat(),
        }
        return self._counters["mem"]

    async def search_memories(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(
                    select(MemoryDB).where(
                        and_(
                            MemoryDB.user_id == user_id,
                            MemoryDB.is_forgotten == False,
                            MemoryDB.content.ilike(f"%{query}%"),
                        )
                    ).order_by(desc(MemoryDB.importance_score)).limit(limit)
                )
                mems = [self._mem_to_dict(m) for m in result.scalars().all()]
                return [m for m in mems if self._memory_not_expired(m)][:limit]
        results = [
            self._mem_to_dict_raw(m) for m in self._mem_memories.values()
            if m["user_id"] == user_id and not m["is_forgotten"]
            and query.lower() in m["content"].lower()
        ]
        results = [m for m in results if self._memory_not_expired(m)]
        return sorted(results, key=lambda x: x["importance_score"], reverse=True)[:limit]

    async def get_user_memories(
        self, user_id: int, memory_type: Optional[str] = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if self._use_db:
            async with self._session() as s:
                q = select(MemoryDB).where(
                    and_(MemoryDB.user_id == user_id, MemoryDB.is_forgotten == False)
                )
                if memory_type:
                    q = q.where(MemoryDB.memory_type == memory_type)
                q = q.order_by(desc(MemoryDB.importance_score)).limit(limit)
                result = await s.execute(q)
                mems = [self._mem_to_dict(m) for m in result.scalars().all()]
                mems = [m for m in mems if self._memory_not_expired(m)]
                return mems[:limit]
        mems = [
            self._mem_to_dict_raw(m) for m in self._mem_memories.values()
            if m["user_id"] == user_id and not m["is_forgotten"]
            and (memory_type is None or m["memory_type"] == memory_type)
        ]
        mems = [m for m in mems if self._memory_not_expired(m)]
        return sorted(mems, key=lambda x: x["importance_score"], reverse=True)[:limit]

    async def forget_memory(self, memory_id: int) -> bool:
        if self._use_db:
            async with self._session() as s:
                await s.execute(
                    update(MemoryDB).where(MemoryDB.id == memory_id).values(
                        is_forgotten=True, forgotten_at=datetime.utcnow()
                    )
                )
                await s.commit()
                return True
        if memory_id in self._mem_memories:
            self._mem_memories[memory_id]["is_forgotten"] = True
            self._mem_memories[memory_id]["forgotten_at"] = datetime.utcnow().isoformat()
            return True
        return False

    async def update_memory_access(self, memory_id: int) -> bool:
        if self._use_db:
            async with self._session() as s:
                result = await s.execute(select(MemoryDB).where(MemoryDB.id == memory_id))
                mem = result.scalar_one_or_none()
                if mem:
                    mem.access_count += 1
                    mem.last_referenced = datetime.utcnow()
                    await s.commit()
                    return True
            return False
        if memory_id in self._mem_memories:
            self._mem_memories[memory_id]["access_count"] += 1
            self._mem_memories[memory_id]["last_referenced"] = datetime.utcnow().isoformat()
            return True
        return False

    async def get_active_reminders(self, user_id: int) -> List[Dict[str, Any]]:
        return await self.get_user_memories(user_id, memory_type="reminder")

    async def get_user_interests(self, user_id: int) -> List[str]:
        mems = await self.get_user_memories(user_id, memory_type="interest")
        return [m.get("content", "") for m in mems if m.get("content")]
    
    async def get_primary_users(self) -> List[Dict[str, Any]]:
        """Get all primary users"""
        try:
            if self._use_db:
                async with self._session() as s:
                    result = await s.execute(
                        select(UserDB).where(UserDB.is_primary_user == True)
                    )
                    users = result.scalars().all()
                    return [
                        {
                            "id": user.id,
                            "telegram_id": user.telegram_id,
                            "username": user.username,
                            "first_name": user.first_name,
                            "preferred_personality": user.preferred_personality,
                        }
                        for user in users
                    ]
            else:
                # In-memory fallback
                return [
                    {**user, "id": user_id}
                    for user_id, user in self._mem_users.items()
                    if user.get("is_primary_user", False)
                ]
        except Exception as e:
            logger.error(f"Error getting primary users: {str(e)}")
            return []
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            if self._use_db:
                async with self._session() as s:
                    result = await s.execute(
                        select(UserDB).where(UserDB.id == user_id)
                    )
                    user = result.scalar_one_or_none()
                    if user:
                        return {
                            "id": user.id,
                            "telegram_id": user.telegram_id,
                            "username": user.username,
                            "first_name": user.first_name,
                            "is_primary_user": user.is_primary_user,
                        }
            else:
                # In-memory fallback
                for uid, user in self._mem_users.items():
                    if uid == user_id:
                        return {**user, "id": uid}
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    async def get_last_interaction(self, user_id: int) -> Optional[datetime]:
        """Get last interaction time for user"""
        try:
            if self._use_db:
                async with self._session() as s:
                    result = await s.execute(
                        select(ConversationDB)
                        .where(ConversationDB.user_id == user_id)
                        .order_by(ConversationDB.created_at.desc())
                        .limit(1)
                    )
                    conversation = result.scalar_one_or_none()
                    return conversation.created_at if conversation else None
            else:
                # In-memory fallback
                user_conversations = [
                    conv for conv in self._mem_conversations.values()
                    if conv.get("user_id") == user_id
                ]
                if user_conversations:
                    return max(conv.get("created_at") for conv in user_conversations)
            return None
        except Exception as e:
            logger.error(f"Error getting last interaction: {str(e)}")
            return None

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    async def log_tool_call(
        self, user_id: int, tool_name: str, action: str, input_data: Dict[str, Any],
    ) -> Optional[int]:
        if self._use_db:
            async with self._session() as s:
                tc = ToolCallDB(
                    user_id=user_id, tool_name=tool_name, action=action,
                    status="pending", input_data=input_data,
                )
                s.add(tc)
                await s.commit()
                await s.refresh(tc)
                return tc.id
        self._counters["tool"] += 1
        self._mem_tool_calls[self._counters["tool"]] = {
            "id": self._counters["tool"], "user_id": user_id,
            "tool_name": tool_name, "action": action, "status": "pending",
            "input_data": input_data, "output_data": None, "error_message": None,
            "created_at": datetime.utcnow().isoformat(), "completed_at": None,
        }
        return self._counters["tool"]

    async def update_tool_call_result(
        self, tool_call_id: int, status: str,
        output_data: Optional[Dict] = None, error_message: Optional[str] = None,
    ) -> bool:
        if self._use_db:
            async with self._session() as s:
                await s.execute(
                    update(ToolCallDB).where(ToolCallDB.id == tool_call_id).values(
                        status=status, output_data=output_data,
                        error_message=error_message, completed_at=datetime.utcnow(),
                    )
                )
                await s.commit()
                return True
        if tool_call_id in self._mem_tool_calls:
            self._mem_tool_calls[tool_call_id].update({
                "status": status, "output_data": output_data,
                "error_message": error_message, "completed_at": datetime.utcnow().isoformat(),
            })
            return True
        return False

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        if self._use_db:
            try:
                async with self._session() as s:
                    await s.execute(select(1))
                return True
            except Exception:
                return False
        return True

    async def close(self):
        if self._engine:
            await self._engine.dispose()
        if self.redis_client:
            await self.redis_client.aclose()
        logger.info("Storage service closed")

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _memory_not_expired(self, mem: Dict[str, Any]) -> bool:
        """
        Expiry is encoded in tags as: "expires_at:<ISO8601>".
        This avoids DB migrations while enabling retention policies.
        """
        try:
            tags = mem.get("tags") or []
            for t in tags:
                if isinstance(t, str) and t.startswith("expires_at:"):
                    iso = t.split("expires_at:", 1)[1].strip()
                    if not iso:
                        continue
                    expires_at = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    return datetime.utcnow() < expires_at.replace(tzinfo=None)
            return True
        except Exception:
            return True

    def _user_to_dict(self, u: "UserDB") -> Dict[str, Any]:
        return {
            "id": u.id, "telegram_id": u.telegram_id,
            "first_name": u.first_name, "last_name": u.last_name,
            "username": u.username, "phone_number": u.phone_number,
            "is_primary_user": u.is_primary_user, "is_stranger": u.is_stranger,
            "preferred_personality": u.preferred_personality,
            "personality_config": u.personality_config or {},
            "language": u.language, "timezone": u.timezone,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_active": u.last_active.isoformat() if u.last_active else None,
        }

    def _conv_to_dict(self, c: "ConversationDB") -> Dict[str, Any]:
        return {
            "id": c.id, "user_id": c.user_id, "message_type": c.message_type,
            "user_message": c.user_message, "eva_response": c.eva_response,
            "detected_personality": c.detected_personality, "detected_intent": c.detected_intent,
            "tools_used": c.tools_used or [], "session_id": c.session_id,
            "context_data": c.context_data or {},
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "completed_at": c.completed_at.isoformat() if c.completed_at else None,
        }

    def _mem_to_dict(self, m: "MemoryDB") -> Dict[str, Any]:
        return {
            "id": m.id, "user_id": m.user_id, "memory_type": m.memory_type,
            "content": m.content, "importance_score": m.importance_score,
            "tags": m.tags or [], "source": m.source, "access_count": m.access_count,
            "last_referenced": m.last_referenced.isoformat() if m.last_referenced else None,
            "is_forgotten": m.is_forgotten,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }

    def _mem_to_dict_raw(self, m: Dict) -> Dict[str, Any]:
        return {k: v for k, v in m.items()}
