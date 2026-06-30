"""
Memory Manager module
Manages short-term and long-term memory for EVA
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages EVA's memory system:
    - Short-term: Current session context (Redis)
    - Long-term: Persistent user data (PostgreSQL)
    - Semantic: Embeddings for similarity search (Pinecone)
    """
    
    def __init__(self, redis_service=None, db_service=None, pinecone_service=None):
        """
        Initialize memory manager
        
        Args:
            redis_service: Redis connection for short-term memory
            db_service: Database connection for long-term memory
            pinecone_service: Pinecone connection for semantic memory
        """
        self.redis = redis_service
        self.db = db_service
        self.pinecone = pinecone_service
    
    async def health_check(self) -> bool:
        """Check if memory manager is healthy"""
        try:
            # Simple health check - test database connection
            if self.db:
                return await self.db.health_check()
            return True  # If no database, consider healthy
        except Exception as e:
            logger.error(f"Memory manager health check failed: {str(e)}")
            return False
    
    # ==================== Short-Term Memory (Session) ====================
    
    async def store_session_context(
        self,
        user_id: int,
        session_id: str,
        context: Dict[str, Any],
        ttl_minutes: int = 60,
    ) -> bool:
        """
        Store session context in Redis
        
        Args:
            user_id: User ID
            session_id: Session ID
            context: Context data
            ttl_minutes: Time to live in minutes
        
        Returns:
            Success boolean
        """
        try:
            if not self.redis:
                logger.warning("Redis not configured, skipping session storage")
                return False
            
            key = f"session:{user_id}:{session_id}"
            ttl_seconds = ttl_minutes * 60
            
            # Store as JSON
            context_json = json.dumps(context, default=str)
            await self.redis.setex(key, ttl_seconds, context_json)
            
            logger.info(f"Stored session context: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing session context: {str(e)}")
            return False
    
    async def get_session_context(
        self,
        user_id: int,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve session context from Redis
        
        Args:
            user_id: User ID
            session_id: Session ID
        
        Returns:
            Context dict or None
        """
        try:
            if not self.redis:
                return None
            
            key = f"session:{user_id}:{session_id}"
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving session context: {str(e)}")
            return None
    
    async def get_recent_conversation(
        self,
        user_id: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation from short-term memory
        
        Args:
            user_id: User ID
            limit: Number of recent messages
        
        Returns:
            List of recent messages
        """
        try:
            if not self.redis:
                return []
            
            key = f"recent_conv:{user_id}"
            data = await self.redis.lrange(key, 0, limit - 1)
            
            if data:
                return [json.loads(item) for item in data]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent conversation: {str(e)}")
            return []
    
    async def add_to_conversation(
        self,
        user_id: int,
        role: str,  # "user" or "assistant"
        content: str,
        ttl_minutes: int = 120,
    ) -> bool:
        """
        Add message to short-term conversation
        
        Args:
            user_id: User ID
            role: Message role
            content: Message content
            ttl_minutes: Session TTL
        
        Returns:
            Success boolean
        """
        try:
            if not self.redis:
                return False
            
            key = f"recent_conv:{user_id}"
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Keep last 20 messages
            await self.redis.lpush(key, json.dumps(message))
            await self.redis.ltrim(key, 0, 19)
            await self.redis.expire(key, ttl_minutes * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to conversation: {str(e)}")
            return False
    
    # ==================== Long-Term Memory (Persistent) ====================
    
    async def store_memory(
        self,
        user_id: int,
        memory_type: str,
        content: str,
        importance: int = 1,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> Optional[int]:
        """
        Store long-term memory in database
        
        Args:
            user_id: User ID
            memory_type: Type of memory (preference, fact, habit, goal, etc)
            content: Memory content
            importance: Importance score 1-10
            tags: List of tags
            source: Source of memory
        
        Returns:
            Memory ID or None
        """
        try:
            if not self.db:
                logger.warning("Database not configured, skipping memory storage")
                return None
            
            # Store in database
            memory_id = await self.db.create_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance_score=importance,
                tags=tags or [],
                source=source,
            )
            
            logger.info(f"Stored long-term memory: {memory_id}")
            
            # Also store in Pinecone for semantic search (Long-term memory)
            if self.pinecone:
                await self._store_semantic_embedding(
                    memory_id, content, user_id
                )
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return None
    
    async def get_relevant_memory(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories for a query
        
        Args:
            user_id: User ID
            query: Query string
            top_k: Number of results
        
        Returns:
            List of relevant memories
        """
        try:
            results = []
            
            # Try semantic search with Pinecone first
            if self.pinecone:
                semantic_results = await self._semantic_search(
                    query, user_id, top_k
                )
                results.extend(semantic_results)
            
            # Fallback to keyword search in database
            if not results and self.db:
                results = await self.db.search_memories(
                    user_id, query, limit=top_k
                )
            
            # Update access count
            for memory in results:
                if self.db:
                    await self.db.update_memory_access(memory["id"])
            
            logger.info(f"Retrieved {len(results)} relevant memories")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant memory: {str(e)}")
            return []
    
    async def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        importance: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Update existing memory
        
        Args:
            memory_id: Memory ID
            content: New content
            importance: New importance score
            tags: New tags
        
        Returns:
            Success boolean
        """
        try:
            if not self.db:
                return False
            
            await self.db.update_memory(
                memory_id,
                content=content,
                importance_score=importance,
                tags=tags,
            )
            
            logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            return False
    
    async def forget_memory(self, memory_id: int) -> bool:
        """
        Soft delete memory (mark as forgotten)
        
        Args:
            memory_id: Memory ID
        
        Returns:
            Success boolean
        """
        try:
            if not self.db:
                return False
            
            await self.db.forget_memory(memory_id)
            
            logger.info(f"Forgotten memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error forgetting memory: {str(e)}")
            return False
    
    async def get_user_memories(
        self,
        user_id: int,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a user
        
        Args:
            user_id: User ID
            memory_type: Filter by type
            limit: Max results
        
        Returns:
            List of memories
        """
        try:
            if not self.db:
                return []
            
            memories = await self.db.get_user_memories(
                user_id, memory_type=memory_type, limit=limit
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting user memories: {str(e)}")
            return []
    
    # ==================== Semantic Memory ====================
    
    async def _store_semantic_embedding(
        self,
        memory_id: int,
        content: str,
        user_id: int,
    ) -> bool:
        """Store semantic embedding in Pinecone"""
        try:
            if not self.pinecone:
                return False
            
            # Generate embedding (would use NVIDIA embeddings or similar)
            embedding = await self._generate_embedding(content)
            
            # Store in Pinecone
            await self.pinecone.upsert(
                vectors=[{
                    "id": f"{user_id}:{memory_id}",
                    "values": embedding,
                    "metadata": {
                        "user_id": user_id,
                        "memory_id": memory_id,
                        "content": content[:500],  # Truncate for metadata
                    }
                }]
            )
            
            logger.info(f"Stored semantic embedding for memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing semantic embedding: {str(e)}")
            return False
    
    async def _semantic_search(
        self,
        query: str,
        user_id: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories semantically using Pinecone"""
        try:
            if not self.pinecone:
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search in Pinecone
            results = await self.pinecone.query(
                vector=query_embedding,
                top_k=top_k,
                filter={"user_id": {"$eq": user_id}},
                include_metadata=True,
            )
            
            # Convert to memory format
            memories = [
                {
                    "id": int(match["metadata"]["memory_id"]),
                    "content": match["metadata"]["content"],
                    "score": match["score"],
                }
                for match in results.get("matches", [])
            ]
            
            return memories
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder)"""
        # In production, use NVIDIA embeddings or OpenAI
        # For now, return dummy embedding
        logger.warning("Using placeholder embedding - implement with real embeddings")
        return [0.0] * 768  # 768-dimensional dummy embedding
    
    # ==================== Memory Analysis ====================
    
    async def add_memory(
        self,
        user_id: int,
        memory_type: str,
        content: str,
        importance_score: int = 3,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> Optional[int]:
        return await self.store_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=importance_score,
            tags=tags,
            source=source,
        )

    async def get_user_interests(self, user_id: int) -> List[str]:
        try:
            if not self.db:
                return []
            memories = await self.db.get_user_memories(user_id, memory_type="interest", limit=20)
            return [m.get("content", "") for m in memories if m.get("content")]
        except Exception as e:
            logger.error(f"Error getting user interests: {e}")
            return []

    async def health_check(self) -> bool:
        return True

    async def analyze_memory_patterns(
        self,
        user_id: int,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze memory patterns over time
        
        Args:
            user_id: User ID
            days: Number of days to analyze
        
        Returns:
            Analysis dictionary
        """
        try:
            if not self.db:
                return {}
            
            since = datetime.utcnow() - timedelta(days=days)
            
            memories = await self.db.get_user_memories(
                user_id, since=since
            )
            
            # Analyze patterns
            memory_types = {}
            importance_avg = 0
            
            for memory in memories:
                mtype = memory.get("memory_type")
                memory_types[mtype] = memory_types.get(mtype, 0) + 1
                importance_avg += memory.get("importance_score", 1)
            
            if memories:
                importance_avg /= len(memories)
            
            return {
                "total_memories": len(memories),
                "memory_types": memory_types,
                "avg_importance": importance_avg,
                "period_days": days,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {str(e)}")
            return {}
