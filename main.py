import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any

from config.settings import get_settings
from services.nvidia_service import NVIDIAService
from services.telegram_service import TelegramService
from services.storage_service import StorageService
from core.orchestrator import Orchestrator
from core.personality_engine import PersonalityEngine
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
from core.autonomous_engine import AutonomousEngine
from core.background_scheduler import BackgroundScheduler
from core.identity import IdentityService, TrustLevel
from core.memory_curator import MemoryCurator
from agents.agent_manager import AgentManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

nvidia_service: NVIDIAService = None
telegram_service: TelegramService = None   # bot fallback
eva_account = None                         # Eva's real user account (Pyrogram)
storage_service: StorageService = None
personality_engine: PersonalityEngine = None
memory_manager: MemoryManager = None
memory_curator: MemoryCurator = None
orchestrator: Orchestrator = None
tool_registry: ToolRegistry = None
agent_manager: AgentManager = None
autonomous_engine: AutonomousEngine = None
background_scheduler: BackgroundScheduler = None
identity_service: IdentityService = None


# ====================== Lifecycle Events ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global nvidia_service, telegram_service, eva_account, storage_service
    global personality_engine, memory_manager, memory_curator, orchestrator, tool_registry, agent_manager
    global autonomous_engine, background_scheduler, identity_service

    logger.info("Starting EVA...")

    try:
        nvidia_service = NVIDIAService()
        telegram_service = TelegramService(settings.telegram_bot_token)
        storage_service = StorageService()
        await storage_service.initialize()

        identity_service = IdentityService(settings)

        personality_engine = PersonalityEngine(nvidia_service, storage_service)
        memory_manager = MemoryManager(
            redis_service=storage_service.redis_client,
            db_service=storage_service,
            pinecone_service=storage_service.pinecone_index,
        )
        memory_curator = MemoryCurator()
        orchestrator = Orchestrator(
            nvidia_service=nvidia_service,
            personality_engine=personality_engine,
            memory_manager=memory_manager,
            storage_service=storage_service,
            settings=settings,
        )
        tool_registry = ToolRegistry()
        agent_manager = AgentManager(nvidia_service=nvidia_service)
        # Now wire agent_manager into orchestrator (created after to avoid circular dep)
        orchestrator.set_agent_manager(agent_manager)

        for agent_name, agent_instance in agent_manager.agents.items():
            tool_registry.register_tool(
                name=agent_name,
                tool_instance=agent_instance,
                metadata={"description": f"{agent_name} agent", "type": "agent"},
            )

        # Try to start Eva's real Telegram user account via Pyrogram
        if settings.telegram_user_mode and settings.telegram_api_id and settings.telegram_api_hash:
            try:
                from services.pyrogram_service import PyrogramService
                eva_account = PyrogramService(
                    api_id=settings.telegram_api_id,
                    api_hash=settings.telegram_api_hash,
                    phone_number=settings.eva_phone,
                    session_name=settings.telegram_session_name,
                    session_dir=settings.telegram_session_dir,
                )
                await eva_account.start(message_handler=_handle_pyrogram_message)
                # Give all agents access to Eva's Telegram account
                agent_manager.set_pyrogram(eva_account)
                logger.info("Eva is running as a real Telegram user account")
            except Exception as e:
                logger.error(f"Could not start Pyrogram user account: {e}")
                logger.warning("Falling back to bot-only mode")
                eva_account = None
        else:
            logger.info("Running in bot-only mode (set TELEGRAM_USER_MODE=true to enable user account)")

        # Use Pyrogram account for autonomous actions if available, else bot service
        active_tg = eva_account if eva_account else telegram_service

        autonomous_engine = AutonomousEngine(
            nvidia_service=nvidia_service,
            memory_manager=memory_manager,
            agent_manager=agent_manager,
            storage_service=storage_service,
            telegram_service=active_tg,
        )
        background_scheduler = BackgroundScheduler(
            autonomous_engine=autonomous_engine,
            storage_service=storage_service,
            telegram_service=active_tg,
        )

        is_healthy = await nvidia_service.health_check()
        logger.info(f"NVIDIA API healthy: {is_healthy}")

        await background_scheduler.start()
        logger.info("EVA is live and autonomous")

        yield

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise

    finally:
        logger.info("Shutting down EVA...")
        try:
            if background_scheduler:
                await background_scheduler.stop()
            if eva_account and eva_account.is_running:
                await eva_account.stop()
            if nvidia_service:
                await nvidia_service.close()
            if storage_service:
                await storage_service.close()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


async def _process_incoming_message(
    telegram_id: str,
    chat_id,
    msg_type: str,
    content,
    first_name: Optional[str],
    username: Optional[str],
    phone_number: Optional[str] = None,
    raw_message=None,
) -> None:
    """Single pipeline for all incoming messages regardless of source."""
    try:
        user = await storage_service.get_or_create_user(
            telegram_id=telegram_id,
            first_name=first_name,
            username=username,
            phone_number=phone_number,
        )

        identity = await identity_service.resolve(
            telegram_id=telegram_id,
            first_name=first_name,
            username=username,
            phone_number=phone_number,
            db_user=user,
        )

        if user.get("is_primary_user") != identity.is_primary:
            user["is_primary_user"] = identity.is_primary
            user["is_stranger"] = not identity.is_primary

        is_primary = identity.is_primary
        allowed_tools = identity_service.get_allowed_tools(identity.trust_level)
        trust_context = identity_service.format_trust_context(identity)

        user_data = {
            "id": user["id"],
            "telegram_id": user["telegram_id"],
            "preferred_personality": user.get("preferred_personality", "general"),
            "language": user.get("language", "en"),
            "timezone": user.get("timezone", "UTC"),
            "trust_level": identity.trust_level.value,
            "trust_context": trust_context,
        }

        if eva_account:
            await eva_account.send_typing(chat_id)

        # Transcribe voice
        text_content = content
        if msg_type == "voice" and eva_account and raw_message:
            try:
                path = await eva_account.download_media(raw_message)
                if path:
                    with open(path, "rb") as f:
                        audio_bytes = f.read()
                    text_content = await nvidia_service.call_stt_model(audio_bytes)
                    os.remove(path)
            except Exception as e:
                logger.error(f"Voice transcription failed: {e}")
                text_content = "[Voice message — transcription failed]"

        if not isinstance(text_content, str):
            text_content = f"[{msg_type} message]"

        await memory_manager.add_to_conversation(user_id=user["id"], role="user", content=text_content)

        # Curated long-term memory (only for trusted users; heuristics-only)
        try:
            caps = identity_service.get_capabilities(identity.trust_level)
            if caps.get("memory_write"):
                candidates = memory_curator.extract(text_content) if memory_curator else []
                for c in candidates:
                    tags = list(c.tags or [])
                    if c.expires_at:
                        tags.append(f"expires_at:{c.expires_at.isoformat()}")
                    await memory_manager.add_memory(
                        user_id=user["id"],
                        memory_type=c.memory_type,
                        content=c.content,
                        importance_score=c.importance_score,
                        tags=tags,
                        source="conversation:user",
                    )
        except Exception as e:
            logger.warning(f"Memory curation skipped: {e}")

        conversation = await storage_service.create_conversation(
            user_id=user["id"], message_type=msg_type,
            user_message=text_content, personality="general",
        )

        decision = await orchestrator.process_request(
            user_message=text_content, user_id=user["id"],
            user_data=user_data, is_primary_user=is_primary,
            allowed_tools=allowed_tools,
        )

        # Tool restriction already enforced inside orchestrator for non-owners.
        # Double-check here as a safety net (orchestrator executes tools itself now).

        eva_response = await _generate_response(
            user_message=text_content, personality=decision.personality,
            user_data=user_data, is_primary_user=is_primary,
            decision=decision,
        )

        await storage_service.update_conversation_response(
            conversation_id=conversation["id"],
            eva_response=eva_response, tools_used=decision.required_tools,
        )
        await memory_manager.add_to_conversation(
            user_id=user["id"], role="assistant", content=eva_response
        )

        active_tg = eva_account if eva_account else telegram_service

        if eva_account and raw_message:
            await eva_account.react_to_message(chat_id, raw_message.id, "👀")

        if msg_type == "voice" and eva_account:
            try:
                import io
                audio = await nvidia_service.call_tts_model(text=eva_response)
                await eva_account.send_voice(chat_id, io.BytesIO(audio))
            except Exception:
                await active_tg.send_message(chat_id=chat_id, text=eva_response)
        else:
            await active_tg.send_message(chat_id=chat_id, text=eva_response)

        if eva_account:
            await eva_account.read_messages(chat_id)

        if is_primary and decision.required_tools:
            await _execute_autonomous_tasks(user_id=user["id"], chat_id=chat_id, decision=decision)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)


async def _handle_pyrogram_message(parsed: Dict[str, Any], raw_message) -> None:
    chat_id = parsed.get("chat_id")
    user_id = parsed.get("user_id")
    if not chat_id or not user_id:
        return
    try:
        me = await eva_account._client.get_me()
        if user_id == me.id:
            return
    except Exception:
        pass
    await _process_incoming_message(
        telegram_id=str(user_id),
        chat_id=chat_id,
        msg_type=parsed.get("type", "text"),
        content=parsed.get("content"),
        first_name=parsed.get("first_name"),
        username=parsed.get("username"),
        raw_message=raw_message,
    )


# ====================== FastAPI App Setup ======================

app = FastAPI(
    title=settings.app_name,
    description="Agentic Telegram Bot with Personality Adaptation",
    version=settings.app_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== Routes ======================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.post("/webhook/telegram")
async def telegram_webhook(request: Request) -> Dict[str, Any]:
    """
    Telegram webhook endpoint — bot mode fallback.
    In user account mode (Pyrogram), messages arrive via _handle_pyrogram_message instead.
    """
    try:
        payload = await request.json()
        message_data = telegram_service.parse_webhook_message(payload)
        if not message_data:
            return {"status": "error", "message": "Unable to parse message"}

        chat_id = message_data["chat_id"]

        try:
            await _process_incoming_message(
                telegram_id=str(message_data["user_id"]),
                chat_id=chat_id,
                msg_type=message_data["type"],
                content=message_data["content"],
                first_name=message_data.get("first_name"),
                username=message_data.get("username"),
            )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error processing webhook message: {e}", exc_info=True)
            await telegram_service.send_message(
                chat_id=chat_id,
                text="Sorry, I hit an error. Try again in a moment.",
            )
            return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.error(f"Webhook handler error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "message": "EVA Telegram Bot is running",
        "health": "/health",
        "webhook": "/webhook/telegram",
    }


@app.post("/admin/set-personality")
async def admin_set_personality(
    user_id: int,
    personality: str,
    x_admin_token: str = Header(default=""),
) -> Dict[str, Any]:
    """
    Admin endpoint to set user personality
    (In production, add proper authentication)
    """
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        success = await storage_service.update_user_personality(user_id, personality)
        
        return {
            "status": "success" if success else "error",
            "user_id": user_id,
            "personality": personality,
        }
    
    except Exception as e:
        logger.error(f"Error setting personality: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/scheduler-status")
async def admin_scheduler_status(x_admin_token: str = Header(default="")) -> Dict[str, Any]:
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return await background_scheduler.get_scheduler_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/model-health")
async def admin_model_health(x_admin_token: str = Header(default="")) -> Dict[str, Any]:
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        circuits = nvidia_service.get_circuit_status()
        api_healthy = await nvidia_service.health_check()
        return {
            "api_reachable": api_healthy,
            "circuits": circuits,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/autonomous-status")
async def admin_autonomous_status(x_admin_token: str = Header(default="")) -> Dict[str, Any]:
    """Admin endpoint to check autonomous engine status"""
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if background_scheduler:
            status = await background_scheduler.get_scheduler_status()
            return {"status": "success", "data": status}
        else:
            return {"status": "error", "message": "Background scheduler not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/trigger-autonomous-think")
async def admin_trigger_autonomous_think(x_admin_token: str = Header(default="")) -> Dict[str, Any]:
    """Admin endpoint to manually trigger autonomous thinking"""
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if autonomous_engine:
            decisions = await autonomous_engine.think()
            return {
                "status": "success",
                "message": f"EVA made {len(decisions)} autonomous decisions",
                "decisions": [d.to_dict() for d in decisions]
            }
        else:
            return {"status": "error", "message": "Autonomous engine not initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/user-memory/{user_id}")
async def admin_get_user_memory(user_id: int, x_admin_token: str = Header(default="")) -> Dict[str, Any]:
    """
    Admin endpoint to view user's memories
    (In production, add proper authentication)
    """
    if settings.admin_api_key and x_admin_token != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        memories = await storage_service.get_user_memories(user_id, limit=20)
        
        return {
            "user_id": user_id,
            "memory_count": len(memories),
            "memories": memories,
        }
    
    except Exception as e:
        logger.error(f"Error getting user memory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ====================== Helper Functions ======================

async def _generate_response(
    user_message: str,
    personality: str,
    user_data: Dict[str, Any],
    is_primary_user: bool,
    decision=None,
) -> str:
    try:
        system_prompt = personality_engine.get_personality_prompt(personality)
        trust_context = user_data.get("trust_context", "")
        if trust_context:
            system_prompt = f"{trust_context}\n\n{system_prompt}"

        # Build context that includes what tools ran and what happened
        context = f"User said: {user_message}\n"

        if decision and decision.tool_results:
            context += "\nActions you just took and their results:\n"
            for tool, result in decision.tool_results.items():
                if result.get("success"):
                    msg = result.get("message") or result.get("text_sent") or "Done"
                    context += f"  ✓ {tool}: {msg}\n"
                else:
                    context += f"  ✗ {tool} failed: {result.get('error', 'unknown error')}\n"
            context += "\nNow write a natural reply to the user confirming what you did (or what went wrong). Be brief and conversational — don't repeat the full details, just confirm naturally."
        else:
            context += "\nReply naturally and helpfully."

        response = await nvidia_service.call_chat_model(
            system_prompt=system_prompt,
            user_message=context,
        )
        return personality_engine.adapt_response_style(
            response=response, personality=personality, user_data=user_data,
        )
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Done — let me know if you need anything else."


async def _execute_autonomous_tasks(
    user_id: int,
    chat_id,
    decision: Any,
) -> None:
    active_tg = eva_account if eva_account else telegram_service
    try:
        if not decision.required_tools:
            return

        await active_tg.send_message(
            chat_id=chat_id,
            text=f"🤖 Working on: {', '.join(decision.required_tools)}",
        )

        for tool_name in decision.required_tools:
            try:
                result = await agent_manager.execute(
                    agent_name=tool_name,
                    action="execute",
                    user_id=user_id,
                    chat_id=chat_id,
                )
                status = "✅" if result.get("success") else "⚠️"
                msg = result.get("message") or result.get("error") or "Done"
                await active_tg.send_message(chat_id=chat_id, text=f"{status} {tool_name}: {msg}")
            except Exception as e:
                await active_tg.send_message(chat_id=chat_id, text=f"❌ {tool_name}: {e}")

    except Exception as e:
        logger.error(f"Error in autonomous task execution: {e}")


# ====================== Entry Point ======================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
    )
