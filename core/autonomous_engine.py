import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AutonomousActionType(str, Enum):
    USER_CHECK = "user_check"
    PROACTIVE_EMAIL = "proactive_email"
    SCHEDULED_REMINDER = "scheduled_reminder"
    DATA_UPDATE = "data_update"
    HEALTH_CHECK = "health_check"
    CONVERSATION_INITIATION = "conversation_initiation"
    TASK_EXECUTION = "task_execution"
    INFORMATION_GATHERING = "information_gathering"
    VOICE_CALL = "voice_call"
    UPDATE_BRIEFING = "update_briefing"


class AutonomousDecision:
    def __init__(
        self,
        action_type: AutonomousActionType,
        priority: int,
        reasoning: str,
        action_data: Dict[str, Any],
        target_user_id: Optional[int] = None,
        scheduled_time: Optional[datetime] = None,
    ):
        self.action_type = action_type
        self.priority = priority
        self.reasoning = reasoning
        self.action_data = action_data
        self.target_user_id = target_user_id
        self.scheduled_time = scheduled_time or datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.status = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "action_data": self.action_data,
            "target_user_id": self.target_user_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


class AutonomousEngine:
    def __init__(self, nvidia_service, memory_manager, agent_manager, storage_service, telegram_service=None):
        self.nvidia_service = nvidia_service
        self.memory_manager = memory_manager
        self.agent_manager = agent_manager
        self.storage_service = storage_service
        self.telegram_service = telegram_service

        self.decision_queue: List[AutonomousDecision] = []
        self.active_actions: Dict[str, Any] = {}
        self.action_history: List[AutonomousDecision] = []

        # Track last AI-driven proactive think per user to avoid spamming
        self._last_proactive_think: Dict[int, datetime] = {}
        # Track last update briefing per user
        self._last_briefing: Dict[int, datetime] = {}

        logger.info("Autonomous Engine initialized")

    def set_telegram_service(self, telegram_service):
        self.telegram_service = telegram_service

    async def think(self) -> List[AutonomousDecision]:
        logger.info("EVA is thinking autonomously...")
        decisions = []

        try:
            scheduled = await self._consider_scheduled_actions()
            decisions.extend(scheduled)

            proactive = await self._consider_proactive_actions()
            decisions.extend(proactive)

            tasks = await self._consider_pending_tasks()
            decisions.extend(tasks)

            info = await self._consider_information_gathering()
            decisions.extend(info)

            if decisions:
                evaluated = await self._evaluate_decisions_with_ai(decisions)
            else:
                evaluated = decisions

            for d in evaluated:
                if not self._already_queued(d):
                    self.decision_queue.append(d)

            logger.info(f"EVA queued {len(evaluated)} autonomous actions")
            return evaluated

        except Exception as e:
            logger.error(f"Error in autonomous thinking: {e}")
            return []

    def _already_queued(self, decision: AutonomousDecision) -> bool:
        for d in self.decision_queue:
            if (
                d.status == "pending"
                and d.action_type == decision.action_type
                and d.target_user_id == decision.target_user_id
                and abs((d.scheduled_time - decision.scheduled_time).total_seconds()) < 60
            ):
                return True
        return False

    async def _consider_scheduled_actions(self) -> List[AutonomousDecision]:
        decisions = []
        now = datetime.utcnow()

        try:
            users = await self.storage_service.get_primary_users()
            for user in users:
                uid = user["id"]

                # Morning check at 9 AM
                if now.hour == 9 and now.minute < 5:
                    if not await self._has_recent_interaction(uid, hours=12):
                        decisions.append(AutonomousDecision(
                            action_type=AutonomousActionType.USER_CHECK,
                            priority=7,
                            reasoning="Morning check-in",
                            action_data={"message": "Good morning! ☀️ Hope you slept well. Got anything lined up today you'd like help with?"},
                            target_user_id=uid,
                        ))

                # Evening check at 6 PM
                if now.hour == 18 and now.minute < 5:
                    if not await self._has_recent_interaction(uid, hours=6):
                        decisions.append(AutonomousDecision(
                            action_type=AutonomousActionType.USER_CHECK,
                            priority=6,
                            reasoning="Evening check-in",
                            action_data={"message": "Hey, how did your day go? Anything I can help you wrap up before tomorrow?"},
                            target_user_id=uid,
                        ))

                # Daily briefing at 8 AM if not briefed today
                if now.hour == 8 and now.minute < 5:
                    last_brief = self._last_briefing.get(uid)
                    if not last_brief or (now - last_brief).total_seconds() > 82800:
                        decisions.append(AutonomousDecision(
                            action_type=AutonomousActionType.UPDATE_BRIEFING,
                            priority=8,
                            reasoning="Daily morning briefing",
                            action_data={},
                            target_user_id=uid,
                        ))

        except Exception as e:
            logger.error(f"Error considering scheduled actions: {e}")

        return decisions

    async def _consider_proactive_actions(self) -> List[AutonomousDecision]:
        decisions = []

        try:
            users = await self.storage_service.get_primary_users()
            for user in users:
                uid = user["id"]
                now = datetime.utcnow()

                # Limit AI-driven proactive checks to once every 30 minutes per user
                last = self._last_proactive_think.get(uid)
                if last and (now - last).total_seconds() < 1800:
                    continue

                memories = await self.memory_manager.get_relevant_memory(
                    uid, "recent activities goals tasks", top_k=15
                )
                history = await self.storage_service.get_conversation_history(uid, limit=10, days=3)

                if not memories and not history:
                    continue

                proactive = await self._ai_driven_proactive_analysis(user, memories, history)
                decisions.extend(proactive)
                self._last_proactive_think[uid] = now

        except Exception as e:
            logger.error(f"Error considering proactive actions: {e}")

        return decisions

    async def _ai_driven_proactive_analysis(
        self,
        user: Dict[str, Any],
        memories: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> List[AutonomousDecision]:
        decisions = []

        try:
            mem_text = "\n".join([f"- [{m.get('memory_type','?')}] {m.get('content','')}" for m in memories[:10]])
            hist_text = "\n".join([
                f"- User: {h.get('user_message','')[:100]}" for h in history[:5]
            ])

            system_prompt = """You are EVA's autonomous reasoning core. Analyze the user's recent memories and conversation history to decide if EVA should take any proactive initiative RIGHT NOW.

Consider:
- Unfinished tasks or follow-ups the user mentioned
- Goals the user expressed that EVA could help advance
- Long gaps in communication (check-in opportunity)
- Upcoming events or deadlines the user mentioned
- Emotional state patterns that warrant attention
- Information updates that might be relevant to the user's interests

Respond ONLY with valid JSON:
{
  "should_act": true/false,
  "actions": [
    {
      "type": "user_check|conversation_initiation|proactive_email|information_gathering|update_briefing",
      "priority": 1-10,
      "reasoning": "why EVA should do this now",
      "message": "the actual message EVA should send (for check/initiation types)",
      "topic": "search topic (for information_gathering type)"
    }
  ]
}

Only suggest an action if there is genuine value. Avoid being intrusive or repetitive."""

            user_context = f"""User: {user.get('first_name') or user.get('username') or 'User'}

Recent memories:
{mem_text or 'No memories yet'}

Recent conversations:
{hist_text or 'No recent conversations'}

Current time: {datetime.utcnow().strftime('%A %H:%M UTC')}"""

            response = await self.nvidia_service.call_orchestrator_model(
                system_prompt=system_prompt,
                user_message=user_context,
            )

            parsed = self._safe_parse_json(response)
            if not parsed or not parsed.get("should_act"):
                return decisions

            for action in parsed.get("actions", []):
                action_type_str = action.get("type", "conversation_initiation")
                try:
                    action_type = AutonomousActionType(action_type_str)
                except ValueError:
                    action_type = AutonomousActionType.CONVERSATION_INITIATION

                action_data = {}
                if action.get("message"):
                    action_data["message"] = action["message"]
                if action.get("topic"):
                    action_data["topic"] = action["topic"]

                decisions.append(AutonomousDecision(
                    action_type=action_type,
                    priority=int(action.get("priority", 5)),
                    reasoning=action.get("reasoning", "AI-driven proactive action"),
                    action_data=action_data,
                    target_user_id=user["id"],
                ))

        except Exception as e:
            logger.error(f"Error in AI proactive analysis: {e}")

        return decisions

    async def _consider_pending_tasks(self) -> List[AutonomousDecision]:
        decisions = []

        try:
            users = await self.storage_service.get_primary_users()
            for user in users:
                reminders = await self.storage_service.get_active_reminders(user["id"])
                for reminder in reminders:
                    if self._is_reminder_due(reminder):
                        decisions.append(AutonomousDecision(
                            action_type=AutonomousActionType.SCHEDULED_REMINDER,
                            priority=9,
                            reasoning=f"Reminder due: {reminder.get('title', 'Unnamed')}",
                            action_data={
                                "reminder_id": reminder.get("id"),
                                "message": f"⏰ Reminder: {reminder.get('title', '')}\n{reminder.get('description', '')}",
                            },
                            target_user_id=user["id"],
                        ))

        except Exception as e:
            logger.error(f"Error considering pending tasks: {e}")

        return decisions

    async def _consider_information_gathering(self) -> List[AutonomousDecision]:
        decisions = []

        try:
            users = await self.storage_service.get_primary_users()
            for user in users:
                interests = await self.storage_service.get_user_interests(user["id"])
                for interest in interests:
                    if await self._should_update_information(user["id"], interest):
                        decisions.append(AutonomousDecision(
                            action_type=AutonomousActionType.INFORMATION_GATHERING,
                            priority=3,
                            reasoning=f"Refresh info on '{interest}' for {user.get('username') or user.get('first_name')}",
                            action_data={"topic": interest},
                            target_user_id=user["id"],
                        ))

        except Exception as e:
            logger.error(f"Error considering information gathering: {e}")

        return decisions

    async def _evaluate_decisions_with_ai(
        self, decisions: List[AutonomousDecision]
    ) -> List[AutonomousDecision]:
        if not decisions:
            return decisions

        # Deduplicate by type+user before AI evaluation
        seen = set()
        unique = []
        for d in decisions:
            key = (d.action_type, d.target_user_id)
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique

    async def _has_recent_interaction(self, user_id: int, hours: int = 24) -> bool:
        try:
            last = await self.storage_service.get_last_interaction(user_id)
            if not last:
                return False
            return (datetime.utcnow() - last).total_seconds() < hours * 3600
        except Exception as e:
            logger.error(f"Error checking recent interaction: {e}")
            return False

    def _is_reminder_due(self, reminder: Dict[str, Any]) -> bool:
        try:
            due_str = reminder.get("due_time") or reminder.get("remind_at")
            if not due_str:
                return False
            if isinstance(due_str, str):
                due = datetime.fromisoformat(due_str)
            else:
                due = due_str
            return datetime.utcnow() >= due
        except Exception:
            return False

    async def _should_update_information(self, user_id: int, topic: str) -> bool:
        try:
            memories = await self.storage_service.search_memories(user_id, f"info_update:{topic}", limit=1)
            if not memories:
                return True
            last_update = memories[0].get("created_at")
            if not last_update:
                return True
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)
            return (datetime.utcnow() - last_update).total_seconds() > 86400
        except Exception:
            return False

    def _safe_parse_json(self, text: str) -> Optional[Dict]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return None

    async def execute_decision(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            logger.info(f"Executing autonomous decision: {decision.action_type.value} for user {decision.target_user_id}")

            result = {"success": False, "error": "Unknown action type"}

            if decision.action_type in (
                AutonomousActionType.USER_CHECK,
                AutonomousActionType.CONVERSATION_INITIATION,
            ):
                result = await self._execute_send_message(decision)

            elif decision.action_type == AutonomousActionType.SCHEDULED_REMINDER:
                result = await self._execute_scheduled_reminder(decision)

            elif decision.action_type == AutonomousActionType.PROACTIVE_EMAIL:
                result = await self._execute_proactive_email(decision)

            elif decision.action_type == AutonomousActionType.INFORMATION_GATHERING:
                result = await self._execute_information_gathering(decision)

            elif decision.action_type == AutonomousActionType.UPDATE_BRIEFING:
                result = await self._execute_daily_briefing(decision)

            elif decision.action_type == AutonomousActionType.VOICE_CALL:
                result = await self._execute_voice_message(decision)

            decision.status = "completed" if result.get("success") else "failed"
            self.action_history.append(decision)
            return result

        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            decision.status = "failed"
            return {"success": False, "error": str(e)}

    async def _get_user_chat_id(self, user_id: int) -> Optional[str]:
        user = await self.storage_service.get_user_by_id(user_id)
        if not user:
            return None
        return user.get("telegram_id")

    async def _execute_send_message(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            if not self.telegram_service:
                return {"success": False, "error": "Telegram service not available"}

            chat_id = await self._get_user_chat_id(decision.target_user_id)
            if not chat_id:
                return {"success": False, "error": "User chat ID not found"}

            message = decision.action_data.get("message", "Hey, just checking in!")
            await self.telegram_service.send_message(chat_id=int(chat_id), text=message)

            await self.memory_manager.add_memory(
                user_id=decision.target_user_id,
                memory_type="autonomous_action",
                content=f"EVA proactively reached out: {message[:100]}",
                importance_score=3,
            )

            return {"success": True, "message": "Message sent"}

        except Exception as e:
            logger.error(f"Error sending proactive message: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_scheduled_reminder(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            if not self.telegram_service:
                return {"success": False, "error": "Telegram service not available"}

            chat_id = await self._get_user_chat_id(decision.target_user_id)
            if not chat_id:
                return {"success": False, "error": "User chat ID not found"}

            message = decision.action_data.get("message", "⏰ You have a reminder!")
            await self.telegram_service.send_message(chat_id=int(chat_id), text=message)

            reminder_id = decision.action_data.get("reminder_id")
            if reminder_id:
                await self.agent_manager.execute(
                    agent_name="reminder",
                    action="complete",
                    reminder_id=reminder_id,
                )

            return {"success": True, "message": "Reminder sent"}

        except Exception as e:
            logger.error(f"Error executing reminder: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_proactive_email(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            result = await self.agent_manager.execute(
                agent_name="email",
                action="send",
                **decision.action_data,
            )
            return result
        except Exception as e:
            logger.error(f"Error executing proactive email: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_information_gathering(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            topic = decision.action_data.get("topic")
            if not topic:
                return {"success": False, "error": "No topic specified"}

            result = await self.agent_manager.execute(
                agent_name="search",
                action="web",
                query=topic,
                num_results=5,
            )

            if result.get("success") and decision.target_user_id:
                summary = result.get("summary") or result.get("results", "")
                if summary:
                    await self.memory_manager.add_memory(
                        user_id=decision.target_user_id,
                        memory_type="info_update",
                        content=f"info_update:{topic} | {str(summary)[:500]}",
                        importance_score=4,
                    )

                    if self.telegram_service:
                        chat_id = await self._get_user_chat_id(decision.target_user_id)
                        if chat_id:
                            msg = f"📡 *Update on {topic}*\n\n{str(summary)[:600]}"
                            await self.telegram_service.send_message(
                                chat_id=int(chat_id), text=msg
                            )

            return result

        except Exception as e:
            logger.error(f"Error gathering information: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_daily_briefing(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            if not self.telegram_service:
                return {"success": False, "error": "Telegram service not available"}

            uid = decision.target_user_id
            chat_id = await self._get_user_chat_id(uid)
            if not chat_id:
                return {"success": False, "error": "User chat ID not found"}

            user = await self.storage_service.get_user_by_id(uid)
            name = (user or {}).get("first_name") or "there"

            reminders = await self.storage_service.get_active_reminders(uid)
            calendar_items = []
            try:
                cal_result = await self.agent_manager.execute(
                    agent_name="calendar",
                    action="list",
                    user_id=uid,
                    days_ahead=1,
                )
                calendar_items = cal_result.get("events", [])
            except Exception:
                pass

            memories = await self.memory_manager.get_relevant_memory(
                uid, "pending tasks goals", top_k=5
            )

            system_prompt = f"""You are EVA. Generate a concise, warm morning briefing for {name}.
Include what you know about their day based on the context below. Be natural, brief, and helpful.
Keep it under 200 words."""

            context = f"""Upcoming reminders: {[r.get('title') for r in reminders[:3]]}
Calendar events today: {[e.get('summary') for e in calendar_items[:3]]}
Recent context/goals: {[m.get('content', '')[:80] for m in memories[:3]]}
Current time: {datetime.utcnow().strftime('%A, %B %d')}"""

            briefing = await self.nvidia_service.call_chat_model(
                system_prompt=system_prompt,
                user_message=context,
            )

            await self.telegram_service.send_message(
                chat_id=int(chat_id),
                text=f"🌅 *Good morning, {name}!*\n\n{briefing}",
            )

            self._last_briefing[uid] = datetime.utcnow()
            return {"success": True, "message": "Briefing sent"}

        except Exception as e:
            logger.error(f"Error executing daily briefing: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_voice_message(self, decision: AutonomousDecision) -> Dict[str, Any]:
        try:
            if not self.telegram_service:
                return {"success": False, "error": "Telegram service not available"}

            chat_id = await self._get_user_chat_id(decision.target_user_id)
            if not chat_id:
                return {"success": False, "error": "User chat ID not found"}

            text = decision.action_data.get("message", "Hey, just wanted to check in!")
            audio_bytes = await self.nvidia_service.call_tts_model(text=text)
            await self.telegram_service.send_audio(
                chat_id=int(chat_id),
                audio_bytes=audio_bytes,
                caption=text[:100],
            )
            return {"success": True, "message": "Voice message sent"}

        except Exception as e:
            logger.error(f"Error sending voice message: {e}")
            return {"success": False, "error": str(e)}

    async def get_pending_decisions(self) -> List[AutonomousDecision]:
        pending = [d for d in self.decision_queue if d.status == "pending"]
        return sorted(pending, key=lambda x: x.priority, reverse=True)

    async def clear_completed_decisions(self):
        self.decision_queue = [d for d in self.decision_queue if d.status == "pending"]
