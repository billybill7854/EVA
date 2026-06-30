import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

from core.tool_contracts import validate_and_normalize_tool_call


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrchestratorDecision:
    def __init__(
        self,
        understanding: str,
        required_tools: List[str],
        tool_inputs: Dict[str, Any],
        plan: str,
        personality: str,
        requires_confirmation: bool = False,
        confidence: float = 0.8,
    ):
        self.understanding = understanding
        self.required_tools = required_tools
        self.tool_inputs = tool_inputs        # {tool_name: {action: ..., params: {...}}}
        self.plan = plan
        self.personality = personality
        self.requires_confirmation = requires_confirmation
        self.confidence = confidence
        self.tool_results: Dict[str, Any] = {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "understanding": self.understanding,
            "required_tools": self.required_tools,
            "plan": self.plan,
            "personality": self.personality,
            "requires_confirmation": self.requires_confirmation,
            "confidence": self.confidence,
            "tool_results": self.tool_results,
            "timestamp": self.timestamp.isoformat(),
        }


class Orchestrator:
    def __init__(self, nvidia_service, personality_engine, memory_manager, agent_manager=None, storage_service=None, settings=None):
        self.nvidia_service = nvidia_service
        self.personality_engine = personality_engine
        self.memory_manager = memory_manager
        self.agent_manager = agent_manager
        self.storage_service = storage_service
        self.settings = settings
        self.active_sessions = {}

    def set_agent_manager(self, agent_manager):
        self.agent_manager = agent_manager

    async def process_request(
        self,
        user_message: str,
        user_id: int,
        user_data: Dict[str, Any],
        is_primary_user: bool = True,
        allowed_tools: List[str] = None,
    ) -> OrchestratorDecision:
        logger.info(f"Processing request from user {user_id}: {user_message[:80]}")

        try:
            personality = await self.personality_engine.detect_personality(
                user_message, user_id, user_data
            )
            logger.info(f"Detected personality: {personality}")

            memory_context = await self.memory_manager.get_relevant_memory(
                user_id, user_message, top_k=5
            )
            logger.info(f"Retrieved {len(memory_context)} relevant memories")

            decision = await self._plan_with_model(
                user_message=user_message,
                personality=personality,
                memory_context=memory_context,
                user_data=user_data,
                is_primary_user=is_primary_user,
            )

            if not is_primary_user:
                decision = self._restrict_stranger_mode(decision)

            # Apply allowed_tools filter before execution
            if allowed_tools is not None:
                decision.required_tools = [t for t in decision.required_tools if t in allowed_tools]
                decision.tool_inputs = {k: v for k, v in decision.tool_inputs.items() if k in allowed_tools}

            # Robustness: enforce tool budget and validate tool call schemas
            decision = self._apply_tool_budget(decision)
            decision = await self._validate_tool_calls(decision, user_id=user_id)

            # Execute tools immediately and collect results
            if decision.required_tools and self.agent_manager:
                await self._execute_tools(decision, user_id, user_data)

            session_id = f"{user_id}_{datetime.utcnow().timestamp()}"
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "decision": decision,
                "status": ExecutionStatus.SUCCESS,
                "created_at": datetime.utcnow(),
            }

            return decision

        except Exception as e:
            logger.error(f"Error in orchestration: {e}", exc_info=True)
            raise

    async def _plan_with_model(
        self,
        user_message: str,
        personality: str,
        memory_context: List[Dict[str, Any]],
        user_data: Dict[str, Any],
        is_primary_user: bool,
    ) -> OrchestratorDecision:
        system_prompt = self._build_planner_prompt(is_primary_user)
        user_context = self._build_context_prompt(user_message, memory_context, user_data)

        response = await self.nvidia_service.call_orchestrator_model(
            system_prompt=system_prompt,
            user_message=user_context,
        )

        return await self._parse_plan_response_with_repair(
            response=response,
            personality=personality,
            system_prompt=system_prompt,
            user_context=user_context,
        )

    async def _parse_plan_response_with_repair(
        self,
        response: str,
        personality: str,
        system_prompt: str,
        user_context: str,
    ) -> OrchestratorDecision:
        decision = self._parse_plan_response(response, personality)
        attempts = 0
        if self.settings and getattr(self.settings, "orchestrator_json_repair_attempts", None) is not None:
            attempts = int(self.settings.orchestrator_json_repair_attempts)

        if attempts <= 0:
            return decision

        # If the model produced non-JSON or empty/malformed tool plan, try a single constrained repair.
        should_repair = (
            (not decision.understanding)
            or (decision.confidence < 0.6 and "General conversation" in decision.understanding)
        )
        if not should_repair:
            return decision

        repair_prompt = (
            "You previously responded with malformed JSON. "
            "Return ONLY valid JSON matching the required schema. "
            "Do not include any prose, markdown, or code fences. "
            "If no tool is needed, set required_tools to [] and tool_inputs to {}."
        )
        try:
            repaired = await self.nvidia_service.call_orchestrator_model(
                system_prompt=f"{system_prompt}\n\n{repair_prompt}",
                user_message=user_context,
            )
            return self._parse_plan_response(repaired, personality)
        except Exception as e:
            logger.warning(f"Plan repair attempt failed: {e}")
            return decision

    def _build_planner_prompt(self, is_primary_user: bool) -> str:
        prompt = """You are EVA's planning brain. Your job is to read what the user wants and decide which tools to use and exactly how to call them.

Respond ONLY with valid JSON in this exact format:
{
  "understanding": "what the user wants in one sentence",
  "required_tools": ["tool1"],
  "tool_inputs": {
    "tool1": {
      "action": "action_name",
      "params": { "param1": "value1" }
    }
  },
  "plan": "brief step description",
  "requires_confirmation": false,
  "confidence": 0.9
}

Available tools and their actions (use these exact action names):
	- telegram: send_message(phone_or_username?, chat_id?, message), find_contact(query), get_dialogs(limit?), get_history(chat_id, limit?), search_messages(chat_id, query), get_contact_info(user_id), send_voice(phone_or_username?, chat_id?, text), send_audio(phone_or_username?, chat_id?, text, caption?)
- email: send(to, subject, body, attachments?), read(inbox?, limit?), search(query, sender?)
- calendar: create(title, date, time, duration?, description?, attendees?), list(date?, days_ahead?), update(event_id, ...fields_to_update), delete(event_id)
- reminder: set(title, date, time, description?, priority?), list(status?), update(reminder_id, ...fields_to_update), complete(reminder_id), delete(reminder_id)
- search: web(query, num_results?, language?), news(topic, num_results?), history(limit?)
- image: generate(prompt, width?, height?), edit(image_id, instruction), upscale(image_id)

Hard rules:
- Never invent tools outside this list.
- Never call more than 3 tools in a single request.
- If unsure about a required parameter, set "requires_confirmation": true and ask for the missing info instead of guessing.

If no tool is needed (just conversation), return: "required_tools": [], "tool_inputs": {}

Examples:
User: "send a message to Kelly (+254113755206) saying we meet tomorrow at 11:30"
→ tool: telegram, action: send_message, params: {"phone_or_username": "+254113755206", "message": "Hi Kelly, Felix says we need to meet tomorrow at 11:30 AM."}

User: "remind me to call mum at 5pm"
→ tool: reminder, action: set, params: {"title": "Call mum", "date": "today", "time": "17:00"}"""

        if not is_primary_user:
            prompt += "\n\nSTRANGER MODE: Only allow search and image tools. No messaging, email, payment, or calendar actions."

        return prompt

    def _build_context_prompt(
        self,
        user_message: str,
        memory_context: List[Dict[str, Any]],
        user_data: Dict[str, Any],
    ) -> str:
        context = f"User message: {user_message}\n"
        if memory_context:
            context += "\nRelevant memories:\n"
            for m in memory_context:
                context += f"  - {m.get('memory_type', '')}: {m.get('content', '')[:100]}\n"
        context += f"\nUser name: {user_data.get('first_name', 'User')}"
        context += f"\nCurrent time: {datetime.utcnow().strftime('%A %d %B %Y %H:%M UTC')}"
        return context

    def _parse_plan_response(self, response: str, personality: str) -> OrchestratorDecision:
        try:
            # Extract JSON even if model adds surrounding text
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
            else:
                raise ValueError("No JSON found")

            return OrchestratorDecision(
                understanding=data.get("understanding", ""),
                required_tools=data.get("required_tools", []),
                tool_inputs=data.get("tool_inputs", {}),
                plan=data.get("plan", ""),
                personality=personality,
                requires_confirmation=data.get("requires_confirmation", False),
                confidence=float(data.get("confidence", 0.8)),
            )
        except Exception as e:
            logger.warning(f"Could not parse orchestrator plan JSON: {e}. Response: {response[:200]}")
            return OrchestratorDecision(
                understanding="General conversation",
                required_tools=[],
                tool_inputs={},
                plan="Respond conversationally",
                personality=personality,
                requires_confirmation=False,
                confidence=0.5,
            )

    def _apply_tool_budget(self, decision: OrchestratorDecision) -> OrchestratorDecision:
        max_tools = 3
        if self.settings and getattr(self.settings, "tool_max_calls_per_request", None):
            max_tools = int(self.settings.tool_max_calls_per_request)
        if len(decision.required_tools) > max_tools:
            decision.required_tools = decision.required_tools[:max_tools]
            decision.tool_inputs = {k: v for k, v in decision.tool_inputs.items() if k in decision.required_tools}
        return decision

    async def _validate_tool_calls(self, decision: OrchestratorDecision, user_id: int) -> OrchestratorDecision:
        """
        Remove/mark invalid tool calls early (schema mismatch, missing params).
        Logs validation failures as tool calls (status=failed) when storage is available.
        """
        valid_tools: List[str] = []
        normalized_inputs: Dict[str, Any] = {}

        for tool in decision.required_tools:
            spec = decision.tool_inputs.get(tool) or {}
            action = spec.get("action") or "execute"
            params = spec.get("params") or {}
            ok, normalized, err = validate_and_normalize_tool_call(tool, action, params)
            if not ok:
                logger.warning(f"Invalid tool call planned: {tool}.{action} err={err}")
                if self.storage_service:
                    tc_id = await self.storage_service.log_tool_call(
                        user_id=user_id, tool_name=tool, action=action, input_data={"params": params, "validation_error": err}
                    )
                    if tc_id:
                        await self.storage_service.update_tool_call_result(
                            tool_call_id=tc_id, status="failed", output_data=None, error_message=f"validation_error: {err}"
                        )
                continue
            valid_tools.append(tool)
            normalized_inputs[tool] = {"action": action, "params": normalized}

        decision.required_tools = valid_tools
        decision.tool_inputs = normalized_inputs
        return decision

    async def _execute_tools(self, decision: OrchestratorDecision, user_id: int, user_data: Dict):
        for tool_name in decision.required_tools:
            tool_spec = decision.tool_inputs.get(tool_name, {})
            action = tool_spec.get("action", "execute")
            params = tool_spec.get("params", {})

            logger.info(f"Executing tool: {tool_name}.{action} params={params}")
            try:
                tool_call_id = None
                if self.storage_service:
                    tool_call_id = await self.storage_service.log_tool_call(
                        user_id=user_id, tool_name=tool_name, action=action, input_data=params
                    )

                timeout_s = 20
                if self.settings and getattr(self.settings, "tool_timeout_seconds", None):
                    timeout_s = int(self.settings.tool_timeout_seconds)

                result = await asyncio.wait_for(
                    self.agent_manager.execute(
                        agent_name=tool_name,
                        action=action,
                        user_id=user_id,
                        **params,
                    ),
                    timeout=timeout_s,
                )
                decision.tool_results[tool_name] = result
                logger.info(f"Tool {tool_name} result: success={result.get('success')}")
                if tool_call_id and self.storage_service:
                    status = "success" if result.get("success") else "failed"
                    await self.storage_service.update_tool_call_result(
                        tool_call_id=tool_call_id,
                        status=status,
                        output_data=result if isinstance(result, dict) else {"result": str(result)},
                        error_message=None if result.get("success") else (result.get("error") or "tool_failed"),
                    )
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                decision.tool_results[tool_name] = {"success": False, "error": str(e)}

    def _restrict_stranger_mode(self, decision: OrchestratorDecision) -> OrchestratorDecision:
        allowed = ["search", "image"]
        decision.required_tools = [t for t in decision.required_tools if t in allowed]
        decision.tool_inputs = {k: v for k, v in decision.tool_inputs.items() if k in allowed}
        decision.personality = "general"
        return decision

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.active_sessions.get(session_id)
