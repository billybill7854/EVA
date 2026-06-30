import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    def __init__(self, autonomous_engine, storage_service, telegram_service):
        self.autonomous_engine = autonomous_engine
        self.storage_service = storage_service
        self.telegram_service = telegram_service

        self.is_running = False
        self._think_task = None
        self._execute_task = None
        self._health_task = None

        self.thinking_interval = 300      # 5 minutes
        self.execution_interval = 30      # 30 seconds
        self.health_check_interval = 3600 # 1 hour

        logger.info("Background Scheduler initialized")

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting background scheduler...")

        # Inject telegram service into autonomous engine so it can send messages
        self.autonomous_engine.set_telegram_service(self.telegram_service)

        # Run thinking and execution as separate independent loops
        self._think_task = asyncio.create_task(self._thinking_loop())
        self._execute_task = asyncio.create_task(self._execution_loop())
        self._health_task = asyncio.create_task(self._health_loop())

        logger.info("Background scheduler started — EVA is now autonomous")

    async def stop(self):
        if not self.is_running:
            return

        logger.info("Stopping background scheduler...")
        self.is_running = False

        for task in (self._think_task, self._execute_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Background scheduler stopped")

    async def _thinking_loop(self):
        # Initial short delay to let app fully start
        await asyncio.sleep(10)

        while self.is_running:
            try:
                decisions = await self.autonomous_engine.think()
                if decisions:
                    logger.info(f"EVA thought up {len(decisions)} actions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in thinking loop: {e}")

            await asyncio.sleep(self.thinking_interval)

    async def _execution_loop(self):
        # Wait a moment before starting execution checks
        await asyncio.sleep(15)

        while self.is_running:
            try:
                await self._execute_pending_decisions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")

            await asyncio.sleep(self.execution_interval)

    async def _health_loop(self):
        await asyncio.sleep(60)

        while self.is_running:
            try:
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health loop: {e}")

            await asyncio.sleep(self.health_check_interval)

    async def _execute_pending_decisions(self):
        pending = await self.autonomous_engine.get_pending_decisions()
        if not pending:
            return

        now = datetime.utcnow()
        executed = 0

        for decision in pending:
            if now >= decision.scheduled_time:
                try:
                    result = await self.autonomous_engine.execute_decision(decision)
                    executed += 1

                    if not result.get("success"):
                        logger.warning(
                            f"Autonomous action '{decision.action_type.value}' failed: {result.get('error')}"
                        )
                except Exception as e:
                    logger.error(f"Error executing decision {decision.action_type.value}: {e}")

        if executed:
            logger.info(f"Executed {executed} autonomous decisions")
            await self.autonomous_engine.clear_completed_decisions()

    async def _run_health_checks(self):
        try:
            db_healthy = await self.storage_service.health_check()
            logger.info(f"Health check — db: {db_healthy}")
        except Exception as e:
            logger.error(f"Health check error: {e}")

    async def schedule_custom_action(
        self,
        action_type: str,
        action_data: Dict[str, Any],
        scheduled_time: datetime,
        user_id: Optional[int] = None,
        priority: int = 5,
    ) -> Dict[str, Any]:
        try:
            from core.autonomous_engine import AutonomousDecision, AutonomousActionType

            decision = AutonomousDecision(
                action_type=AutonomousActionType(action_type),
                priority=priority,
                reasoning="Custom scheduled action",
                action_data=action_data,
                target_user_id=user_id,
                scheduled_time=scheduled_time,
            )
            self.autonomous_engine.decision_queue.append(decision)
            logger.info(f"Scheduled custom action: {action_type} at {scheduled_time}")
            return {"success": True}
        except Exception as e:
            logger.error(f"Error scheduling custom action: {e}")
            return {"success": False, "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        pending = await self.autonomous_engine.get_pending_decisions()
        return {
            "is_running": self.is_running,
            "pending_decisions": len(pending),
            "action_history": len(self.autonomous_engine.action_history),
            "thinking_interval_seconds": self.thinking_interval,
            "execution_interval_seconds": self.execution_interval,
        }
